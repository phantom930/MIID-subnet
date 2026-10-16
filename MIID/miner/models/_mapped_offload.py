"""CPU offload that keeps the weights where diffusers loaded them.

diffusers loads safetensors weights as memory-mapped tensors: the bytes stay
in the kernel's page cache, shared by every process that opens the same
file, and the process itself owns almost nothing. Both built-in offload
modes then lose that during generation:

* ``enable_model_cpu_offload`` moves whole models with ``module.to()``. The
  move back to the CPU after each model's turn (and the pipeline re-arming
  the hooks after every call) allocates fresh private memory for every
  weight — 15 GB per generation for FLUX.2 klein — so each miner process
  ends up with its own private copy, and the churn fragments the heap:
  measured on 2026-10-14, four miners on one box went from 40 GB to 55 GB
  RSS each in four hours.
* ``enable_sequential_cpu_offload`` keeps the mapping but re-uploads the
  transformer on every denoising step.

This module does what model offload does — uploads a model when its forward
runs — but uploads from the mapped tensors and evicts to the meta device,
never copying back. The process keeps about 1.5 GB of private memory for the
pipeline instead of 15 GB, and sibling miner processes share one copy of the
weights in the page cache.

Models stay on the GPU as long as they fit a budget (RESIDENT_BYTES,
``MIID_RESIDENT_GB``); when one does not fit, the largest resident model
goes first.
FLUX.2 klein's transformer (7.8 GB) and VAE (0.2 GB) fit together; its text
encoder (8.0 GB) does not fit beside the transformer, so a generation that
needs the text encoder swaps the two. One that is handed its prompt
embeddings (see generate_variations.PromptCache) touches only the
transformer and the VAE, and with both resident from the previous attempt
uploads nothing at all. Eviction otherwise happens when the process hands
the GPU to another one (``release_offloaded`` from the GPU scheduler's
release hook) and before the voice clone.

Usage (see ``_cuda_place.place_diffusers_pipeline``)::

    manager = enable_mapped_cpu_offload(pipe, "cuda")
    pipe(...)                 # uploads what it needs, keeps it
    release_offloaded(pipe)   # every model back off the GPU
"""

from __future__ import annotations

import collections
import itertools
import os
import time
from typing import Any, Dict, List, Optional

import torch
from accelerate.hooks import ModelHook, add_hook_to_module
from accelerate.utils import send_to_device, set_module_tensor_to_device

ATTR = "_miid_mapped_offload"
# Weights kept on the GPU at once. 9 GB holds the klein transformer and VAE
# (8.0 GB) with room for the denoising activations (~1.5 GB) beside the idle
# footprints of five sibling processes (~0.5 GB each) on a 16 GB card.
RESIDENT_BYTES = int(float(os.environ.get("MIID_RESIDENT_GB", "9")) * 1e9)


class _ModelState:
    """One pipeline component: its CPU weights and whether it is on the GPU."""

    def __init__(self, name: str, module: torch.nn.Module, device: torch.device):
        self.name = name
        self.module = module
        self.device = device
        self.loaded = False
        self.restored = False
        # state_dict() hands back the parameters' own tensors, so these are
        # the mapped tensors diffusers loaded, not copies.
        self.weights: Dict[str, torch.Tensor] = dict(module.state_dict())
        params = [n for n, _ in module.named_parameters(remove_duplicate=False)]
        buffers = [n for n, _ in module.named_buffers(remove_duplicate=False)]
        missing = [n for n in params if n not in self.weights]
        if missing:
            raise RuntimeError(f"{name}: parameters absent from state_dict: {missing[:3]}")
        # Only parameters travel. Buffers are small (batch-norm statistics,
        # rotary tables) and pipelines read some of them directly rather than
        # through a forward — Flux2KleinPipeline takes vae.bn.running_mean —
        # so they live on the GPU for the life of the hook.
        self.managed: List[str] = params
        self.resident: List[str] = buffers
        # Tied weights (Qwen3's lm_head shares embed_tokens) share one upload.
        counts = collections.Counter(self.weights[n].data_ptr() for n in self.managed)
        self.tied: Dict[int, Dict[Any, torch.Tensor]] = {
            ptr: {} for ptr, c in counts.items() if c > 1 and ptr != 0
        }
        self.bytes = sum(
            t.numel() * t.element_size()
            for t in {self.weights[n].data_ptr(): self.weights[n] for n in self.managed}.values()
        )

    def init(self) -> None:
        for name in self.managed:
            set_module_tensor_to_device(self.module, name, "meta")
        for name in self.resident:
            set_module_tensor_to_device(self.module, name, self.device, clear_cache=False)

    def upload(self) -> None:
        for name in self.managed:
            set_module_tensor_to_device(
                self.module, name, self.device, value=self.weights[name],
                tied_params_map=self.tied, clear_cache=False,
            )
        self.loaded = True

    def evict(self) -> None:
        for name in self.managed:
            set_module_tensor_to_device(self.module, name, "meta")
        for per_device in self.tied.values():
            per_device.clear()
        self.loaded = False

    def restore(self) -> None:
        """Put the mapped CPU tensors back on the module (hook removal)."""
        if self.restored:
            return
        if self.loaded:
            self.evict()
        for name in self.managed:
            set_module_tensor_to_device(self.module, name, "cpu", value=self.weights[name])
        for name in self.resident:
            set_module_tensor_to_device(self.module, name, "cpu")
        self.restored = True


class _EntryHook(ModelHook):
    """Fires on the component's forward (or an ``apply_forward_hook`` method)."""

    no_grad = False

    def __init__(self, state: _ModelState, manager: "MappedOffload"):
        self.state = state
        self.manager = manager
        # diffusers reads this to know where a component's inputs must be.
        self.execution_device = state.device

    def pre_forward(self, module, *args, **kwargs):
        self.manager.activate(self.state)
        return send_to_device(args, self.execution_device), send_to_device(kwargs, self.execution_device)

    def post_forward(self, module, output):
        return output

    def detach_hook(self, module):
        self.state.restore()
        return module


class MappedOffload:
    """Keeps components on the GPU within RESIDENT_BYTES, uploading from mapped weights."""

    def __init__(self, pipe: Any, device: torch.device, budget: Optional[int] = None):
        self.device = device
        self.budget = RESIDENT_BYTES if budget is None else budget
        self.states: List[_ModelState] = []
        self.uploads = 0
        self.upload_seconds = 0.0
        self.evictions = 0
        self._clock = itertools.count()
        self._last_used: Dict[str, int] = {}
        components = {
            k: v for k, v in pipe.components.items() if isinstance(v, torch.nn.Module)
        }
        order = [n for n in (getattr(pipe, "model_cpu_offload_seq", None) or "").split("->") if n]
        names = [n for n in order if n in components] + [n for n in components if n not in order]
        exclude = set(getattr(pipe, "_exclude_from_cpu_offload", None) or [])
        for name in names:
            module = components[name]
            if name in exclude:
                module.to(device)
                continue
            state = _ModelState(name, module, device)
            state.init()
            add_hook_to_module(module, _EntryHook(state, self))
            self.states.append(state)

    def loaded_bytes(self) -> int:
        return sum(s.bytes for s in self.states if s.loaded)

    def loaded(self) -> List[str]:
        return [s.name for s in self.states if s.loaded]

    def activate(self, state: _ModelState) -> None:
        """Make ``state`` resident, evicting others — the largest first, the
        least recently used among equals — only as far as the budget requires.

        Largest first keeps the small models (the VAE) resident through a
        swap of the two big ones; least-recently-used alone would throw the
        VAE out first and then the transformer anyway.
        """
        self._last_used[state.name] = next(self._clock)
        if state.loaded:
            return
        others = sorted(
            (s for s in self.states if s is not state and s.loaded),
            key=lambda s: (-s.bytes, self._last_used.get(s.name, -1)),
        )
        freed = False
        for other in others:
            if self.loaded_bytes() + state.bytes <= self.budget:
                break
            other.evict()
            self.evictions += 1
            freed = True
        if freed and self.device.type == "cuda":
            torch.cuda.empty_cache()
        started = time.time()
        try:
            state.upload()
        except Exception:
            state.evict()
            raise
        self.upload_seconds += time.time() - started
        self.uploads += 1

    def evict_all(self) -> None:
        evicted = False
        for state in self.states:
            if state.loaded:
                state.evict()
                self.evictions += 1
                evicted = True
        if evicted and self.device.type == "cuda":
            torch.cuda.empty_cache()

    def describe(self) -> str:
        parts = ", ".join(f"{s.name} {s.bytes / 1e9:.1f} GB" for s in self.states)
        return (
            f"mapped CPU offload on {self.device} ({parts}): up to "
            f"{self.budget / 1e9:.0f} GB of weights stay resident, the largest evicted first"
        )


def enable_mapped_cpu_offload(pipe: Any, device: str = "cuda") -> MappedOffload:
    """Replace any offload hooks on ``pipe`` with mapped offload to ``device``."""
    dev = torch.device(device)
    if dev.type == "cuda" and dev.index is None:
        dev = torch.device("cuda", torch.cuda.current_device())
    if hasattr(pipe, "remove_all_hooks"):
        pipe.remove_all_hooks()
    manager = MappedOffload(pipe, dev)
    setattr(pipe, ATTR, manager)
    return manager


def release_offloaded(pipe: Any) -> None:
    """Move every mapped-offload component off the GPU (no-op otherwise)."""
    manager: Optional[MappedOffload] = getattr(pipe, ATTR, None)
    if manager is not None:
        manager.evict_all()
