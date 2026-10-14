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

This module does what model offload does — one upload per model per
generation, one model resident on the GPU at a time — but uploads from the
mapped tensors and evicts to the meta device, never copying back. The
process keeps about 1.5 GB of private memory for the pipeline instead of
15 GB, and sibling miner processes share one copy of the weights in the
page cache.

Usage (see ``_cuda_place.place_diffusers_pipeline``)::

    manager = enable_mapped_cpu_offload(pipe, "cuda")
    try:
        pipe(...)
    finally:
        release_offloaded(pipe)   # every model back off the GPU
"""

from __future__ import annotations

import collections
import time
from typing import Any, Dict, List, Optional

import torch
from accelerate.hooks import ModelHook, add_hook_to_module
from accelerate.utils import send_to_device, set_module_tensor_to_device

ATTR = "_miid_mapped_offload"


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
    """Keeps one component on the GPU at a time, uploading from mapped weights."""

    def __init__(self, pipe: Any, device: torch.device):
        self.device = device
        self.states: List[_ModelState] = []
        self.uploads = 0
        self.upload_seconds = 0.0
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

    def activate(self, state: _ModelState) -> None:
        for other in self.states:
            if other is not state and other.loaded:
                other.evict()
        if state.loaded:
            return
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
                evicted = True
        if evicted and self.device.type == "cuda":
            torch.cuda.empty_cache()

    def describe(self) -> str:
        parts = ", ".join(f"{s.name} {s.bytes / 1e9:.1f} GB" for s in self.states)
        return f"mapped CPU offload on {self.device} ({parts}): one model on the GPU at a time"


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
