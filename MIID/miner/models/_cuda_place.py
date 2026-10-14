"""Shared CUDA placement for model_testing scripts (CPU offload vs full GPU)."""

from __future__ import annotations

import os
import sys
from typing import Any

# Cards at or below this get sequential (per-layer) offload automatically, because
# model offload moves a whole submodule at a time and a 4B bf16 transformer is ~8 GB
# on its own — more than such a card has.
SMALL_VRAM_GIB = float(os.environ.get("MIID_SMALL_VRAM_GIB", "12"))

# How weights move between CPU memory and the GPU when offload is on:
#   mapped     one model on the GPU at a time, uploaded from the memory-mapped
#              safetensors and evicted without copying back (default; the
#              process keeps ~1.5 GB private memory for FLUX.2 klein instead
#              of 15 GB, and sibling processes share the page cache). Same
#              peak VRAM as model offload.
#   model      diffusers' enable_model_cpu_offload: whole models, copied back
#              to fresh CPU memory after each generation.
#   sequential diffusers' enable_sequential_cpu_offload: layer by layer, the
#              lowest VRAM and the slowest (the transformer is re-uploaded
#              every denoising step).
OFFLOAD_MODES = ("mapped", "model", "sequential")


def _total_vram_gib() -> float:
    """Total VRAM of cuda:0 in GiB, or 0.0 when it cannot be determined."""
    try:
        import torch

        return torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    except Exception:  # noqa: BLE001
        return 0.0


def offload_mode() -> str:
    """The offload flavour for CUDA: ``MIID_OFFLOAD_MODE``, else the older
    ``MIID_SEQUENTIAL_CPU_OFFLOAD`` switch (1 → sequential, 0 → model), else mapped."""
    mode = os.environ.get("MIID_OFFLOAD_MODE", "").strip().lower()
    if mode in OFFLOAD_MODES:
        return mode
    seq_env = os.environ.get("MIID_SEQUENTIAL_CPU_OFFLOAD", "").strip().lower()
    if seq_env in ("1", "true", "yes"):
        return "sequential"
    if seq_env in ("0", "false", "no"):
        return "model"
    return "mapped"


def offload_wanted(default_offload_on_cuda: bool) -> bool:
    env = os.environ.get("MIID_ENABLE_CPU_OFFLOAD", "").strip().lower()
    if default_offload_on_cuda:
        return env not in ("0", "false", "no")
    return env in ("1", "true", "yes")


def release_offloaded(pipe: Any) -> None:
    """After a generation: move every mapped-offload model off the GPU.

    No-op for the other modes (their hooks already left the GPU empty) and for
    pipelines that run fully on one device.
    """
    if getattr(pipe, "_miid_mapped_offload", None) is None:
        return
    from ._mapped_offload import release_offloaded as _release

    _release(pipe)


def place_diffusers_pipeline(
    pipe: Any,
    dev: str,
    *,
    default_offload_on_cuda: bool,
    prefer_sequential_offload: bool = False,
) -> None:
    """Move pipeline to device, or enable CPU offload on CUDA when requested.

    - If ``default_offload_on_cuda`` is True and ``dev == "cuda"``, offload is used unless
      ``MIID_ENABLE_CPU_OFFLOAD`` is ``0`` / ``false`` / ``no``.
    - If ``default_offload_on_cuda`` is False, offload is used only when
      ``MIID_ENABLE_CPU_OFFLOAD`` is ``1`` / ``true`` / ``yes``.
    - ``MIID_OFFLOAD_MODE`` picks mapped (default), model or sequential offload; see
      ``OFFLOAD_MODES``. Huge models (``prefer_sequential_offload=True``) and cards below
      ``SMALL_VRAM_GIB``, where a whole transformer does not fit, use sequential offload
      unless the mode was set explicitly.
    """
    if dev != "cuda":
        pipe.to(dev)
        return

    if not offload_wanted(default_offload_on_cuda):
        pipe.to(dev)
        return

    mode = offload_mode()
    explicit = bool(os.environ.get("MIID_OFFLOAD_MODE", "").strip()) or bool(
        os.environ.get("MIID_SEQUENTIAL_CPU_OFFLOAD", "").strip()
    )
    if not explicit:
        vram = _total_vram_gib()
        small_card = 0.0 < vram < SMALL_VRAM_GIB
        if small_card:
            print(
                f"cuda:0 has {vram:.1f} GiB VRAM (< {SMALL_VRAM_GIB:.0f} GiB); using "
                "sequential CPU offload. Generation will be slower — set "
                "MIID_OFFLOAD_MODE=mapped to opt out.",
                file=sys.stderr,
            )
        if small_card or prefer_sequential_offload:
            mode = "sequential"

    if mode == "mapped":
        try:
            from ._mapped_offload import enable_mapped_cpu_offload

            manager = enable_mapped_cpu_offload(pipe, dev)
            print(f"CPU offload: {manager.describe()}", file=sys.stderr)
            return
        except Exception as exc:  # noqa: BLE001
            print(
                f"mapped CPU offload failed ({exc}); trying sequential CPU offload",
                file=sys.stderr,
            )
            mode = "sequential"

    if mode == "sequential":
        try:
            pipe.enable_sequential_cpu_offload()
            print("CPU offload: sequential (layer by layer)", file=sys.stderr)
            return
        except Exception as exc:  # noqa: BLE001
            print(
                f"enable_sequential_cpu_offload failed ({exc}); trying model CPU offload",
                file=sys.stderr,
            )

    try:
        pipe.enable_model_cpu_offload()
        print("CPU offload: model (whole models, copied back to the CPU after each run)", file=sys.stderr)
        return
    except Exception as exc:  # noqa: BLE001
        print(f"enable_model_cpu_offload failed ({exc}); falling back to pipe.to(cuda)", file=sys.stderr)
    pipe.to(dev)
