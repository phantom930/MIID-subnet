"""Shared CUDA placement for model_testing scripts (CPU offload vs full GPU)."""

from __future__ import annotations

import os
import sys
from typing import Any

# Cards at or below this get sequential (per-layer) offload automatically, because
# model offload moves a whole submodule at a time and a 4B bf16 transformer is ~8 GB
# on its own — more than such a card has.
SMALL_VRAM_GIB = float(os.environ.get("MIID_SMALL_VRAM_GIB", "12"))


def _total_vram_gib() -> float:
    """Total VRAM of cuda:0 in GiB, or 0.0 when it cannot be determined."""
    try:
        import torch

        return torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    except Exception:  # noqa: BLE001
        return 0.0


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
    - For huge models (e.g. FLUX Kontext), ``prefer_sequential_offload=True`` tries
      ``enable_sequential_cpu_offload()`` first (lower peak VRAM than model offload on ~16GB).
      Sequential offload is also chosen automatically on cards below ``SMALL_VRAM_GIB``,
      where model offload cannot fit a single transformer submodule.
      Force with ``MIID_SEQUENTIAL_CPU_OFFLOAD=1``, disable with ``0``.
    """
    if dev != "cuda":
        pipe.to(dev)
        return

    env = os.environ.get("MIID_ENABLE_CPU_OFFLOAD", "").strip().lower()
    if default_offload_on_cuda:
        want_offload = env not in ("0", "false", "no")
    else:
        want_offload = env in ("1", "true", "yes")

    if not want_offload:
        pipe.to(dev)
        return

    seq_env = os.environ.get("MIID_SEQUENTIAL_CPU_OFFLOAD", "").strip().lower()
    if seq_env in ("0", "false", "no"):
        use_sequential = False
    elif seq_env in ("1", "true", "yes"):
        use_sequential = True
    else:
        vram = _total_vram_gib()
        small_card = 0.0 < vram < SMALL_VRAM_GIB
        if small_card and not prefer_sequential_offload:
            print(
                f"cuda:0 has {vram:.1f} GiB VRAM (< {SMALL_VRAM_GIB:.0f} GiB); using "
                "sequential CPU offload. Generation will be slower — set "
                "MIID_SEQUENTIAL_CPU_OFFLOAD=0 to opt out.",
                file=sys.stderr,
            )
        use_sequential = prefer_sequential_offload or small_card

    if use_sequential:
        try:
            pipe.enable_sequential_cpu_offload()
            return
        except Exception as exc:  # noqa: BLE001
            print(
                f"enable_sequential_cpu_offload failed ({exc}); trying model CPU offload",
                file=sys.stderr,
            )

    try:
        pipe.enable_model_cpu_offload()
        return
    except Exception as exc:  # noqa: BLE001
        print(f"enable_model_cpu_offload failed ({exc}); falling back to pipe.to(cuda)", file=sys.stderr)
    pipe.to(dev)
