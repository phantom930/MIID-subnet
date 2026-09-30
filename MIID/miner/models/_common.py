#!/usr/bin/env python3
"""Shared helpers for the pipelines in this package.

Three things every backend needs and none of them should reimplement:

* ``supported_kwargs`` — diffusers pipelines disagree about which of
  ``width`` / ``height`` / ``generator`` / ``negative_prompt`` they accept, and
  an unexpected keyword is a hard ``TypeError``. Filtering against the real
  ``__call__`` signature lets one call site serve every backend, and lets a
  diffusers upgrade add support without another edit here.

* ``generation_size`` / ``fit_to_target`` — the validator asks for a
  passport-style 3:4 portrait at a fixed resolution (see
  ``IMAGE_VARIATION_REQUIREMENTS`` in MIID/validator/image_variations.py, which
  is appended to every request's ``detail``). Models return whatever their own
  defaults produce — usually square — so generation is asked for the nearest
  latent-friendly 3:4 size and the result is cropped and resized to exactly
  what was requested. The grading API scores compliance, so an off-aspect
  image loses points on every variation.

* ``make_generator`` — a per-attempt seed, so an identity retry actually
  re-rolls instead of reproducing the attempt that already failed.
"""

import inspect
from typing import Any, Dict, Optional, Tuple

from PIL import Image

# Latent stride: diffusion pipelines want both sides on a multiple of this, and
# silently round otherwise (which reintroduces the aspect error we are fixing).
LATENT_MULTIPLE = 16


def supported_kwargs(pipe: Any, **kwargs: Any) -> Dict[str, Any]:
    """Keep only the keywords this pipeline's ``__call__`` actually names.

    ``None`` values are dropped as well, so a caller can pass every optional
    knob unconditionally and let the pipeline decide what it understands.
    """
    try:
        params = inspect.signature(pipe.__call__).parameters
    except (TypeError, ValueError):  # C-level or wrapped callable
        return {}

    return {
        key: value
        for key, value in kwargs.items()
        if value is not None
        and key in params
        and params[key].kind is not inspect.Parameter.VAR_KEYWORD
    }


def make_generator(seed: Optional[int]):
    """A CPU torch generator for ``seed``, or None to leave sampling random.

    CPU rather than device-local: it is the portable choice across the CUDA,
    MPS and offloaded-pipeline paths this miner runs on.
    """
    if seed is None:
        return None
    import torch

    return torch.Generator("cpu").manual_seed(int(seed))


def _snap(value: int, multiple: int = LATENT_MULTIPLE) -> int:
    return max(multiple, int(round(value / multiple)) * multiple)


def generation_size(
    width: Optional[int], height: Optional[int]
) -> Tuple[Optional[int], Optional[int]]:
    """Latent-friendly size to generate at for a requested output size.

    The requested 1015x1350 is not a multiple of 16 in either axis, so it is
    generated at 1008x1344 and resized on the way out by ``fit_to_target``.
    """
    if not width or not height:
        return None, None
    return _snap(width), _snap(height)


def fit_to_target(
    image: Image.Image, width: Optional[int], height: Optional[int]
) -> Image.Image:
    """Center-crop to the target aspect ratio, then resize to exactly w x h."""
    if not width or not height:
        return image
    if image.size == (width, height):
        return image

    src_w, src_h = image.size
    if not src_w or not src_h:
        return image

    target_ratio = width / height
    src_ratio = src_w / src_h

    if src_ratio > target_ratio:
        # Too wide — trim the sides evenly.
        new_w = max(1, round(src_h * target_ratio))
        left = (src_w - new_w) // 2
        image = image.crop((left, 0, left + new_w, src_h))
    elif src_ratio < target_ratio:
        # Too tall — trim mostly from the bottom. A head-and-shoulders crop can
        # afford to lose chest; cropping the forehead costs identity.
        new_h = max(1, round(src_w / target_ratio))
        top = (src_h - new_h) // 4
        image = image.crop((0, top, src_w, top + new_h))

    return image.resize((width, height), Image.LANCZOS)
