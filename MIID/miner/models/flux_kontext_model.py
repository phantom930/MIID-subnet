#!/usr/bin/env python3
"""FLUX.1 Kontext image variation — pipeline loader and standalone test.

Model: black-forest-labs/FLUX.1-Kontext-dev
Type:  Context-aware text-guided image editing
VRAM:  ~24 GB recommended (uses CPU offload by default on CUDA)

Reusable API (imported by generate_variations.py):
    load_pipeline(device, token)  -> pipeline
    generate(pipe, image, prompt) -> PIL Image

Standalone test:
    python flux_kontext_model.py [path/to/seed_image.png]
"""

import gc
import os
import sys
from typing import Optional

import torch
from PIL import Image

MODEL_ID = "black-forest-labs/FLUX.1-Kontext-dev"
# Kontext is NOT step-distilled — unlike the FLUX.2 Klein models it degrades
# below ~20 steps, so this default stays where it is even though it is the
# slowest backend.
DEFAULT_STEPS = int(os.environ.get("MIID_INFERENCE_STEPS", "20"))
DEFAULT_GUIDANCE = float(os.environ.get("MIID_GUIDANCE_SCALE", "3.5"))
INTENSITY_GUIDANCE_MULT = {"light": 0.92, "medium": 1.0, "far": 1.12}
INTENSITY_TO_STRENGTH = {"light": 0.35, "medium": 0.55, "far": 0.75}


def _device() -> str:
    explicit = os.environ.get("FLUX_DEVICE", "").strip()
    if explicit:
        return explicit
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_pipeline(
    device: Optional[str] = None,
    token: Optional[str] = None,
):
    """Load and return a ready-to-use FluxKontextPipeline."""
    from diffusers import FluxKontextPipeline

    dev = device or _device()
    dtype = torch.bfloat16 if dev != "cpu" else torch.float32
    tok = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

    pipe = FluxKontextPipeline.from_pretrained(
        MODEL_ID, torch_dtype=dtype, token=tok,
    )

    try:
        from ._cuda_place import place_diffusers_pipeline
    except ImportError:
        from _cuda_place import place_diffusers_pipeline

    place_diffusers_pipeline(
        pipe, dev, default_offload_on_cuda=True, prefer_sequential_offload=True,
    )
    if dev == "cuda":
        gc.collect()
        torch.cuda.empty_cache()

    return pipe


def generate(
    pipe,
    image: Image.Image,
    prompt: str,
    intensity: str = "medium",
    num_steps: Optional[int] = None,
    guidance_scale: Optional[float] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    seed: Optional[int] = None,
    identity_bias: float = 0.0,
    negative_prompt: Optional[str] = None,
) -> Image.Image:
    """Generate a single variation and return the PIL Image.

    ``identity_bias`` (0–1) is raised on an identity retry: it pulls the
    denoising strength down so less of the reference face is repainted.
    """
    try:
        from ._common import fit_to_target, generation_size, make_generator, supported_kwargs
    except ImportError:
        from _common import fit_to_target, generation_size, make_generator, supported_kwargs

    bias = max(0.0, min(1.0, identity_bias))
    strength = INTENSITY_TO_STRENGTH.get(intensity, 0.55) * (1.0 - 0.25 * bias)
    guidance = guidance_scale or DEFAULT_GUIDANCE
    steps = num_steps or DEFAULT_STEPS

    # MIID_KONTEXT_WIDTH/HEIGHT stay an operator override for the size we
    # *generate* at (it exists to fit a smaller card); otherwise follow the
    # caller. Either way fit_to_target below still delivers the requested
    # output size, so the override trades detail for VRAM, not compliance.
    _h = os.environ.get("MIID_KONTEXT_HEIGHT", "").strip()
    _w = os.environ.get("MIID_KONTEXT_WIDTH", "").strip()
    if _h.isdigit() and _w.isdigit():
        gen_w, gen_h = int(_w), int(_h)
    else:
        gen_w, gen_h = generation_size(width, height)

    out = pipe(
        prompt=prompt,
        image=image,
        num_inference_steps=steps,
        guidance_scale=guidance,
        strength=strength,
        **supported_kwargs(
            pipe,
            width=gen_w,
            height=gen_h,
            generator=make_generator(seed),
            negative_prompt=negative_prompt,
        ),
    )
    return fit_to_target(out.images[0], width, height)


# ── Testing ──────────────────────────────────────────────────────────────────

TEST_PROMPT = (
    "Same person, same identity, Change background environment while keeping "
    "subject unchanged. Add religious head covering, Change environment type "
    "(office to outdoor, solid color to gradient). Additionally, include: "
    "Religious head covering (hijab, turban, kippah, taqiyah, etc.) "
    "appropriate to subject. Preserve face identity."
)


def main() -> None:
    """Run a test generation.

    Usage: python flux_kontext_model.py [path/to/seed_image.png]
    """
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        raise SystemExit("Set HF_TOKEN or HUGGINGFACE_TOKEN for Hugging Face model access.")

    seed_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "seed_images", "475c5c38e38b_m_doc.png",
    )

    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(here, "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "flux_kontext_test.png")

    base = Image.open(seed_path).convert("RGB")
    pipe = load_pipeline(token=token)
    result = generate(pipe, base, TEST_PROMPT)
    result.save(out_path)
    print(out_path)


if __name__ == "__main__":
    main()
