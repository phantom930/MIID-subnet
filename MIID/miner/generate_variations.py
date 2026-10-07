# The MIT License (MIT)
# Copyright © 2025 YANEZ
# MIID miner: Multi-model image variation generation.

"""
This file makes the miner generate image variations.

What this file does:
1. Keeps a list of image models the miner can use.
2. Picks one of the active models at random (or uses the forced model).
3. Loads that model (importing from MIID.miner.models.*).
4. Uses the model to make the requested image variations.
5. Keeps the same return format so the rest of the miner still works.

=============================================================================
Base models (default pool)
=============================================================================

1. ``flux_klein``
   - Model: ``black-forest-labs/FLUX.2-klein-4B``
   - Why: fastest and lightest, good baseline for GPU-constrained setups.

2. ``pulid``
   - Model: PuLID via Nunchaku (falls back to FLUX.1 Kontext)
   - Why: very high identity fidelity without changing background/lighting.
   - Extra packages: ``pip install nunchaku`` (CUDA required for true PuLID).

3. ``pulid_flux2``
   - Model: ``black-forest-labs/FLUX.2-klein-4B`` (PuLID-FLUX2 adapter compatible)
   - Why: strong identity preservation with the FLUX.2 Klein backbone,
     compatible with Fayens PuLID-FLUX2 adapter weights.

=============================================================================
Recommended alternatives (easy to set up)
=============================================================================

4. ``flux_kontext``
   - Model: ``black-forest-labs/FLUX.1-Kontext-dev``
   - Why: best text-guided editing quality with minimal visual drift.
   - Requires ≥24 GB VRAM on GPU.

5. ``qwen``
   - Model: ``Qwen/Qwen-Image-Edit-2511``
   - Why: strong instruction-following image editor from Qwen.
   - Install latest diffusers from source for this model.

=============================================================================
Paid / API model recommendations (not integrated — for future work)
=============================================================================

These models offer excellent quality via paid APIs.  They are not yet
integrated into this file but are recommended for miners who want top-tier
output and are willing to use API credits:

- **Soul**        — high-quality identity-preserving generation
- **Grok Imagination** — xAI's image generation API
- **Seedream**    — ByteDance's SeedreamDiT image model
- **Nonobana**    — advanced image editing model
- **Nonobana2**   — next-gen Nonobana with improved identity preservation

=============================================================================
How model choice works
=============================================================================

1. If ``MIID_MODEL`` is set, this file uses that exact model.
2. If ``MIID_MODEL`` is not set, ``MIID_MODEL_RANDOM`` controls random selection.
3. ``MIID_MODEL_RANDOM`` defaults to ``1`` (random among the 3 base models).
4. Model selection happens once at the start of each query and is reused for all
   variations in that query.

=============================================================================
How intensity works
=============================================================================

- ``light`` keeps the new image closer to the original.
- ``medium`` makes a balanced edit.
- ``far`` allows a bigger change.

Three of the five slots in a standard round are *combined* ones, where the
validator sends ``"lighting_edit+expression_edit"`` against ``"light+far"``.
Each component is phrased at its own intensity (see ``EDIT_PHRASES``), and the
slot as a whole is driven at its strongest component's bin.

=============================================================================
Simple setup steps
=============================================================================

1. Create a Hugging Face token.
2. Put it in your environment:
   ``export HF_TOKEN="hf_..."``
3. Install the packages for the model you want to use.

Packages for ``flux_klein`` and ``pulid_flux2``:
- ``pip install diffusers transformers accelerate``

Packages for ``pulid``:
- ``pip install diffusers transformers accelerate``
- For true PuLID: ``pip install nunchaku`` (requires CUDA)
- Without nunchaku, falls back to FLUX.1 Kontext automatically.

Packages for ``flux_kontext``:
- ``pip install diffusers transformers accelerate``

Packages for ``qwen``:
- ``pip install git+https://github.com/huggingface/diffusers``
- ``pip install transformers accelerate torchvision``

=============================================================================
Helpful environment variables
=============================================================================

- ``MIID_MODEL``: force one model, for example ``flux_klein``
- ``FLUX_DEVICE``: choose ``cuda``, ``mps``, or ``cpu``
- ``MIID_MODEL_RANDOM``: set to ``1`` to randomly pick among base models
- ``MIID_INFERENCE_STEPS``: change number of generation steps
- ``MIID_GUIDANCE_SCALE``: change prompt strength
- ``MIID_OUTPUT_WIDTH`` / ``MIID_OUTPUT_HEIGHT``: output size, default the
  1015 x 1350 the validator asks for; set both to 0 to ship raw model output
- ``MIID_IDENTITY_TARGET``: AdaFace similarity a variation must reach before it
  is accepted without a retry (default 0.6 — the validator's own gate)
- ``MIID_IDENTITY_RETRIES``: extra attempts per variation when it misses that
  target, or when its measured head turn misses the requested pose range
  (default 2)
- ``MIID_GENERATION_BUDGET_SECONDS``: wall-clock budget for one request's
  generation, retries included (default 900, inside the validator's 1200s)
- ``HF_TOKEN``: Hugging Face access token

=============================================================================
Testing individual models
=============================================================================

Each model can be tested standalone from MIID/miner/models/:
    python MIID/miner/models/flux_klein_model.py [seed_image.png]
    python MIID/miner/models/pulid_model.py [seed_image.png]
    python MIID/miner/models/pulid_flux2_model.py [seed_image.png]
    python MIID/miner/models/flux_kontext_model.py [seed_image.png]
    python MIID/miner/models/qwen_model.py [seed_image.png]

This lets you see the output for a given prompt and check if the model works
on your hardware before running the full miner.
"""

import gc
import os
import random
import re
import logging
from typing import List, Dict, Any, Optional, Tuple

import torch
from PIL import Image

logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================


def _resolve_device() -> str:
    """Prefer GPU when available."""
    explicit = os.environ.get("FLUX_DEVICE", "").strip()
    if explicit:
        return explicit
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = _resolve_device()

# None means "let each backend use its own default", which is what you want:
# the FLUX.2-klein models are step-distilled and are done in ~8 steps, while
# FLUX.1 Kontext is not and needs ~20. A single global number either wastes
# minutes on the distilled models or starves the others. MIID_INFERENCE_STEPS
# still overrides every backend when an operator sets it.
_STEPS_ENV = os.environ.get("MIID_INFERENCE_STEPS", "").strip()
NUM_INFERENCE_STEPS: Optional[int] = int(_STEPS_ENV) if _STEPS_ENV.isdigit() else None

GUIDANCE_SCALE = float(os.environ.get("MIID_GUIDANCE_SCALE", "3.5"))

DEFAULT_INTENSITY = "medium"

INTENSITY_TO_STRENGTH: Dict[str, float] = {
    "light":  0.35,
    "medium": 0.55,
    "far":    0.75,
}

# Output geometry the validator asks for. IMAGE_VARIATION_REQUIREMENTS (in
# MIID/validator/image_variations.py) is appended to every request's `detail`:
# "Professional passport-style portraits, 3:4 aspect ratio, head-and-shoulders
# composition from chest up. Recommended output resolution: 1015 x 1350 pixels."
# The grading API scores compliance, so producing the model's default square
# frame costs points on every single variation. Set MIID_OUTPUT_WIDTH /
# MIID_OUTPUT_HEIGHT to 0 to disable resizing and ship the raw model output.
TARGET_WIDTH = int(os.environ.get("MIID_OUTPUT_WIDTH", "1015"))
TARGET_HEIGHT = int(os.environ.get("MIID_OUTPUT_HEIGHT", "1350"))

# Steered away from on every generation. The grading API judges photographic
# quality alongside identity, and these are the failure modes a portrait edit
# falls into on its own. Note the FLUX.2 Klein pipelines do not accept a
# negative prompt at all (supported_kwargs drops it), so the framing defence
# below has to hold on its own in the positive prompt.
NEGATIVE_PROMPT = (
    "different person, face swap, distorted or deformed face, asymmetric eyes, "
    "extra fingers, blurry, low resolution, jpeg artifacts, oversaturated, "
    "cartoon, anime, illustration, painting, 3d render, plastic skin, "
    "watermark, text, logo, extreme close-up, selfie crop, cropped head, "
    "shoulders out of frame, multiple people"
)

# Stated first on every generation, never suppressed.
#
# The validator already appends this requirement to each request's `detail`
# ("All images are Professional passport-style portraits, 3:4 aspect ratio,
# head-and-shoulders composition from chest up..."), but it lands at the tail of
# a long instruction string where the model under-weights it — observed result:
# correct background and accessory, but a tight selfie crop, which the grading
# API scored negative. Leading with it, in imperative form and with the failure
# mode named explicitly, is what keeps the frame compliant.
FRAMING_CLAUSE = (
    "Professional passport-style portrait photograph, 3:4 vertical format. "
    "Frame the subject head-and-shoulders from mid-chest up: the whole head "
    "with headroom above it and both shoulders fully inside the frame, subject "
    "centred and facing the camera. Not an extreme close-up, not a selfie crop "
    "— do not fill the frame with the face. Sharp focus, natural skin texture, "
    "even photographic exposure."
)

# The validator's own copy of the requirement, stripped from the edit text so it
# is stated once (above, forcefully) instead of twice (here, weakly).
_REQUIREMENTS_PREFIX = "All images are Professional passport-style portraits"

IDENTITY_CLAUSE = (
    "Keep the identity of the reference person exactly: same facial structure "
    "and proportions, same eyes, nose, mouth and jawline, same skin tone, same "
    "hair, same apparent age and gender. Do not restyle or beautify the face."
)

# The validator joins a combined slot's components with this, in both `type`
# ("lighting_edit+expression_edit") and `intensity` ("light+far").  See
# COMBINED_VARIATION_SEPARATOR in MIID/validator/image_variations.py.
COMBINED_SEPARATOR = "+"

# Ordering over the intensity bins, so a combined slot can be reduced to the
# single scalar the backends' guidance multiplier expects.
INTENSITY_RANK: Dict[str, int] = {"light": 0, "medium": 1, "far": 2}
_RANK_TO_INTENSITY = {rank: name for name, rank in INTENSITY_RANK.items()}

# The validator appends the accessory to a background slot's `detail` behind
# this marker, in a richer wording than the copy it puts in `description`
# ("Baseball cap or similar sports cap" vs "Add baseball cap").  It is graded
# content, so it is lifted out and stated once, from the better source.
_ACCESSORY_MARKER = "Additionally, include: "

# The validator's religious-covering request is "Religious head covering
# (hijab, turban, kippah, taqiyah, etc.) appropriate to subject". Relayed as
# is, the model reads the first item and draws a hijab on men too — and the
# grading score sheet gives -3 for a covering that does not match the seed's
# gender. That was the largest single loss in the archive (22 of 145
# variations), so the covering is chosen here, from the subject's gender.
_RELIGIOUS_MARKER = "religious head covering"
# Men get a taqiyah only. The grader classifies the accessory with a
# confidence cut-off (score sheet: "background accessory path insufficient
# match" = 0, or -1 without a background), and CLIP-style models read a
# wrapped turban as a bandana: raw CLIP put "bandana" first on 9 of 9 of our
# turbans, and the turban graded -1 by RoundTable21 on 2026-10-07 read 94%
# bandana / 6% religious covering.
RELIGIOUS_COVERINGS: Dict[str, Tuple[str, ...]] = {
    "m": (
        "a white Muslim prayer cap (taqiyah / kufi) with a fine crocheted "
        "pattern, fitted on the crown of his head, with his ears, jawline and "
        "neck left uncovered — not a headscarf, not a turban",
    ),
    "f": (
        "a hijab headscarf wrapped around her head that covers her hair, ears "
        "and neck while her whole face stays visible",
    ),
}


def subject_gender_from_filename(name: Optional[str]) -> Optional[str]:
    """"m" / "f" from a validator base filename (e.g. ``070f3abdfcf9_m.png``)."""
    if not name:
        return None
    match = re.search(r"_([mf])(?:_[a-z]+)?\.[a-z0-9]+$", name.lower())
    return match.group(1) if match else None


# Concrete renderings of the validator's other accessories, keyed by a phrase
# from its `detail` text. The grader scores the accessory by classifier
# confidence, and on 2026-10-07 the muted versions the bare text produced
# failed it: CLIP with the validator's own wording read 7 of 9 snug grey
# beanies and 3 of 12 dark caps as "no head accessory" (< 0.5). Contrast with
# the hair and the item's defining shape are what carry the confidence.
ACCESSORY_RENDERINGS: Tuple[Tuple[str, str], ...] = (
    # First: the brim-hat text itself says "(not baseball cap)".
    ("brim hat", "a wide-brim hat (a fedora or sun hat) with a clearly "
                 "visible brim all the way round — not a baseball cap"),
    ("knit hat", "a chunky cable-knit winter beanie in a bright solid colour, "
                 "with a thick folded cuff, pulled down over the top of the "
                 "head so it clearly sits above the hair"),
    ("baseball cap", "a baseball cap in a bright solid colour with a curved "
                     "visor pointing forward, clearly a sports cap"),
    ("bandana", "a bandana with a bold paisley print tied over the top of the "
                "head, knotted at the back"),
)


def _resolve_accessory(accessory: str, subject_gender: Optional[str]) -> str:
    """A concrete accessory to draw, gender-appropriate for a religious covering."""
    lowered = accessory.lower()
    if _RELIGIOUS_MARKER not in lowered:
        for phrase, rendering in ACCESSORY_RENDERINGS:
            if phrase in lowered:
                return rendering
        return accessory
    options = RELIGIOUS_COVERINGS.get(subject_gender or "")
    if not options:
        return (
            f"{accessory}; choose the covering that matches the subject's own "
            "gender — a hijab only for a woman, a taqiyah prayer cap only for "
            "a man"
        )
    return random.choice(options)

# One handwritten instruction per (type, intensity) the validator can ask for.
#
# The protocol's own `description` is a *taxonomy entry* ("Modify illumination
# direction, intensity, or color temperature") and `detail` is the actual
# directive ("Subtle brightness or contrast change") — so relaying both, which
# is what merging the two fields does, spends half the prompt restating the
# category in vaguer words that contradict the directive.  On a combined slot
# it also drags the schema tokens in with it: "Combined variation — apply all
# of the following while preserving identity: lighting_edit (light): ...".
#
# The type vocabulary is closed (5 types x 3 bins), so the miner can carry its
# own phrasing and skip the taxonomy entirely.  Anything outside this table
# falls back to the merge — see _get_prompt_from_request.
#
# Pose phrases ask for more rotation than the bin names. FLUX.2 Klein ignores
# guidance_scale (it is step-distilled), so wording is the only lever, and the
# reference image pulls the head back toward frontal: across 145 archived
# variations the measured yaw medians were 4° for "light" (asked ±15°), 19° for
# "medium" (±30°) and 22° for "far" (>45°). Asking past the target lands in it.
# {side} is filled with left/right per generation.
EDIT_PHRASES: Dict[Tuple[str, str], str] = {
    ("pose_edit", "light"): (
        "Rotate the head about 20 degrees to the {side}: the nose points "
        "clearly off-centre and one cheek shows more than the other, while "
        "both eyes stay fully visible. The head must not face the camera "
        "straight on."
    ),
    ("pose_edit", "medium"): (
        "Rotate the head about 40 degrees to the {side} into a clear "
        "three-quarter view: the far cheek recedes, the far ear is hidden and "
        "the nose overlaps the far cheek."
    ),
    ("pose_edit", "far"): (
        "Rotate the head 70 degrees to the {side} into a near-profile view: "
        "the nose seen in profile against the background, the far eye almost "
        "hidden, the near ear fully visible. Shoulders may stay toward the "
        "camera, but the face points to the side."
    ),
    # Generators brighten the face on their own; "a gentle shift in
    # brightness" on top of that read as a medium change in most rounds.
    ("lighting_edit", "light"): (
        "Change the lighting only subtly: keep the overall exposure close to "
        "the reference, with slightly softer or slightly warmer light and "
        "faint, diffuse shadows."
    ),
    ("lighting_edit", "medium"): (
        "Light the subject from one clear directional source on the {side}: "
        "that half of the face is bright, the other half falls into a clearly "
        "visible shadow down the cheek, nose and jaw."
    ),
    ("lighting_edit", "far"): (
        "Light the subject with a single hard off-axis source: deep, "
        "high-contrast shadows across the face and a strong warm or cool "
        "colour cast."
    ),
    # "Faint, barely-there" landed on both sides of light: no change at all
    # on some rounds, a full smile on others. A named, closed-mouth smile is a
    # visible but bounded change.
    ("expression_edit", "light"): (
        "Give the subject a small closed-mouth smile: the corners of the lips "
        "lifted, lips together with no teeth showing, and a relaxed brow."
    ),
    ("expression_edit", "medium"): (
        "Give the subject a clearly changed expression — an open smile, a "
        "serious set to the mouth, or mild surprise."
    ),
    ("expression_edit", "far"): (
        "Give the subject a pronounced expression — laughing with the mouth "
        "open, or visibly surprised or concerned, with the eyes and brow fully "
        "engaged."
    ),
    # The base image is a plain studio backdrop, so "keep the same room" kept
    # the blank backdrop — and the grader's background check found no
    # background. Light still has to be a real, if quiet, setting.
    # Backgrounds must read as a real scene to the grader's background check:
    # a white-walled gallery graded -1 ("no background detected") on
    # 2026-10-07, so every level names concrete, visible furniture or scenery
    # and light blur only.
    ("background_in", "light"): (
        "Replace the plain backdrop with a real indoor room behind the "
        "subject — a coloured wall with a shelf, a plant or a framed picture, "
        "slightly out of focus but clearly recognisable, never a blank or "
        "white backdrop."
    ),
    ("background_in", "medium"): (
        "Place the subject in a different but plausible indoor setting — an "
        "office with desks and shelves, a cafe with tables and warm lights, "
        "or a hotel lobby with furniture — entirely indoors, with the room "
        "clearly visible behind them."
    ),
    ("background_in", "far"): (
        "Place the subject in a clearly different indoor setting with its own "
        "interior design and visible depth, such as a library with full "
        "bookshelves, a busy open-plan office or a richly furnished hotel "
        "lobby — colourful and detailed, not white walls. Keep the background "
        "fully indoors, with no outdoor elements anywhere in the frame."
    ),
    ("background_out", "light"): (
        "Replace the plain backdrop with a real outdoor scene in daylight — "
        "trees, a path or building fronts behind the subject, slightly out of "
        "focus but clearly recognisable as outdoors, never a blank backdrop."
    ),
    ("background_out", "medium"): (
        "Place the subject in a different but plausible outdoor setting — a "
        "park, a street corner, a waterfront."
    ),
    ("background_out", "far"): (
        "Place the subject in a clearly different outdoor setting with its own "
        "composition and depth, such as an urban street, a beach promenade, a "
        "mountain overlook or a garden path."
    ),
}

# Appended on a retry after image_generator measured the head turn outside
# the requested range. Klein under-rotates far more often than it over-rotates:
# the first rounds on the "70 degrees" wording still landed at 21–34°.
POSE_HINTS: Dict[str, str] = {
    "more": (
        "The previous attempt barely turned the head: turn it clearly further "
        "this time — the change of angle must be obvious at a glance."
    ),
    "less": (
        "Keep the head turn modest — no more than the angle described, with "
        "the face still mostly toward the camera."
    ),
}

# =============================================================================
# Available models
# =============================================================================

AVAILABLE_MODELS: Dict[str, Dict[str, Any]] = {
    # ── Base models (default pool for random selection) ──
    "flux_klein": {
        "model_id": "black-forest-labs/FLUX.2-klein-4B",
        "type": "flux_klein",
        "params": "4B",
        "license": "FLUX.1 dev non-commercial (commercial available from BFL)",
        "base": True,
        "notes": (
            "Original default. Lightweight, fast inference. "
            "Good baseline for GPU-constrained setups."
        ),
    },
    "pulid": {
        "model_id": "guozinan/PuLID (Nunchaku) / FLUX.1-Kontext-dev (fallback)",
        "type": "pulid",
        "params": "~12B",
        "license": "NeurIPS 2024 (ByteDance) / FLUX dev non-commercial",
        # Out of the random pool: without nunchaku it is FLUX.1 Kontext, which
        # kept the seed's face pixels (copy-paste risk) and returned unchanged
        # images for pose/expression slots. Still selectable via MIID_MODEL.
        "base": False,
        "notes": (
            "PuLID: Pure and Lightning ID Customization. Very high identity "
            "fidelity. Uses Nunchaku PuLIDFluxPipeline on CUDA; falls back to "
            "FLUX.1 Kontext when Nunchaku is unavailable."
        ),
    },
    "pulid_flux2": {
        "model_id": "black-forest-labs/FLUX.2-klein-4B",
        "type": "pulid_flux2",
        "params": "4B",
        "license": "FLUX.1 dev non-commercial",
        "base": True,
        "notes": (
            "FLUX.2 Klein backbone compatible with Fayens PuLID-FLUX2 adapter "
            "weights (pulid_flux2_klein_v1/v2.safetensors). Strong identity "
            "preservation with reduced artifacts."
        ),
    },
    # ── Recommended alternatives ──
    "flux_kontext": {
        "model_id": "black-forest-labs/FLUX.1-Kontext-dev",
        "type": "flux_kontext",
        "params": "12B",
        "license": "FLUX.1 dev non-commercial (commercial $999/mo from BFL)",
        "base": False,
        "notes": (
            "Context-aware editing model from Black Forest Labs. Best text-guided "
            "editing quality with minimal visual drift. Requires ≥24 GB VRAM."
        ),
    },
    "qwen": {
        "model_id": "Qwen/Qwen-Image-Edit-2511",
        "type": "qwen",
        "params": "~14B",
        "license": "Check Qwen model card",
        "base": False,
        "notes": (
            "Qwen's instruction-based image editor. Strong prompt following. "
            "Requires latest diffusers from source and torchvision."
        ),
    },
}

BASE_MODELS = [k for k, v in AVAILABLE_MODELS.items() if v.get("base")]

# =============================================================================
# Module state
# =============================================================================

_cached_pipeline: Any = None
_cached_model_key: Optional[str] = None

# =============================================================================
# Model selection
# =============================================================================


def _select_model() -> str:
    """Choose the model for this generation round."""

    forced = os.environ.get("MIID_MODEL", "").strip().lower()
    # Default behavior: randomly pick among base models each call.
    random_flag = os.environ.get("MIID_MODEL_RANDOM", "1").strip().lower() in (
        "1", "true", "yes",
    )

    if forced and forced in AVAILABLE_MODELS:
        selected_model = forced
        logger.info("Model forced via MIID_MODEL env var: %s", selected_model)
    elif random_flag:
        selected_model = random.choice(BASE_MODELS)
        logger.info("Randomly selected model for this round: %s", selected_model)
    else:
        selected_model = "flux_klein"
        logger.info(
            "Using default model: %s (override with MIID_MODEL=..., or MIID_MODEL_RANDOM=1)",
            selected_model,
        )

    cfg = AVAILABLE_MODELS[selected_model]
    logger.info(
        "  model_id=%s  params=%s  license=%s",
        cfg["model_id"], cfg["params"], cfg["license"],
    )
    return selected_model


def get_selected_model_info(model_key: Optional[str] = None) -> Dict[str, Any]:
    """Return config dict for a selected model key (or choose one now)."""
    key = model_key or _select_model()
    return {"key": key, **AVAILABLE_MODELS[key]}


# =============================================================================
# HF token helper
# =============================================================================


def _get_hf_token() -> str:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN") or ""
    if not token:
        raise RuntimeError(
            "Missing Hugging Face token.  Set HF_TOKEN or HUGGINGFACE_TOKEN "
            "in your environment, e.g.\n  export HF_TOKEN=\"hf_...\""
        )
    return token


# =============================================================================
# Loaders — each imports from the corresponding model module.
# Lazy imports keep startup fast and avoid loading unneeded dependencies.
# =============================================================================


def _load_flux_klein() -> Any:
    from .models.flux_klein_model import load_pipeline
    return load_pipeline(device=DEVICE, token=_get_hf_token())


def _load_pulid() -> Any:
    from .models.pulid_model import load_pipeline
    return load_pipeline(device=DEVICE, token=_get_hf_token())


def _load_pulid_flux2() -> Any:
    from .models.pulid_flux2_model import load_pipeline
    return load_pipeline(device=DEVICE, token=_get_hf_token())


def _load_flux_kontext() -> Any:
    from .models.flux_kontext_model import load_pipeline
    return load_pipeline(device=DEVICE, token=_get_hf_token())


def _load_qwen() -> Any:
    from .models.qwen_model import load_pipeline
    return load_pipeline(device=DEVICE, token=_get_hf_token())


_MODEL_LOADERS: Dict[str, Any] = {
    "flux_klein":   _load_flux_klein,
    "pulid":        _load_pulid,
    "pulid_flux2":  _load_pulid_flux2,
    "flux_kontext": _load_flux_kontext,
    "qwen":         _load_qwen,
}


def _release_pipeline() -> None:
    """Drop the cached pipeline and hand its VRAM back.

    A new pipeline must never be loaded while the previous one is still referenced:
    both would be resident at once, which OOMs anything short of a very large card.
    """
    global _cached_pipeline, _cached_model_key

    if _cached_pipeline is None:
        return

    logger.info("Releasing cached pipeline: %s", _cached_model_key)
    try:
        _cached_pipeline.remove_all_hooks()
    except Exception:  # noqa: BLE001 — not all pipelines carry offload hooks
        pass

    _cached_pipeline = None
    _cached_model_key = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _get_pipeline(model_key: str) -> Any:
    """Load the selected model pipeline and reuse it.

    If loading fails, we fall back to ``flux_klein`` so the miner can still run.
    """
    global _cached_pipeline, _cached_model_key

    if _cached_pipeline is not None and _cached_model_key == model_key:
        return _cached_pipeline

    # Model choice is re-rolled per query, so this is the swap path: free the
    # outgoing pipeline before the incoming one starts allocating.
    _release_pipeline()

    loader = _MODEL_LOADERS.get(model_key)
    if loader is None:
        raise ValueError(
            f"No loader for model '{model_key}'.  "
            f"Available: {list(_MODEL_LOADERS.keys())}"
        )

    logger.info(
        "Loading pipeline: %s (%s) …",
        model_key, AVAILABLE_MODELS[model_key]["model_id"],
    )

    try:
        _cached_pipeline = loader()
    except Exception as exc:
        if model_key == "flux_klein":
            raise
        logger.warning(
            "Failed to load %s: %s — falling back to flux_klein", model_key, exc,
        )
        # A failed load (OOM in particular) can leave partial weights on the card;
        # the cache is already empty here, so reclaim directly before retrying.
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _cached_pipeline = _load_flux_klein()
        _cached_model_key = "flux_klein"
        return _cached_pipeline

    _cached_model_key = model_key
    logger.info("Pipeline ready: %s", model_key)
    return _cached_pipeline


# =============================================================================
# Helper functions for reading the request and building the prompt.
# =============================================================================


def _canonical_background_type(req: Any, var_type: str) -> str:
    """Use background_in / background_out in S3 keys; never a bare background type."""
    if var_type in ("background_in", "background_out"):
        return var_type
    if var_type not in ("background_edit", "background"):
        return var_type
    description = getattr(req, "description", None) or (
        req.get("description") if isinstance(req, dict) else None
    ) or ""
    detail = getattr(req, "detail", None) or (
        req.get("detail") if isinstance(req, dict) else None
    ) or ""
    blob = f"{description} {detail}".lower()
    # The description names the environment outright ("Indoor background
    # change"). A bare "outdoor" search is wrong: the indoor far detail ends
    # "...with no outdoor elements", which sent indoor requests out as
    # background_out — wrong label and an outdoor scene.
    if "indoor background" in blob:
        return "background_in"
    if "outdoor background" in blob:
        return "background_out"
    blob = blob.replace("no outdoor", "").replace("no indoor", "")
    if "outdoor" in blob:
        return "background_out"
    return "background_in"


def _split_components(var_type: str, raw_intensity: str) -> List[Tuple[str, str]]:
    """Split a slot into its (type, intensity) components.

    Three of the five slots in a standard round are combined ones, where the
    validator sends ``"lighting_edit+expression_edit"`` against
    ``"light+far"``. A single slot simply yields one pair.

    A malformed pairing — a differing number of types and intensities, which
    the protocol does not currently produce but miners and validators upgrade
    independently — spreads the first intensity across every component rather
    than dropping the slot.
    """
    types = [t for t in var_type.split(COMBINED_SEPARATOR) if t]
    levels = [i for i in raw_intensity.split(COMBINED_SEPARATOR) if i]

    if len(levels) != len(types):
        head = levels[0] if levels else DEFAULT_INTENSITY
        levels = [head] * len(types)

    return [
        (t, level if level in INTENSITY_RANK else DEFAULT_INTENSITY)
        for t, level in zip(types, levels)
    ]


def requested_pose_level(req: Any) -> Optional[str]:
    """The intensity a slot asks of its pose_edit component, or None."""
    var_type = _req_field(req, "type")
    for comp_type, level in _split_components(var_type, _req_field(req, "intensity")):
        if comp_type == "pose_edit":
            return level
    return None


def _effective_intensity(components: List[Tuple[str, str]]) -> str:
    """The single intensity bin that stands for a whole slot.

    The backends take one scalar ``intensity`` and turn it into a guidance
    multiplier, so a combined slot has to collapse to one bin. The strongest
    component is the honest reading: ``pose_edit(far)+expression_edit(light)``
    is a far edit that happens to carry a light component, and treating it as
    anything less under-drives the edit the grading API is looking for.
    """
    if not components:
        return DEFAULT_INTENSITY
    return _RANK_TO_INTENSITY[max(INTENSITY_RANK[level] for _, level in components)]


def _get_type_and_intensity(req: Any) -> Tuple[str, str]:
    """Extract .type and .intensity from a VariationRequest-like object or dict.

    The type is returned verbatim (beyond the background canonicalisation),
    because it becomes ``variation_type`` on the result and thence the S3 key
    the grading API reads — a combined slot stays ``"lighting_edit+pose_edit"``.

    The intensity is *not* verbatim: it is reduced to the single bin the
    backends understand. Rejecting the combined form outright, which is what
    a membership test against the three bins does, silently sent every
    combined slot through as ``medium`` — and ``medium`` is the one bin whose
    guidance multiplier is exactly 1.0, so the validator's intensity reached
    the model through nothing at all on three of the five scored slots.
    """
    var_type = getattr(req, "type", None) or (req.get("type") if isinstance(req, dict) else None)
    intensity = getattr(req, "intensity", None) or (req.get("intensity") if isinstance(req, dict) else None)
    if not var_type:
        raise ValueError("variation_requests entry missing 'type'")

    canonical = _canonical_background_type(req, var_type)
    return (canonical, _effective_intensity(_split_components(canonical, intensity or "")))


def _strip_requirements(text: Optional[str]) -> str:
    """Drop the validator's trailing requirements boilerplate from one field.

    FRAMING_CLAUSE restates it at the front of the prompt, so leaving the
    original in place would only repeat the constraint in its weaker wording.
    Anything after the marker is that boilerplate, so the tail goes with it.
    """
    cleaned = (text or "").strip()
    marker = cleaned.find(_REQUIREMENTS_PREFIX)
    if marker != -1:
        cleaned = cleaned[:marker]
    return cleaned.strip().rstrip(".").strip()


def _req_field(req: Any, name: str) -> str:
    """Read one field off a VariationRequest-like object or a dict."""
    value = getattr(req, name, None)
    if value is None and isinstance(req, dict):
        value = req.get(name)
    return (value or "").strip()


def _accessory_from_detail(detail: str) -> str:
    """The accessory a background slot asks for, or "" when there is none.

    The validator appends it to `detail` behind _ACCESSORY_MARKER, after the
    requirements boilerplate has been stripped off the tail.
    """
    cleaned = _strip_requirements(detail)
    marker = cleaned.find(_ACCESSORY_MARKER)
    if marker == -1:
        return ""
    return cleaned[marker + len(_ACCESSORY_MARKER):].strip().rstrip(".").strip()


def _soften(level: str) -> str:
    """One intensity bin down, floored at ``light``."""
    return _RANK_TO_INTENSITY[max(0, INTENSITY_RANK[level] - 1)]


def _edit_clause(
    components: List[Tuple[str, str]], identity_bias: float,
) -> Optional[str]:
    """Written instructions for a slot's components, or None if unrecognised.

    None means some component is outside EDIT_PHRASES — a variation type added
    on the validator side against a miner that has not been upgraded yet. The
    caller falls back to relaying the protocol text, so an unknown type still
    generates something rather than raising.
    """
    phrases = []
    # One side per slot, so a pose turn and a directional light agree.
    side = random.choice(("left", "right"))
    for comp_type, level in components:
        # A retry means the face already drifted past the identity floor, and
        # pose is what moves an embedding furthest — so give that component a
        # bin back rather than flattening the whole edit. The alternative is a
        # variation dropped at the 0.4 floor in submission_builder, which
        # scores zero; a slightly under-driven edit still scores.
        if identity_bias > 0 and comp_type == "pose_edit":
            level = _soften(level)
        phrase = EDIT_PHRASES.get((comp_type, level))
        if phrase is None:
            return None
        phrases.append(phrase.replace("{side}", side))

    return " ".join(phrases) if phrases else None


def _get_prompt_from_request(
    req: Any, var_type: str, intensity: str, identity_bias: float = 0.0,
    subject_gender: Optional[str] = None,
    pose_hint: Optional[str] = None,
) -> str:
    """Build the generation prompt from the protocol fields.

    Order matters, and it is: framing, then the edit, then identity. The
    grading API's validation_score is the whole ranking signal (identity is a
    1e-5 tiebreaker), and it scores both whether the requested edit landed and
    whether the result obeys the passport-portrait composition — so those two
    take the front of the prompt, where the model weights them most.

    The edit itself is written here from (type, intensity) rather than
    assembled out of the request's `description` and `detail`. Those two
    fields are a category label and a directive respectively, so merging them
    states each instruction twice in two strengths, and on a combined slot it
    carries the schema's own tokens into the prompt. EDIT_PHRASES covers every
    type the validator currently sends; anything else falls back to the merge.

    ``identity_bias`` > 0 marks a retry after the face drifted too far, and
    holds the edit back — generally, and on the pose component specifically.

    ``subject_gender`` ("m" / "f") turns the validator's generic religious
    head covering into one that matches the subject — see _resolve_accessory.

    ``pose_hint`` ("more" / "less") follows a measured head turn that missed
    the requested range, and pushes the next attempt the other way.
    """
    description = _req_field(req, "description")
    detail = _req_field(req, "detail")

    components = _split_components(var_type, _req_field(req, "intensity"))
    written = _edit_clause(components, identity_bias)
    if written is not None and pose_hint in POSE_HINTS:
        written = f"{written} {POSE_HINTS[pose_hint]}"

    parts = [
        FRAMING_CLAUSE,
        "The subject is the SAME person as the reference image.",
    ]

    if written is not None:
        parts.append(written)
        # The accessory is randomised per round, so it cannot live in the
        # table — but it is graded content, so it is carried over from the
        # request. `detail` holds the fuller of the validator's two wordings.
        accessory = _accessory_from_detail(detail)
        if accessory:
            # Exactly one item: the score sheet treats a second accessory as
            # a mismatch, and generators like to add a scarf under a cap.
            parts.append(
                f"The subject is wearing {_resolve_accessory(accessory, subject_gender)}. "
                "This is the only accessory added — no other hat, scarf or "
                "head covering — and the subject keeps their own clothing."
            )
    else:
        edit = ". ".join(
            _strip_requirements(p).rstrip(".")
            for p in (description, detail)
            if _strip_requirements(p)
        )
        if not edit:
            edit = f"{var_type} variation at {intensity} intensity"
        parts.append(f"Apply this edit: {edit}.")

    parts.append(IDENTITY_CLAUSE)

    if identity_bias > 0:
        parts.append(
            "Apply the edit conservatively — the face must stay clearly "
            "recognizable as the reference person."
        )

    return " ".join(parts)


# =============================================================================
# Generators — each imports from the corresponding model module.
# =============================================================================


def _common_generate_kwargs(
    intensity: str, seed: Optional[int], identity_bias: float,
) -> Dict[str, Any]:
    """The arguments every backend's ``generate()`` takes the same way."""
    return {
        "intensity": intensity,
        "num_steps": NUM_INFERENCE_STEPS,
        "guidance_scale": GUIDANCE_SCALE,
        "width": TARGET_WIDTH,
        "height": TARGET_HEIGHT,
        "seed": seed,
        "identity_bias": identity_bias,
        "negative_prompt": NEGATIVE_PROMPT,
    }


def _generate_with_flux_klein(
    pipe: Any, base_image: Image.Image, prompt: str, **kwargs: Any,
) -> Image.Image:
    from .models.flux_klein_model import generate
    return generate(pipe, base_image, prompt, **kwargs)


def _generate_with_pulid(
    pipe: Any, base_image: Image.Image, prompt: str, **kwargs: Any,
) -> Image.Image:
    from .models.pulid_model import generate
    return generate(pipe, base_image, prompt, **kwargs)


def _generate_with_pulid_flux2(
    pipe: Any, base_image: Image.Image, prompt: str, **kwargs: Any,
) -> Image.Image:
    from .models.pulid_flux2_model import generate
    return generate(pipe, base_image, prompt, **kwargs)


def _generate_with_flux_kontext(
    pipe: Any, base_image: Image.Image, prompt: str, **kwargs: Any,
) -> Image.Image:
    from .models.flux_kontext_model import generate
    return generate(pipe, base_image, prompt, **kwargs)


def _generate_with_qwen(
    pipe: Any, base_image: Image.Image, prompt: str, **kwargs: Any,
) -> Image.Image:
    from .models.qwen_model import generate
    return generate(pipe, base_image, prompt, **kwargs)


_GENERATORS: Dict[str, Any] = {
    "flux_klein":   _generate_with_flux_klein,
    "pulid":        _generate_with_pulid,
    "pulid_flux2":  _generate_with_pulid_flux2,
    "flux_kontext": _generate_with_flux_kontext,
    "qwen":         _generate_with_qwen,
}


# =============================================================================
# Public API
# =============================================================================


def active_model_key() -> Optional[str]:
    """The model actually loaded right now — not necessarily the one selected.

    ``_get_pipeline()`` falls back to ``flux_klein`` when the chosen model
    fails to load, and the miner records which approach really ran.
    """
    return _cached_model_key


def generate_one(
    base_image: Image.Image,
    req: Any,
    model_key: Optional[str] = None,
    attempt: int = 0,
    subject_gender: Optional[str] = None,
    identity_bias: Optional[float] = None,
    pose_hint: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate a single variation for one validator request.

    Separate from :func:`generate_variations` so a caller that measured the
    result — the identity check in ``image_generator`` — can re-roll one slot
    without regenerating the others. The pipeline is cached, so a retry costs
    one denoising pass, not another model load.

    Args:
        base_image: PIL Image of the base face.
        req: One validator request with ``.type`` and ``.intensity``.
        model_key: Model to use; selected fresh when omitted.
        attempt: 0 for the first try. Higher values re-roll the seed and raise
            ``identity_bias``, trading edit strength for identity retention.
        subject_gender: "m" / "f" (see subject_gender_from_filename) or None.
        identity_bias: Overrides the attempt-derived bias. A retry for a head
            turn that fell short passes 0: the face held, the edit did not.
        pose_hint: "more" / "less" when an earlier attempt's measured head
            turn missed the requested range (see POSE_HINTS).

    Returns:
        Dict with ``image``, ``variation_type``, ``model_key``, ``model_id``
        and ``attempt``.
    """
    model_key = model_key or _select_model()
    pipe = _get_pipeline(model_key)

    # _get_pipeline may have fallen back; generate with what is actually loaded.
    loaded_key = _cached_model_key or model_key
    model_type = AVAILABLE_MODELS[loaded_key]["type"]
    generator = _GENERATORS.get(model_type)
    if generator is None:
        raise RuntimeError(
            f"No generation handler for model type '{model_type}'.  "
            f"Registered types: {list(_GENERATORS.keys())}"
        )

    var_type, intensity = _get_type_and_intensity(req)
    # Each attempt steps 40% further toward "hold the face, soften the edit",
    # capped at 0.8 so a variation never collapses into a copy of the input.
    if identity_bias is None:
        identity_bias = min(0.8, 0.4 * attempt)
    prompt = _get_prompt_from_request(
        req, var_type, intensity, identity_bias, subject_gender=subject_gender,
        pose_hint=pose_hint,
    )
    seed = random.randint(0, 2**31 - 1)

    try:
        gen_image = generator(
            pipe, base_image, prompt,
            **_common_generate_kwargs(intensity, seed, identity_bias),
        )
    except Exception as e:
        raise RuntimeError(
            f"Variation failed for {var_type}({intensity}) "
            f"with {loaded_key} on attempt {attempt}: {e}"
        ) from e

    return {
        "image": gen_image,
        "variation_type": var_type,
        "model_key": loaded_key,
        "model_id": AVAILABLE_MODELS[loaded_key]["model_id"],
        "attempt": attempt,
    }


def generate_variations(
    base_image: Image.Image,
    variation_requests: List[Any],
    model_key: Optional[str] = None,
    subject_gender: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Generate one image variation per validator request, first attempt only.

    Picks and loads the model once, then runs :func:`generate_one` for each
    request. No identity checking happens here — ``image_generator`` owns that,
    because it is the layer that can re-roll a slot that came back too far from
    the reference face.

    Args:
        base_image: PIL Image of the base face.
        variation_requests: List of validator requests; each has ``.type`` and
            ``.intensity`` (e.g. ``VariationRequest`` or dict).

    Returns:
        List of dicts as returned by :func:`generate_one` — ``image``,
        ``variation_type``, ``model_key``, ``model_id``, ``attempt``.
    """
    if not variation_requests:
        return []

    model_key = model_key or _select_model()

    logger.info(
        "Generating %d variation(s) with %s (%s)",
        len(variation_requests), model_key, AVAILABLE_MODELS[model_key]["model_id"],
    )

    return [
        generate_one(base_image, req, model_key=model_key, subject_gender=subject_gender)
        for req in variation_requests
    ]
