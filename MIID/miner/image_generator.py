# MIID/miner/image_generator.py
#
# Phase 4: Image variation generator for miners.
# Uses model-based generation from generate_variations.py.
# See MIID.miner.generate_variations module docstring for setup.

import base64
import hashlib
import io
import os
import time
import bittensor as bt
from typing import Dict, List, Optional, Tuple
from PIL import Image

from MIID.miner.generate_variations import (
    active_model_key,
    generate_one,
    get_selected_model_info,
    requested_pose_level,
)
from MIID.miner.ada_face_compare import (
    embed_image,
    face_yaw_delta,
    face_yaw_signature,
    similarity_to_embedding,
    validate_single_variation,
)


# AdaFace similarity a variation should reach before it is accepted as-is.
# Matched to the validator's own identity gate (--neuron.quality_threshold,
# default 0.6): a miner whose average identity preservation lands under that
# is filtered out of the KAV ranking entirely, however good the images look.
IDENTITY_TARGET = float(os.environ.get("MIID_IDENTITY_TARGET", "0.6"))

# Extra attempts per variation when the first one misses IDENTITY_TARGET.
# Two, because ~14% of archived slots still landed under 0.6 after one retry,
# and such a slot scores 0 whenever it is the one the validator samples. The
# time-budget guard below stops retrying before the dendrite timeout.
IDENTITY_RETRIES = int(os.environ.get("MIID_IDENTITY_RETRIES", "2"))

# Wall-clock budget for one request's generation, retries included. The
# validator's dendrite timeout is --neuron.timeout (default 1200s) and the
# response still has to be encrypted and uploaded inside it, so retries stop
# well before that: a late response scores nothing at all.
GENERATION_BUDGET_SECONDS = float(
    os.environ.get("MIID_GENERATION_BUDGET_SECONDS", "900")
)

# Head turn, in degrees, that each requested pose intensity should land in.
# The validator describes light / medium / far as ±15° / ±30° / >±45°, but
# the grader is more lenient than that: on 2026-10-07 lighting+pose slots
# whose "far" turns measured ~30° graded 5 every time, while a "light" slot
# with no visible turn graded 3. So the ranges guard against a missing or
# clearly undersized turn rather than chasing the nominal angle, which costs
# identity. An attempt outside its range is retried at full edit strength
# (the face held; the turn did not), budget permitting, and the closest
# attempt that clears the identity target wins.
POSE_RANGE_DEG = {"light": (6.0, 30.0), "medium": (15.0, 45.0), "far": (24.0, 180.0)}


def decode_base_image(base64_image: str) -> Image.Image:
    """Decode a Base64 encoded image to a PIL Image.

    Args:
        base64_image: Base64 encoded image string

    Returns:
        PIL Image object in RGB — the pipelines and the PNG encoder below both
        assume three channels, and the validator's base image is not
        guaranteed to arrive that way.
    """
    image_bytes = base64.b64decode(base64_image)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def encode_image_to_bytes(image: Image.Image, format: str = "PNG") -> bytes:
    """Encode a PIL Image to bytes.

    Args:
        image: PIL Image object
        format: Image format (PNG, JPEG, etc.)

    Returns:
        Image as bytes
    """
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    return buffer.getvalue()


def calculate_image_hash(image_bytes: bytes) -> str:
    """Calculate SHA256 hash of image bytes.

    Args:
        image_bytes: Raw image bytes

    Returns:
        SHA256 hash as hex string
    """
    return hashlib.sha256(image_bytes).hexdigest()


def _base_embedding(base_image: Image.Image):
    """Embed the base face once, or None if AdaFace cannot read it.

    None disables identity checking for the whole request rather than failing
    it: an unscored variation still earns whatever the grading API gives it,
    and no variation at all earns zero.
    """
    try:
        embedding = embed_image(base_image)
    except Exception as e:
        bt.logging.warning(
            f"AdaFace unavailable ({e}); generating without identity retries."
        )
        return None
    if embedding is None:
        bt.logging.warning(
            "AdaFace found no face in the base image; generating without "
            "identity retries."
        )
    return embedding


def _similarity(base_embedding, image: Image.Image) -> Optional[float]:
    """AdaFace similarity of one variation to the base face, or None."""
    if base_embedding is None:
        return None
    try:
        return similarity_to_embedding(base_embedding, image)
    except Exception as e:
        bt.logging.warning(f"AdaFace comparison failed: {e}")
        return None


def _base_yaw(base_image: Image.Image):
    """The base face's yaw signature, or None (pose checks then switch off)."""
    try:
        return face_yaw_signature(base_image)
    except Exception as e:
        bt.logging.warning(f"Head-pose check unavailable ({e}).")
        return None


def _yaw(base_signature, image: Image.Image) -> Optional[float]:
    """Head turn of one variation relative to the base face, or None."""
    if base_signature is None:
        return None
    try:
        return face_yaw_delta(base_signature, image)
    except Exception as e:
        bt.logging.warning(f"Head-pose check failed: {e}")
        return None


def _pose_miss(level: Optional[str], yaw: Optional[float]) -> Tuple[float, Optional[str]]:
    """(degrees outside the requested range, "more" / "less" / None).

    No requested pose, or no measurement, counts as a hit: an unmeasured turn
    is no reason to spend a retry.
    """
    if level not in POSE_RANGE_DEG or yaw is None:
        return 0.0, None
    low, high = POSE_RANGE_DEG[level]
    if yaw < low:
        return low - yaw, "more"
    if yaw > high:
        return yaw - high, "less"
    return 0.0, None


def generate_variations(
    base_image: Image.Image,
    variation_requests: List,
    identity_target: Optional[float] = None,
    max_retries: Optional[int] = None,
    time_budget: Optional[float] = None,
    subject_gender: Optional[str] = None,
) -> List[Dict]:
    """Generate image variations from a base image (see generate_variations.py).

    Flow: validator sends image_request with variation_requests (each has type +
    intensity) -> miner passes base_image + variation_requests here -> selected
    model gets the base image and a prompt built from the request's type,
    intensity, description and detail -> one variation image per request.

    Each result is scored against the base face with AdaFace before it is
    returned. A variation below ``identity_target`` is regenerated with the
    edit held back (see ``generate_one(attempt=...)``) and the best-scoring
    attempt is kept. This is where retrying is cheap: the pipeline is already
    loaded and the base embedding is already computed, so one retry costs one
    denoising pass.

    Retries stop early once ``time_budget`` seconds have gone, because a
    response that misses the validator's dendrite timeout scores zero for every
    slot, not just the one being retried.

    Args:
        base_image: PIL Image of the base face (decoded from image_request.base_image).
        variation_requests: List of validator variation requests; each has .type and .intensity
            (e.g. protocol.VariationRequest, or dict with "type"/"intensity").
            Order is preserved: first request -> first result.
        identity_target: AdaFace similarity to reach before accepting a
            variation without a retry. Defaults to IDENTITY_TARGET.
        max_retries: Extra attempts per variation. Defaults to IDENTITY_RETRIES.
        time_budget: Seconds allowed for the whole call. Defaults to
            GENERATION_BUDGET_SECONDS.
        subject_gender: "m" / "f" from the base filename, or None. Picks a
            gender-appropriate religious head covering (see generate_one).

    Returns:
        List of dicts, each containing:
            - image: PIL Image object
            - variation_type: str - which type this is (matches request order)
            - image_bytes: bytes - raw image data
            - image_hash: str - SHA256 hash for verification
            - model_key / model_id: str - which model produced it, so the
              caller can record the approach used for this request
            - identity_similarity: float | None - AdaFace score against the
              base face, None when no face could be read
            - attempts: int - how many generations this slot took
            - winning_attempt: int - which of them is the one returned
    """
    if not variation_requests:
        return []

    target = IDENTITY_TARGET if identity_target is None else identity_target
    retries = IDENTITY_RETRIES if max_retries is None else max_retries
    budget = GENERATION_BUDGET_SECONDS if time_budget is None else time_budget
    started = time.monotonic()

    model_info = get_selected_model_info()
    bt.logging.info(
        f"Using model: {model_info['key']} ({model_info['model_id']}, "
        f"{model_info['params']})"
    )

    base_embedding = _base_embedding(base_image)
    # No base embedding means no identity signal for any slot this round, so
    # retrying would only re-roll blind. A *variation* that scores None is a
    # different matter — see below.
    attempts_allowed = (retries + 1) if base_embedding is not None else 1

    # Worst observed cost of one generate+score cycle, used to decide whether a
    # retry can still finish. Checking only elapsed time is not enough: on the
    # 12B Kontext path a single cycle runs several minutes, so a retry started
    # just under the budget lands well past it — and past the budget is where
    # the validator's dendrite timeout is, which scores every slot zero.
    slowest_cycle = 0.0
    base_yaw = _base_yaw(base_image)

    variations = []
    for req in variation_requests:
        best = None
        best_key = None
        best_similarity = None
        generated = 0
        pose_level = requested_pose_level(req)
        # Settings for the next attempt. An identity miss holds the edit back
        # (more with each miss); a head turn outside its range re-rolls at
        # full strength with a nudge in the right direction.
        identity_misses = 0
        identity_bias = 0.0
        pose_hint = None

        for attempt in range(attempts_allowed):
            cycle_started = time.monotonic()
            result = generate_one(
                base_image, req, model_key=model_info["key"], attempt=attempt,
                subject_gender=subject_gender, identity_bias=identity_bias,
                pose_hint=pose_hint,
            )
            generated += 1
            similarity = _similarity(base_embedding, result["image"])
            yaw = _yaw(base_yaw, result["image"]) if pose_level else None
            slowest_cycle = max(slowest_cycle, time.monotonic() - cycle_started)
            result["identity_similarity"] = similarity
            result["pose_yaw"] = yaw
            result["winning_attempt"] = attempt + 1

            # Rank: clearing the identity target first (under it the grade is
            # capped whatever the edit), then the head turn's distance from
            # its range, then similarity. A variation AdaFace cannot read a
            # face in (None) is the worst outcome there is — the grading API
            # scores it 0 for identity — so it ranks below every scored one.
            identity_ok = similarity is not None and similarity >= target
            miss, direction = _pose_miss(pose_level, yaw)
            key = (
                identity_ok,
                -miss if identity_ok else 0.0,
                -1.0 if similarity is None else similarity,
            )
            if best is None or key > best_key:
                best, best_key, best_similarity = result, key, similarity

            if identity_ok and miss == 0.0:
                break
            if attempt >= attempts_allowed - 1:
                break

            shown = "no face detected" if similarity is None else f"{similarity:.3f}"
            if identity_ok:
                reason = (
                    f"head turn {yaw:.0f}° outside the {pose_level} range "
                    f"{POSE_RANGE_DEG[pose_level][0]:.0f}–"
                    f"{POSE_RANGE_DEG[pose_level][1]:.0f}°"
                )
                plan = f"retrying at full strength, turn {direction}."
                identity_bias, pose_hint = 0.0, direction
            else:
                reason = f"identity {shown} < {target}"
                plan = "retrying with the edit held back."
                identity_misses += 1
                identity_bias, pose_hint = min(0.8, 0.4 * identity_misses), None
            elapsed = time.monotonic() - started
            # Remaining slots must still get their first attempt, which is
            # never skipped — a missing slot scores zero. Reserve their time
            # before spending any of it on a retry.
            slots_left = len(variation_requests) - len(variations) - 1
            reserved = slots_left * slowest_cycle
            if elapsed + slowest_cycle + reserved > budget:
                bt.logging.warning(
                    f"{result['variation_type']}: {reason}, "
                    f"but a retry needs ~{slowest_cycle:.0f}s and only "
                    f"{max(0.0, budget - elapsed - reserved):.0f}s of the "
                    f"{budget:.0f}s budget is free — keeping the best attempt."
                )
                break
            bt.logging.info(f"{result['variation_type']}: {reason} — {plan}")

        var_type = best["variation_type"]
        image_bytes = encode_image_to_bytes(best["image"])
        image_hash = calculate_image_hash(image_bytes)

        variations.append({
            "image": best["image"],
            "variation_type": var_type,
            "image_bytes": image_bytes,
            "image_hash": image_hash,
            "model_key": best.get("model_key") or active_model_key() or model_info["key"],
            "model_id": best.get("model_id") or model_info["model_id"],
            "identity_similarity": best_similarity,
            "attempts": generated,
            "winning_attempt": best["winning_attempt"],
        })

        bt.logging.info(
            f"Generated {var_type} variation "
            f"(identity="
            f"{'n/a' if best_similarity is None else f'{best_similarity:.3f}'}"
            + (
                f", head turn {best['pose_yaw']:.0f}° for {pose_level}"
                if best.get("pose_yaw") is not None else ""
            )
            + f", kept attempt {best['winning_attempt']} of {generated})"
            f", hash: {image_hash[:16]}..."
        )

    bt.logging.info(
        f"Generated {len(variations)} variations in "
        f"{time.monotonic() - started:.0f}s"
    )
    return variations


def validate_face_variation(
    variation: Dict,
    base_image: Image.Image,
    min_similarity: float = 0.7
) -> bool:
    """Validate that a variation maintains face identity using AdaFace.

    Prefers the score generate_variations already computed; falls back to a
    fresh comparison for a variation dict built some other way.

    Args:
        variation: Variation dict from generate_variations (must have "image" key).
        base_image: Original base image (PIL Image).
        min_similarity: Minimum AdaFace cosine similarity threshold (default 0.7).

    Returns:
        True if variation maintains face identity, False otherwise.
    """
    if "identity_similarity" in variation:
        similarity = variation["identity_similarity"]
        return similarity is not None and similarity >= min_similarity

    return validate_single_variation(
        base_image,
        variation["image"],
        min_similarity=min_similarity,
        device=None,
    )
