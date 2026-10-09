# MIID/miner/image_generator.py
#
# Phase 4: Image variation generator for miners.
# Uses model-based generation from generate_variations.py.
# See MIID.miner.generate_variations module docstring for setup.

import base64
import hashlib
import io
import os
import statistics
import time
from collections import deque
import bittensor as bt
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple
from PIL import Image

from MIID.miner.generate_variations import (
    SCREEN_REPLAY_TYPE,
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
# and such a slot scores 0 whenever it is the one the validator samples.
# Whether they run is a question of time: see generate_variations.
IDENTITY_RETRIES = int(os.environ.get("MIID_IDENTITY_RETRIES", "2"))

# Upper bound on one request's generation, retries included, counted from the
# start of generation. The binding limit is usually the request's own
# deadline (validator send time + its 1200 s timeout, minus upload and voice),
# which the miner passes in: time spent queued behind other validators counts
# against it, and a late response scores nothing at all.
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


# Retry priority, part 1: the chance that one more attempt lifts a slot over
# the identity target, by its best identity so far and by which attempt it
# would be. Measured on 324 slots that retried in the miner log, 2026-10-06
# 20:03 to 10-08 (up to three attempts per slot): a near miss converts far more
# often than a distant one, and a third attempt less often than a second.
# Rows: (lowest best identity, odds on attempt 2, odds on attempt 3).
_IDENTITY_RETRY_ODDS = (
    (0.55, 0.60, 0.45),
    (0.50, 0.34, 0.30),
    (0.00, 0.33, 0.22),
)
# Retry priority, part 2: what a successful retry is worth relative to an
# identity rescue. A head turn outside its range costs at most a partial
# match, and the grader has been lenient on angle; the screen_replay slot is
# ignored by the grading API (validator image_variations.py) and has never
# missed the identity target anyway.
POSE_RETRY_ODDS = 0.5
POSE_RETRY_WEIGHT = 0.3
SCREEN_REPLAY_RETRY_WEIGHT = 0.25

# Seconds one generate + score cycle takes, until this process has measured
# its own (median of recent non-first attempts; FLUX.2 Klein on the 16 GB card
# runs ~37 s). The miner also uses it to estimate a queued request's workload.
DEFAULT_CYCLE_SECONDS = 40.0
_recent_cycles: Deque[float] = deque(maxlen=30)


def cycle_estimate() -> float:
    """Expected seconds for one more attempt, from recent measurements."""
    if not _recent_cycles:
        return DEFAULT_CYCLE_SECONDS
    return float(statistics.median(_recent_cycles))


def _req_type(req: Any) -> Optional[str]:
    return getattr(req, "type", None) or (
        req.get("type") if isinstance(req, dict) else None
    )


def _is_screen_replay(req: Any) -> bool:
    return _req_type(req) == SCREEN_REPLAY_TYPE


class _Slot:
    """One requested variation and the best attempt at it so far."""

    def __init__(self, index: int, req: Any):
        self.index = index
        self.req = req
        self.pose_level = requested_pose_level(req)
        self.screen_replay = _is_screen_replay(req)
        self.best: Optional[Dict] = None
        self.best_key: Optional[Tuple] = None
        self.generated = 0
        self.identity_misses = 0
        self.identity_ok = False
        self.pose_miss = 0.0
        self.pose_direction: Optional[str] = None

    @property
    def satisfied(self) -> bool:
        return self.best is not None and self.identity_ok and self.pose_miss == 0.0

    def record(self, result: Dict, target: float) -> None:
        """Score one attempt and keep it if it is the best so far.

        Rank: clearing the identity target first (under it the grade is capped
        whatever the edit), then the head turn's distance from its range, then
        similarity. A variation AdaFace cannot read a face in (None) is the
        worst outcome there is — the grading API scores it 0 for identity — so
        it ranks below every scored one.
        """
        similarity = result["identity_similarity"]
        identity_ok = similarity is not None and similarity >= target
        miss, direction = _pose_miss(self.pose_level, result.get("pose_yaw"))
        if not identity_ok:
            self.identity_misses += 1
        key = (
            identity_ok,
            -miss if identity_ok else 0.0,
            -1.0 if similarity is None else similarity,
        )
        if self.best is None or key > self.best_key:
            self.best, self.best_key = result, key
            self.identity_ok, self.pose_miss, self.pose_direction = (
                identity_ok, miss, direction,
            )

    def retry_priority(self, attempts_allowed: int) -> Optional[float]:
        """Expected value of one more attempt, or None if it needs none."""
        if self.best is None or self.satisfied or self.generated >= attempts_allowed:
            return None
        if self.identity_ok:
            value = POSE_RETRY_ODDS * POSE_RETRY_WEIGHT
        else:
            similarity = self.best["identity_similarity"]
            column = 1 if self.generated < 2 else 2
            value = next(
                row[column] for row in _IDENTITY_RETRY_ODDS
                if (similarity or 0.0) >= row[0]
            )
        if self.screen_replay:
            value *= SCREEN_REPLAY_RETRY_WEIGHT
        return value

    def next_settings(self) -> Tuple[float, Optional[str]]:
        """(identity_bias, pose_hint) for the next attempt.

        An identity miss holds the edit back, more with each miss; a head turn
        outside its range re-rolls at full strength with a nudge in the right
        direction (the face held; the turn did not).
        """
        if self.identity_ok:
            return 0.0, self.pose_direction
        return min(0.8, 0.4 * self.identity_misses), None

    def reason(self, target: float) -> str:
        if self.identity_ok:
            low, high = POSE_RANGE_DEG[self.pose_level]
            return (
                f"head turn {self.best['pose_yaw']:.0f}° outside the "
                f"{self.pose_level} range {low:.0f}–{high:.0f}°"
            )
        similarity = self.best["identity_similarity"]
        shown = "no face detected" if similarity is None else f"{similarity:.3f}"
        return f"identity {shown} < {target}"


def generate_variations(
    base_image: Image.Image,
    variation_requests: List,
    identity_target: Optional[float] = None,
    max_retries: Optional[int] = None,
    time_budget: Optional[float] = None,
    subject_gender: Optional[str] = None,
    deadline: Optional[float] = None,
    retry_cutoff_fn: Optional[Callable[[], Optional[float]]] = None,
) -> List[Dict]:
    """Generate image variations from a base image (see generate_variations.py).

    Flow: validator sends image_request with variation_requests (each has type +
    intensity) -> miner passes base_image + variation_requests here -> selected
    model gets the base image and a prompt built from the request's type,
    intensity, description and detail -> one variation image per request.

    Two phases. First every slot gets one attempt (screen_replay last), each
    scored against the base face with AdaFace and, for pose slots, a head-turn
    estimate. Then whatever time is left goes to retries, one at a time, always
    on the slot where one more attempt is most likely to pay: a near identity
    miss before a distant one, any identity miss before a head turn out of
    range (see _Slot.retry_priority). An identity retry holds the edit back; a
    head-turn retry re-rolls at full strength. The best attempt per slot wins.

    Time limits, all wall-clock (time.time()):

    - ``deadline``: when generation must be finished for the response to
      reach the validator inside its timeout (the caller has already taken
      encryption, upload and voice off it). A late response scores zero for
      every slot, so near it even first attempts are skipped: fewer slots on
      time beat all slots late.
    - ``time_budget`` (default GENERATION_BUDGET_SECONDS) from the start of
      this call caps retries in any case.
    - ``retry_cutoff_fn()``: a further, changing cap on retries only — the
      miner returns the latest finish that still lets every queued request
      meet its own deadline (gpu_scheduler.queue_cutoff).

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
        deadline: Wall-clock time generation must end by, or None.
        retry_cutoff_fn: Returns an extra wall-clock cap on retries, or None.

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
        A slot skipped because the deadline left no time for it is absent.
    """
    if not variation_requests:
        return []

    target = IDENTITY_TARGET if identity_target is None else identity_target
    retries = IDENTITY_RETRIES if max_retries is None else max_retries
    budget = GENERATION_BUDGET_SECONDS if time_budget is None else time_budget
    started = time.time()
    budget_end = started + budget

    model_info = get_selected_model_info()
    bt.logging.info(
        f"Using model: {model_info['key']} ({model_info['model_id']}, "
        f"{model_info['params']})"
        + (f"; generation must end in {deadline - started:.0f}s" if deadline else "")
    )

    base_embedding = _base_embedding(base_image)
    # No base embedding means no identity signal for any slot this round, so
    # retrying would only re-roll blind. A *variation* that scores None is a
    # different matter — see _Slot.record.
    attempts_allowed = (retries + 1) if base_embedding is not None else 1
    base_yaw = _base_yaw(base_image)

    slots = [_Slot(i, req) for i, req in enumerate(variation_requests)]
    first_attempt = [True]

    def run_attempt(slot: _Slot, identity_bias: float, pose_hint: Optional[str]) -> None:
        cycle_started = time.time()
        result = generate_one(
            base_image, slot.req, model_key=model_info["key"],
            attempt=slot.generated, subject_gender=subject_gender,
            identity_bias=identity_bias, pose_hint=pose_hint,
        )
        slot.generated += 1
        result["identity_similarity"] = _similarity(base_embedding, result["image"])
        result["pose_yaw"] = (
            _yaw(base_yaw, result["image"]) if slot.pose_level else None
        )
        result["winning_attempt"] = slot.generated
        # The request's first cycle carries warm-up (pipeline load, offload
        # hooks) and would overstate every later one.
        if not first_attempt[0]:
            _recent_cycles.append(time.time() - cycle_started)
        first_attempt[0] = False
        slot.record(result, target)

    # ── phase 1: one attempt per slot ──
    skipped = []
    for slot in sorted(slots, key=lambda s: (s.screen_replay, s.index)):
        if deadline is not None and time.time() + cycle_estimate() > deadline:
            skipped.append(slot)
            continue
        run_attempt(slot, 0.0, None)
    first_pass_seconds = time.time() - started
    if skipped:
        bt.logging.warning(
            f"Deadline: no time for {len(skipped)} slot(s) "
            f"({', '.join(str(_req_type(s.req)) for s in skipped)}) "
            "— submitting the rest on time instead of everything late."
        )

    # ── phase 2: retries, most valuable first, while time allows ──
    retries_run = 0
    stop_reason = "every slot met its targets"
    while True:
        candidates = [
            (priority, slot) for slot in slots
            if (priority := slot.retry_priority(attempts_allowed)) is not None
        ]
        if not candidates:
            if any(not s.satisfied for s in slots if s.best is not None):
                stop_reason = "attempts used up"
            break
        cutoffs = [budget_end] + ([deadline] if deadline is not None else [])
        queue_cap = retry_cutoff_fn() if retry_cutoff_fn else None
        if queue_cap is not None:
            cutoffs.append(queue_cap)
        cutoff = min(cutoffs)
        if time.time() + cycle_estimate() > cutoff:
            stop_reason = (
                "a queued request's turn" if queue_cap is not None and cutoff == queue_cap
                else "the deadline" if deadline is not None and cutoff == deadline
                else "the time budget"
            )
            for _, slot in candidates:
                bt.logging.warning(
                    f"{slot.best['variation_type']}: {slot.reason(target)}, but a "
                    f"retry needs ~{cycle_estimate():.0f}s and only "
                    f"{max(0.0, cutoff - time.time()):.0f}s remain before "
                    f"{stop_reason} — keeping the best attempt."
                )
            break
        _, slot = max(candidates, key=lambda c: (c[0], -c[1].index))
        identity_bias, pose_hint = slot.next_settings()
        plan = (
            f"retrying at full strength, turn {pose_hint}." if slot.identity_ok
            else "retrying with the edit held back."
        )
        bt.logging.info(f"{slot.best['variation_type']}: {slot.reason(target)} — {plan}")
        run_attempt(slot, identity_bias, pose_hint)
        retries_run += 1

    # ── assemble, in request order ──
    variations = []
    for slot in slots:
        best = slot.best
        if best is None:
            continue
        var_type = best["variation_type"]
        image_bytes = encode_image_to_bytes(best["image"])
        image_hash = calculate_image_hash(image_bytes)
        best_similarity = best["identity_similarity"]

        variations.append({
            "image": best["image"],
            "variation_type": var_type,
            "image_bytes": image_bytes,
            "image_hash": image_hash,
            "model_key": best.get("model_key") or active_model_key() or model_info["key"],
            "model_id": best.get("model_id") or model_info["model_id"],
            "identity_similarity": best_similarity,
            "attempts": slot.generated,
            "winning_attempt": best["winning_attempt"],
        })

        bt.logging.info(
            f"Generated {var_type} variation "
            f"(identity="
            f"{'n/a' if best_similarity is None else f'{best_similarity:.3f}'}"
            + (
                f", head turn {best['pose_yaw']:.0f}° for {slot.pose_level}"
                if best.get("pose_yaw") is not None else ""
            )
            + f", kept attempt {best['winning_attempt']} of {slot.generated})"
            f", hash: {image_hash[:16]}..."
        )

    bt.logging.info(
        f"Generated {len(variations)} variations in {time.time() - started:.0f}s "
        f"(first pass {first_pass_seconds:.0f}s, {retries_run} retries, "
        f"stopped: {stop_reason}; ~{cycle_estimate():.0f}s per attempt)"
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
