# MIID/miner/image_generator.py
#
# Phase 4: Image variation generator for miners.
# Uses model-based generation from generate_variations.py.
# See MIID.miner.generate_variations module docstring for setup.

import base64
import contextlib
import hashlib
import io
import os
import statistics
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
import bittensor as bt
import torch
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple
from PIL import Image

from MIID.miner.generate_variations import (
    SCREEN_REPLAY_TYPE,
    active_model_key,
    build_prompt,
    canonical_variation_type,
    free_gpu,
    generate_one,
    get_selected_model_info,
    prepare_prompt_cache,
    requested_accessory,
    requested_levels,
)
from MIID.miner.ada_face_compare import (
    embed_image,
    face_yaw_delta,
    face_yaw_signature,
    similarity_to_embedding,
    validate_single_variation,
)
from MIID.miner.background_check import (
    background_miss,
    describe as describe_environment,
    environment_probs,
    wanted_environment,
)
from MIID.miner.edit_check import (
    base_context as edit_base_context,
    describe as describe_edit,
    measure as measure_edit,
    misses as edit_misses,
    preserved as edit_preserved,
)
from MIID.miner import face_blend
from MIID.miner import sibling_registry as siblings


# AdaFace similarity a variation should reach before it is accepted as-is.
# The validator's own identity gate is 0.6 (--neuron.quality_threshold): a
# miner whose identity preservation lands under that is filtered out of the
# KAV ranking entirely, however good the images look. But the gate is not
# where the money is: only the top 50 of a graded round are paid, most
# miners grade 5/5, and the 50th place was decided by identity at 0.79–0.93
# on 2026-10-15 (see docs/miner.md, "Face blend"). 0.7 therefore: a slot
# between 0.6 and 0.7 is rescued too (an attempt with the edit held back),
# and between two takes of a slot the one over 0.7 wins before edit
# strength is compared. Raised from 0.6 on 2026-10-15 at the user's request
# to put identity first.
IDENTITY_TARGET = float(os.environ.get("MIID_IDENTITY_TARGET", "0.7"))

# Extra attempts per variation after the first pass: identity rescues (the
# edit held back) and edit retries (a hint naming the measured miss). Zero
# since 2026-10-16: six hotkeys share one 16 GB card, their first passes
# alone fill most of it, and the user chose to spend what is left on polish
# — full-strength re-rolls that keep the best take. With retries off, polish
# also covers the slots that missed their targets (_Slot.polish_priority):
# a new seed at full strength, a hint when an edit was measured short, and
# the ranking in _Slot.record decides which take ships. Three is right for
# a card with time to spare: over 128 archived rounds (2026-10-12/13) the
# retry loop stopped on "attempts used up" more often than on "every slot
# met its targets" (76 vs 67) while the deadline never bound once.
IDENTITY_RETRIES = int(os.environ.get("MIID_IDENTITY_RETRIES", "0"))

# Identity worth polishing toward once a slot already clears the target.
# The validator ranks miners by validation score and breaks ties on identity
# preservation, so between two miners whose edits both grade 5 the higher
# AdaFace similarity ranks higher. The first-attempt identity of a slot
# scatters with sd ≈ 0.05 (128 archived rounds), so re-rolling a satisfied
# slot at full edit strength can lift its identity — but the higher-identity
# take of two is usually the weaker edit, and choosing on identity alone
# (2026-10-13 19:49 to 2026-10-14) pulled the validation score down while
# winning only 2 of 22 polish attempts. A polish take is therefore kept only
# if every measured edit — head turn, lighting, expression, scene, accessory
# (edit_check.preserved) — is at least as strong as the incumbent's, and
# polishing runs only while no other validator's request waits for the GPU
# (one waited 204 s behind eight polish attempts). Lowest-identity slot
# first, after every rescue retry, stopping at the goal.
# MIID_POLISH_ATTEMPTS=0 switches it off.
IDENTITY_GOAL = float(os.environ.get("MIID_IDENTITY_GOAL", "0.85"))
POLISH_ATTEMPTS = int(os.environ.get("MIID_POLISH_ATTEMPTS", "2"))

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
#
# Light and medium floors raised from 6° / 15° on 2026-10-13: with those,
# the offline benchmark (MediaPipe head pose) still measured 13 of 32 "light"
# slots and 5 of 33 "medium" slots under 7°, i.e. no visible turn at all —
# and two of that day's four real -1 grades fell on combined slots with a
# "light" pose component. The far floor stays: the same grader gave 5 to
# ~30° "far" turns, and pushing further costs identity.
POSE_RANGE_DEG = {"light": (10.0, 30.0), "medium": (20.0, 45.0), "far": (24.0, 180.0)}


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
# identity rescue. A head turn, lighting change or expression outside its
# level costs at most a partial match, and the grader has been lenient on
# angle; the screen_replay slot is ignored by the grading API (validator
# image_variations.py) and has never missed the identity target anyway.
POSE_RETRY_ODDS = 0.5
POSE_RETRY_WEIGHT = 0.3
SCREEN_REPLAY_RETRY_WEIGHT = 0.25
# A background slot whose scene read as the wrong kind, or whose accessory
# is missing or something else, is a -1 or 0 against a possible 5 — a bigger
# swing than a partial match — so it outranks a pose retry and sits level
# with a near identity miss. The hint names the failure to avoid.
BACKGROUND_RETRY_ODDS = 0.6
BACKGROUND_RETRY_WEIGHT = 0.8

# Estimated validation-score cost of each kind of measured miss, used to rank
# the takes of one slot against each other (see _Slot.record): a wrong scene
# or accessory on a background slot is the difference between a 5 and a -1
# or 0; a head turn, lighting change or expression outside its level is a
# partial match, a 3 against a 5. Any miss outranks any identity margin: the
# validator scores the edit first and breaks ties on identity.
MISS_PENALTY: Dict[str, float] = {
    "scene": 6.0, "accessory": 5.0, "pose": 2.0, "lighting": 2.0, "expression": 2.0,
    # Near-identical to another hotkey's take (sibling_registry): the grader's
    # duplicate penalty is -3 to -5 and can watchlist every hotkey involved.
    "duplicate": 6.0,
}

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


def _is_oom(error: BaseException) -> bool:
    """Is ``error`` (or what it wraps — generate_one re-raises as
    RuntimeError) a CUDA out-of-memory error?"""
    seen = 0
    while error is not None and seen < 8:
        if isinstance(error, torch.cuda.OutOfMemoryError) or "out of memory" in str(error):
            return True
        error = error.__cause__ or error.__context__
        seen += 1
    return False


def _recover_gpu_memory() -> None:
    """Give back everything this process holds on the GPU between attempts:
    the image model's resident weights (re-uploaded by the next attempt) and
    PyTorch's cached blocks. What the sibling processes hold stays theirs."""
    import gc

    gc.collect()
    free_gpu()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


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
        self.levels = requested_levels(req)
        self.pose_level = self.levels.get("pose_edit")
        self.screen_replay = _is_screen_replay(req)
        # Keyed on the canonical type: a generic "background_edit" request is
        # still a background slot, resolved to in/out from its description.
        self.scene = wanted_environment(canonical_variation_type(req))
        self.background = self.scene is not None
        self.accessory = requested_accessory(req) if self.background else None
        self.best: Optional[Dict] = None
        self.best_key: Optional[Tuple] = None
        self.generated = 0
        self.polished = 0
        self.identity_misses = 0
        self.identity_ok = False
        self.pose_miss = 0.0
        self.pose_direction: Optional[str] = None
        self.bg_miss: Optional[str] = None
        self.edit_misses: List[Tuple[str, str]] = []
        # The sibling take this one is near-identical to, when it is.
        self.dup: Optional[Dict] = None

    @property
    def satisfied(self) -> bool:
        return (
            self.best is not None and self.identity_ok
            and self.pose_miss == 0.0 and self.bg_miss is None
            and not self.edit_misses and self.dup is None
        )

    def snapshot(self) -> Tuple:
        """The current best take and its judgement, for restore()."""
        return (
            self.best, self.best_key, self.identity_ok, self.pose_miss,
            self.pose_direction, self.bg_miss, list(self.edit_misses), self.dup,
        )

    def restore(self, snapshot: Tuple) -> None:
        """Put a snapshot() back as the best take (a polish take that lost an edit)."""
        (
            self.best, self.best_key, self.identity_ok, self.pose_miss,
            self.pose_direction, self.bg_miss, self.edit_misses, self.dup,
        ) = snapshot

    def polish_priority(
        self, goal: float, polish_allowed: int, include_unsatisfied: bool = False,
    ) -> Optional[float]:
        """Identity headroom a full-strength re-roll could still close, or None.

        A slot that meets its targets is polished while it is short of the
        goal. One that does not is the retry loop's business — unless
        ``include_unsatisfied`` (retries are off), when it is polished first:
        a missed target outranks any identity headroom, since under the 0.6
        gate the whole challenge is capped and a measured edit miss is a
        grade lost whenever that slot is sampled. screen_replay is skipped:
        the grading API ignores that slot.
        """
        if self.best is None or self.screen_replay or self.polished >= polish_allowed:
            return None
        similarity = self.best["identity_similarity"]
        if not self.satisfied:
            if not include_unsatisfied:
                return None
            return 1.0 + (goal - min(similarity or 0.0, goal))
        if similarity is None or similarity >= goal:
            return None
        return goal - similarity

    def record(self, result: Dict, target: float) -> None:
        """Score one attempt and keep it if it is the best so far.

        Rank: clearing the identity target first (under it the grade is capped
        whatever the edit), then the estimated validation-score cost of every
        measured miss — a head turn outside its range, a background slot
        showing the wrong kind of scene or missing its accessory, a lighting
        change or expression that reads a level short (MISS_PENALTY) — and
        only then similarity. A variation AdaFace cannot read a face in
        (None) is the worst outcome there is — the grading API scores it 0
        for identity — so it ranks below every scored one.
        """
        similarity = result["identity_similarity"]
        identity_ok = similarity is not None and similarity >= target
        miss, direction = _pose_miss(self.pose_level, result.get("pose_yaw"))
        bg_miss = (
            background_miss(result["variation_type"], result.get("environment"))
            if self.background else None
        )
        misses = edit_misses(self.levels, result.get("edit"))
        sibling = result.get("sibling")
        dup = sibling if sibling and sibling.get("too_close") else None
        penalty = 0.0
        if miss:
            penalty += MISS_PENALTY["pose"] + miss / 45.0
        if bg_miss:
            penalty += MISS_PENALTY["scene"]
        for component, _ in misses:
            penalty += MISS_PENALTY[component]
        if dup:
            penalty += MISS_PENALTY["duplicate"]
        if not identity_ok:
            self.identity_misses += 1
        key = (
            identity_ok,
            -penalty if identity_ok else 0.0,
            -1.0 if similarity is None else similarity,
        )
        if self.best is None or key > self.best_key:
            self.best, self.best_key = result, key
            self.identity_ok, self.pose_miss, self.pose_direction = (
                identity_ok, miss, direction,
            )
            self.bg_miss = bg_miss
            self.edit_misses = misses
            self.dup = dup

    def rerecord(self, result: Dict, target: float) -> None:
        """Re-judge a take whose measurements changed after it was recorded
        (a face blend that finished in the background raised its identity
        and re-measured its edit). The best take is re-keyed in place; any
        other take is simply recorded again and wins if it now ranks higher."""
        if self.best is not result:
            self.record(result, target)
            return
        self.best, self.best_key = None, None
        misses_before = self.identity_misses
        self.record(result, target)
        self.identity_misses = misses_before

    def retry_priority(self, attempts_allowed: int) -> Optional[float]:
        """Expected value of one more attempt, or None if it needs none."""
        if self.best is None or self.satisfied or self.generated >= attempts_allowed:
            return None
        if self.identity_ok and (
            self.bg_miss is not None or self.dup is not None
            or any(component == "accessory" for component, _ in self.edit_misses)
        ):
            value = BACKGROUND_RETRY_ODDS * BACKGROUND_RETRY_WEIGHT
        elif self.identity_ok:
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

    def next_settings(self) -> Tuple[float, Dict[str, str]]:
        """(identity_bias, hints by component) for the next attempt.

        An identity miss holds the edit back, more with each miss; a measured
        miss of the edit — head turn, scene, lighting, expression, accessory
        — re-rolls at full strength with a hint naming it (the face held; the
        edit did not).
        """
        if not self.identity_ok:
            return min(0.8, 0.4 * self.identity_misses), {}
        hints: Dict[str, str] = {}
        if self.pose_direction:
            hints["pose"] = self.pose_direction
        if self.bg_miss:
            hints["background"] = self.bg_miss
        for component, hint in self.edit_misses:
            hints[component] = hint
        if self.dup:
            hints["duplicate"] = "vary"
        return 0.0, hints

    def reason(self, target: float) -> str:
        if not self.identity_ok:
            similarity = self.best["identity_similarity"]
            shown = "no face detected" if similarity is None else f"{similarity:.3f}"
            return f"identity {shown} < {target}"
        parts = []
        if self.dup:
            parts.append(
                f"near-identical to {siblings.Match(**{k: self.dup[k] for k in ('hotkey', 'variation_type', 'attempt', 'bits', 'cosine')}).describe()}"
            )
        if self.bg_miss is not None:
            parts.append(
                f"background reads as an {self.bg_miss} scene, {self.scene} scene requested "
                f"({describe_environment(self.best['variation_type'], self.best.get('environment'))})"
            )
        if self.pose_miss:
            low, high = POSE_RANGE_DEG[self.pose_level]
            parts.append(
                f"head turn {self.best['pose_yaw']:.0f}° outside the "
                f"{self.pose_level} range {low:.0f}–{high:.0f}°"
            )
        edit = self.best.get("edit") or {}
        for component, hint in self.edit_misses:
            if component == "lighting":
                m = edit["lighting"]
                parts.append(
                    f"lighting change reads {m['level']} ({m['magnitude']:.2f}), "
                    f"{self.levels.get('lighting_edit')} requested"
                )
            elif component == "expression":
                m = edit["expression"]
                parts.append(
                    f"expression still reads neutral ({m['neutral']:.2f}), "
                    f"{self.levels.get('expression_edit')} requested"
                )
            elif component == "accessory":
                m = edit["accessory"]
                parts.append(
                    f"accessory {m['requested']} {hint} (read as {m['top']} "
                    f"{m['top_confidence']:.2f})"
                )
        return "; ".join(parts) or "edit measured short"


def generate_variations(
    base_image: Image.Image,
    variation_requests: List,
    identity_target: Optional[float] = None,
    max_retries: Optional[int] = None,
    time_budget: Optional[float] = None,
    subject_gender: Optional[str] = None,
    deadline: Optional[float] = None,
    retry_cutoff_fn: Optional[Callable[[], Optional[float]]] = None,
    identity_goal: Optional[float] = None,
    polish_attempts: Optional[int] = None,
    sibling: Optional[siblings.Context] = None,
    yield_fn: Optional[Callable[[float, int], Optional[float]]] = None,
) -> List[Dict]:
    """Generate image variations from a base image (see generate_variations.py).

    Flow: validator sends image_request with variation_requests (each has type +
    intensity) -> miner passes base_image + variation_requests here -> selected
    model gets the base image and a prompt built from the request's type,
    intensity, description and detail -> one variation image per request.

    Three phases. First every slot gets one attempt (screen_replay last), each
    scored against the base face with AdaFace and, for pose slots, a head-turn
    estimate. Then whatever time is left goes to retries, one at a time, always
    on the slot where one more attempt is most likely to pay: a near identity
    miss before a distant one, any identity miss before a head turn out of
    range (see _Slot.retry_priority). An identity retry holds the edit back; a
    head-turn retry re-rolls at full strength. Last, time permitting, slots
    that already meet their targets but sit under ``identity_goal`` are
    re-rolled at full strength, lowest identity first, up to
    ``polish_attempts`` times each (see _Slot.polish_priority) — the ranking
    tiebreaker is identity preservation, so a higher-identity take of the same
    edit is worth the time. The best attempt per slot wins throughout.

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
        yield_fn: Called before each retry and polish attempt with the latest
            time it could still start and its phase — 1 for an identity
            rescue (the slot is under the identity target), 2 for an edit
            retry or polish. It may hand the GPU to more urgent queued work
            (a first pass, a rescue, an earlier-deadline request at the same
            phase) and block until this request's turn comes round again,
            returning the seconds waited — or None when that latest time
            passed first, in which case the GPU is gone and no further
            attempt is made.
        identity_goal: AdaFace similarity above which a satisfied slot is left
            alone. Defaults to IDENTITY_GOAL.
        polish_attempts: Full-strength re-rolls allowed per satisfied slot.
            Defaults to POLISH_ATTEMPTS; 0 disables polishing.
        sibling: This hotkey's view of the challenge in the sibling ledger
            (sibling_registry.context), or None. With it, every attempt is
            compared with the other hotkeys' takes of the same slot, a
            near-identical one is re-rolled, and each attempt is recorded for
            the siblings to steer away from.

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
    goal = IDENTITY_GOAL if identity_goal is None else identity_goal
    polish_allowed = POLISH_ATTEMPTS if polish_attempts is None else polish_attempts
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
    try:
        edit_context = edit_base_context(base_image)
    except Exception as e:
        bt.logging.warning(f"Edit measurements unavailable ({e}); takes are ranked on identity and pose only.")
        edit_context = {}
    # The base face's landmarks, for blending it back into background takes.
    base_geometry = face_blend.base_geometry(base_image, edit_context)

    slots = [_Slot(i, req) for i, req in enumerate(variation_requests)]
    first_attempt = [True]

    # The request's prompts, built ahead so FLUX.2 klein can encode them in
    # one text-encoder pass (generate_variations.PromptCache) instead of one
    # per attempt — the 8 GB text encoder then leaves the GPU to the
    # transformer for the rest of the request — and so a polish re-roll can
    # re-use a slot's prompt exactly. None for models without embeddings.
    prompt_cache = prepare_prompt_cache(model_info["key"])
    first_prompts: Dict[int, str] = {}
    for slot in slots:
        try:
            first_prompts[slot.index] = build_prompt(
                slot.req, attempt=0, identity_bias=0.0, subject_gender=subject_gender,
            )
        except Exception as e:  # noqa: BLE001 — generate_one builds its own then
            bt.logging.warning(f"Could not build slot {slot.index}'s prompt ahead: {e}")
    if prompt_cache is not None and first_prompts:
        try:
            encode_started = time.time()
            encoded = prompt_cache.warm(list(first_prompts.values()))
            bt.logging.debug(f"Encoded {encoded} first-pass prompt(s) in {time.time() - encode_started:.1f}s")
        except Exception as e:  # noqa: BLE001
            bt.logging.warning(f"Prompt embeddings unavailable ({e}); encoding per attempt.")
            prompt_cache = None

    # Everything that judges a take after it is generated — identity, head
    # turn, scene, edit measurements, the face blend, the sibling check — is
    # CPU work or a small GPU model taking a few seconds, and the big model
    # would sit idle for it. During the first pass it runs in one worker
    # thread while the next slot generates; a slot is settled — judged and
    # recorded — before anything depends on it (a retry, polish, ranking).
    judge_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="judge")
    pending: Dict[int, Future] = {}
    # The judging thread's GPU work (AdaFace, its face detector, the blend's
    # identity reads) runs on its own CUDA stream. On the default stream it
    # queues behind the thousands of denoising kernels the main thread keeps
    # launching and finishes only when they do — measured live on
    # 2026-10-16: identity 13–44 s per slot instead of 0.7 s, the first pass
    # no shorter than before the overlap, and the attempt estimate inflated
    # to 45 s. The current stream is per thread, so the main thread is not
    # affected.
    judge_stream = torch.cuda.Stream() if torch.cuda.is_available() else None

    def judge(slot: _Slot, result: Dict, cycle_started: float, generated_at: float) -> None:
        """Measure, blend, check against the siblings and record one attempt."""
        with (torch.cuda.stream(judge_stream) if judge_stream is not None else contextlib.nullcontext()):
            result["identity_similarity"] = _similarity(base_embedding, result["image"])
            identity_at = time.time()
            result["pose_yaw"] = (
                _yaw(base_yaw, result["image"]) if slot.pose_level else None
            )
            result["environment"] = (
                environment_probs(result["image"]) if slot.background else None
            )
            try:
                result["edit"] = measure_edit(edit_context, result["image"], slot.levels, slot.accessory)
            except Exception as e:
                bt.logging.warning(f"Edit measurement failed: {e}")
                result["edit"] = None
            # A take short of the blend goal on a blendable slot gets the base
            # face blended back in (face_blend): identity rises, the edit stays,
            # and the take is re-measured so the ranking sees the final image.
            result["blend"] = None
            if face_blend.wanted(slot.background, result):
                try:
                    result["blend"] = face_blend.apply(
                        base_image, base_embedding, base_geometry, result,
                        edit_context=edit_context, levels=slot.levels,
                        accessory=slot.accessory, scene=slot.scene,
                    )
                except Exception as e:
                    bt.logging.warning(f"Face blend failed: {e}")
            result["sibling"] = None
            if sibling is not None and not slot.screen_replay:
                try:
                    fingerprint = siblings.fingerprint(result["image"])
                    match = sibling.closest(result["variation_type"], fingerprint)
                    result["sibling"] = match.as_dict() if match else None
                    sibling.register(result["variation_type"], result["winning_attempt"], fingerprint)
                except Exception as e:
                    bt.logging.warning(f"Sibling ledger failed: {e}")
            if judge_stream is not None:
                judge_stream.synchronize()
        checked_at = time.time()
        # The request's first cycle carries warm-up (pipeline load, offload
        # hooks) and would overstate every later one. A cycle is the GPU
        # time plus the judging, whether or not the two overlapped: it is
        # what a retry or polish attempt, which cannot overlap, will cost.
        if not first_attempt[0]:
            _recent_cycles.append((generated_at - cycle_started) + (checked_at - generated_at))
        first_attempt[0] = False
        slot.record(result, target)
        measured = describe_edit(result.get("edit"))
        blended = face_blend.describe(result.get("blend"))
        bt.logging.debug(
            f"{result['variation_type']}: attempt {result['winning_attempt']} — generate "
            f"{generated_at - cycle_started:.1f}s, identity {identity_at - generated_at:.1f}s, "
            f"checks {checked_at - identity_at:.1f}s"
            + (f"; {measured}" if measured else "")
            + (f"; {blended}" if blended else "")
        )
        blend = result.get("blend")
        if blend and blend.get("applied"):
            bt.logging.info(f"{result['variation_type']}: {face_blend.describe(blend)}, {blend['seconds']:.1f}s")
        elif blend and blend.get("reason"):
            bt.logging.debug(f"{result['variation_type']}: face blend skipped — {blend['reason']}")

    def settle(slot: _Slot) -> None:
        """Wait for the slot's deferred judgement, if any. A judgement that
        failed loses that one take, not the request."""
        future = pending.pop(slot.index, None)
        if future is not None:
            try:
                future.result()
            except Exception as e:  # noqa: BLE001
                bt.logging.error(f"{_req_type(slot.req)}: judging the take failed ({e}); dropping it.")

    def settle_all() -> None:
        for slot in slots:
            settle(slot)

    def run_attempt(
        slot: _Slot, identity_bias: float, hints: Optional[Dict[str, str]] = None,
        defer: bool = False, prompt: Optional[str] = None,
    ) -> None:
        settle(slot)
        cycle_started = time.time()
        result = generate_one(
            base_image, slot.req, model_key=model_info["key"],
            attempt=slot.generated, subject_gender=subject_gender,
            identity_bias=identity_bias, hints=hints,
            prompt=prompt, prompt_cache=prompt_cache,
        )
        generated_at = time.time()
        slot.generated += 1
        result["winning_attempt"] = slot.generated
        if defer:
            pending[slot.index] = judge_pool.submit(judge, slot, result, cycle_started, generated_at)
        else:
            judge(slot, result, cycle_started, generated_at)

    def attempt(slot: _Slot, *args: Any, **kwargs: Any) -> bool:
        """run_attempt that a failure cannot take the request down with.

        Before this, one CUDA out-of-memory error on the first slot ended the
        whole request with nothing submitted — every round from 2026-10-16
        14:13 until the restart, once the six processes' leaked VRAM had
        filled the card. Now an out-of-memory attempt gives back this
        process's GPU memory and runs once more; an attempt that still fails
        (or fails any other way) loses only that slot. False when the
        attempt did not produce a take.
        """
        for retry in (False, True):
            try:
                run_attempt(slot, *args, **kwargs)
                return True
            except Exception as e:  # noqa: BLE001
                fits = deadline is None or time.time() + cycle_estimate() <= deadline
                if not retry and fits and _is_oom(e):
                    bt.logging.warning(
                        f"{_req_type(slot.req)}: CUDA out of memory; freeing this process's "
                        "GPU memory and trying the slot once more."
                    )
                    _recover_gpu_memory()
                    continue
                bt.logging.error(f"{_req_type(slot.req)}: attempt failed ({e}).")
                return False
        return False

    # ── phase 1: one attempt per slot ──
    skipped = []
    failed: List[_Slot] = []
    for slot in sorted(slots, key=lambda s: (s.screen_replay, s.index)):
        if deadline is not None and time.time() + cycle_estimate() > deadline:
            skipped.append(slot)
            continue
        if not attempt(slot, 0.0, None, defer=True, prompt=first_prompts.get(slot.index)):
            failed.append(slot)
    settle_all()
    judge_pool.shutdown(wait=True)
    first_pass_seconds = time.time() - started
    if prompt_cache is not None:
        bt.logging.debug(prompt_cache.describe())
    if skipped:
        bt.logging.warning(
            f"Deadline: no time for {len(skipped)} slot(s) "
            f"({', '.join(str(_req_type(s.req)) for s in skipped)}) "
            "— submitting the rest on time instead of everything late."
        )
    if failed:
        bt.logging.warning(
            f"{len(failed)} slot(s) failed to generate "
            f"({', '.join(str(_req_type(s.req)) for s in failed)}) — submitting the rest."
        )

    def next_cutoff() -> Tuple[float, str]:
        """(wall-clock cap on the next attempt, what imposes it)."""
        cutoffs = [budget_end] + ([deadline] if deadline is not None else [])
        queue_cap = retry_cutoff_fn() if retry_cutoff_fn else None
        if queue_cap is not None:
            cutoffs.append(queue_cap)
        cutoff = min(cutoffs)
        reason = (
            "a queued request's turn" if queue_cap is not None and cutoff == queue_cap
            else "the deadline" if deadline is not None and cutoff == deadline
            else "the time budget"
        )
        return cutoff, reason

    # ── phase 2: retries, most valuable first, while time allows ──
    retries_run = 0
    stop_reason = "every slot met its targets"
    gpu_lost = False  # set when a hand-over never came back: no more GPU work
    while True:
        candidates = [
            (priority, slot) for slot in slots
            if (priority := slot.retry_priority(attempts_allowed)) is not None
        ]
        if not candidates:
            if any(not s.satisfied for s in slots if s.best is not None):
                stop_reason = "attempts used up" if retries > 0 else "retries off, polish takes over"
            break
        cutoff, stop_reason = next_cutoff()
        if time.time() + cycle_estimate() > cutoff:
            for _, slot in candidates:
                bt.logging.warning(
                    f"{slot.best['variation_type']}: {slot.reason(target)}, but a "
                    f"retry needs ~{cycle_estimate():.0f}s and only "
                    f"{max(0.0, cutoff - time.time()):.0f}s remain before "
                    f"{stop_reason} — keeping the best attempt."
                )
            break
        _, slot = max(candidates, key=lambda c: (c[0], -c[1].index))
        if yield_fn is not None:
            # More urgent queued work goes first — every first pass, then
            # identity rescues, then edit retries by deadline: hand the GPU
            # over and queue again behind it.
            phase = 1 if not slot.identity_ok else 2
            waited = yield_fn(cutoff - cycle_estimate(), phase)
            if waited is None:
                gpu_lost = True
                stop_reason = "the deadline, reached while queued for the GPU"
                bt.logging.warning(
                    f"{len(candidates)} slot(s) could use a retry, but the GPU went "
                    "to more urgent queued work until no time was left — keeping the best attempts."
                )
                break
            if waited > 0:
                bt.logging.info(
                    f"Handed the GPU to more urgent queued work for {waited:.0f}s; "
                    f"back for {'identity rescues' if phase == 1 else 'retries'}."
                )
                continue
        identity_bias, hints = slot.next_settings()
        plan = (
            "retrying at full strength with hints ("
            + ", ".join(f"{component}: {hint}" for component, hint in hints.items())
            + ")."
            if slot.identity_ok and hints
            else "retrying at full strength." if slot.identity_ok
            else "retrying with the edit held back."
        )
        bt.logging.info(f"{slot.best['variation_type']}: {slot.reason(target)} — {plan}")
        if not attempt(slot, identity_bias, hints):
            # Another attempt would most likely fail the same way; ship
            # what the first pass made.
            gpu_lost = True
            stop_reason = "a failed attempt"
            break
        retries_run += 1

    # ── phase 3: polish — full-strength re-rolls, while time allows ──
    # Satisfied slots short of the identity goal; with retries off, first the
    # slots that missed a target (identity or a measured edit), since nothing
    # else will re-roll them.
    polish_unsatisfied = retries == 0
    polish_run = 0
    polish_kept = 0
    polish_stop = "every slot at the goal or polished out"
    while polish_allowed > 0 and not gpu_lost:
        candidates = [
            (headroom, slot) for slot in slots
            if (headroom := slot.polish_priority(goal, polish_allowed, polish_unsatisfied)) is not None
        ]
        if not candidates:
            polish_stop = "every slot at the goal or polished out"
            break
        # Polish is surplus work: any queued first pass, identity rescue or
        # earlier-deadline request goes first. With a scheduler the GPU is
        # handed over and taken back (phase 2); without one, polish simply
        # stops as soon as another request waits.
        if yield_fn is None and retry_cutoff_fn is not None and retry_cutoff_fn() is not None:
            polish_stop = "a queued request"
            bt.logging.info(
                f"Polish: {len(candidates)} slot(s) under identity {goal:.2f}, but "
                "another request is waiting for the GPU — keeping the best attempts."
            )
            break
        cutoff, polish_stop = next_cutoff()
        if time.time() + cycle_estimate() > cutoff:
            bt.logging.info(
                f"Polish: {len(candidates)} slot(s) under identity {goal:.2f}, but an "
                f"attempt needs ~{cycle_estimate():.0f}s and only "
                f"{max(0.0, cutoff - time.time()):.0f}s remain before {polish_stop} "
                "— keeping the best attempts."
            )
            break
        _, slot = max(candidates, key=lambda c: (c[0], -c[1].index))
        if yield_fn is not None:
            # A slot still under the identity target queues at rescue
            # priority (phase 1); everything else is surplus work (phase 2).
            phase = 1 if not slot.identity_ok else 2
            waited = yield_fn(cutoff - cycle_estimate(), phase)
            if waited is None:
                gpu_lost = True
                polish_stop = "the deadline, reached while queued for the GPU"
                bt.logging.info(
                    f"Polish: {len(candidates)} slot(s) under identity {goal:.2f}, but the "
                    "GPU went to more urgent queued work until no time was left — keeping the best attempts."
                )
                break
            if waited > 0:
                bt.logging.info(
                    f"Handed the GPU to more urgent queued work for {waited:.0f}s; back for polish."
                )
                continue
        incumbent = slot.snapshot()
        incumbent_satisfied = slot.satisfied
        before = incumbent[0]["identity_similarity"]
        slot.polished += 1
        hints: Optional[Dict[str, str]] = None
        if incumbent_satisfied:
            # Full strength, no hints: the edit landed, only the face is being
            # re-rolled. _Slot.record adopts the new take if it ranks higher —
            # and then it stays only if it kept every measured edit of the one
            # it replaced, because a higher-identity take is usually a weaker
            # edit. The same prompt as the first pass: only the seed moves.
            ran = attempt(slot, 0.0, None, prompt=first_prompts.get(slot.index))
        else:
            # A missed target: full strength again (never the edit held back),
            # with a hint naming a measured edit miss; the ranking in record()
            # keeps whichever take is better.
            _, hints = slot.next_settings()
            bt.logging.info(
                f"{slot.best['variation_type']}: {slot.reason(target)} — re-rolling at full strength"
                + (" with hints (" + ", ".join(f"{c}: {h}" for c, h in hints.items()) + ")." if hints else ".")
            )
            ran = attempt(slot, 0.0, hints or None, prompt=None if hints else first_prompts.get(slot.index))
        if not ran:
            polish_stop = "a failed attempt"
            break
        polish_run += 1
        if slot.best is incumbent[0]:
            verdict = "kept the earlier take"
        elif not incumbent_satisfied:
            polish_kept += 1
            verdict = "kept the new take" + ("" if slot.satisfied else f" ({slot.reason(target)})")
        else:
            kept, lost = edit_preserved(slot.levels, incumbent[0], slot.best, slot.scene)
            if kept:
                polish_kept += 1
                verdict = "kept the new take"
            else:
                slot.restore(incumbent)
                verdict = f"kept the earlier take: the new one weakened the {lost}"
        after = slot.best["identity_similarity"]
        shown = lambda v: "n/a" if v is None else f"{v:.3f}"  # noqa: E731
        bt.logging.info(
            f"{slot.best['variation_type']}: polish {slot.polished}/{polish_allowed} — "
            f"identity {shown(before)} → {shown(after)} ({verdict})"
        )

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
            "environment": best.get("environment"),
            "pose_yaw": best.get("pose_yaw"),
            "edit": best.get("edit"),
            "sibling": best.get("sibling"),
            "blend": best.get("blend"),
        })

        environment = describe_environment(var_type, best.get("environment"))
        measured = describe_edit(best.get("edit"))
        blended = face_blend.describe(best.get("blend"))
        bt.logging.info(
            f"Generated {var_type} variation "
            f"(identity="
            f"{'n/a' if best_similarity is None else f'{best_similarity:.3f}'}"
            + (
                f", head turn {best['pose_yaw']:.0f}° for {slot.pose_level}"
                if best.get("pose_yaw") is not None else ""
            )
            + (f", background {environment}" if environment else "")
            + (f", {measured}" if measured else "")
            + (f", {blended}" if blended else "")
            + (f", {slot.reason(target)}" if not slot.satisfied and slot.identity_ok else "")
            + f", kept attempt {best['winning_attempt']} of {slot.generated})"
            f", hash: {image_hash[:16]}..."
        )

    bt.logging.info(
        f"Generated {len(variations)} variations in {time.time() - started:.0f}s "
        f"(first pass {first_pass_seconds:.0f}s, {retries_run} retries, "
        f"stopped: {stop_reason}; {polish_run} polish attempts, {polish_kept} kept, "
        f"stopped: {polish_stop}; ~{cycle_estimate():.0f}s per attempt)"
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
