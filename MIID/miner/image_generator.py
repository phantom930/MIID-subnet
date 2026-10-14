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
    canonical_variation_type,
    generate_one,
    get_selected_model_info,
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
from MIID.miner import sibling_registry as siblings


# AdaFace similarity a variation should reach before it is accepted as-is.
# Matched to the validator's own identity gate (--neuron.quality_threshold,
# default 0.6): a miner whose average identity preservation lands under that
# is filtered out of the KAV ranking entirely, however good the images look.
IDENTITY_TARGET = float(os.environ.get("MIID_IDENTITY_TARGET", "0.6"))

# Extra attempts per variation when the first one misses IDENTITY_TARGET.
# Three: over 128 archived rounds (2026-10-12/13) the retry loop stopped on
# "attempts used up" more often than on "every slot met its targets" (76 vs
# 67) while the deadline never bound once, and a tenth of the
# pose+expression slots still shipped under 0.6 — a slot that scores 0
# whenever it is the one the validator samples. Whether they run is a
# question of time: see generate_variations.
IDENTITY_RETRIES = int(os.environ.get("MIID_IDENTITY_RETRIES", "3"))

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

    def polish_priority(self, goal: float, polish_allowed: int) -> Optional[float]:
        """Identity headroom a full-strength re-roll could still close, or None.

        Only a slot that already meets its targets is polished — an unsatisfied
        one is the rescue loop's business — and only while it is short of the
        goal. screen_replay is skipped: the grading API ignores that slot.
        """
        if self.best is None or not self.satisfied or self.screen_replay:
            return None
        if self.polished >= polish_allowed:
            return None
        similarity = self.best["identity_similarity"]
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

    slots = [_Slot(i, req) for i, req in enumerate(variation_requests)]
    first_attempt = [True]

    def run_attempt(
        slot: _Slot, identity_bias: float, hints: Optional[Dict[str, str]] = None,
    ) -> None:
        cycle_started = time.time()
        result = generate_one(
            base_image, slot.req, model_key=model_info["key"],
            attempt=slot.generated, subject_gender=subject_gender,
            identity_bias=identity_bias, hints=hints,
        )
        generated_at = time.time()
        slot.generated += 1
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
        result["sibling"] = None
        if sibling is not None and not slot.screen_replay:
            try:
                fingerprint = siblings.fingerprint(result["image"])
                match = sibling.closest(result["variation_type"], fingerprint)
                result["sibling"] = match.as_dict() if match else None
                sibling.register(result["variation_type"], slot.generated, fingerprint)
            except Exception as e:
                bt.logging.warning(f"Sibling ledger failed: {e}")
        checked_at = time.time()
        result["winning_attempt"] = slot.generated
        # The request's first cycle carries warm-up (pipeline load, offload
        # hooks) and would overstate every later one.
        if not first_attempt[0]:
            _recent_cycles.append(checked_at - cycle_started)
        first_attempt[0] = False
        slot.record(result, target)
        measured = describe_edit(result.get("edit"))
        bt.logging.debug(
            f"{result['variation_type']}: attempt {slot.generated} — generate "
            f"{generated_at - cycle_started:.1f}s, identity {identity_at - generated_at:.1f}s, "
            f"checks {checked_at - identity_at:.1f}s"
            + (f"; {measured}" if measured else "")
        )

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
    while True:
        candidates = [
            (priority, slot) for slot in slots
            if (priority := slot.retry_priority(attempts_allowed)) is not None
        ]
        if not candidates:
            if any(not s.satisfied for s in slots if s.best is not None):
                stop_reason = "attempts used up"
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
        run_attempt(slot, identity_bias, hints)
        retries_run += 1

    # ── phase 3: polish — full-strength re-rolls of satisfied slots, while time allows ──
    polish_run = 0
    polish_kept = 0
    polish_stop = "every slot at the goal or polished out"
    while polish_allowed > 0:
        candidates = [
            (headroom, slot) for slot in slots
            if (headroom := slot.polish_priority(goal, polish_allowed)) is not None
        ]
        if not candidates:
            polish_stop = "every slot at the goal or polished out"
            break
        # Polish is surplus work: another validator's queued request is worth
        # more than a tiebreak here, so hand the GPU over as soon as one waits.
        if retry_cutoff_fn is not None and retry_cutoff_fn() is not None:
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
        incumbent = slot.snapshot()
        before = incumbent[0]["identity_similarity"]
        slot.polished += 1
        # Full strength, no hints: the edit landed, only the face is being
        # re-rolled. _Slot.record adopts the new take if it ranks higher —
        # and then it stays only if it kept every measured edit of the one it
        # replaced, because a higher-identity take is usually a weaker edit.
        run_attempt(slot, 0.0, None)
        polish_run += 1
        if slot.best is incumbent[0]:
            verdict = "kept the earlier take"
        else:
            kept, lost = edit_preserved(slot.levels, incumbent[0], slot.best, slot.scene)
            if kept:
                polish_kept += 1
                verdict = "kept the new take"
            else:
                slot.restore(incumbent)
                verdict = f"kept the earlier take: the new one weakened the {lost}"
        after = slot.best["identity_similarity"]
        bt.logging.info(
            f"{slot.best['variation_type']}: polish {slot.polished}/{polish_allowed} — "
            f"identity {before:.3f} → {after:.3f} ({verdict})"
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
        })

        environment = describe_environment(var_type, best.get("environment"))
        measured = describe_edit(best.get("edit"))
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
