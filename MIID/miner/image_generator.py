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
import random
import statistics
import threading
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
import bittensor as bt
import torch
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple
from PIL import Image

from MIID.miner.generate_variations import (
    SCREEN_REPLAY_TYPE,
    _req_field,
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
    POSE_TARGET_DEG,
    base_context as edit_base_context,
    describe as describe_edit,
    measure as measure_edit,
    misses as edit_misses,
    preserved as edit_preserved,
)
from MIID.miner import face_blend, liveness_check, screen_replay


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

# The identity a face variation must reach to count at all. Above it the
# grade is decided by the edit — a 5 needs every component in its requested
# intensity bin — so takes are ranked edit first from here up. Under it the
# variation fails whatever the edit, so a take over the gate beats every
# take under it. The subnet owner's stated bar (2026-10-26) is identity
# above 0.7 on each of the five face variations, with the prompt followed;
# the gate was 0.6 (the KAV filter) from 2026-10-23 to 10-26.
#
# A slot whose best take is under the gate is retried (see next_settings):
# full-strength re-rolls alternate with ones that hold the edit back, so
# the edit is not given up first — held-back rescues alone shipped far
# pose requests as medium or light turns (7 of 7 on 2026-10-23's morning
# rounds) — and it gets GATE_RETRIES attempts beyond IDENTITY_RETRIES.
IDENTITY_GATE = float(os.environ.get("MIID_IDENTITY_GATE", "0.7"))
# Extra attempts for a slot still under the gate once IDENTITY_RETRIES are
# spent. Retries stopped on "attempts used up" in 129 of 150 requests on
# 2026-10-26 and on time in 16: there was GPU time to spare for them.
GATE_RETRIES = int(os.environ.get("MIID_GATE_RETRIES", "2"))
# A best take this far under the gate is held back at once: a full-strength
# re-roll scatters by ~0.05 and will not close a gap that size.
GATE_FAR_MARGIN = 0.12

# Extra attempts per variation after the first pass: identity rescues (the
# edit held back) and edit retries (a hint naming the measured miss). They
# run at a lower GPU priority than any queued first pass, so they only use
# time no other request needs. (Zero from 2026-10-16 while six hotkeys
# shared the card; back to three on 2026-10-19.) With
# retries set to 0, polish also covers the slots that missed their targets
# (_Slot.polish_priority). Over 128 archived rounds (2026-10-12/13) the
# retry loop stopped on "attempts used up" more often than on "every slot
# met its targets" (76 vs 67) while the deadline never bound once.
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
#
# 0.85 and two attempts until 2026-10-29, when the user asked for identity
# above 0.9. Polish had stopped on "every slot at the goal or polished out"
# in 220 of 228 requests (2026-10-27/28) — time was never what ended it —
# and a re-roll beat its incumbent on 30% of attempts, by +0.03.
IDENTITY_GOAL = float(os.environ.get("MIID_IDENTITY_GOAL", "0.90"))
POLISH_ATTEMPTS = int(os.environ.get("MIID_POLISH_ATTEMPTS", "3"))

# Upper bound on one request's generation, retries included, counted from the
# start of generation. The binding limit is usually the request's own
# deadline (validator send time + its 1200 s timeout, minus upload and voice),
# which the miner passes in: time spent queued behind other validators counts
# against it, and a late response scores nothing at all.
GENERATION_BUDGET_SECONDS = float(
    os.environ.get("MIID_GENERATION_BUDGET_SECONDS", "900")
)

# Head turn, in degrees, that each requested pose intensity should land in:
# the benchmark's bins in this miner's units (edit_check.POSE_TARGET_DEG).
# An attempt outside its range — short or past it — is retried at full edit
# strength with a hint (the face held; the turn did not), budget
# permitting, and the closest attempt over the identity gate wins.
#
# History: from 2026-10-07 the ranges only guarded against a missing turn
# (far from 24°), on a grader that then gave 5 to ~30° far turns. On the
# 2026-10-22/23 rounds only 10 of 152 miners graded 5 on a pose_edit
# challenge, and the benchmark read 90% of our far requests as medium or
# light, so the ranges now follow the bins in both directions.
POSE_RANGE_DEG = POSE_TARGET_DEG

# Seconds a request that is otherwise done may wait for the screen replay's
# pending /is_live verdicts (liveness_check; calls are a minute apart) and
# for new content after a flagged take, as long as no other request is
# queued for the GPU. Three rungs and one new take need about four minutes.
PL_WAIT_SECONDS = float(os.environ.get("MIID_PL_WAIT_SECONDS", "240"))


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


def _screen_take(
    result: Dict, req: Any, base_image: Image.Image, base_embedding: Any,
    base_geometry: Optional[Tuple], edit_context: Dict[str, Any],
) -> None:
    """Turn a screen_replay generation into its composites, in place.

    ``result["image"]`` is what FLUX drew for the screen (the reference
    person in an everyday photo). Its face is blended toward the base like
    a background take's when short of the blend goal, then screen_replay
    puts it on the requested device with the requested cues, once per rung
    of the variant ladder (screen_replay.variants), and each composite is
    scored for identity. The take's image becomes the first composite over
    the identity gate (else the best-scoring one); the liveness thread may
    later swap in a rung that /is_live passes. ``result["screen"]`` holds
    the ladder, ``result["screen_images"]`` its images.
    """
    from MIID.miner.ada_face_compare import face_geometry

    device, cues = screen_replay.parse_request(_req_field(req, "description"), _req_field(req, "detail"))
    content = result["image"]
    result["content_identity"] = _similarity(base_embedding, content)
    blend = None
    if face_blend.ENABLED and result["content_identity"] is not None and result["content_identity"] < face_blend.GOAL:
        stand_in = {
            "variation_type": "background_in", "image": content,
            "identity_similarity": result["content_identity"],
            "environment": None, "edit": None, "pose_yaw": None,
        }
        try:
            blend = face_blend.apply(
                base_image, base_embedding, base_geometry, stand_in,
                edit_context=edit_context, levels={}, accessory=None, scene=None,
            )
            content = stand_in["image"]
        except Exception as e:  # noqa: BLE001
            bt.logging.warning(f"screen_replay: face blend failed: {e}")
    result["blend"] = blend
    geometry = face_geometry(content)
    box = geometry[0] if geometry else None

    seed = random.randrange(2**31)
    rungs, images = [], []
    for settings in screen_replay.variants(device, cues):
        image = screen_replay.compose(content, device, cues, face_box=box, seed=seed, **settings)
        images.append(image)
        rungs.append({"settings": settings, "identity": _similarity(base_embedding, image), "pl": None})
    over = [i for i, r in enumerate(rungs) if (r["identity"] or 0.0) >= IDENTITY_GATE]
    chosen = over[0] if over else max(range(len(rungs)), key=lambda i: rungs[i]["identity"] or -1.0)
    result["screen"] = {
        "device": device, "cues": cues, "rungs": rungs, "chosen": chosen,
        "passed": None, "failed": False,
    }
    result["screen_images"] = images
    result["image"] = images[chosen]
    result["identity_similarity"] = rungs[chosen]["identity"]
    result["pose_yaw"] = result["environment"] = result["edit"] = None


def _describe_screen(screen: Dict) -> str:
    """Log form: "phone, moire+glare, rung 2, /is_live passed (0.81)"."""
    rung = screen["rungs"][screen["chosen"]]
    verdict = rung.get("pl")
    checked = sum(1 for r in screen["rungs"] if r.get("pl"))
    if verdict is None:
        pl = f"/is_live {'flagged ' + str(checked) + ' other rung(s)' if checked else 'not asked'}"
    else:
        pl = f"/is_live {'flagged it' if verdict['is_screen_replay'] else 'passed'} ({verdict['confidence']:.2f})"
    return f"{screen['device']}, {'+'.join(screen['cues'])}, rung {screen['chosen'] + 1}, {pl}"


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
# angle. The screen_replay slot is reviewed by hand like the others (and must
# break Passive Liveness v3), so its retries weigh the same since 2026-10-26.
POSE_RETRY_ODDS = 0.5
POSE_RETRY_WEIGHT = 0.3
SCREEN_REPLAY_RETRY_WEIGHT = 1.0
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


_JUDGE_STREAM: Optional[Any] = None


def _judge_stream() -> Optional[Any]:
    """The CUDA stream the judging thread runs on, created once per process.

    PyTorch gives every (cuBLAS handle, stream) pair its own cuBLAS workspace
    (8.1 MiB on this card) and never frees it. A new stream per request — as
    from 2026-10-16 09:28 — handed out the next of the pool's 32 streams each
    time, so every miner process grew by one workspace per request up to
    ~0.27 GiB, and with the fragmentation around them ~0.6 GiB; six processes
    filled the card and every round went empty (2026-10-16 14:13 and the
    weekend after). Found by the gpu_health leak trace: +8.1 MiB per request
    allocated at MTCNN's RNet linear layer.
    """
    global _JUDGE_STREAM
    if _JUDGE_STREAM is None and torch.cuda.is_available():
        _JUDGE_STREAM = torch.cuda.Stream()
    return _JUDGE_STREAM


def _recover_gpu_memory() -> None:
    """Give back everything this process holds on the GPU between attempts:
    the image model's resident weights (re-uploaded by the next attempt) and
    PyTorch's cached blocks."""
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
        self.gate_ok = False
        # Rescue attempts made while under the gate, by kind (next_settings).
        self.held_back = 0
        self.full_rescues = 0
        self.pose_miss = 0.0
        self.pose_direction: Optional[str] = None
        self.bg_miss: Optional[str] = None
        self.edit_misses: List[Tuple[str, str]] = []

    @property
    def satisfied(self) -> bool:
        return (
            self.best is not None and self.identity_ok
            and self.pose_miss == 0.0 and self.bg_miss is None
            and not self.edit_misses and not self.screen_flagged
        )

    @property
    def screen_flagged(self) -> bool:
        """Did /is_live call every composite of the best take a replay?"""
        return bool(self.best and (self.best.get("screen") or {}).get("failed"))

    def snapshot(self) -> Tuple:
        """The current best take and its judgement, for restore()."""
        return (
            self.best, self.best_key, self.identity_ok, self.gate_ok, self.pose_miss,
            self.pose_direction, self.bg_miss, list(self.edit_misses),
        )

    def restore(self, snapshot: Tuple) -> None:
        """Put a snapshot() back as the best take (a polish take that lost an edit)."""
        (
            self.best, self.best_key, self.identity_ok, self.gate_ok, self.pose_miss,
            self.pose_direction, self.bg_miss, self.edit_misses,
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
        its composites are judged by /is_live, not polished for identity.
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

        Rank: clearing the identity gate first (under it the grade is capped
        whatever the edit), then the estimated validation-score cost of every
        measured miss — a head turn outside its range, a background slot
        showing the wrong kind of scene or missing its accessory, a lighting
        change outside its level or an expression that still reads neutral
        (MISS_PENALTY) — and only then similarity. ``target`` decides whether
        the slot still wants more identity (identity_ok), not the ranking. A
        variation AdaFace cannot read a face in (None) is the worst outcome
        there is — the grading API scores it 0 for identity — so it ranks
        below every scored one.
        """
        similarity = result["identity_similarity"]
        identity_ok = similarity is not None and similarity >= target
        gate_ok = similarity is not None and similarity >= IDENTITY_GATE
        miss, direction = _pose_miss(self.pose_level, result.get("pose_yaw"))
        bg_miss = (
            background_miss(result["variation_type"], result.get("environment"))
            if self.background else None
        )
        misses = edit_misses(self.levels, result.get("edit"))
        penalty = 0.0
        if miss:
            penalty += MISS_PENALTY["pose"] + miss / 45.0
        if bg_miss:
            penalty += MISS_PENALTY["scene"]
        for component, _ in misses:
            penalty += MISS_PENALTY[component]
        if not identity_ok:
            self.identity_misses += 1
        # A screen take /is_live flagged in every rung ranks below one it
        # has not (yet) flagged; for every other slot this is always True.
        screen_ok = not (result.get("screen") or {}).get("failed")
        key = (
            screen_ok,
            gate_ok,
            -penalty if gate_ok else 0.0,
            -1.0 if similarity is None else similarity,
        )
        if self.best is None or key > self.best_key:
            self.best, self.best_key = result, key
            self.identity_ok, self.gate_ok, self.pose_miss, self.pose_direction = (
                identity_ok, gate_ok, miss, direction,
            )
            self.bg_miss = bg_miss
            self.edit_misses = misses

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
        """Expected value of one more attempt, or None if it needs none.
        A slot still under the identity gate may take GATE_RETRIES more
        attempts than the rest."""
        if not self.gate_ok:
            attempts_allowed += GATE_RETRIES
        if self.best is None or self.satisfied or self.generated >= attempts_allowed:
            return None
        if self.screen_flagged and self.identity_ok:
            # New content, a new setting: the composite's verdict hangs on it.
            return BACKGROUND_RETRY_ODDS * BACKGROUND_RETRY_WEIGHT
        if self.identity_ok and (
            self.bg_miss is not None
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

        A best take under the identity gate is rescued two ways in turn: a
        full-strength re-roll (with hints for any edit miss) and one with
        the edit held back, a little more each time. The ranking keeps a
        held-back take only if no full-strength one clears the gate, so the
        edit is given up last. A take far under the gate (GATE_FAR_MARGIN),
        or with no face at all, is held back straight away. A slot over the
        gate, or one with a measured miss of the edit — head turn, scene,
        lighting, expression, accessory — re-rolls at full strength, with a
        hint naming any edit miss (the face held; the edit did not).
        """
        similarity = self.best["identity_similarity"] if self.best else None
        under_gate = similarity is None or similarity < IDENTITY_GATE
        far_under = similarity is None or similarity < IDENTITY_GATE - GATE_FAR_MARGIN
        if under_gate and (far_under or self.full_rescues > self.held_back):
            return min(0.8, 0.3 * (self.held_back + 1) + (0.2 if far_under else 0.0)), {}
        hints: Dict[str, str] = {}
        if self.pose_direction:
            hints["pose"] = self.pose_direction
        if self.bg_miss:
            hints["background"] = self.bg_miss
        for component, hint in self.edit_misses:
            hints[component] = hint
        return 0.0, hints

    def reason(self, target: float) -> str:
        if not self.identity_ok:
            similarity = self.best["identity_similarity"]
            shown = "no face detected" if similarity is None else f"{similarity:.3f}"
            return f"identity {shown} < {target}"
        parts = []
        if self.screen_flagged:
            parts.append("/is_live called every composite a screen replay")
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
    # turn, scene, edit measurements, the face blend — is
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
    # affected. One stream for the life of the process (_judge_stream).
    judge_stream = _judge_stream()

    def judge(slot: _Slot, result: Dict, cycle_started: float, generated_at: float) -> None:
        """Measure, blend and record one attempt."""
        if slot.screen_replay:
            judge_screen(slot, result, cycle_started, generated_at)
            return
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

    # The screen_replay slot: its take is composed onto the device in the
    # judging step, and a thread of its own asks /is_live about the
    # composites while the GPU works on the other slots (one call a minute,
    # liveness_check). Both it and the main thread touch the slot's best
    # take, under screen_lock.
    screen_lock = threading.Lock()
    screen_stop = threading.Event()
    screen_thread: List[threading.Thread] = []
    liveness_end = min([budget_end] + ([deadline] if deadline is not None else [])) - 5.0

    def judge_screen(slot: _Slot, result: Dict, cycle_started: float, generated_at: float) -> None:
        with (torch.cuda.stream(judge_stream) if judge_stream is not None else contextlib.nullcontext()):
            try:
                _screen_take(result, slot.req, base_image, base_embedding, base_geometry, edit_context)
            except Exception as e:  # noqa: BLE001 — the raw content is still a take
                bt.logging.warning(f"screen_replay: composing the take failed ({e}); shipping the content as is")
                result["identity_similarity"] = _similarity(base_embedding, result["image"])
                result["pose_yaw"] = result["environment"] = result["edit"] = result["blend"] = None
            if judge_stream is not None:
                judge_stream.synchronize()
        if not first_attempt[0]:
            _recent_cycles.append(time.time() - cycle_started)
        first_attempt[0] = False
        with screen_lock:
            slot.record(result, target)
        screen = result.get("screen")
        if screen:
            bt.logging.info(
                f"screen_replay: {screen['device']} with {', '.join(screen['cues'])} — identity "
                + ", ".join("n/a" if r["identity"] is None else f"{r['identity']:.3f}" for r in screen["rungs"])
                + f" by rung (content {result.get('content_identity') or 0:.3f}"
                + (f", {face_blend.describe(result.get('blend'))}" if face_blend.describe(result.get("blend")) else "")
                + f"); attempt {result['winning_attempt']}, generate {generated_at - cycle_started:.1f}s"
            )
            if liveness_check.available() and not screen_thread:
                worker = threading.Thread(target=liveness_worker, args=(slot,), name="liveness", daemon=True)
                screen_thread.append(worker)
                worker.start()

    def liveness_worker(slot: _Slot) -> None:
        """Check the best screen take's rungs with /is_live, strongest cues
        first, until one passes; a take whose every rung over the identity
        gate is flagged is marked failed (the retry loop then draws new
        content) and the next best take's rungs are checked in turn."""
        while not screen_stop.is_set():
            with screen_lock:
                take = slot.best
                screen = (take or {}).get("screen")
                if not screen or screen["passed"] is not None:
                    rung = None
                else:
                    rung = next(
                        (i for i, r in enumerate(screen["rungs"])
                         if r["pl"] is None and (r["identity"] or 0.0) >= IDENTITY_GATE),
                        None,
                    )
                    checked = any(r["pl"] is not None for r in screen["rungs"])
                    if rung is None and checked and not screen["failed"]:
                        screen["failed"] = True
                        slot.best_key = (False,) + slot.best_key[1:]
                        bt.logging.info("screen_replay: /is_live flagged every rung of this take; new content next")
            if screen and screen["passed"] is not None:
                return
            if rung is None:
                # Nothing left to ask about this take: wait for another one.
                if screen_stop.wait(2.0) or time.time() > liveness_end:
                    return
                continue
            verdict = liveness_check.check(take["screen_images"][rung], latest_start=liveness_end)
            if verdict is None:
                if not liveness_check.available() or time.time() + liveness_check.seconds_until_allowed() > liveness_end:
                    return
                continue
            with screen_lock:
                screen["rungs"][rung]["pl"] = verdict
                passed = not verdict["is_screen_replay"]
                if passed and not screen_stop.is_set():
                    screen["passed"] = screen["chosen"] = rung
                    take["image"] = take["screen_images"][rung]
                    take["identity_similarity"] = screen["rungs"][rung]["identity"]
            bt.logging.info(
                f"screen_replay: rung {rung + 1} (strength {screen['rungs'][rung]['settings']['strength']}) — "
                f"/is_live {'passed' if passed else 'flagged it'} "
                f"(confidence {verdict['confidence']:.3f})"
            )

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
        if slot.best is not None and not slot.gate_ok:
            if identity_bias > 0:
                slot.held_back += 1
            else:
                slot.full_rescues += 1
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
    # The screen replay first: its /is_live checks (a minute apart) then run
    # while the other slots generate.
    for slot in sorted(slots, key=lambda s: (not s.screen_replay, s.index)):
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

    # The screen replay's /is_live verdicts come a minute apart, so a request
    # whose slots are all settled may still be waiting on one — and on a
    # flagged take, on new content and its verdict. While no other request
    # is queued for the GPU and the deadline allows, it waits for them, up to
    # PL_WAIT_SECONDS in all.
    def screen_pending() -> bool:
        """Is the best screen take neither passed nor yet given up on (the
        liveness thread marks it failed once every rung is flagged)?"""
        with screen_lock:
            for slot in slots:
                screen = (slot.best or {}).get("screen") if slot.screen_replay else None
                if screen and screen["passed"] is None and not screen["failed"] and any(
                    (r["identity"] or 0.0) >= IDENTITY_GATE for r in screen["rungs"]
                ):
                    return True
        return False

    waited_for_pl = [0.0]

    def wait_for_screen_verdict() -> bool:
        """Wait two seconds for a pending verdict; False when there is none
        to wait for, or no time or GPU to wait with."""
        if not (screen_thread and screen_thread[0].is_alive() and screen_pending()):
            return False
        if waited_for_pl[0] >= PL_WAIT_SECONDS or time.time() + 2.0 > liveness_end:
            return False
        if retry_cutoff_fn is not None and retry_cutoff_fn() is not None:
            return False
        screen_thread[0].join(timeout=2.0)
        waited_for_pl[0] += 2.0
        return True

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
            if wait_for_screen_verdict():
                continue  # a flagged screen take becomes a candidate
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
            "retrying with the edit held back." if identity_bias > 0
            else "retrying at full strength with hints ("
            + ", ".join(f"{component}: {hint}" for component, hint in hints.items())
            + ")."
            if hints
            else "retrying at full strength."
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

    # A verdict still pending once everything else is done gets the rest of
    # the wait (see wait_for_screen_verdict); then the liveness thread stops
    # and a verdict still in flight is dropped.
    while wait_for_screen_verdict():
        pass
    if waited_for_pl[0]:
        bt.logging.info(f"screen_replay: waited {waited_for_pl[0]:.0f}s for /is_live")

    # ── assemble, in request order ──
    screen_stop.set()
    variations = []
    for slot in slots:
        with screen_lock:
            best = slot.best
            screen = dict(best["screen"]) if best and best.get("screen") else None
            if screen:
                screen["rungs"] = [dict(r) for r in screen["rungs"]]
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
            "blend": best.get("blend"),
            "screen": screen,
            "content_identity": best.get("content_identity"),
        })
        best.pop("screen_images", None)

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
            + (f", {_describe_screen(screen)}" if screen else "")
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
