# MIID/miner/grade_check.py
#
# Client for the subnet owner's grade API (P5 C2, miner_api/api_white_paper.md):
# POST /face_grade with the variation, the seed, its variation type and
# intensity returns the exact grade a reviewer's rubric gives it —
# {"breaks": bool, "score": -5..5, "comment": str}. The score is the UAV
# score of that face (docs/scoring.md): 5 only when Deepfake V1 breaks and the
# ask was met, at most 3 when the detector holds, negative for a duplicate, a
# misleading or a spam image. The live KAV grade (/grade_v3) uses the same
# rubric on one sampled face.
#
# image_generator grades each face slot's takes with it while the GPU works on
# the others, and ships the best graded take. The API allows one call per
# hotkey per rate window (60 s in the white paper): one worker thread per
# process makes every call, earliest request deadline first, and a job whose
# take is no longer wanted by the time its turn comes is dropped unsent.
#
# On 2026-11-25 the API answered 403 "Unauthorized hotkey" for hotkey_0 (uid
# 25, registered since 2026-11-13); the white paper says miner hotkeys are
# allow-listed from the chain once a day. A refusal switches the check off and
# it is re-probed every PROBE_SECONDS with the free /rate-limit/status call,
# so it comes on by itself once the hotkey is allowed.
#
# MIID_GRADE_CHECK=0 switches it off; MIID_GRADE_URL points elsewhere.

import base64
import heapq
import io
import itertools
import os
import re
import threading
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import bittensor as bt
from PIL import Image


ENABLED = os.environ.get("MIID_GRADE_CHECK", "1").strip().lower() not in ("0", "false", "no")
API_URL = os.environ.get("MIID_GRADE_URL", "https://api.yanez.ai:5000").rstrip("/")
# Seconds between calls until the server's /rate-limit/status says otherwise.
RATE_SECONDS = float(os.environ.get("MIID_GRADE_RATE_SECONDS", "61"))
# How often a refused or unreachable API is asked again (free status call).
PROBE_SECONDS = float(os.environ.get("MIID_GRADE_PROBE_SECONDS", "600"))
TIMEOUT_SECONDS = 90.0

_lock = threading.Condition()
_queue: List = []                 # heap of (priority, seq, job)
_seq = itertools.count()
_keypair: Any = None
_worker: Optional[threading.Thread] = None
_spacing = RATE_SECONDS           # seconds between calls
_next_allowed = 0.0
# None: not known yet; True: calls go through; False: refused / unreachable.
_access: Optional[bool] = None
_access_reason = "not probed yet"
_next_probe = 0.0
_probing = False


class Job:
    """One grade request. ``done`` is set when it has an answer or never will."""

    def __init__(
        self, kind: str, payload: Dict[str, Any], label: str,
        latest_start: Optional[float], wanted: Optional[Callable[[], bool]],
    ):
        self.kind = kind
        self.payload = payload
        self.label = label
        self.latest_start = latest_start
        self.wanted = wanted
        self.result: Optional[Dict[str, Any]] = None
        self.reason: Optional[str] = None   # why there is no result
        self.done = threading.Event()

    def finish(self, result: Optional[Dict[str, Any]], reason: Optional[str] = None) -> None:
        self.result, self.reason = result, reason
        self.done.set()


def configure(wallet: Any) -> None:
    """Sign calls with this wallet's hotkey and probe access (the miner calls it at startup)."""
    global _keypair
    _keypair = wallet.hotkey
    _probe_async()


def available() -> bool:
    """True when calls currently go through. A refused API is re-probed in
    the background every PROBE_SECONDS; this never blocks."""
    if not ENABLED or _keypair is None:
        return False
    if _access is not True and time.time() >= _next_probe:
        _probe_async()
    return _access is True


def describe() -> str:
    if not ENABLED:
        return "off (MIID_GRADE_CHECK=0)"
    if _access is True:
        return f"on ({API_URL}, one call per {_spacing:.0f}s)"
    return f"waiting for access ({API_URL}: {_access_reason}; re-checked every {PROBE_SECONDS / 60:.0f} min)"


def seconds_until_allowed() -> float:
    return max(0.0, _next_allowed - time.time())


def queued() -> int:
    with _lock:
        return len(_queue)


def png_b64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def submit_face(
    image: Image.Image,
    seed_b64: str,
    variation_type: str,
    intensity: str,
    priority: float,
    latest_start: Optional[float] = None,
    wanted: Optional[Callable[[], bool]] = None,
    label: str = "",
) -> Optional[Job]:
    """Queue an exact face grade; None when the check is unavailable.

    ``priority``: lower goes first (the request's deadline). ``latest_start``:
    the call is dropped if it cannot start by then. ``wanted()`` is asked just
    before the call; False drops it unsent (the take lost its place).
    """
    if not available():
        return None
    payload = {
        "image": png_b64(image),
        "seed_image": seed_b64,
        "variation_type": variation_type,
        "intensity": intensity,
    }
    job = Job("face_grade", payload, label or variation_type, latest_start, wanted)
    _enqueue(priority, job)
    return job


def submit_screen(
    image: Image.Image,
    seed_b64: str,
    cues: List[str],
    priority: float,
    latest_start: Optional[float] = None,
    wanted: Optional[Callable[[], bool]] = None,
    label: str = "screen_replay",
) -> Optional[Job]:
    """Queue an exact screen-replay grade (cue names as in the white paper)."""
    if not available():
        return None
    payload = {"image": png_b64(image), "seed_image": seed_b64, "cues": list(cues)}
    job = Job("screen_replay_grade", payload, label, latest_start, wanted)
    _enqueue(priority, job)
    return job


def _enqueue(priority: float, job: Job) -> None:
    global _worker
    with _lock:
        heapq.heappush(_queue, (priority, next(_seq), job))
        if _worker is None or not _worker.is_alive():
            _worker = threading.Thread(target=_run, name="grade-check", daemon=True)
            _worker.start()
        _lock.notify_all()


def _signature() -> str:
    now = datetime.now()
    message = f"<Bytes>On {now} {now.astimezone().tzname()} API Request</Bytes>"
    signed = _keypair.sign(data=message)
    return f"{message}\n\tSigned by: {_keypair.ss58_address}\n\tSignature: {signed.hex()}"


def _set_access(ok: bool, reason: str) -> None:
    global _access, _access_reason, _next_probe
    changed = _access is not ok
    _access, _access_reason = ok, reason
    _next_probe = time.time() + PROBE_SECONDS
    if changed:
        if ok:
            bt.logging.info(f"Grade check: {API_URL} accepts this hotkey — face grades on ({reason})")
        else:
            bt.logging.warning(
                f"Grade check: {API_URL} unavailable ({reason}); takes ship ungraded, "
                f"asking again every {PROBE_SECONDS / 60:.0f} min"
            )


def _apply_status(status: Dict[str, Any]) -> None:
    """Space calls by the server's window: rate_limit_seconds / max_requests."""
    global _spacing, _next_allowed
    try:
        window = float(status.get("rate_limit_seconds") or RATE_SECONDS)
        per_window = max(1, int(status.get("max_requests") or 1))
        _spacing = window / per_window + 1.0
        wait = float(status.get("seconds_until_allowed") or 0.0)
        if not status.get("can_request", True):
            _next_allowed = max(_next_allowed, time.time() + wait + 1.0)
    except (TypeError, ValueError):
        pass


def _probe() -> None:
    """The free /rate-limit/status call: does the API accept this hotkey?"""
    global _probing
    import requests

    try:
        response = requests.post(
            f"{API_URL}/rate-limit/status", json={"signature": _signature()}, timeout=30,
        )
        if response.status_code == 200:
            _apply_status(response.json())
            _set_access(True, f"one call per {_spacing:.0f}s")
        else:
            _set_access(False, f"HTTP {response.status_code} {response.text.strip()[:120]}")
    except Exception as e:  # noqa: BLE001
        _set_access(False, f"{type(e).__name__}: {e}")
    finally:
        _probing = False


def _probe_async() -> None:
    global _probing, _next_probe
    if _probing or _keypair is None or not ENABLED:
        return
    _probing = True
    _next_probe = time.time() + PROBE_SECONDS
    threading.Thread(target=_probe, name="grade-probe", daemon=True).start()


def _next_job() -> Optional[Job]:
    """The most urgent job still worth sending, waiting for the rate window."""
    with _lock:
        while True:
            if not _queue:
                _lock.wait(timeout=30.0)
                if not _queue:
                    return None
                continue
            _, _, job = _queue[0]
            now = time.time()
            if _access is False:
                heapq.heappop(_queue)
                job.finish(None, f"API unavailable ({_access_reason})")
                continue
            start_at = max(now, _next_allowed)
            if job.latest_start is not None and start_at > job.latest_start:
                heapq.heappop(_queue)
                job.finish(None, "no rate-limit window before its deadline")
                continue
            if start_at > now:
                # A more urgent job may arrive while we wait for the window.
                _lock.wait(timeout=min(start_at - now, 5.0))
                continue
            heapq.heappop(_queue)
            try:
                if job.wanted is not None and not job.wanted():
                    job.finish(None, "no longer wanted")
                    continue
            except Exception:  # noqa: BLE001
                pass
            return job


def _run() -> None:
    while True:
        job = _next_job()
        if job is None:
            with _lock:
                if not _queue:
                    return
            continue
        _call(job)


def _call(job: Job) -> None:
    global _next_allowed
    import requests

    started = time.time()
    _next_allowed = started + _spacing
    payload = dict(job.payload, signature=_signature())
    try:
        response = requests.post(f"{API_URL}/{job.kind}", json=payload, timeout=TIMEOUT_SECONDS)
    except Exception as e:  # noqa: BLE001
        bt.logging.warning(f"Grade check: /{job.kind} unreachable ({type(e).__name__}: {e})")
        job.finish(None, f"unreachable: {type(e).__name__}")
        return
    if response.status_code == 429:
        try:
            _next_allowed = time.time() + float(response.json()["retry_after_seconds"]) + 1.0
        except Exception:  # noqa: BLE001
            _next_allowed = time.time() + _spacing
        bt.logging.info(f"Grade check: rate limited; next call in {seconds_until_allowed():.0f}s")
        job.finish(None, "rate limited")
        return
    if response.status_code in (401, 403):
        _set_access(False, f"HTTP {response.status_code} {response.text.strip()[:120]}")
        job.finish(None, f"HTTP {response.status_code}")
        return
    if response.status_code != 200:
        bt.logging.warning(
            f"Grade check: /{job.kind} answered HTTP {response.status_code}: {response.text[:200]}"
        )
        job.finish(None, f"HTTP {response.status_code}")
        return
    try:
        body = response.json()
        result = {
            "breaks": bool(body["breaks"]),
            "score": float(body["score"]),
            "comment": str(body.get("comment") or ""),
            "seconds": round(time.time() - started, 2),
        }
    except Exception as e:  # noqa: BLE001
        bt.logging.warning(f"Grade check: unreadable answer ({e}): {response.text[:200]}")
        job.finish(None, "unreadable answer")
        return
    if _access is not True:
        _set_access(True, "a grade came back")
    bt.logging.info(f"Grade check: {job.label} — {describe_grade(result)}")
    job.finish(result)


# What the grader's comment names as the failure, turned into the prompt hint
# that answers it (generate_variations.BACKGROUND_HINTS / FRAMING_HINTS).
# 2026-11-26/27: every background_out 0 it gave us read "Score 0. Background
# environment mismatch: expected outdoor, CLIP classified indoor." (41% of 139
# graded attempts, 62% of light ones), and the -1s on far backgrounds "Face
# not dominant (face_ratio=0.0946, threshold=0.10)".
_ENVIRONMENT_MISMATCH = re.compile(r"expected (indoor|outdoor), CLIP classified (indoor|outdoor)")


def hints(grade: Optional[Dict[str, Any]]) -> Dict[str, str]:
    """Prompt hints, by component, for the failure a grade's comment names:
    ``background`` = the environment the grader's CLIP saw ("indoor" /
    "outdoor"), ``framing`` = "closer" when the face was too small."""
    comment = (grade or {}).get("comment") or ""
    found: Dict[str, str] = {}
    mismatch = _ENVIRONMENT_MISMATCH.search(comment)
    if mismatch and mismatch.group(1) != mismatch.group(2):
        found["background"] = mismatch.group(2)
    if "face not dominant" in comment.lower():
        found["framing"] = "closer"
    return found


def describe_grade(grade: Optional[Dict[str, Any]]) -> str:
    if not grade:
        return "not graded"
    verdict = "breaks the detector" if grade["breaks"] else "detector holds"
    comment = grade.get("comment") or ""
    return f"score {grade['score']:+.0f}, {verdict}" + (f" ({comment[:160]})" if comment else "")
