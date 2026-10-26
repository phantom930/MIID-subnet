# MIID/miner/liveness_check.py
#
# Client for the subnet's Passive Liveness API (PL v3; see
# miner_api/api_white_paper.md): POST /is_live with a base64 image and a
# message signed by the miner's hotkey, answered with
# {"is_screen_replay": bool, "overall_confidence": float}.
#
# The screen_replay slot has to break PL v3 — /is_live must call it "not a
# screen replay" — so the miner checks its composite before submitting it
# (image_generator). The API allows one call per hotkey per rate window (60 s
# on 2026-10-26; the white paper still says 300): calls are serialised in
# this process and spaced by the window, and a 429 pushes the next one back
# by what /rate-limit/status reports.
#
# MIID_PL_CHECK=0 switches the check off; MIID_PL_URL points elsewhere.

import base64
import io
import os
import threading
import time
from datetime import datetime
from typing import Any, Dict, Optional

import bittensor as bt
from PIL import Image


ENABLED = os.environ.get("MIID_PL_CHECK", "1").strip().lower() not in ("0", "false", "no")
API_URL = os.environ.get("MIID_PL_URL", "http://98.90.28.118:5001").rstrip("/")
# Seconds between calls from this hotkey, until the server says otherwise.
RATE_SECONDS = float(os.environ.get("MIID_PL_RATE_SECONDS", "61"))
TIMEOUT_SECONDS = 30.0

_lock = threading.Lock()
_keypair: Any = None
_next_allowed = 0.0
_disabled_reason: Optional[str] = None


def configure(wallet: Any) -> None:
    """Sign calls with this wallet's hotkey (the miner calls it at startup)."""
    global _keypair
    _keypair = wallet.hotkey


def available() -> bool:
    return ENABLED and _keypair is not None and _disabled_reason is None


def seconds_until_allowed() -> float:
    return max(0.0, _next_allowed - time.time())


def _signature() -> str:
    now = datetime.now()
    message = f"<Bytes>On {now} {now.astimezone().tzname()} API Request</Bytes>"
    signed = _keypair.sign(data=message)
    return f"{message}\n\tSigned by: {_keypair.ss58_address}\n\tSignature: {signed.hex()}"


def _png_b64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _sync_rate_limit() -> None:
    """After a 429: ask the server when the next call is allowed (free)."""
    global _next_allowed
    import requests

    try:
        status = requests.post(
            f"{API_URL}/rate-limit/status", json={"signature": _signature()}, timeout=TIMEOUT_SECONDS,
        ).json()
        _next_allowed = time.time() + float(status.get("seconds_until_allowed", RATE_SECONDS)) + 1.0
    except Exception:  # noqa: BLE001
        _next_allowed = time.time() + RATE_SECONDS


def check(image: Image.Image, latest_start: Optional[float] = None) -> Optional[Dict[str, Any]]:
    """Ask /is_live about ``image``.

    Waits for this hotkey's rate window, but never past ``latest_start``
    (wall clock): None when the call could not start by then, the check is
    off or unconfigured, or the API failed. Otherwise
    {"is_screen_replay", "confidence", "seconds"}.
    """
    global _next_allowed, _disabled_reason
    if not available():
        return None
    import requests

    with _lock:
        wait = _next_allowed - time.time()
        if latest_start is not None and time.time() + max(0.0, wait) > latest_start:
            return None
        if wait > 0:
            time.sleep(wait)
        started = time.time()
        _next_allowed = started + RATE_SECONDS
        try:
            response = requests.post(
                f"{API_URL}/is_live",
                json={"image": _png_b64(image), "signature": _signature()},
                timeout=TIMEOUT_SECONDS,
            )
        except Exception as e:  # noqa: BLE001
            bt.logging.warning(f"PL check: /is_live unreachable ({type(e).__name__}: {e})")
            return None
        if response.status_code == 429:
            try:
                _next_allowed = time.time() + float(response.json()["retry_after_seconds"]) + 1.0
            except Exception:  # noqa: BLE001
                _sync_rate_limit()
            bt.logging.info(f"PL check: rate limited; next call in {seconds_until_allowed():.0f}s")
            return None
        if response.status_code in (401, 403):
            # Not a passing condition: the hotkey is not allowed to call.
            _disabled_reason = f"HTTP {response.status_code}"
            bt.logging.warning(f"PL check: /is_live refused this hotkey ({response.text[:200]}); checks off")
            return None
        if response.status_code != 200:
            bt.logging.warning(f"PL check: /is_live answered HTTP {response.status_code}: {response.text[:200]}")
            return None
        try:
            body = response.json()
            return {
                "is_screen_replay": bool(body["is_screen_replay"]),
                "confidence": float(body.get("overall_confidence", 0.0)),
                "seconds": round(time.time() - started, 2),
            }
        except Exception as e:  # noqa: BLE001
            bt.logging.warning(f"PL check: unreadable answer ({e}): {response.text[:200]}")
            return None
