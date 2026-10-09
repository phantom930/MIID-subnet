# MIID/miner/voice_generator.py
#
# Voice-clone hook for miners (P5 C1).
#
# generate_voice_clone() takes the validator's reference WAV and the prompt
# text, and returns WAV bytes that keep the reference speaker's identity while
# speaking the prompt. The model (Chatterbox Multilingual, English + Spanish,
# zero-shot cloning) runs in a separate worker process inside voice_env/ —
# see MIID/miner/voice_worker.py and scripts/miner/setup_voice.sh. It cannot
# live in miner_env: its pinned torch/transformers would break the FLUX path.
#
# The worker generates up to MIID_VOICE_ATTEMPTS takes and keeps the one with
# the highest SpeechBrain ECAPA similarity to the reference, stopping early
# once it reaches MIID_VOICE_IDENTITY_TARGET. The validator counts a voice as
# identity-preserving above 0.6 (speech_brain_compare enforces that floor).
#
# Prompt text is already joined for you as target_text, e.g.
#   "three orange six two nine apple table one"
# (same as " ".join(target_words)).

import base64
import hashlib
import json
import os
import queue
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import bittensor as bt


REPO_ROOT = Path(__file__).resolve().parents[2]
VOICE_ENV = Path(os.environ.get("MIID_VOICE_ENV", str(REPO_ROOT / "voice_env")))
WORKER_SCRIPT = Path(__file__).resolve().with_name("voice_worker.py")

# Takes generated per request; the best by speaker similarity is kept.
VOICE_ATTEMPTS = int(os.environ.get("MIID_VOICE_ATTEMPTS", "3"))
# Stop re-rolling once a take reaches this similarity. Above the validator's
# 0.6 floor so that scoring noise on their side does not tip it under.
VOICE_IDENTITY_TARGET = float(os.environ.get("MIID_VOICE_IDENTITY_TARGET", "0.7"))
# Seconds to wait on one worker call (includes the model load on first use).
VOICE_TIMEOUT = float(os.environ.get("MIID_VOICE_TIMEOUT", "600"))
# The worker holds ~3-4 GB of VRAM while loaded, on the GPU the FLUX image
# path needs. Voice comes once per round, so by default it is shut down after
# each request; set 1 to keep it warm.
VOICE_KEEP_LOADED = os.environ.get("MIID_VOICE_KEEP_LOADED", "0").strip().lower() in (
    "1", "true", "yes",
)

# Speaker similarity of the last clip generate_voice_clone returned, for logs
# and the request archive.
last_similarity: Optional[float] = None


def voice_env_available() -> bool:
    """True when voice_env/ has been set up (scripts/miner/setup_voice.sh)."""
    return (VOICE_ENV / "bin" / "python").is_file()


class _VoiceWorker:
    """One long-lived voice_worker.py process, spoken to over JSON lines."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._proc: Optional[subprocess.Popen] = None
        self._replies: "queue.Queue[Optional[str]]" = queue.Queue()
        self._next_id = 0

    def _start(self) -> None:
        # -I: isolated mode, so neither this checkout nor miner_env's
        # PYTHONPATH can shadow voice_env's packages. stderr is inherited and
        # lands in the miner's log.
        self._proc = subprocess.Popen(
            [str(VOICE_ENV / "bin" / "python"), "-I", str(WORKER_SCRIPT)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._replies = queue.Queue()
        threading.Thread(
            target=self._read, args=(self._proc, self._replies), daemon=True,
        ).start()

    @staticmethod
    def _read(proc: subprocess.Popen, replies: "queue.Queue[Optional[str]]") -> None:
        for line in proc.stdout:
            replies.put(line)
        replies.put(None)

    def _alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def call(self, op: str, timeout: float = VOICE_TIMEOUT, **fields: Any) -> Optional[Dict[str, Any]]:
        """Send one request; the reply dict, or None if the worker failed."""
        with self._lock:
            if not self._alive():
                self._start()
            self._next_id += 1
            req_id = self._next_id
            try:
                self._proc.stdin.write(json.dumps({"id": req_id, "op": op, **fields}) + "\n")
                self._proc.stdin.flush()
            except (BrokenPipeError, OSError) as e:
                bt.logging.warning(f"Voice worker: could not send {op}: {e}")
                self._stop()
                return None

            while True:
                try:
                    line = self._replies.get(timeout=timeout)
                except queue.Empty:
                    bt.logging.warning(f"Voice worker: {op} timed out after {timeout:.0f}s")
                    self._stop()
                    return None
                if line is None:
                    bt.logging.warning(f"Voice worker: exited during {op}")
                    self._stop()
                    return None
                try:
                    reply = json.loads(line)
                except ValueError:
                    continue
                if reply.get("id") == req_id:
                    return reply

    def _stop(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            return
        try:
            proc.stdin.close()
            proc.wait(timeout=15)
        except Exception:
            proc.kill()

    def stop(self) -> None:
        with self._lock:
            self._stop()


_worker = _VoiceWorker()


def worker_call(op: str, **fields: Any) -> Optional[Dict[str, Any]]:
    """Call the voice worker, or None when voice_env is missing or it failed."""
    if not voice_env_available():
        return None
    return _worker.call(op, **fields)


def release_worker(force: bool = False) -> None:
    """Shut the worker down (frees its VRAM) unless MIID_VOICE_KEEP_LOADED."""
    if force or not VOICE_KEEP_LOADED:
        _worker.stop()


def decode_base_voice(base64_voice: str) -> bytes:
    """Decode a base64-encoded WAV into raw bytes."""
    return base64.b64decode(base64_voice)


def generate_voice_clone(
    base_wav_bytes: bytes,
    target_words: List[str],
    language: str,
    target_text: str = "",
) -> Optional[bytes]:
    """Generate a voice-cloned WAV speaking the target prompt.

    Args:
        base_wav_bytes: Reference speaker WAV bytes from the validator.
        target_words: Token list the generated audio should contain.
        language: ``"en"`` or ``"es"``.
        target_text: Ready-to-speak prompt string (preferred for TTS).
            Falls back to ``" ".join(target_words)`` if empty.

    Returns:
        Generated WAV bytes, or None if generation failed.
    """
    global last_similarity
    last_similarity = None

    prompt = (target_text or " ".join(target_words or [])).strip()
    if not prompt or not base_wav_bytes:
        bt.logging.warning("Voice: empty prompt or reference — skipping")
        return None
    if not voice_env_available():
        bt.logging.warning(
            f"Voice: {VOICE_ENV} not found — run scripts/miner/setup_voice.sh "
            "to enable voice clones."
        )
        return None

    with tempfile.TemporaryDirectory(prefix="miid_voice_") as tmp:
        ref_wav = os.path.join(tmp, "reference.wav")
        out_wav = os.path.join(tmp, "clone.wav")
        with open(ref_wav, "wb") as f:
            f.write(base_wav_bytes)

        reply = worker_call(
            "clone",
            ref_wav=ref_wav,
            out_wav=out_wav,
            text=prompt,
            language=language,
            attempts=VOICE_ATTEMPTS,
            target_similarity=VOICE_IDENTITY_TARGET,
        )
        if not reply or not reply.get("ok"):
            error = (reply or {}).get("error", "worker unavailable")
            bt.logging.warning(f"Voice: clone failed ({error})")
            return None

        last_similarity = reply.get("similarity")
        bt.logging.info(
            f"Voice: cloned {len(prompt.split())} words (lang={language}) — "
            f"speaker similarity {last_similarity:.3f} after "
            f"{reply.get('attempts')} attempt(s)"
        )
        with open(out_wav, "rb") as f:
            return f.read()


def hash_voice_bytes(wav_bytes: bytes) -> str:
    """SHA256 hex digest of WAV bytes (same role as image_hash)."""
    return hashlib.sha256(wav_bytes).hexdigest()
