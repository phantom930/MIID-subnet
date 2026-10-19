# MIID/miner/voice_generator.py
#
# Voice-clone hook for miners (P5 C1), running inside the miner process.
#
# generate_voice_clone() takes the validator's reference WAV and the prompt
# text, and returns WAV bytes that keep the reference speaker's identity while
# speaking the prompt. Chatterbox Multilingual (zero-shot cloning, English +
# Spanish) and the SpeechBrain ECAPA speaker encoder (the validator's identity
# model) are loaded lazily into this process, on the same GPU as the FLUX
# image path, and parked in CPU memory between rounds so their VRAM is free.
#
# Install into miner_env with scripts/miner/setup_voice.sh. chatterbox-tts
# publishes pins (torch==2.6, transformers==5.2, diffusers==0.29, numpy<2)
# that the image stack cannot satisfy, so the script installs it with
# --no-deps and adds what it actually needs from requirements-voice.txt. The
# combination was verified on torch 2.14 / transformers 5.19 / diffusers 0.41.
#
# Up to MIID_VOICE_ATTEMPTS takes are generated and the one with the highest
# ECAPA similarity to the reference is kept, stopping early once it reaches
# MIID_VOICE_IDENTITY_TARGET. The validator counts a voice as identity-
# preserving above 0.6 (speech_brain_compare enforces that floor).
#
# Prompt text is already joined for you as target_text, e.g.
#   "three orange six two nine apple table one"
# (same as " ".join(target_words)).
#
# Smoke test from the command line:
#   python -m MIID.miner.voice_generator --reference ref.wav \
#       --text "three orange six two nine" --language en --out clone.wav

import base64
import contextlib
import hashlib
import io
import logging
import os
import sys
import tempfile
import threading
import time
from typing import Any, Dict, Iterator, List, Optional, Tuple

import bittensor as bt


# Takes generated per request; the best by speaker similarity is kept.
VOICE_ATTEMPTS = int(os.environ.get("MIID_VOICE_ATTEMPTS", "3"))
# Stop re-rolling once a take reaches this similarity. Above the validator's
# 0.6 floor so that scoring noise on their side does not tip it under.
VOICE_IDENTITY_TARGET = float(os.environ.get("MIID_VOICE_IDENTITY_TARGET", "0.7"))
# The models hold ~3 GB of VRAM while loaded, on the GPU the FLUX image path
# needs. Voice comes once per round, so by default they are parked in CPU
# memory after each request; set 1 to keep them on the GPU.
VOICE_KEEP_LOADED = os.environ.get("MIID_VOICE_KEEP_LOADED", "0").strip().lower() in (
    "1", "true", "yes",
)
# Load the models (downloading the weights if needed) into CPU memory in the
# background at miner startup, so the first voice request does not spend its
# time budget on that. Set 0 to load on first use instead.
VOICE_PRELOAD = os.environ.get("MIID_VOICE_PRELOAD", "1").strip().lower() in (
    "1", "true", "yes",
)
# Chatterbox conditions on at most ~10 s of reference; the rest is ignored.
REF_SECONDS = float(os.environ.get("MIID_VOICE_REF_SECONDS", "10"))

ECAPA_ID = "speechbrain/spkrec-ecapa-voxceleb"
ECAPA_SR = 16000
COND_SR = 24000  # sample rate Chatterbox conditions on
SUPPORTED_LANGUAGES = {"en", "es"}

# Speaker similarity of the last clip generate_voice_clone returned, for logs
# and the request archive.
last_similarity: Optional[float] = None

_availability: Optional[bool] = None
_import_error: Optional[str] = None


@contextlib.contextmanager
def _preserve_root_logging() -> Iterator[None]:
    """Undo root-logger changes made by the voice stack while it loads.

    Loading Chatterbox calls ``logging.basicConfig()``, which attaches a
    stderr handler to the root logger. From then on every logger in the
    process prints there too, and with ``--logging.debug`` (which bittensor
    applies to its chain client as well) that floods the miner log with one
    ``DEBUG:async_substrate_interface`` block per chain RPC. Anything the
    voice stack attaches to the root logger is removed again here.
    """
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    # The stdlib only auto-configures when the root logger has no handler
    # at all, so a placeholder for the duration of the call prevents it.
    placeholder = logging.NullHandler() if not handlers else None
    if placeholder is not None:
        root.addHandler(placeholder)
    try:
        yield
    finally:
        for handler in list(root.handlers):
            if handler not in handlers:
                root.removeHandler(handler)
        root.setLevel(level)


@contextlib.contextmanager
def _drop_new_hooks(root: Any) -> Iterator[None]:
    """Remove the forward hooks a Chatterbox call adds under ``root``.

    chatterbox-tts 0.1.7's T3.inference resets ``self.compiled`` just before
    testing it, so every generate builds a new AlignmentStreamAnalyzer and
    registers its attention hooks on the transformer without ever removing
    them. Each one keeps its analyzer alive and keeps firing on every later
    call: three more hooks per generate, so a miner process that had cloned
    for a few hours ran dozens of stale analyzers on every decoding step.
    The current call's own hooks are left in place while it runs.
    """
    before = {id(m): set(m._forward_hooks) for m in root.modules()}
    try:
        yield
    finally:
        for module in root.modules():
            known = before.get(id(module), set())
            for key in [k for k in module._forward_hooks if k not in known]:
                del module._forward_hooks[key]


def voice_available() -> bool:
    """True when the in-process voice stack (Chatterbox + SpeechBrain) imports."""
    global _availability, _import_error
    if _availability is None:
        try:
            with _preserve_root_logging():
                import soundfile  # noqa: F401
                import torch  # noqa: F401
                import torchaudio.functional  # noqa: F401
                import chatterbox.mtl_tts  # noqa: F401
                import speechbrain.inference.speaker  # noqa: F401
                import perth  # noqa: F401
            _availability = True
        except ImportError as e:
            _availability = False
            _import_error = f"{type(e).__name__}: {e}"
    return _availability


def _requested_device() -> str:
    wanted = os.environ.get("MIID_VOICE_DEVICE", "").strip().lower()
    if wanted:
        return wanted
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def _indexed(dev: str) -> str:
    """'cuda' -> 'cuda:0' (SpeechBrain wants an index); anything else as is."""
    return "cuda:0" if dev == "cuda" else dev


def describe() -> str:
    """One-line summary for the miner's startup log."""
    return f"Chatterbox Multilingual + SpeechBrain ECAPA, device={_requested_device()}"


def _swap_module(module: Any, dev: str, parked: Dict[int, Dict[str, Any]]) -> None:
    """Move ``module`` between the CPU and ``dev`` without allocating CPU memory.

    ``module.to("cpu")`` allocates a fresh CPU tensor for every weight on
    every call; after a day of voice rounds that churn had fragmented the
    heap by gigabytes per process. The CPU tensors a module was loaded with
    are kept in ``parked`` and re-attached on the way back, so a round trip
    allocates only on the GPU.
    """
    tensors = list(module.named_parameters(remove_duplicate=False)) + list(
        module.named_buffers(remove_duplicate=False)
    )
    if dev == "cpu":
        kept = parked.get(id(module), {})
        for name, tensor in tensors:
            if tensor.device.type == "cpu":
                continue
            cpu = kept.get(name)
            if cpu is not None and cpu.shape == tensor.shape and cpu.dtype == tensor.dtype:
                tensor.data = cpu
            else:
                tensor.data = tensor.data.to("cpu")
    else:
        kept = parked.setdefault(id(module), {})
        for name, tensor in tensors:
            if tensor.device.type != "cpu":
                continue
            kept[name] = tensor.data
            tensor.data = tensor.data.to(dev)


class _Models:
    """Chatterbox and ECAPA, loaded once and moved between GPU and CPU."""

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.tts: Any = None
        self.ecapa: Any = None
        self.device: Optional[str] = None  # where the models are right now
        self.cpu_fallback = False          # set after a CUDA out-of-memory
        self._parked: Dict[int, Dict[str, Any]] = {}  # CPU tensors kept across moves

    def target_device(self) -> str:
        return "cpu" if self.cpu_fallback else _requested_device()

    def ensure(self, dev: str, need_tts: bool = True) -> Tuple[Any, Any]:
        """Load whatever is missing onto ``dev`` and move the rest there."""
        with _preserve_root_logging():
            return self._ensure(dev, need_tts)

    def _ensure(self, dev: str, need_tts: bool) -> Tuple[Any, Any]:
        import torch

        if need_tts and self.tts is None:
            from chatterbox.mtl_tts import ChatterboxMultilingualTTS
            started = time.time()
            try:
                self.tts = ChatterboxMultilingualTTS.from_pretrained(device=dev)
            except torch.cuda.OutOfMemoryError:
                # The GPU is shared with the FLUX image path; voice is not
                # worth failing over, so it drops to CPU (slower, same output).
                bt.logging.warning(
                    "Voice: CUDA out of memory loading Chatterbox; using the CPU for voice"
                )
                torch.cuda.empty_cache()
                self.cpu_fallback, dev = True, "cpu"
                self.tts = ChatterboxMultilingualTTS.from_pretrained(device="cpu")
            bt.logging.info(
                f"Voice: Chatterbox Multilingual loaded on {dev} "
                f"in {time.time() - started:.1f}s"
            )
            if self.ecapa is not None and self.device != dev:
                self._move(dev)
            self.device = dev

        if self.ecapa is None:
            from speechbrain.inference.speaker import EncoderClassifier
            savedir = os.path.join(
                os.environ.get("HF_HOME", os.path.expanduser("~/.cache")),
                "speechbrain", "spkrec-ecapa-voxceleb",
            )
            self.ecapa = EncoderClassifier.from_hparams(
                source=ECAPA_ID, savedir=savedir, run_opts={"device": _indexed(dev)},
            )
            bt.logging.info(f"Voice: SpeechBrain ECAPA loaded on {dev}")
            if self.tts is not None and self.device != dev:
                self._move(dev)
            self.device = dev

        if self.device != dev:
            self._move(dev)
        return self.tts, self.ecapa

    def for_inference(self, need_tts: bool = True) -> Tuple[Any, Any]:
        return self.ensure(self.target_device(), need_tts)

    def preload(self) -> None:
        """Load both models into CPU memory (no VRAM is touched)."""
        with self.lock:
            if self.tts is None or self.ecapa is None:
                self.ensure("cpu")

    def _move(self, dev: str) -> None:
        if self.tts is not None:
            for module in (self.tts.t3, self.tts.s3gen, self.tts.ve):
                _swap_module(module, dev, self._parked)
            self.tts.device = dev
            self.tts.conds = None  # reference conditionals are rebuilt per request
        if self.ecapa is not None:
            self.ecapa.to(_indexed(dev))
            self.ecapa.device = _indexed(dev)
        self.device = dev

    def park(self) -> None:
        """Move the models to CPU memory and release their VRAM."""
        with self.lock:
            if self.device not in (None, "cpu"):
                self._move("cpu")
                self._empty_cache()

    def unload(self) -> None:
        """Drop the models entirely; they are reloaded from disk on next use."""
        with self.lock:
            self.tts = self.ecapa = None
            self.device = None
            self._parked.clear()
            import gc
            gc.collect()
            self._empty_cache()

    def fall_back_to_cpu(self) -> None:
        bt.logging.warning(
            "Voice: CUDA out of memory during generation; moving voice to the CPU"
        )
        self.cpu_fallback = True
        with self.lock:
            self._move("cpu")
            self._empty_cache()

    @staticmethod
    def _empty_cache() -> None:
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


_MODELS = _Models()


def preload_async() -> None:
    """Warm the models in a background thread (see MIID_VOICE_PRELOAD)."""
    if not VOICE_PRELOAD or not voice_available():
        return

    def _run() -> None:
        try:
            _MODELS.preload()
            bt.logging.info("Voice: models preloaded and parked in CPU memory")
        except Exception as e:
            bt.logging.warning(
                f"Voice: preload failed ({type(e).__name__}: {e}); "
                "will load on the first voice request"
            )

    threading.Thread(target=_run, name="voice-preload", daemon=True).start()


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def _load_wav(data: bytes, sr: int):
    """Mono float32 [T] tensor at ``sr`` from WAV/FLAC/OGG/MP3 bytes."""
    import soundfile as sf
    import torch
    import torchaudio.functional as AF

    audio, file_sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    wav = torch.from_numpy(audio.mean(axis=1))
    if file_sr != sr:
        wav = AF.resample(wav, file_sr, sr)
    return wav


def _trim_silence(wav, sr: int, db: float = -40.0):
    """Drop leading/trailing frames quieter than ``db`` below the peak."""
    import torch

    frame = int(sr * 0.02)
    if wav.numel() < frame * 4:
        return wav
    frames = wav[: wav.numel() // frame * frame].view(-1, frame)
    rms = frames.pow(2).mean(dim=1).sqrt()
    floor = rms.max() * (10 ** (db / 20))
    voiced = torch.nonzero(rms > floor).flatten()
    if voiced.numel() == 0:
        return wav
    return wav[voiced[0] * frame:(voiced[-1] + 1) * frame]


def _write_reference(data: bytes, path: str) -> None:
    """The clip Chatterbox conditions on: trimmed, at most REF_SECONDS, 24 kHz."""
    import soundfile as sf

    wav = _trim_silence(_load_wav(data, COND_SR), COND_SR)
    wav = wav[: int(REF_SECONDS * COND_SR)]
    sf.write(path, wav.numpy(), COND_SR, subtype="PCM_16")


def _embed(ecapa, wav16k):
    import torch

    with torch.inference_mode():
        emb = ecapa.encode_batch(wav16k.unsqueeze(0).to(ecapa.device))
    return emb.squeeze().float().cpu()


def _similarity(ecapa, ref16k, gen16k) -> float:
    """Cosine similarity of ECAPA embeddings — the validator's identity score."""
    import torch

    return float(torch.nn.functional.cosine_similarity(
        _embed(ecapa, ref16k), _embed(ecapa, gen16k), dim=0,
    ))


def _wav_bytes(samples, sr: int) -> bytes:
    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, samples, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Cloning
# ---------------------------------------------------------------------------

def _clone(base_wav: bytes, text: str, language: str) -> Tuple[float, bytes, int]:
    """Best-of-N clone of ``base_wav`` saying ``text``: (similarity, wav bytes, takes)."""
    import torch
    import torchaudio.functional as AF

    tts, ecapa = _MODELS.for_inference()
    ref16k = _load_wav(base_wav, ECAPA_SR)

    with tempfile.TemporaryDirectory(prefix="miid_voice_") as tmp:
        cond_path = os.path.join(tmp, "reference.wav")
        _write_reference(base_wav, cond_path)
        # Chatterbox builds the conditionals with autograd on, and the graph
        # they carry pins every activation of the speaker encoder and mel
        # extractor — ~270 MB per clone that the garbage collector never
        # reclaims (measured on CPU), and the GPU memory every miner process
        # gained per voice request until the card ran out (2026-10-16 14:13).
        with torch.inference_mode():
            tts.prepare_conditionals(cond_path, exaggeration=0.5)

    attempts = max(1, VOICE_ATTEMPTS)
    best = None
    made = 0
    for attempt in range(attempts):
        made = attempt + 1
        torch.manual_seed(int(time.time() * 1000) % 2**31 + attempt)
        # Later attempts lean harder on the reference: less exaggeration and
        # temperature keep the timbre closer to the speaker.
        with _drop_new_hooks(tts.t3):
            wav = tts.generate(
                text,
                language_id=language,
                exaggeration=0.5 if attempt == 0 else 0.4,
                cfg_weight=0.5,
                temperature=0.8 if attempt == 0 else 0.6,
            )
        wav = wav.squeeze().float().cpu()
        sim = _similarity(ecapa, ref16k, AF.resample(wav, tts.sr, ECAPA_SR))
        bt.logging.info(f"Voice: take {made}/{attempts} — ECAPA similarity {sim:.3f}")
        if best is None or sim > best[0]:
            best = (sim, wav)
        if sim >= VOICE_IDENTITY_TARGET:
            break

    sim, wav = best
    peak = float(wav.abs().max())
    if peak > 1.0:
        wav = wav / peak
    return sim, _wav_bytes(wav.numpy(), tts.sr), made


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
        Generated WAV bytes (16-bit PCM, 24 kHz), or None if generation failed.
    """
    global last_similarity
    last_similarity = None

    prompt = (target_text or " ".join(target_words or [])).strip()
    if not prompt or not base_wav_bytes:
        bt.logging.warning("Voice: empty prompt or reference — skipping")
        return None
    if not voice_available():
        bt.logging.warning(
            f"Voice: packages not installed ({_import_error}) — run "
            "scripts/miner/setup_voice.sh to enable voice clones."
        )
        return None

    lang = (language or "en").strip().lower()[:2]
    if lang not in SUPPORTED_LANGUAGES:
        bt.logging.warning(f"Voice: unsupported language {language!r}; using English")
        lang = "en"

    import torch

    started = time.time()
    try:
        with _MODELS.lock, _preserve_root_logging():
            try:
                sim, wav_bytes, takes = _clone(base_wav_bytes, prompt, lang)
            except torch.cuda.OutOfMemoryError:
                _MODELS.fall_back_to_cpu()
                sim, wav_bytes, takes = _clone(base_wav_bytes, prompt, lang)
    except Exception as e:
        bt.logging.warning(f"Voice: clone failed ({type(e).__name__}: {e})")
        return None

    last_similarity = sim
    bt.logging.info(
        f"Voice: cloned {len(prompt.split())} words (lang={lang}) in "
        f"{time.time() - started:.1f}s — speaker similarity {sim:.3f} "
        f"after {takes} take(s)"
    )
    return wav_bytes


def voice_similarity(base_wav: bytes, generated_wav: bytes) -> Optional[float]:
    """ECAPA cosine similarity of two clips, or None if it could not be measured."""
    if not voice_available():
        return None
    try:
        with _MODELS.lock, _preserve_root_logging():
            _, ecapa = _MODELS.for_inference(need_tts=False)
            return _similarity(
                ecapa, _load_wav(base_wav, ECAPA_SR), _load_wav(generated_wav, ECAPA_SR),
            )
    except Exception as e:
        bt.logging.warning(f"Voice identity: similarity failed ({type(e).__name__}: {e})")
        return None


def release_worker(force: bool = False) -> None:
    """Free the voice models' VRAM between rounds.

    By default they are parked in CPU memory (moving back takes about a
    second); MIID_VOICE_KEEP_LOADED=1 leaves them on the GPU. ``force`` drops
    them entirely and they are reloaded from disk on the next request.
    """
    if force:
        _MODELS.unload()
    elif not VOICE_KEEP_LOADED:
        _MODELS.park()


def hash_voice_bytes(wav_bytes: bytes) -> str:
    """SHA256 hex digest of WAV bytes (same role as image_hash)."""
    return hashlib.sha256(wav_bytes).hexdigest()


# ---------------------------------------------------------------------------
# Command-line smoke test
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Clone a voice with the miner's in-process voice stack.",
    )
    parser.add_argument("--reference", required=True, help="reference clip (wav/flac/ogg/mp3)")
    parser.add_argument("--text", required=True, help="what the clone should say")
    parser.add_argument("--language", default="en", choices=sorted(SUPPORTED_LANGUAGES))
    parser.add_argument("--out", default="voice_clone.wav")
    args = parser.parse_args(argv)

    if not voice_available():
        print(
            f"voice stack not installed ({_import_error}); run scripts/miner/setup_voice.sh",
            file=sys.stderr,
        )
        return 2

    with open(args.reference, "rb") as f:
        reference = f.read()

    started = time.time()
    wav = generate_voice_clone(reference, args.text.split(), args.language, target_text=args.text)
    if not wav:
        print("no clone produced (see the log above)", file=sys.stderr)
        return 1
    with open(args.out, "wb") as f:
        f.write(wav)
    check = voice_similarity(reference, wav)
    release_worker()
    print(
        f"wrote {args.out} ({len(wav)} bytes) in {time.time() - started:.1f}s; "
        f"ECAPA similarity {last_similarity:.3f}"
        + (f" (re-measured on the file: {check:.3f})" if check is not None else "")
        + " — validator floor 0.6"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
