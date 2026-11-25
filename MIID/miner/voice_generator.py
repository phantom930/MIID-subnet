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
# A clone is graded twice: ECAPA similarity to the reference above 0.6
# (speech_brain_compare enforces that floor), then a reviewer listens for
# every target word: all of them heard is 5 points, a couple missing or
# wrong is 3 (docs/scoring.md; voice is half of the P5 C1 UAV score). Up
# to MIID_VOICE_ATTEMPTS takes are generated; each is scored with ECAPA and
# transcribed by Whisper (word_check), and the best take is the one over the
# floor that says every word, then the one with fewest word errors, then the
# highest similarity. Generation stops early once a take says every word at
# MIID_VOICE_IDENTITY_TARGET.
#
# The model is not given the target text as is. Every prompt is wrapped in
# the hotkey's first and last four characters ("5 H P O ... R D 8 L"), and
# read raw the Spanish voice ran them into words ("Hachapo", "Rebocho
# Locho"): Whisper heard every word of 0 of 57 archived Spanish clones
# (2026-10-25/26) and 26 of 66 English ones. word_check.spoken_text names
# each letter in the clip's language ("hache, pe, o") with a pause between
# tokens; on the same references Spanish went from 0 of 4 takes to 3 of 4,
# and English held level (7 of 10 either way), so English alternates the
# two spellings between takes and lets the word check pick.
#
# Phase 5 Cycle 2 (from 2026-11-19) asks for five clips a round, all the
# same words and speaker: one per style (neutral, surprised, excited, angry,
# whisper), each over a background-noise level the validator draws (none,
# light, high). The validator's style prompt cannot be handed to Chatterbox,
# which reads aloud whatever text it gets. Styles are set with its
# exaggeration / cfg_weight knobs and the prompt's end punctuation
# (VOICE_STYLES). A whisper and the noise are added to each take afterwards
# (_render), as strongly as the clip's ECAPA similarity allows. Each clip
# is scored out of 5 (every word, ECAPA >= 0.6, the style audible) and the
# five are averaged (docs/scoring.md).
#
# Prompt text is already joined for you as target_text, e.g.
#   "5 G U P three orange book apple table door light music phone school six K V D U"
# (same as " ".join(target_words); includes hotkey bookends when present).

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

from MIID.miner import word_check


# Takes generated per request at most (a take is ~2–3 s on the GPU). Raised
# from 3 on 2026-10-26 with the word check: about a third of single takes
# drop or garble a word.
VOICE_ATTEMPTS = int(os.environ.get("MIID_VOICE_ATTEMPTS", "6"))
# Stop re-rolling once a take reaches this similarity. Above the validator's
# 0.6 floor so that scoring noise on their side does not tip it under.
VOICE_IDENTITY_TARGET = float(os.environ.get("MIID_VOICE_IDENTITY_TARGET", "0.7"))
# The validator's floor: a take under it is rejected whatever it says.
VOICE_FLOOR = 0.6
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

# Chatterbox has no emotion labels. Its exaggeration knob (0.25-2) sets how
# expressive a take is, and at high exaggeration a lower cfg_weight keeps
# the pace natural (Chatterbox README). The prompt's end punctuation steers
# the closing contour: "?!" for the surprised rise. A whisper is rendered
# after generation from a calm take (_render).
VOICE_STYLES: Dict[str, Dict[str, Any]] = {
    "neutral": {"exaggeration": 0.5, "cfg_weight": 0.5, "end": "."},
    "surprised": {"exaggeration": 0.8, "cfg_weight": 0.35, "end": "?!"},
    "excited": {"exaggeration": 1.0, "cfg_weight": 0.3, "end": "!"},
    "angry": {"exaggeration": 1.2, "cfg_weight": 0.3, "end": "!"},
    "whisper": {"exaggeration": 0.3, "cfg_weight": 0.5, "end": "."},
}
# Background-noise SNR (dB under the speech) per level, strongest first; a
# weaker step is used only when the stronger one costs too much similarity.
NOISE_SNR_DB: Dict[str, Tuple[float, ...]] = {
    "none": (),
    "light": (20.0, 26.0),
    "high": (10.0, 14.0, 18.0),
}
# Share of breath in a whisper, strongest first. A full whisper keeps no
# pitch, and ECAPA loses the speaker with it (0.39 on average, see _render).
WHISPER_MIX: Tuple[float, ...] = (0.5, 0.4, 0.3, 0.2)
# Peak level of a whispered clip (about -10 dBFS): quiet, still clear.
WHISPER_PEAK = 0.3

# Speaker similarity, word check and rendering (style, noise) of the last
# clip generate_voice_clone returned, for logs and the request archive.
last_similarity: Optional[float] = None
last_words: Optional[Dict[str, Any]] = None
last_render: Optional[Dict[str, Any]] = None

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
        self.cond_key: Optional[str] = None  # reference the tts conditionals were built from

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
            self.cond_key = None
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
            self.device = self.cond_key = None
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
            word_check.preload()
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
# Styles and background noise (P5 C2)
# ---------------------------------------------------------------------------

def _style(variation_type: str) -> Tuple[str, Dict[str, Any]]:
    """The style preset for a requested variation type (neutral if unknown)."""
    name = (variation_type or "").strip().lower()
    if name not in VOICE_STYLES:
        if name and name != "voice":
            bt.logging.warning(f"Voice: unknown style {variation_type!r}; reading it neutral")
        name = "neutral"
    return name, VOICE_STYLES[name]


def _noise_level(noise: str) -> str:
    level = (noise or "none").strip().lower()
    if level not in NOISE_SNR_DB:
        bt.logging.warning(f"Voice: unknown noise level {noise!r}; adding none")
        level = "none"
    return level


def _whisper(wav, sr: int):
    """``wav`` with its voicing replaced by breath: each frame's spectral
    envelope (cepstrum liftered below the shortest pitch period) over
    random-phase noise, high-passed above the pitch range, at the same RMS."""
    import torch
    import torchaudio.functional as AF

    n_fft, hop, lifter = 1024, 256, int(sr * 0.0017)
    window = torch.hann_window(n_fft)
    spec = torch.stft(wav, n_fft, hop, window=window, return_complex=True)
    cepstrum = torch.fft.irfft(spec.abs().clamp_min(1e-7).log(), n=n_fft, dim=0)
    cepstrum[lifter:n_fft - lifter + 1] = 0
    envelope = torch.fft.rfft(cepstrum, dim=0).real.exp()
    phase = torch.rand(envelope.shape) * (2 * torch.pi)
    out = torch.istft(torch.polar(envelope, phase), n_fft, hop, window=window, length=wav.numel())
    out = AF.highpass_biquad(out, sr, 300.0)
    return out * (_rms(wav) / _rms(out).clamp_min(1e-9))


def _rms(wav):
    return wav.pow(2).mean().sqrt()


def _speech_rms(wav, sr: int):
    """RMS over the frames within 40 dB of the loudest: the level the SNR is
    set against, so pauses between tokens do not make the noise louder."""
    frame = max(1, int(sr * 0.02))
    if wav.numel() < frame:
        return _rms(wav)
    power = wav[: wav.numel() // frame * frame].view(-1, frame).pow(2).mean(dim=1)
    active = power[power > power.max() * 1e-4]
    return active.mean().sqrt() if active.numel() else _rms(wav)


def _noise_bed(n: int, level: str):
    """Unit-RMS background noise: pink for a light room tone, pink over brown
    (traffic, wind, a busy room's rumble) for a high one."""
    import torch

    def colored(exponent: float):
        spectrum = torch.fft.rfft(torch.randn(n))
        freq = torch.arange(spectrum.numel(), dtype=torch.float32).clamp_min(1.0)
        x = torch.fft.irfft(spectrum / freq.pow(exponent), n=n)
        return x / _rms(x).clamp_min(1e-9)

    bed = colored(0.5) if level == "light" else 0.8 * colored(0.5) + 0.6 * colored(1.0)
    return bed / _rms(bed).clamp_min(1e-9)


def _render(wav, sr: int, style: str, level: str, ecapa, ref16k):
    """Apply the style's post-processing and the background noise to a take.

    Whispered speech and noise both pull ECAPA similarity down (on 12
    archived clones, 2026-11-25: noise at 10 dB SNR -0.10 on average and
    down to 0.59; a full whisper 0.80 -> 0.39, a 30 % breath blend 0.73).
    The strongest rendering that keeps the clip at VOICE_IDENTITY_TARGET is
    used — whisper blend first, then the noise backed off — and the most
    similar one when none does. Returns (wav, similarity, what was applied).
    """
    import torchaudio.functional as AF

    whispered = _whisper(wav, sr) if style == "whisper" else None
    mixes = WHISPER_MIX if whispered is not None else (0.0,)
    snrs = NOISE_SNR_DB[level] or (None,)
    bed = _noise_bed(wav.numel(), level) if snrs[0] is not None else None

    best = None
    for snr in snrs:
        for mix in mixes:
            out = wav if not mix else mix * whispered + (1.0 - mix) * wav
            if snr is not None:
                out = out + bed * (_speech_rms(out, sr) * 10 ** (-snr / 20))
            peak = float(out.abs().max())
            # A whisper is quiet; everything else just must not clip.
            target_peak = WHISPER_PEAK if whispered is not None else min(peak, 0.99)
            if peak > 0:
                out = out * (target_peak / peak)
            sim = _similarity(ecapa, ref16k, AF.resample(out, sr, ECAPA_SR))
            applied = {"whisper_mix": mix or None, "snr_db": snr}
            if best is None or sim > best[1]:
                best = (out, sim, applied)
            if sim >= VOICE_IDENTITY_TARGET:
                return out, sim, applied
    return best


# ---------------------------------------------------------------------------
# Cloning
# ---------------------------------------------------------------------------

def _texts(words: List[str], text: str, language: str, end: str = ".") -> List[str]:
    """What the model reads, by attempt (cycled): the letters named for a
    Spanish voice, both spellings in turn for an English one (see the top
    of this module), ending on the style's punctuation."""
    spoken = word_check.spoken_text(words, language)
    texts = [spoken] if language == "es" or not text else [spoken, text]
    return [t.rstrip(" .") + end for t in texts]


def _rank(sim: float, words: Optional[Dict[str, Any]]) -> Tuple:
    """Order of takes: over the ECAPA floor, every word heard, fewest word
    errors, then similarity. An unchecked take (no Whisper) counts as
    neither a pass nor a fail."""
    errors = 0 if words is None else words["errors"]
    return (sim >= VOICE_FLOOR, words is None or words["ok"], -errors, sim)


def _clone(
    base_wav: bytes, words: List[str], text: str, language: str,
    style: str = "neutral", level: str = "none", deadline: Optional[float] = None,
) -> Tuple[float, bytes, int, Optional[Dict[str, Any]], Dict[str, Any]]:
    """Best-of-N clone of ``base_wav`` saying ``words`` in ``style`` over
    ``level`` background noise: (similarity, wav bytes, takes, word check of
    the kept take, what _render applied to it)."""
    import torch
    import torchaudio.functional as AF

    tts, ecapa = _MODELS.for_inference()
    ref16k = _load_wav(base_wav, ECAPA_SR)
    preset = VOICE_STYLES[style]
    texts = _texts(words, text, language, preset["end"])

    # The five clips of a round share one reference: its conditionals are
    # built once and kept until the models move (generate() swaps in each
    # take's exaggeration itself).
    cond_key = hashlib.sha256(base_wav).hexdigest()
    if tts.conds is None or _MODELS.cond_key != cond_key:
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
        _MODELS.cond_key = cond_key

    attempts = max(1, VOICE_ATTEMPTS)
    best = None
    made = 0
    take_seconds = 0.0
    for attempt in range(attempts):
        if best is not None and deadline is not None and time.time() + take_seconds > deadline:
            bt.logging.info(f"Voice: no time for take {attempt + 1}; keeping the best of {made}")
            break
        started = time.time()
        made = attempt + 1
        torch.manual_seed(int(time.time() * 1000) % 2**31 + attempt)
        # Later attempts lean harder on the reference: less exaggeration and
        # temperature keep the timbre closer to the speaker.
        with _drop_new_hooks(tts.t3):
            wav = tts.generate(
                texts[attempt % len(texts)],
                language_id=language,
                exaggeration=preset["exaggeration"] * (1.0 if attempt == 0 else 0.8),
                cfg_weight=preset["cfg_weight"],
                temperature=0.8 if attempt == 0 else 0.6,
            )
        wav = wav.squeeze().float().cpu()
        wav, sim, applied = _render(wav, tts.sr, style, level, ecapa, ref16k)
        heard = word_check.check(
            AF.resample(wav, tts.sr, ECAPA_SR), words, language, device=_MODELS.device or "cpu",
        )
        bt.logging.info(
            f"Voice: {style} take {made}/{attempts} — ECAPA similarity {sim:.3f}"
            f"{_describe_render(applied)}, {word_check.describe(heard)}"
        )
        if best is None or _rank(sim, heard) > _rank(best[0], best[2]):
            best = (sim, wav, heard, applied)
        take_seconds = max(take_seconds, time.time() - started)
        if sim >= VOICE_IDENTITY_TARGET and (heard is None or heard["ok"]):
            break

    sim, wav, heard, applied = best
    peak = float(wav.abs().max())
    if peak > 1.0:
        wav = wav / peak
    return sim, _wav_bytes(wav.numpy(), tts.sr), made, heard, applied


def _describe_render(applied: Dict[str, Any]) -> str:
    parts = []
    if applied.get("whisper_mix"):
        parts.append(f"whisper blend {applied['whisper_mix']:.0%}")
    if applied.get("snr_db") is not None:
        parts.append(f"noise at {applied['snr_db']:.0f} dB SNR")
    return f" ({', '.join(parts)})" if parts else ""


def decode_base_voice(base64_voice: str) -> bytes:
    """Decode a base64-encoded WAV into raw bytes."""
    return base64.b64decode(base64_voice)


def generate_voice_clone(
    base_wav_bytes: bytes,
    target_words: List[str],
    language: str,
    target_text: str = "",
    variation_type: str = "",
    noise: str = "",
    variation_detail: str = "",
    deadline: Optional[float] = None,
) -> Optional[bytes]:
    """Generate a voice-cloned WAV speaking the target prompt.

    A P5 C2 round asks for five clips. Each call is one clip: same speaker
    and same words, in the requested style over the requested background
    noise.

    Args:
        base_wav_bytes: Reference speaker WAV bytes from the validator.
        target_words: Token list the generated audio should contain.
        language: ``"en"`` or ``"es"``.
        target_text: Ready-to-speak prompt string (preferred for TTS).
            Falls back to ``" ".join(target_words)`` if empty.
        variation_type: ``neutral``, ``surprised``, ``excited``, ``angry``,
            or ``whisper`` (anything else is read neutral).
        noise: ``none``, ``light``, or ``high``.
        variation_detail: The validator's style and noise prompt, logged
            only: Chatterbox reads every word it is given, so the style is
            set through VOICE_STYLES and _render instead.
        deadline: Wall-clock time the last take must end by; no new take is
            started that would run past it (the first always runs).

    Returns:
        Generated WAV bytes (16-bit PCM, 24 kHz), or None if generation failed.
    """
    global last_similarity, last_words, last_render
    last_similarity = last_words = last_render = None

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
    words = [w for w in (target_words or prompt.split()) if w.strip()]
    style, _ = _style(variation_type)
    level = _noise_level(noise)
    if variation_detail:
        bt.logging.debug(f"Voice: {style}/{level} asked for: {variation_detail}")

    import torch

    started = time.time()
    try:
        with _MODELS.lock, _preserve_root_logging():
            try:
                sim, wav_bytes, takes, heard, applied = _clone(
                    base_wav_bytes, words, prompt, lang, style, level, deadline,
                )
            except torch.cuda.OutOfMemoryError:
                _MODELS.fall_back_to_cpu()
                sim, wav_bytes, takes, heard, applied = _clone(
                    base_wav_bytes, words, prompt, lang, style, level, deadline,
                )
    except Exception as e:
        bt.logging.warning(f"Voice: {style} clone failed ({type(e).__name__}: {e})")
        return None

    last_similarity, last_words = sim, heard
    last_render = {"style": style, "noise": level, **applied}
    bt.logging.info(
        f"Voice: cloned {len(words)} words {style}/{level} noise (lang={lang}) in "
        f"{time.time() - started:.1f}s — speaker similarity {sim:.3f}"
        f"{_describe_render(applied)}, {word_check.describe(heard)}, after {takes} take(s)"
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
    word_check.release()


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
    parser.add_argument("--style", default="neutral", choices=list(VOICE_STYLES))
    parser.add_argument("--noise", default="none", choices=list(NOISE_SNR_DB))
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
    wav = generate_voice_clone(
        reference, args.text.split(), args.language, target_text=args.text,
        variation_type=args.style, noise=args.noise,
    )
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
        + f" — validator floor 0.6; {word_check.describe(last_words)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
