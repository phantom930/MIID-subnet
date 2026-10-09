# MIID/miner/voice_worker.py
#
# Voice-clone worker. Runs inside voice_env/ (scripts/miner/setup_voice.sh),
# NOT miner_env: Chatterbox pins a torch/transformers stack that would break
# the FLUX image path. The miner talks to it through voice_generator.py.
#
# Deliberately standalone — no MIID / bittensor imports — and started by path
# with `python -I`, so it never touches miner_env's packages.
#
# Protocol: one JSON object per line on stdin, one reply per line on stdout.
#   {"id": 1, "op": "ping"}
#   {"id": 2, "op": "clone", "ref_wav": "/x/ref.wav", "out_wav": "/x/out.wav",
#    "text": "three apple ...", "language": "en", "attempts": 3,
#    "target_similarity": 0.7}
#   {"id": 3, "op": "similarity", "ref_wav": "/x/a.wav", "gen_wav": "/x/b.wav"}
# Replies carry {"id", "ok"} plus "similarity" / "attempts" / "error".
#
# Model output that would otherwise go to stdout is redirected to stderr at
# startup, so the protocol channel only ever carries replies.

import json
import os
import sys
import time
import traceback

# Claim the protocol channel before any library can print into it.
_PROTO = os.fdopen(os.dup(1), "w", buffering=1)
os.dup2(2, 1)
sys.stdout = sys.stderr

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402
import torchaudio.functional as AF  # noqa: E402

ECAPA_ID = "speechbrain/spkrec-ecapa-voxceleb"
ECAPA_SR = 16000
# Chatterbox conditions on at most ~10 s of reference; the rest is ignored.
REF_SECONDS = float(os.environ.get("MIID_VOICE_REF_SECONDS", "10"))
SUPPORTED_LANGUAGES = {"en", "es"}

_tts = None
_ecapa = None
_device_override = None


def _log(msg: str) -> None:
    print(f"[voice_worker] {msg}", file=sys.stderr, flush=True)


def _device() -> str:
    if _device_override:
        return _device_override
    wanted = os.environ.get("MIID_VOICE_DEVICE", "").strip().lower()
    if wanted:
        return wanted
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_audio(path: str, sr: int) -> torch.Tensor:
    """Mono float32 [T] at ``sr``."""
    data, file_sr = sf.read(path, dtype="float32", always_2d=True)
    wav = torch.from_numpy(data.mean(axis=1))
    if file_sr != sr:
        wav = AF.resample(wav, file_sr, sr)
    return wav


def _trim_silence(wav: torch.Tensor, sr: int, db: float = -40.0) -> torch.Tensor:
    """Drop leading/trailing frames quieter than ``db`` below the peak."""
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


def _tts_model():
    global _tts, _ecapa, _device_override
    if _tts is None:
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS
        started = time.time()
        try:
            _tts = ChatterboxMultilingualTTS.from_pretrained(device=_device())
        except torch.cuda.OutOfMemoryError:
            # The GPU is shared with the FLUX image path; voice is not worth
            # failing over, so it drops to CPU (slower, same output).
            _log("CUDA out of memory loading Chatterbox; falling back to CPU")
            torch.cuda.empty_cache()
            _device_override, _ecapa = "cpu", None
            _tts = ChatterboxMultilingualTTS.from_pretrained(device="cpu")
        _log(f"Chatterbox Multilingual loaded on {_device()} "
             f"in {time.time() - started:.1f}s")
    return _tts


def _ecapa_model():
    global _ecapa
    if _ecapa is None:
        from speechbrain.inference.speaker import EncoderClassifier
        savedir = os.path.join(
            os.environ.get("HF_HOME", os.path.expanduser("~/.cache")),
            "speechbrain", "spkrec-ecapa-voxceleb",
        )
        _ecapa = EncoderClassifier.from_hparams(
            source=ECAPA_ID, savedir=savedir,
            run_opts={"device": _device()},
        )
        _log(f"SpeechBrain ECAPA loaded on {_device()}")
    return _ecapa


def _embed(wav16k: torch.Tensor) -> torch.Tensor:
    with torch.inference_mode():
        emb = _ecapa_model().encode_batch(wav16k.unsqueeze(0).to(_device()))
    return emb.squeeze().float().cpu()


def _similarity(ref16k: torch.Tensor, gen16k: torch.Tensor) -> float:
    """Cosine similarity of ECAPA embeddings — the validator's identity score."""
    a, b = _embed(ref16k), _embed(gen16k)
    return float(torch.nn.functional.cosine_similarity(a, b, dim=0))


def _write_reference(src: str, dst: str) -> None:
    """The reference clip Chatterbox conditions on: trimmed, ~REF_SECONDS."""
    sr = 24000
    wav = _trim_silence(_load_audio(src, sr), sr)
    wav = wav[: int(REF_SECONDS * sr)]
    sf.write(dst, wav.numpy(), sr, subtype="PCM_16")


def _clone(req: dict) -> dict:
    language = (req.get("language") or "en").lower()[:2]
    if language not in SUPPORTED_LANGUAGES:
        language = "en"
    text = (req.get("text") or "").strip()
    if not text:
        return {"ok": False, "error": "empty text"}

    attempts = max(1, int(req.get("attempts", 3)))
    target = float(req.get("target_similarity", 0.7))
    out_wav = req["out_wav"]
    cond_wav = f"{out_wav}.ref.wav"
    _write_reference(req["ref_wav"], cond_wav)

    ref16k = _load_audio(req["ref_wav"], ECAPA_SR)
    model = _tts_model()
    best = None
    try:
        for attempt in range(attempts):
            torch.manual_seed(int(time.time() * 1000) % 2**31 + attempt)
            # Later attempts lean harder on the reference: less exaggeration
            # and temperature keep the timbre closer to the speaker.
            wav = model.generate(
                text,
                language_id=language,
                audio_prompt_path=cond_wav,
                exaggeration=0.5 if attempt == 0 else 0.4,
                cfg_weight=0.5,
                temperature=0.8 if attempt == 0 else 0.6,
            )
            wav = wav.squeeze().float().cpu()
            sim = _similarity(ref16k, AF.resample(wav, model.sr, ECAPA_SR))
            _log(f"attempt {attempt + 1}/{attempts}: similarity {sim:.3f}")
            if best is None or sim > best[0]:
                best = (sim, wav)
            if sim >= target:
                break
    finally:
        try:
            os.remove(cond_wav)
        except OSError:
            pass

    sim, wav = best
    peak = float(wav.abs().max()) or 1.0
    sf.write(out_wav, (wav / max(peak, 1.0)).numpy(), model.sr,
             subtype="PCM_16")
    return {"ok": True, "similarity": sim, "attempts": attempt + 1,
            "sample_rate": model.sr}


def _similarity_op(req: dict) -> dict:
    ref = _load_audio(req["ref_wav"], ECAPA_SR)
    gen = _load_audio(req["gen_wav"], ECAPA_SR)
    return {"ok": True, "similarity": _similarity(ref, gen)}


OPS = {
    "ping": lambda req: {"ok": True, "device": _device()},
    "clone": _clone,
    "similarity": _similarity_op,
}


def main() -> None:
    _log(f"ready (device={_device()}, torch {torch.__version__})")
    for line in sys.stdin:
        if not line.strip():
            continue
        req_id = None
        try:
            req = json.loads(line)
            req_id = req.get("id")
            op = OPS.get(req.get("op"))
            reply = op(req) if op else {"ok": False, "error": "unknown op"}
        except Exception as e:  # report, never die on one bad request
            traceback.print_exc()
            reply = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        reply["id"] = req_id
        _PROTO.write(json.dumps(reply) + "\n")
        _PROTO.flush()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
