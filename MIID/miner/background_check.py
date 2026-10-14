# MIID/miner/background_check.py
#
# CLIP zero-shot checks that run inside the miner, and the environment check
# for background slots built on them: is the scene behind the subject the kind
# the validator asked for?
#
# Two of the four -1 grades on 2026-10-13 fell on background slots. CLIP with
# the benchmark's own environment prompts (MIID/benchmark/analyzers.py) tells
# an indoor scene from an outdoor one reliably — the probabilities computed
# here match the benchmark's to three decimals — so a background attempt is
# checked before it is accepted and re-rolled with a hint when it shows the
# wrong kind of scene (see image_generator._Slot and
# generate_variations.BACKGROUND_HINTS).
#
# What this check does NOT do any more: fail an attempt as a "plain studio
# backdrop". The benchmark's studio prompt fires on passport framing, not on
# blank backdrops — every one of the 13 archived images it flagged on
# 2026-10-13 was a furnished room, and the benchmark itself scored 12 of them
# a 5 — so that rule only burned retries and could swap a higher-identity take
# for a lower one. A genuinely blank backdrop is left to the scene-change
# test the benchmark runs (edge density), which has no in-miner port yet.
#
# The same loaded model serves the expression and accessory measurements in
# edit_check.py through zero_shot(). Cost: ~0.4 s per call on CPU with
# openai/clip-vit-large-patch14, loaded once on first use. MIID_BG_CHECK=0
# switches the environment check off; a model that cannot load only disables
# every CLIP check (attempts are then accepted unchecked, as before) — it
# never fails a round.

import os
import threading
from typing import Any, Dict, Optional, Tuple

import bittensor as bt
from PIL import Image


CLIP_MODEL_ID = os.environ.get("MIID_BG_CLIP", "openai/clip-vit-large-patch14")
ENABLED = os.environ.get("MIID_BG_CHECK", "1").strip().lower() not in ("0", "false", "no")
# CPU by default: the GPU is the image pipeline's, and ViT-L/14 is well under
# a second on a handful of CPU threads.
DEVICE = os.environ.get("MIID_BG_CHECK_DEVICE", "cpu").strip() or "cpu"


def cpu_quota() -> int:
    """CPU cores this process may actually use: the cgroup quota, else the count.

    A container often advertises every core of the host (os.cpu_count() said
    128 on 2026-10-13) while its cgroup caps it far lower (15.36 cores there).
    torch sizes its thread pool from the advertised count, and 128 threads on
    15 cores turned a 0.4 s CLIP forward into 62 s.
    """
    count = os.cpu_count() or 1
    try:
        count = min(count, len(os.sched_getaffinity(0)))
    except Exception:
        pass
    for path in ("/sys/fs/cgroup/cpu.max", "/sys/fs/cgroup/cpu/cpu.cfs_quota_us"):
        try:
            with open(path) as f:
                parts = f.read().split()
        except Exception:
            continue
        try:
            if path.endswith("cpu.max"):
                if parts[0] == "max":
                    break
                quota, period = float(parts[0]), float(parts[1])
            else:
                quota = float(parts[0])
                if quota <= 0:
                    break
                with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as f:
                    period = float(f.read().strip())
            if quota > 0 and period > 0:
                count = min(count, max(1, int(quota / period)))
        except Exception:
            pass
        break
    return max(1, count)


# Threads for the CLIP forward on CPU: measured 0.43 s at 16, 0.47 s at 8,
# 0.80 s at 32 and 62 s at the 128 default on the 15-core quota above.
THREADS = int(os.environ.get("MIID_BG_CHECK_THREADS", "0")) or min(16, cpu_quota())
# The other kind of scene at least this likely fails the attempt.
MISMATCH_PROB = float(os.environ.get("MIID_BG_MISMATCH_PROB", "0.5"))

# Same wording as the benchmark, so its reports and this check agree. The
# studio prompt is kept so the logged probabilities stay comparable with the
# benchmark's; it no longer fails anything.
ENV_PROMPTS: Dict[str, str] = {
    "indoor": "a portrait photo of a person indoors, inside a room or building",
    "outdoor": "a portrait photo of a person outdoors, outside in the open air",
    "studio": "a passport photo of a person against a plain empty studio backdrop",
}
WANTED_ENV: Dict[str, str] = {"background_in": "indoor", "background_out": "outdoor"}

_lock = threading.Lock()
_clip: Optional[Tuple[Any, Any, Any]] = None
_failed = False
_text_cache: Dict[Tuple[str, ...], Any] = {}


def wanted_environment(var_type: Optional[str]) -> Optional[str]:
    """"indoor" / "outdoor" for a background slot, None for any other."""
    return WANTED_ENV.get(var_type or "")


def _model() -> Optional[Tuple[Any, Any, Any]]:
    global _clip, _failed
    if _clip is not None or _failed:
        return _clip
    with _lock:
        if _clip is None and not _failed:
            try:
                import torch
                from transformers import CLIPModel, CLIPProcessor

                # Cached weights first: the online path spends ~6 s asking the
                # hub whether the cache is current, every time the miner starts.
                try:
                    model = CLIPModel.from_pretrained(CLIP_MODEL_ID, local_files_only=True)
                    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID, local_files_only=True)
                except Exception:
                    model = CLIPModel.from_pretrained(CLIP_MODEL_ID)
                    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
                model = model.eval().to(DEVICE)
                _clip = (model, processor, torch)
                bt.logging.info(
                    f"CLIP checks: loaded {CLIP_MODEL_ID} on {DEVICE} "
                    f"({THREADS} CPU threads for the forward)"
                )
            except Exception as e:
                _failed = True
                bt.logging.warning(
                    f"CLIP checks unavailable ({e}); background, expression and "
                    "accessory checks are skipped and attempts accepted unchecked."
                )
    return _clip


def available() -> bool:
    """True once the model is loaded, loading it on the first call."""
    return _model() is not None


def _features(features: Any) -> Any:
    """transformers 5 returns a ModelOutput from get_*_features; 4.x a tensor."""
    if hasattr(features, "pooler_output") and features.pooler_output is not None:
        return features.pooler_output
    if hasattr(features, "last_hidden_state"):
        return features.last_hidden_state
    return features


def zero_shot(image: Image.Image, prompts: Dict[str, str]) -> Optional[Dict[str, float]]:
    """CLIP softmax over ``prompts`` for one image — label -> probability.

    None when the model is unavailable. Text embeddings are cached per prompt
    set, so a call costs one image forward.
    """
    loaded = _model()
    if loaded is None:
        return None
    model, processor, torch = loaded
    labels = list(prompts)
    texts = tuple(prompts[k] for k in labels)
    # torch's thread pool is process-wide; cap it for this forward only and
    # put it back, so nothing else in the miner changes.
    previous_threads = torch.get_num_threads()
    try:
        if DEVICE == "cpu" and previous_threads != THREADS:
            torch.set_num_threads(THREADS)
        with torch.no_grad():
            text_emb = _text_cache.get(texts)
            if text_emb is None:
                tokens = processor(text=list(texts), return_tensors="pt", padding=True)
                emb = _features(model.get_text_features(
                    **{k: v.to(DEVICE) for k, v in tokens.items()}
                ))
                text_emb = emb / emb.norm(dim=-1, keepdim=True)
                _text_cache[texts] = text_emb
            pixels = processor(images=image.convert("RGB"), return_tensors="pt")["pixel_values"]
            img_emb = _features(model.get_image_features(pixel_values=pixels.to(DEVICE)))
            img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
            logits = model.logit_scale.exp() * img_emb @ text_emb.T
            probs = logits.softmax(dim=-1)[0].tolist()
        return {label: float(p) for label, p in zip(labels, probs)}
    except Exception as e:
        bt.logging.warning(f"CLIP check failed: {e}")
        return None
    finally:
        if torch.get_num_threads() != previous_threads:
            torch.set_num_threads(previous_threads)


def image_embedding(image: Image.Image) -> Optional[list]:
    """Unit-length CLIP image embedding of one image, or None when unavailable.

    Used by the sibling ledger (sibling_registry.py) to tell whether two
    hotkeys' takes of the same slot are near-identical.
    """
    loaded = _model()
    if loaded is None:
        return None
    model, processor, torch = loaded
    previous_threads = torch.get_num_threads()
    try:
        if DEVICE == "cpu" and previous_threads != THREADS:
            torch.set_num_threads(THREADS)
        with torch.no_grad():
            pixels = processor(images=image.convert("RGB"), return_tensors="pt")["pixel_values"]
            emb = _features(model.get_image_features(pixel_values=pixels.to(DEVICE)))
            emb = emb / emb.norm(dim=-1, keepdim=True)
        return [float(x) for x in emb[0].tolist()]
    except Exception as e:
        bt.logging.warning(f"CLIP embedding failed: {e}")
        return None
    finally:
        if torch.get_num_threads() != previous_threads:
            torch.set_num_threads(previous_threads)


def environment_probs(image: Image.Image) -> Optional[Dict[str, float]]:
    """CLIP softmax over ENV_PROMPTS for one image, or None when unavailable."""
    if not ENABLED:
        return None
    return zero_shot(image, ENV_PROMPTS)


def background_miss(var_type: Optional[str], probs: Optional[Dict[str, float]]) -> Optional[str]:
    """Why a background attempt fails, or None when it passes.

    "indoor" / "outdoor": the other kind of scene than the one requested.
    A non-background slot, or an attempt without probabilities, passes.
    """
    wanted = wanted_environment(var_type)
    if wanted is None or not probs:
        return None
    other = "outdoor" if wanted == "indoor" else "indoor"
    if probs.get(other, 0.0) >= MISMATCH_PROB:
        return other
    return None


def describe(var_type: Optional[str], probs: Optional[Dict[str, float]]) -> str:
    """Short log form, e.g. "indoor 0.85 / studio 0.05"."""
    wanted = wanted_environment(var_type)
    if wanted is None or not probs:
        return ""
    return f"{wanted} {probs.get(wanted, 0.0):.2f} / studio {probs.get('studio', 0.0):.2f}"
