# MIID/benchmark/analyzers.py
#
# Measurements the grading API's score sheet is written in terms of: AdaFace
# identity, MediaPipe face count / pose / expression, face-region lighting,
# background, accessory, and a copy-paste residual. Nothing here assigns a
# score — scoring.py does that from the plain dict pair_metrics() returns, so
# rules can be tuned and re-run without loading a single model.
#
# Everything runs on CPU by default: the live miner owns the GPU, and every
# model here is small enough that a round of five variations takes seconds.

from __future__ import annotations

import hashlib
import math
import os
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

MODELS_DIR = Path(
    os.environ.get(
        "MIID_BENCH_MODELS",
        Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
        / "miid_benchmark",
    )
)

_MP_BASE = "https://storage.googleapis.com/mediapipe-models"
MEDIAPIPE_MODELS = {
    "face_landmarker.task":
        f"{_MP_BASE}/face_landmarker/face_landmarker/float16/latest/"
        "face_landmarker.task",
    "blaze_face_short_range.tflite":
        f"{_MP_BASE}/face_detector/blaze_face_short_range/float16/latest/"
        "blaze_face_short_range.tflite",
    "selfie_multiclass_256x256.tflite":
        f"{_MP_BASE}/image_segmenter/selfie_multiclass_256x256/float32/latest/"
        "selfie_multiclass_256x256.tflite",
}

CLIP_MODEL_ID = os.environ.get("MIID_BENCH_CLIP", "openai/clip-vit-large-patch14")

# Bump when a measurement changes meaning, so cached metrics are recomputed.
ANALYZER_VERSION = 4

# selfie_multiclass_256x256 categories.
SEG_BACKGROUND, SEG_HAIR, SEG_BODY, SEG_FACE, SEG_CLOTHES, SEG_OTHER = range(6)

NOSE_TIP = 1

# Blendshapes that move with an expression edit. Left/right pairs are
# averaged so a symmetric smile and a lopsided one read the same.
EXPRESSION_CHANNELS = {
    "smile": ("mouthSmileLeft", "mouthSmileRight"),
    "frown": ("mouthFrownLeft", "mouthFrownRight"),
    "jaw_open": ("jawOpen",),
    "brow_up": ("browInnerUp",),
    "brow_down": ("browDownLeft", "browDownRight"),
    "eye_wide": ("eyeWideLeft", "eyeWideRight"),
    "eye_squint": ("eyeSquintLeft", "eyeSquintRight"),
    "cheek_squint": ("cheekSquintLeft", "cheekSquintRight"),
    "mouth_stretch": ("mouthStretchLeft", "mouthStretchRight"),
    "mouth_press": ("mouthPressLeft", "mouthPressRight"),
    # Upper lip raised off the teeth: separates a toothy smile from a
    # closed-mouth one, which mouthSmile alone cannot (it reads ~0.75 for
    # both). scoring.expression_level uses it; it never wins on its own.
    "upper_lip_up": ("mouthUpperUpLeft", "mouthUpperUpRight"),
}

# Zero-shot prompts for what is on the subject's head, one per accessory key
# in ACCESSORY_TYPES (MIID/validator/image_variations.py). One prompt per
# group on purpose: splitting religious coverings into four prompts let their
# summed mass outvote a single, correct "knit beanie".
ACCESSORY_PROMPTS = {
    "religious_head_covering":
        "a photo of a person wearing a religious head covering such as a hijab, "
        "turban, kippah or prayer cap",
    "brim_hat": "a photo of a person wearing a fedora or wide-brim hat",
    # Shape words matter: plain "beanie" lost to "bare head" on snug knits,
    # and "baseball cap" without the visor won on them.
    "knit_winter_hat": "a photo of a person wearing a knit beanie, a soft stretchy "
                       "wool hat with no brim pulled over the head",
    "bandana": "a photo of a person wearing a bandana tied on the head",
    "baseball_cap": "a photo of a person wearing a baseball cap with a stiff curved "
                    "visor sticking out at the front",
    "headphones": "a photo of a person wearing headphones over the ears",
    "none": "a photo of a person with a bare head, their own hair or scalp visible on top",
}

# Second stage, only to name a religious covering for the gender check.
# Worded by coverage because CLIP's notion of "taqiyah" swallows most wraps;
# scoring.py trusts the segmentation side-coverage cue over these.
RELIGIOUS_PROMPTS = {
    "hijab": "a person wearing a hijab: a headscarf wrapped around the face that "
             "hides the hair, ears and neck",
    "turban": "a person wearing a turban: a large cloth wrapped around the top of "
              "the head, with the ears and neck visible",
    "kippah": "a person wearing a kippah: a tiny flat skullcap on the back of the "
              "head with hair visible",
    "taqiyah": "a person wearing a taqiyah: a short rounded prayer cap on top of "
               "the head, ears and hair at the sides visible",
}

# Third stage, only when the head-covering group comes out as a religious
# covering or a bandana: CLIP's group stage calls most plain turbans
# "bandana" (9 of 9 on 2026-10-07), and the turban-vs-cap stage above has no
# bandana option at all. Print and bulk are what tell them apart here.
WRAP_PROMPTS = {
    "turban": "a person wearing a turban: plain solid-colour fabric wound in "
              "thick folds around the whole top of the head",
    "bandana": "a person wearing a bandana: a thin printed kerchief, such as "
               "paisley, tied over the hair with a knot",
}

ENVIRONMENT_PROMPTS = {
    "indoor": "a portrait photo of a person indoors, inside a room or building",
    "outdoor": "a portrait photo of a person outdoors, outside in the open air",
    "studio": "a passport photo of a person against a plain empty studio backdrop",
}

GENDER_PROMPTS = {
    "m": "a portrait photo of a man",
    "f": "a portrait photo of a woman",
}


def ensure_mediapipe_models(models_dir: Path = MODELS_DIR) -> Path:
    """Download the three MediaPipe task files on first use."""
    models_dir.mkdir(parents=True, exist_ok=True)
    for name, url in MEDIAPIPE_MODELS.items():
        target = models_dir / name
        if target.is_file() and target.stat().st_size > 0:
            continue
        tmp = target.with_suffix(target.suffix + ".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(target)
    return models_dir


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dhash(image: Image.Image, size: int = 16) -> int:
    """256-bit difference hash — catches re-encoded or resized duplicates.

    64 bits is too coarse for this data: two different expressions of the
    same passport framing land within 4 bits. At 256 bits distinct
    submissions sit 34+ apart, re-saved copies under 10.
    """
    gray = image.convert("L").resize((size + 1, size), Image.LANCZOS)
    px = np.asarray(gray, dtype=np.int16)
    bits = (px[:, 1:] > px[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def gender_from_filename(name: Optional[str]) -> Optional[str]:
    """Validator base filenames end in _m / _f (e.g. 070f3abdfcf9_m.png)."""
    if not name:
        return None
    match = re.search(r"_([mf])(?:_[a-z]+)?\.[a-z0-9]+$", name.lower())
    return match.group(1) if match else None


def _euler_degrees(matrix: np.ndarray) -> Tuple[float, float, float]:
    """(yaw, pitch, roll) from MediaPipe's facial transformation matrix."""
    rot = np.asarray(matrix)[:3, :3]
    pitch = math.degrees(math.atan2(rot[2, 1], rot[2, 2]))
    yaw = math.degrees(math.asin(max(-1.0, min(1.0, -rot[2, 0]))))
    roll = math.degrees(math.atan2(rot[1, 0], rot[0, 0]))
    return yaw, pitch, roll


@dataclass
class ImageFeatures:
    """Per-image measurements, shared by every pair the image is part of."""

    path: Path
    width: int
    height: int
    sha256: str
    dhash: int
    face_count: int
    landmark_faces: int
    landmarks: Optional[np.ndarray] = None  # (478, 2) pixel coordinates
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    expression: Optional[Dict[str, float]] = None
    face_wh: Optional[float] = None
    face_area: Optional[float] = None
    face_light: Optional[Dict[str, float]] = None
    background: Optional[Dict[str, float]] = None
    bg_hist: Optional[np.ndarray] = None
    seg_fractions: Optional[List[float]] = None
    head: Optional[Dict[str, float]] = None
    gray: Optional[np.ndarray] = None
    face_mask: Optional[np.ndarray] = None

    def summary(self) -> Dict[str, Any]:
        return {
            "size": [self.width, self.height],
            "sha256": self.sha256,
            "dhash": f"{self.dhash:064x}",
            "face_count": self.face_count,
            "landmark_faces": self.landmark_faces,
            "yaw": self.yaw,
            "pitch": self.pitch,
            "roll": self.roll,
            "expression": self.expression,
            "face_wh": self.face_wh,
            "face_area": self.face_area,
            "face_light": self.face_light,
            "background": self.background,
            "seg_fractions": self.seg_fractions,
            "head": self.head,
        }


class Analyzer:
    """Lazily loads each model the first time a measurement needs it."""

    def __init__(self, device: str = "cpu", use_clip: bool = True):
        self.device = device
        self.use_clip = use_clip
        self._landmarker = None
        self._detector = None
        self._segmenter = None
        self._adaface = None
        self._clip = None
        self._features: Dict[str, ImageFeatures] = {}
        self._embeddings: Dict[str, Any] = {}
        self._text_cache: Dict[Tuple[str, ...], Any] = {}

    def close(self) -> None:
        """Release MediaPipe tasks now; their __del__ fails at interpreter exit."""
        for task in (self._landmarker, self._detector, self._segmenter):
            if task is not None:
                try:
                    task.close()
                except Exception:  # noqa: BLE001
                    pass
        self._landmarker = self._detector = self._segmenter = None

    # ── model loading ────────────────────────────────────────────────────

    def _mediapipe(self):
        if self._landmarker is not None:
            return
        from mediapipe.tasks.python import BaseOptions, vision

        models = ensure_mediapipe_models()
        self._landmarker = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=BaseOptions(
                    model_asset_path=str(models / "face_landmarker.task")
                ),
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=True,
                num_faces=4,
            )
        )
        self._detector = vision.FaceDetector.create_from_options(
            vision.FaceDetectorOptions(
                base_options=BaseOptions(
                    model_asset_path=str(models / "blaze_face_short_range.tflite")
                ),
                min_detection_confidence=0.5,
            )
        )
        self._segmenter = vision.ImageSegmenter.create_from_options(
            vision.ImageSegmenterOptions(
                base_options=BaseOptions(
                    model_asset_path=str(models / "selfie_multiclass_256x256.tflite")
                ),
                output_category_mask=True,
            )
        )

    def _adaface_model(self):
        if self._adaface is None:
            from MIID.miner import ada_face_compare

            self._adaface = ada_face_compare
            ada_face_compare.get_shared_model(
                "cuda:0" if self.device.startswith("cuda") else "cpu"
            )
        return self._adaface

    def _clip_model(self):
        if self._clip is None:
            import torch
            from transformers import CLIPModel, CLIPProcessor

            model = CLIPModel.from_pretrained(CLIP_MODEL_ID).eval()
            if self.device.startswith("cuda"):
                model = model.to("cuda")
            processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
            self._clip = (model, processor, torch)
        return self._clip

    # ── per-image features ───────────────────────────────────────────────

    def features(self, path: Path) -> ImageFeatures:
        key = str(Path(path).resolve())
        if key in self._features:
            return self._features[key]

        import mediapipe as mp

        self._mediapipe()
        pil = Image.open(path).convert("RGB")
        rgb = np.ascontiguousarray(np.asarray(pil))
        h, w = rgb.shape[:2]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        detections = self._detector.detect(mp_image).detections
        marks = self._landmarker.detect(mp_image)
        category = self._segmenter.segment(mp_image).category_mask.numpy_view()
        category = cv2.resize(
            np.asarray(category, dtype=np.uint8), (w, h),
            interpolation=cv2.INTER_NEAREST,
        )

        feats = ImageFeatures(
            path=Path(path),
            width=w,
            height=h,
            sha256=sha256_file(path),
            dhash=dhash(pil),
            face_count=len(detections),
            landmark_faces=len(marks.face_landmarks),
            seg_fractions=[float((category == k).mean()) for k in range(6)],
            gray=cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY),
        )

        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

        if marks.face_landmarks:
            # The largest face is the subject when the detector sees extras.
            best = max(
                range(len(marks.face_landmarks)),
                key=lambda i: _landmark_area(marks.face_landmarks[i]),
            )
            pts = np.array(
                [[p.x * w, p.y * h] for p in marks.face_landmarks[best]],
                dtype=np.float32,
            )
            feats.landmarks = pts
            feats.yaw, feats.pitch, feats.roll = _euler_degrees(
                marks.facial_transformation_matrixes[best]
            )
            shapes = {b.category_name: b.score for b in marks.face_blendshapes[best]}
            feats.expression = {
                name: float(np.mean([shapes.get(k, 0.0) for k in keys]))
                for name, keys in EXPRESSION_CHANNELS.items()
            }
            x0, y0 = pts.min(axis=0)
            x1, y1 = pts.max(axis=0)
            feats.face_wh = float((x1 - x0) / max(y1 - y0, 1.0))
            feats.face_area = float((x1 - x0) * (y1 - y0) / (w * h))

            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillConvexPoly(mask, cv2.convexHull(pts.astype(np.int32)), 1)
            erode = max(3, int(0.03 * (x1 - x0)))
            mask = cv2.erode(mask, np.ones((erode, erode), np.uint8))
            feats.face_mask = mask.astype(bool)
            feats.face_light = _face_light(lab, feats.face_mask, pts[NOSE_TIP][0])
            feats.head = _head_coverage(category, x0, y0, x1, y1)

        feats.background, feats.bg_hist = _background_stats(
            lab, hsv, feats.gray, category == SEG_BACKGROUND
        )

        self._features[key] = feats
        return feats

    # ── models over pairs ────────────────────────────────────────────────

    def identity(self, base: Path, variation: Path) -> Optional[float]:
        """AdaFace cosine similarity, aligned with MTCNN — the grader's metric."""
        ada = self._adaface_model()
        dev = "cuda:0" if self.device.startswith("cuda") else "cpu"
        base_key = str(Path(base).resolve())
        if base_key not in self._embeddings:
            self._embeddings[base_key] = ada.embed_image(str(base), device=dev)
        base_emb = self._embeddings[base_key]
        if base_emb is None:
            return None
        var_emb = ada.embed_image(str(variation), device=dev)
        if var_emb is None:
            return None
        return float(ada.compute_cosine_similarity(base_emb, var_emb))

    def zero_shot(self, image: Image.Image, prompts: Dict[str, str]) -> Dict[str, float]:
        """CLIP softmax over ``prompts`` — label -> probability."""
        model, processor, torch = self._clip_model()
        labels = list(prompts)
        texts = tuple(prompts[k] for k in labels)
        dev = next(model.parameters()).device
        with torch.no_grad():
            if texts not in self._text_cache:
                tok = processor(text=list(texts), return_tensors="pt", padding=True)
                emb = model.get_text_features(**{k: v.to(dev) for k, v in tok.items()})
                emb = _as_tensor(emb)
                self._text_cache[texts] = emb / emb.norm(dim=-1, keepdim=True)
            text_emb = self._text_cache[texts]
            pix = processor(images=image, return_tensors="pt")["pixel_values"].to(dev)
            img_emb = _as_tensor(model.get_image_features(pixel_values=pix))
            img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
            logits = model.logit_scale.exp() * img_emb @ text_emb.T
            probs = logits.softmax(dim=-1)[0].cpu().numpy()
        return {label: float(p) for label, p in zip(labels, probs)}

    def head_crop(self, feats: ImageFeatures) -> Image.Image:
        """Head plus headwear: the face box grown up, out, and a little down."""
        pil = Image.open(feats.path).convert("RGB")
        if feats.landmarks is None:
            return pil
        x0, y0 = feats.landmarks.min(axis=0)
        x1, y1 = feats.landmarks.max(axis=0)
        fw, fh = x1 - x0, y1 - y0
        # Down past the chin so a hijab's neck wrap is in frame.
        box = (
            max(0, int(x0 - 0.7 * fw)),
            max(0, int(y0 - 0.9 * fh)),
            min(feats.width, int(x1 + 0.7 * fw)),
            min(feats.height, int(y1 + 0.7 * fh)),
        )
        return pil.crop(box)

    def pair_metrics(
        self,
        base: Path,
        variation: Path,
        *,
        background_slot: bool,
        accessory_requested: bool,
        base_gender: Optional[str],
    ) -> Dict[str, Any]:
        """Every measurement scoring.py needs for one (base, variation) pair."""
        b = self.features(base)
        v = self.features(variation)
        out: Dict[str, Any] = {
            "base": b.summary(),
            "variation": v.summary(),
            "identity": self.identity(base, variation),
        }

        if b.landmarks is not None and v.landmarks is not None:
            out["pose_delta"] = {
                "yaw": v.yaw - b.yaw,
                "pitch": v.pitch - b.pitch,
                "roll": v.roll - b.roll,
            }
            out["expression_delta"] = {
                k: v.expression[k] - b.expression[k] for k in EXPRESSION_CHANNELS
            }
            out["face_wh_delta"] = (
                v.face_wh / b.face_wh - 1.0 if b.face_wh else None
            )
            out["lighting_delta"] = _lighting_delta(b.face_light, v.face_light)
            out["copy_paste"] = _copy_paste_residual(b, v)

        out["background_delta"] = _background_delta(b, v)

        if self.use_clip:
            gender = base_gender
            if gender is None:
                probs = self.zero_shot(Image.open(base).convert("RGB"), GENDER_PROMPTS)
                gender = max(probs, key=probs.get)
                out["base_gender_source"] = "clip"
            else:
                out["base_gender_source"] = "filename"
            out["base_gender"] = gender

            if background_slot or accessory_requested:
                env = self.zero_shot(Image.open(variation).convert("RGB"), ENVIRONMENT_PROMPTS)
                out["environment"] = env
                crop = self.head_crop(v)
                out["accessory_groups"] = self.zero_shot(crop, ACCESSORY_PROMPTS)
                out["religious_style_probs"] = self.zero_shot(crop, RELIGIOUS_PROMPTS)
                out["wrap_probs"] = self.zero_shot(crop, WRAP_PROMPTS)
        else:
            out["base_gender"] = base_gender
            out["base_gender_source"] = "filename" if base_gender else None

        return out


# ── helpers ──────────────────────────────────────────────────────────────


def _as_tensor(features):
    """transformers 5 returns a ModelOutput from get_*_features; 4.x a tensor."""
    if hasattr(features, "pooler_output") and features.pooler_output is not None:
        return features.pooler_output
    return features


def _landmark_area(landmarks) -> float:
    xs = [p.x for p in landmarks]
    ys = [p.y for p in landmarks]
    return (max(xs) - min(xs)) * (max(ys) - min(ys))


def _face_light(lab: np.ndarray, mask: np.ndarray, nose_x: float) -> Optional[Dict[str, float]]:
    """Brightness, contrast, colour cast and left/right balance of the face."""
    if mask.sum() < 200:
        return None
    L = lab[..., 0].astype(np.float32)
    a = lab[..., 1].astype(np.float32) - 128.0
    bb = lab[..., 2].astype(np.float32) - 128.0
    cols = np.arange(mask.shape[1])[None, :]
    left = mask & (cols < nose_x)
    right = mask & (cols >= nose_x)
    l_left = float(L[left].mean()) if left.any() else float(L[mask].mean())
    l_right = float(L[right].mean()) if right.any() else float(L[mask].mean())
    mean = float(L[mask].mean())
    return {
        "L": mean,
        "contrast": float(L[mask].std()),
        "a": float(a[mask].mean()),
        "b": float(bb[mask].mean()),
        "asym": (l_left - l_right) / max(mean, 1.0),
    }


def _head_coverage(category: np.ndarray, x0, y0, x1, y1) -> Dict[str, float]:
    """What sits beside the face from the eyes down to the chin.

    A hijab wraps the cheeks and jaw, so the segmenter labels those side
    strips as cloth; a turban, kippah or taqiyah sits on top of the head and
    leaves ears, hair or background there. CLIP cannot tell these apart
    reliably (it calls most dark wraps a taqiyah), this geometry can.
    """
    h, w = category.shape
    fw, fh = x1 - x0, y1 - y0
    top, bottom = int(y0 + 0.35 * fh), int(y1)
    left = category[top:bottom, max(0, int(x0 - 0.22 * fw)):max(0, int(x0 + 0.02 * fw))]
    right = category[top:bottom, min(w, int(x1 - 0.02 * fw)):min(w, int(x1 + 0.22 * fw))]
    side = np.concatenate([left.ravel(), right.ravel()])
    # Crown: the band above the hairline, across the face width. Anything worn
    # on the head shows up here as cloth; a bare head as hair, skin or
    # background. CLIP calls a snug beanie "bare head" more often than not.
    crown = category[
        max(0, int(y0 - 0.45 * fh)):max(0, int(y0 + 0.02 * fh)),
        max(0, int(x0 + 0.1 * fw)):min(w, int(x1 - 0.1 * fw)),
    ].ravel()
    cloth = (SEG_CLOTHES, SEG_OTHER)
    return {
        "side_cloth": float(np.isin(side, cloth).mean()) if side.size else 0.0,
        "side_skin": float(np.isin(side, (SEG_BODY, SEG_FACE)).mean()) if side.size else 0.0,
        "side_hair": float((side == SEG_HAIR).mean()) if side.size else 0.0,
        "crown_cloth": float(np.isin(crown, cloth).mean()) if crown.size else 0.0,
        "crown_hair": float((crown == SEG_HAIR).mean()) if crown.size else 0.0,
    }


def _lighting_delta(base: Optional[Dict[str, float]], var: Optional[Dict[str, float]]):
    if not base or not var:
        return None
    da = var["a"] - base["a"]
    db = var["b"] - base["b"]
    return {
        "brightness": var["L"] - base["L"],
        "contrast": var["contrast"] - base["contrast"],
        "asym": abs(var["asym"] - base["asym"]),
        "cast": math.hypot(da, db),
    }


def _background_stats(lab, hsv, gray, mask):
    """Plainness and colour of whatever the segmenter calls background."""
    if mask.sum() < 0.02 * mask.size:
        return {"fraction": float(mask.mean()), "valid": False}, None
    inner = cv2.erode(mask.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    if inner.sum() < 1000:
        inner = mask
    edges = cv2.Canny(gray, 80, 160) > 0
    L = lab[..., 0].astype(np.float32)
    sat = hsv[..., 1].astype(np.float32)
    stats = {
        "fraction": float(mask.mean()),
        "valid": True,
        "L": float(L[inner].mean()),
        "L_std": float(L[inner].std()),
        "saturation": float(sat[inner].mean()),
        "edge_density": float(edges[inner].mean()),
    }
    stats["plain_white"] = bool(
        stats["L"] > 200 and stats["saturation"] < 30 and stats["edge_density"] < 0.01
    )
    hist = cv2.calcHist(
        [hsv], [0, 1, 2], inner.astype(np.uint8), [18, 8, 8], [0, 180, 0, 256, 0, 256]
    )
    cv2.normalize(hist, hist, 1.0, 0.0, cv2.NORM_L1)
    return stats, hist


def _background_delta(b: ImageFeatures, v: ImageFeatures) -> Optional[Dict[str, Any]]:
    if not b.background or not v.background:
        return None
    if not (b.background.get("valid") and v.background.get("valid")):
        return {"valid": False}
    return {
        "valid": True,
        "hist_distance": float(
            cv2.compareHist(b.bg_hist, v.bg_hist, cv2.HISTCMP_BHATTACHARYYA)
        ),
        "edge_delta": v.background["edge_density"] - b.background["edge_density"],
        "brightness_delta": v.background["L"] - b.background["L"],
        "base_plain_white": b.background["plain_white"],
        "variation_plain_white": v.background["plain_white"],
    }


def _copy_paste_residual(b: ImageFeatures, v: ImageFeatures) -> Optional[Dict[str, float]]:
    """How closely the variation's face pixels reproduce the base's.

    The base is warped onto the variation with a similarity transform fitted
    to the 478 landmarks, then compared inside the variation's face hull. A
    cut-and-paste keeps skin texture intact, so the high-frequency correlation
    stays near 1; a regenerated face re-synthesises texture and drops well
    below that even when the identity is very close.
    """
    if v.face_mask is None:
        return None
    matrix, _ = cv2.estimateAffinePartial2D(
        b.landmarks, v.landmarks, method=cv2.RANSAC, ransacReprojThreshold=3.0
    )
    if matrix is None:
        return None
    warped = cv2.warpAffine(
        b.gray, matrix, (v.width, v.height), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    valid = cv2.warpAffine(
        np.ones_like(b.gray), matrix, (v.width, v.height), flags=cv2.INTER_NEAREST
    ).astype(bool)
    mask = v.face_mask & valid
    if mask.sum() < 500:
        return None

    def high_pass(img):
        img = img.astype(np.float32)
        return img - cv2.GaussianBlur(img, (0, 0), 3.0)

    hp_base = high_pass(warped)[mask]
    hp_var = high_pass(v.gray)[mask]
    denom = float(np.linalg.norm(hp_base) * np.linalg.norm(hp_var))
    corr = float(np.dot(hp_base, hp_var) / denom) if denom > 0 else 0.0
    mae = float(np.abs(warped.astype(np.float32)[mask] - v.gray.astype(np.float32)[mask]).mean())
    scale = float(math.hypot(matrix[0, 0], matrix[1, 0]))
    return {"hf_corr": corr, "mae": mae, "scale": scale}
