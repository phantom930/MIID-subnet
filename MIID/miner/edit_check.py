# MIID/miner/edit_check.py
#
# In-round measurements of how strongly each requested edit landed, so that a
# take is chosen for its edit first and for its identity second.
#
# The validator ranks miners by validation score — did the requested edit
# land at the requested intensity — and breaks ties on identity. Between two
# takes of the same slot, the higher-identity one is usually the weaker edit:
# the face held because less of it changed. Choosing by identity alone, which
# the polish phase did from 2026-10-13 19:49, therefore drifts every slot
# toward its mildest take, and the validation score falls while identity
# rises. These measurements put the edit back in front: a take whose edit
# reads weaker than the requested level, or weaker than the take it would
# replace, does not win on identity.
#
# What is measured, and with what:
#   lighting    the benchmark's lighting delta on the face — brightness,
#               contrast, left/right asymmetry and colour cast in Lab, the
#               largest of the four normalised terms (lighting_magnitude in
#               MIID/benchmark/scoring.py) — on an ellipse inside the MTCNN
#               face box, where the benchmark uses the landmark hull
#   expression  CLIP zero-shot over four expression classes on the head crop,
#               read as an expected ordinal (0 neutral .. 3 laughing) against
#               the base image's
#   accessory   CLIP zero-shot with the benchmark's accessory prompts on the
#               head crop (ACCESSORY_PROMPTS in MIID/benchmark/analyzers.py),
#               for a background slot that asks for one
# Pose (head turn) and scene (indoor / outdoor) are measured in
# image_generator and background_check and only compared here.
#
# The intensity bins are the benchmark's estimates of the grader's, not
# published rules, so only gross misses spend a retry: no measurable lighting
# change at all, a far lighting request that reads light, a medium or far
# expression request that still reads neutral, a missing or wrong accessory.
# The finer use — refusing a polish take whose edit reads weaker than the
# incumbent's — is a relative comparison between two takes of the same slot
# and needs no calibration at all.
#
# MIID_EDIT_CHECK=0 switches the lighting, expression and accessory
# measurements off (pose and scene checks have their own switches). Anything
# that cannot be measured is left unmeasured, never failed.

import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from MIID.miner import background_check
from MIID.miner.ada_face_compare import face_geometry


ENABLED = os.environ.get("MIID_EDIT_CHECK", "1").strip().lower() not in ("0", "false", "no")

# Lighting magnitude cut-offs between none / light / medium / far, as the
# benchmark estimates them (scoring.py T["light_bins"]).
LIGHT_BINS: Tuple[float, float, float] = (0.60, 1.20, 2.00)
# Expression: cut-offs on the change in expected ordinal (see EXPRESSION_ORDINAL).
EXPRESSION_BINS: Tuple[float, float, float] = (0.5, 1.5, 2.5)
# A medium or far expression request whose take still reads neutral at least
# this surely is a miss worth a retry.
NEUTRAL_PROB = float(os.environ.get("MIID_EXPR_NEUTRAL_PROB", "0.6"))
# Accessory confidence the grader is assumed to need (benchmark accessory_conf).
ACCESSORY_CONF = 0.5

# A polish take must keep at least this much of the incumbent's edit.
LIGHT_KEEP = 0.85          # fraction of the incumbent's lighting magnitude
EXPRESSION_KEEP = 0.3      # ordinal points below the incumbent's expression
POSE_KEEP_DEG = 5.0        # degrees of head turn below the incumbent's
SCENE_KEEP = 0.10          # probability of the wanted scene below the incumbent's
ACCESSORY_KEEP = 0.10      # accessory confidence below the incumbent's

INTENSITY_ORDER = ("light", "medium", "far")

# The validator's expression levels are a slight smile, a smile and laughing
# (or visible surprise / concern), so the classes follow what the mouth does.
EXPRESSION_PROMPTS: Dict[str, str] = {
    "neutral": (
        "a portrait photo of a person with a neutral, relaxed face: mouth "
        "closed, not smiling, calm eyes and brow"
    ),
    "light": (
        "a portrait photo of a person with a slight closed-mouth smile, the "
        "lips together and the corners lifted"
    ),
    "medium": (
        "a portrait photo of a person smiling openly with the teeth showing"
    ),
    "far": (
        "a portrait photo of a person laughing with the mouth wide open, or "
        "looking very surprised or worried with wide eyes and raised brows"
    ),
}
EXPRESSION_ORDINAL: Dict[str, int] = {"neutral": 0, "light": 1, "medium": 2, "far": 3}

# One prompt per accessory group, the benchmark's wording (one prompt per
# group on purpose: split religious coverings outvoted a correct beanie).
ACCESSORY_PROMPTS: Dict[str, str] = {
    "religious_head_covering":
        "a photo of a person wearing a religious head covering such as a hijab, "
        "turban, kippah or prayer cap",
    "brim_hat": "a photo of a person wearing a fedora or wide-brim hat",
    "knit_winter_hat": "a photo of a person wearing a knit beanie, a soft stretchy "
                       "wool hat with no brim pulled over the head",
    "bandana": "a photo of a person wearing a bandana tied on the head",
    "baseball_cap": "a photo of a person wearing a baseball cap with a stiff curved "
                    "visor sticking out at the front",
    "headphones": "a photo of a person wearing headphones over the ears",
    "none": "a photo of a person with a bare head, their own hair or scalp visible on top",
}

# Mirrors ACCESSORY_TYPES in MIID/validator/image_variations.py (and the
# benchmark loader). Order matters: the brim-hat detail says "(not baseball
# cap)", so it is matched before the baseball cap.
ACCESSORY_KEYWORDS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("religious_head_covering", ("religious head covering",)),
    ("brim_hat", ("brim hat", "fedora", "wide-brim")),
    ("knit_winter_hat", ("knit hat", "beanie", "winter hat")),
    ("bandana", ("bandana",)),
    ("baseball_cap", ("baseball cap", "sports cap")),
    ("headphones", ("headphones", "headset")),
)


def accessory_key(detail: Optional[str]) -> Optional[str]:
    """The accessory a background request asks for, as the grader keys it."""
    lowered = (detail or "").lower()
    if "additionally, include" not in lowered and "add " not in lowered:
        return None
    tail = lowered.split("additionally, include", 1)[-1]
    for key, words in ACCESSORY_KEYWORDS:
        if any(w in tail for w in words):
            return key
    return None


def level_of(value: Optional[float], bins: Tuple[float, float, float]) -> Optional[str]:
    """none / light / medium / far for a magnitude, or None when unmeasured."""
    if value is None:
        return None
    if value < bins[0]:
        return "none"
    if value < bins[1]:
        return "light"
    if value < bins[2]:
        return "medium"
    return "far"


# ── face region ──────────────────────────────────────────────────────────


def _face_mask(shape: Tuple[int, int], box: Tuple[float, float, float, float]) -> np.ndarray:
    """An ellipse inscribed in the face box, inset so hair and background at
    the box corners stay out — the stand-in for the benchmark's landmark hull."""
    h, w = shape
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    rx, ry = max(1.0, 0.42 * (x1 - x0)), max(1.0, 0.46 * (y1 - y0))
    ys, xs = np.ogrid[:h, :w]
    return (((xs - cx) / rx) ** 2 + ((ys - cy) / ry) ** 2) <= 1.0


def lighting_stats(image: Image.Image, geometry: Tuple) -> Optional[Dict[str, float]]:
    """Brightness, contrast, colour cast and left/right balance of the face.

    The benchmark's _face_light: L, its standard deviation, mean a and b
    (128-centred) and (left - right) / mean, all in OpenCV's 8-bit Lab.
    """
    import cv2

    box, points = geometry
    rgb = np.asarray(image.convert("RGB"))
    mask = _face_mask(rgb.shape[:2], box)
    if mask.sum() < 200:
        return None
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    L = lab[..., 0].astype(np.float32)
    a = lab[..., 1].astype(np.float32) - 128.0
    b = lab[..., 2].astype(np.float32) - 128.0
    nose_x = float(points[2][0])
    cols = np.arange(mask.shape[1])[None, :]
    left = mask & (cols < nose_x)
    right = mask & (cols >= nose_x)
    mean = float(L[mask].mean())
    l_left = float(L[left].mean()) if left.any() else mean
    l_right = float(L[right].mean()) if right.any() else mean
    return {
        "L": mean,
        "contrast": float(L[mask].std()),
        "a": float(a[mask].mean()),
        "b": float(b[mask].mean()),
        "asym": (l_left - l_right) / max(mean, 1.0),
    }


def lighting_delta(base: Optional[Dict[str, float]], var: Optional[Dict[str, float]]) -> Optional[Dict[str, float]]:
    if not base or not var:
        return None
    return {
        "brightness": var["L"] - base["L"],
        "contrast": var["contrast"] - base["contrast"],
        "asym": abs(var["asym"] - base["asym"]),
        "cast": float(np.hypot(var["a"] - base["a"], var["b"] - base["b"])),
    }


def lighting_magnitude(delta: Optional[Dict[str, float]]) -> Optional[float]:
    """The benchmark's lighting magnitude: the largest normalised term."""
    if not delta:
        return None
    return max(
        abs(delta["brightness"]) / 25.0,
        abs(delta["contrast"]) / 12.0,
        delta["asym"] / 0.15,
        delta["cast"] / 10.0,
    )


def head_crop(image: Image.Image, geometry: Tuple) -> Image.Image:
    """Head plus headwear: the face box grown up, out, and a little down."""
    box, _ = geometry
    x0, y0, x1, y1 = box
    fw, fh = x1 - x0, y1 - y0
    w, h = image.size
    crop = (
        max(0, int(x0 - 0.7 * fw)),
        max(0, int(y0 - 0.9 * fh)),
        min(w, int(x1 + 0.7 * fw)),
        min(h, int(y1 + 0.7 * fh)),
    )
    return image.crop(crop)


# ── expression ───────────────────────────────────────────────────────────


def expression_probs(image: Image.Image, geometry: Tuple) -> Optional[Dict[str, float]]:
    return background_check.zero_shot(head_crop(image, geometry), EXPRESSION_PROMPTS)


def expression_score(probs: Optional[Dict[str, float]]) -> Optional[float]:
    """Expected expression ordinal, 0 (neutral) to 3 (laughing)."""
    if not probs:
        return None
    return float(sum(p * EXPRESSION_ORDINAL[k] for k, p in probs.items()))


# ── accessory ────────────────────────────────────────────────────────────


def accessory_verdict(requested: str, groups: Optional[Dict[str, float]]) -> Optional[Dict[str, Any]]:
    """Is the requested accessory there, something else, or nothing?

    The benchmark's accessory_verdict without its segmentation cues: presence
    is CLIP's "bare head" probability under 0.5, and the worn groups are
    renormalised among themselves to name the top one.
    """
    if not groups:
        return None
    present = groups.get("none", 0.0) < 0.5
    worn = {k: v for k, v in groups.items() if k != "none"}
    total = sum(worn.values()) or 1.0
    worn = {k: v / total for k, v in worn.items()}
    top = max(worn, key=worn.get)
    conf = worn.get(requested, 0.0) if present else groups.get(requested, 0.0)
    if not present:
        status = "missing"
    elif top == requested and conf >= ACCESSORY_CONF:
        status = "match"
    elif top != requested and worn[top] >= ACCESSORY_CONF:
        status = "other"
    else:
        status = "insufficient"
    return {
        "requested": requested,
        "status": status,
        "confidence": round(conf, 3),
        "top": top if present else "none",
        "top_confidence": round(worn[top] if present else groups.get("none", 0.0), 3),
    }


# ── per-request context and per-attempt measurement ──────────────────────


def base_context(base_image: Image.Image) -> Dict[str, Any]:
    """What every attempt of a request is measured against: the base face's
    geometry, lighting and expression. Missing parts switch that measurement
    off for the request."""
    if not ENABLED:
        return {}
    geometry = face_geometry(base_image)
    if geometry is None:
        return {}
    context: Dict[str, Any] = {"geometry": geometry}
    try:
        context["lighting"] = lighting_stats(base_image, geometry)
    except Exception:
        context["lighting"] = None
    context["expression"] = expression_score(expression_probs(base_image, geometry))
    return context


def measure(
    context: Dict[str, Any],
    image: Image.Image,
    levels: Dict[str, str],
    accessory: Optional[str],
) -> Dict[str, Any]:
    """Measure the edits a slot asks for on one attempt.

    Returns a dict with "lighting", "expression" and "accessory" entries (each
    None when not requested or not measurable) and the seconds each took.
    """
    out: Dict[str, Any] = {"lighting": None, "expression": None, "accessory": None, "seconds": {}}
    if not ENABLED or not context:
        return out
    wants_lighting = "lighting_edit" in levels and context.get("lighting")
    wants_expression = "expression_edit" in levels and context.get("expression") is not None
    if not (wants_lighting or wants_expression or accessory):
        return out
    started = time.time()
    geometry = face_geometry(image)
    out["seconds"]["detect"] = round(time.time() - started, 2)
    if geometry is None:
        return out

    if wants_lighting:
        started = time.time()
        try:
            delta = lighting_delta(context["lighting"], lighting_stats(image, geometry))
        except Exception:
            delta = None
        magnitude = lighting_magnitude(delta)
        if magnitude is not None:
            out["lighting"] = {
                "magnitude": round(magnitude, 3),
                "level": level_of(magnitude, LIGHT_BINS),
                "delta": {k: round(v, 2) for k, v in delta.items()},
            }
        out["seconds"]["lighting"] = round(time.time() - started, 2)

    if wants_expression:
        started = time.time()
        probs = expression_probs(image, geometry)
        score = expression_score(probs)
        if score is not None:
            change = score - context["expression"]
            out["expression"] = {
                "score": round(score, 3),
                "change": round(change, 3),
                "level": level_of(max(0.0, change), EXPRESSION_BINS),
                "neutral": round(probs.get("neutral", 0.0), 3),
                "top": max(probs, key=probs.get),
            }
        out["seconds"]["expression"] = round(time.time() - started, 2)

    if accessory:
        started = time.time()
        out["accessory"] = accessory_verdict(
            accessory, background_check.zero_shot(head_crop(image, geometry), ACCESSORY_PROMPTS),
        )
        out["seconds"]["accessory"] = round(time.time() - started, 2)
    return out


# ── judgements ───────────────────────────────────────────────────────────


def misses(levels: Dict[str, str], measured: Optional[Dict[str, Any]]) -> List[Tuple[str, str]]:
    """Gross misses worth a retry: (component, hint) pairs.

    Only what the bins can be trusted for: no lighting change at all, a far
    lighting request that reads light, a light one that reads far, a medium or
    far expression request that still reads neutral, and an accessory that is
    missing or something else. Unmeasured components never miss.
    """
    out: List[Tuple[str, str]] = []
    if not measured:
        return out
    lighting = measured.get("lighting")
    if lighting and "lighting_edit" in levels:
        magnitude = lighting["magnitude"]
        requested = levels["lighting_edit"]
        if magnitude < LIGHT_BINS[0]:
            out.append(("lighting", "more"))
        elif requested == "far" and magnitude < LIGHT_BINS[1]:
            out.append(("lighting", "more"))
        elif requested == "light" and magnitude >= LIGHT_BINS[2]:
            out.append(("lighting", "less"))
    expression = measured.get("expression")
    if expression and levels.get("expression_edit") in ("medium", "far"):
        if expression["neutral"] >= NEUTRAL_PROB:
            out.append(("expression", "more"))
    accessory = measured.get("accessory")
    if accessory and accessory["status"] in ("missing", "other"):
        out.append(("accessory", accessory["status"]))
    return out


def preserved(
    levels: Dict[str, str],
    incumbent: Dict[str, Any],
    candidate: Dict[str, Any],
    scene: Optional[str] = None,
) -> Tuple[bool, str]:
    """Does ``candidate`` keep every measured edit of ``incumbent``?

    Both are attempt results carrying ``pose_yaw``, ``environment`` and
    ``edit`` (this module's measurement). A component the incumbent measured
    but the candidate did not counts as lost: an unmeasured take is not a
    safe replacement. Returns (kept, what was lost).
    """
    inc_edit = incumbent.get("edit") or {}
    cand_edit = candidate.get("edit") or {}

    if "pose_edit" in levels and incumbent.get("pose_yaw") is not None:
        yaw = candidate.get("pose_yaw")
        if yaw is None or yaw < incumbent["pose_yaw"] - POSE_KEEP_DEG:
            return False, "head turn"
    inc = inc_edit.get("lighting")
    if inc:
        cand = cand_edit.get("lighting")
        if not cand or cand["magnitude"] < inc["magnitude"] * LIGHT_KEEP:
            return False, "lighting"
    inc = inc_edit.get("expression")
    if inc:
        cand = cand_edit.get("expression")
        if not cand or cand["score"] < inc["score"] - EXPRESSION_KEEP:
            return False, "expression"
    inc = inc_edit.get("accessory")
    if inc:
        cand = cand_edit.get("accessory")
        if not cand or cand["confidence"] < inc["confidence"] - ACCESSORY_KEEP:
            return False, "accessory"
        if inc["status"] == "match" and cand["status"] != "match":
            return False, "accessory"
    if scene and incumbent.get("environment"):
        cand_env = candidate.get("environment") or {}
        if cand_env.get(scene, 0.0) < incumbent["environment"].get(scene, 0.0) - SCENE_KEEP:
            return False, "scene"
    return True, ""


def describe(measured: Optional[Dict[str, Any]]) -> str:
    """Short log form, e.g. "lighting 1.31 (medium), expression +0.9 (light), accessory baseball_cap 0.91"."""
    if not measured:
        return ""
    parts = []
    lighting = measured.get("lighting")
    if lighting:
        parts.append(f"lighting {lighting['magnitude']:.2f} ({lighting['level']})")
    expression = measured.get("expression")
    if expression:
        parts.append(f"expression {expression['change']:+.1f} ({expression['level']})")
    accessory = measured.get("accessory")
    if accessory:
        parts.append(
            f"accessory {accessory['requested']} {accessory['confidence']:.2f}"
            + (f" ({accessory['status']}: {accessory['top']})" if accessory["status"] != "match" else "")
        )
    return ", ".join(parts)
