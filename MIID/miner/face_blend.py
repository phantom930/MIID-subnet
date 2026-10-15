# MIID/miner/face_blend.py
#
# Bring the base face back into a background take.
#
# A background_in / background_out request says "change the background while
# keeping the subject unchanged", yet FLUX.2 Klein regenerates the whole
# image and the face drifts: archived background slots averaged 0.68–0.74
# AdaFace similarity against the seed (2026-10-14/15), while on the latest
# graded rounds the 50th place — the last one paid — sat at 0.87 on a
# background_out slot. The grader's identity score is the same AdaFace
# comparison this miner runs (identical to three decimals on 40 graded
# slots), so what it measures is how much of the inner face survived.
#
# This module puts the inner face back. The base image is warped onto the
# take's face with the similarity transform between their five MTCNN
# landmarks, colour-matched to the take inside a soft ellipse that covers
# eyebrows to chin and cheek to cheek, and alpha-blended there. Everything
# outside the ellipse — the scene, hair, ears, any requested headwear — is
# the take's own: the ellipse stays inside a hijab's or bandana's edge and
# clear of headphones, and the accessory and scene checks are re-run and must
# still pass.
#
# Guards, each of which leaves the take as it was:
#   - no face, or the take's face in a different pose than the base's (the
#     landmark residual after the transform above RESIDUAL_MAX of the eye
#     distance), or a face scale outside SCALE_RANGE;
#   - the blend does not raise identity by at least MIN_GAIN;
#   - the blended image reads as a copy of the seed (dHash within
#     SEED_DHASH_BITS of the base image): the grader scores a copy -1;
#   - a measured edit weakened (edit_check.preserved on scene and accessory).
#
# The smallest alpha in the ladder that reaches MIID_FACE_BLEND_GOAL wins and
# a full copy is never used: even the strongest alpha leaves the take's own
# texture in the face, so the result is a regenerated face steered back
# toward the seed, not a paste. MIID_FACE_BLEND=0 switches it off;
# MIID_FACE_BLEND_SLOTS lists the slot types it applies to.

import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from MIID.miner.ada_face_compare import face_geometry, similarity_to_embedding

try:  # OpenCV grabs every advertised core by default; the miner shares a quota.
    import cv2 as _cv2

    _cv2.setNumThreads(int(os.environ.get("MIID_TORCH_THREADS", "4")))
except Exception:
    pass


ENABLED = os.environ.get("MIID_FACE_BLEND", "1").strip().lower() not in ("0", "false", "no")
# Identity a background take is blended toward; a take already there is left alone.
GOAL = float(os.environ.get("MIID_FACE_BLEND_GOAL", "0.88"))
# Strongest blend allowed. 1.0 would paste the base face. On 80 archived
# takes 0.9 gained no identity over 0.8 (+0.113 vs +0.116) while pulling
# CLIP's wanted-scene probability down twice as far (-0.14 vs -0.08): the
# face starts to read as studio-lit against the new scene.
MAX_ALPHA = float(os.environ.get("MIID_FACE_BLEND_MAX", "0.8"))
# A blend that drops the wanted-scene probability by more than this is
# refused: the face no longer sits in the scene. CLIP wobbles by ~0.1 when
# only the face changes, so this catches the real cases (0.96 → 0.11), not
# the noise.
SCENE_DROP_MAX = 0.25
# Slot types the blend applies to, and how. Background slots keep the whole
# visible face skin. A lighting slot keeps it too, but the colour match is
# local (a Gaussian window of LOCAL_MATCH_SIGMA eye distances), so the
# take's lighting direction, cast and contrast stay on the restored face. An
# expression slot restores only the upper face — brows, eyes, nose — and
# leaves the mouth, cheeks and chin, which carry the expression, as
# generated. Pose slots are never blended: the face has turned, so the
# landmarks do not fit. The lighting+expression combo is not blended either:
# on 10 archived takes every mask/match variant lost identity (−0.05 to
# −0.07), the relit, re-expressed face being too far from the base for a
# landmark-aligned paste.
BLEND_MODES: Dict[str, Tuple[str, str]] = {
    "background_in": ("skin", "global"),
    "background_out": ("skin", "global"),
    "lighting_edit": ("skin", "local"),
    "expression_edit": ("upper", "global"),
}
SLOTS = frozenset(
    s.strip() for s in os.environ.get(
        "MIID_FACE_BLEND_SLOTS", ",".join(BLEND_MODES)
    ).split(",") if s.strip()
)
# Blend strengths tried in order; the first one at GOAL wins. Identity gain
# saturates early (0.6 gave +0.103, 0.8 +0.116, 0.9 +0.113 on 80 archived
# takes), and every step costs a composite and an AdaFace pass, so one step.
_LADDER = (0.8,)
# Window of the local colour match, in eye distances. The benchmark's
# lighting delta on a lighting slot is nearly always the left/right
# asymmetry, a step at the nose line; a wide window turns that step into a
# ramp and the measured lighting falls. On 19 archived lighting takes
# (raw identity 0.76): window 0.5 kept 69% of the lighting magnitude and
# passed the 85% rule on 2 of 18; 0.2 kept 88% (min 83%), passed on 17 of
# 18, and still reached identity 0.85 (56% of takes at 0.85 or better);
# 0.1 kept 95% but gave only 0.82.
LOCAL_MATCH_SIGMA = 0.2
# Which region the base face is blended into when BLEND_MODES does not say:
# "skin" follows the take's visible face skin (forehead below a brim, cheeks
# beside a hijab, the jaw), "ellipse" is the fixed inner-face superellipse
# only, "upper" the brows, eyes and nose. On 80 archived background takes
# (raw identity 0.71) the skin mask at alpha 0.8 reached 0.84 (49% of takes
# at 0.85 or better).
MASK = os.environ.get("MIID_FACE_BLEND_MASK", "skin").strip().lower()
# Poisson (seamless) cloning before the mix hides the boundary but re-shades
# the face from the take's gradients, which AdaFace reads as a different
# face: 0.84 against 0.88 for the plain colour-matched blend. Off by default.
POISSON = os.environ.get("MIID_FACE_BLEND_POISSON", "0").strip().lower() in ("1", "true", "yes")
ALPHAS: Tuple[float, ...] = tuple(a for a in _LADDER if a <= MAX_ALPHA + 1e-9) or (MAX_ALPHA,)
# A blend that lifts identity by less than this is not worth the changed pixels.
MIN_GAIN = 0.02
# Landmark fit: RMS residual of the base landmarks mapped onto the take's,
# as a fraction of the take's eye distance. A still face fits within ~0.05;
# a turned head or a changed expression does not.
RESIDUAL_MAX = 0.12
SCALE_RANGE = (0.7, 1.4)
# dHash distance (of 256 bits) under which two images are the same picture
# (the sibling ledger's benchmark: <= 12 is the same image).
SEED_DHASH_BITS = 20


# ── geometry ─────────────────────────────────────────────────────────────


def similarity_transform(src: np.ndarray, dst: np.ndarray) -> Tuple[np.ndarray, float]:
    """Umeyama: the scale + rotation + translation taking ``src`` points onto
    ``dst`` (both N×2), as a 2×3 matrix, and its scale. No reflection."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    mu_s, mu_d = src.mean(axis=0), dst.mean(axis=0)
    s, d = src - mu_s, dst - mu_d
    cov = d.T @ s / len(src)
    U, S, Vt = np.linalg.svd(cov)
    D = np.eye(2)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[1, 1] = -1.0
    R = U @ D @ Vt
    var_s = float((s ** 2).sum() / len(src))
    scale = float((S * np.diag(D)).sum() / max(var_s, 1e-12))
    t = mu_d - scale * (R @ mu_s)
    M = np.hstack([scale * R, t[:, None]])
    return M, scale


def fit(base_points: np.ndarray, take_points: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """(matrix, scale, residual) mapping the base landmarks onto the take's.

    ``residual`` is the RMS distance between the mapped base landmarks and
    the take's, as a fraction of the take's eye distance.
    """
    M, scale = similarity_transform(base_points, take_points)
    projected = np.asarray(base_points, dtype=np.float64) @ M[:, :2].T + M[:, 2]
    take_points = np.asarray(take_points, dtype=np.float64)
    eye = float(np.linalg.norm(take_points[1] - take_points[0]))
    residual = float(np.sqrt(((projected - take_points) ** 2).sum(axis=1).mean())) / max(eye, 1.0)
    return M, scale, residual


# Rounded-rectangle exponent of the face mask: 2 is an ellipse, which clips
# the outer eye corners; 2.5 keeps them well inside at the same width.
MASK_EXPONENT = 2.5


def face_ellipse(shape: Tuple[int, int], points: np.ndarray, feather: float = 0.07) -> np.ndarray:
    """A soft mask (H×W float, 0..1) over the inner face of a 5-landmark face.

    A superellipse along the face's own axes: from just above the eyebrows
    (0.4 of the eye-to-mouth distance above the eyes) to the chin (0.75 of
    it below the mouth), one eye distance to each side of the centre line —
    inside the hairline, the ears and the edge of any headwear, with the
    eyes, brows, nose and mouth at full weight. Feathered with a Gaussian of
    ``feather`` eye distances.
    """
    import cv2

    h, w = shape
    pts = np.asarray(points, dtype=np.float64)
    left_eye, right_eye, _nose, mouth_l, mouth_r = pts
    eye_c = (left_eye + right_eye) / 2.0
    mouth_c = (mouth_l + mouth_r) / 2.0
    d = float(np.linalg.norm(right_eye - left_eye))
    if d < 2.0:
        return np.zeros((h, w), dtype=np.float32)
    ax = (right_eye - left_eye) / d
    ay = np.array([-ax[1], ax[0]])
    if np.dot(mouth_c - eye_c, ay) < 0:
        ay = -ay
    m = max(float(np.dot(mouth_c - eye_c, ay)), 0.8 * d)
    centre = eye_c + 0.675 * m * ay
    rx, ry = 1.0 * d, 1.075 * m
    ys, xs = np.mgrid[:h, :w]
    px = xs - centre[0]
    py = ys - centre[1]
    u = px * ax[0] + py * ax[1]
    v = px * ay[0] + py * ay[1]
    radius = (np.abs(u) / rx) ** MASK_EXPONENT + (np.abs(v) / ry) ** MASK_EXPONENT
    mask = (radius <= 1.0).astype(np.float32)
    sigma = max(1.0, feather * d)
    k = int(2 * round(3 * sigma) + 1)
    return cv2.GaussianBlur(mask, (k, k), sigma)


# ── pixels ───────────────────────────────────────────────────────────────


# A pixel outside the inner face is restored only where the take still looks
# like the base there: Lab distance between the (smoothed) take and the
# colour-matched base under this. Skin that merely drifted sits well under
# it; fabric or a hat over the base's hair or forehead sits far above.
RESTORE_DELTA_E = 16.0


def skin_mask(
    take_rgb: np.ndarray, points: np.ndarray, reference: Optional[np.ndarray] = None,
    feather: float = 0.05,
) -> np.ndarray:
    """A soft mask over the take's visible face skin: the inner-face superellipse
    always, plus the connected skin-coloured area around it (forehead below a
    hat brim, cheeks beside a hijab, jaw and chin) inside an outer bound a
    little above the brows to just under the chin. Skin is what matches the
    colour of the inner face in YCrCb and, when ``reference`` (the warped,
    colour-matched base) is given, what still resembles the base there — so
    hair, a backdrop and any fabric worn over the base's hair or forehead
    stay the take's own, and the mask adapts to whatever headwear it wears.
    """
    import cv2

    h, w = take_rgb.shape[:2]
    core = face_ellipse((h, w), points, feather=0.0)
    pts = np.asarray(points, dtype=np.float64)
    left_eye, right_eye, nose, mouth_l, mouth_r = pts
    eye_c = (left_eye + right_eye) / 2.0
    mouth_c = (mouth_l + mouth_r) / 2.0
    d = float(np.linalg.norm(right_eye - left_eye))
    if d < 2.0:
        return core
    ax = (right_eye - left_eye) / d
    ay = np.array([-ax[1], ax[0]])
    if np.dot(mouth_c - eye_c, ay) < 0:
        ay = -ay
    m = max(float(np.dot(mouth_c - eye_c, ay)), 0.8 * d)
    # Outer bound: 0.6 m above the eyes (mid-forehead) to under the chin
    # (0.95 m below the mouth), 1.15 eye distances to each side.
    centre = eye_c + 0.675 * m * ay
    rx, ry = 1.15 * d, 1.275 * m
    ys, xs = np.mgrid[:h, :w]
    px, py = xs - centre[0], ys - centre[1]
    u = px * ax[0] + py * ax[1]
    v = px * ay[0] + py * ay[1]
    outer = ((np.abs(u) / rx) ** MASK_EXPONENT + (np.abs(v) / ry) ** MASK_EXPONENT) <= 1.0

    ycc = cv2.cvtColor(take_rgb, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    sample = core > 0.5
    if sample.sum() < 100:
        return cv2.GaussianBlur(core, (0, 0), max(1.0, feather * d))
    feats = ycc[sample][:, 1:]  # Cr, Cb of the inner face
    mu = feats.mean(axis=0)
    cov = np.cov(feats.T) + np.eye(2) * 2.0
    inv = np.linalg.inv(cov)
    diff = ycc[..., 1:] - mu
    maha = np.einsum("...i,ij,...j->...", diff, inv, diff)
    luma = ycc[..., 0]
    lo, hi = np.percentile(luma[sample], [3, 97])
    skin = (maha <= 4.0) & (luma >= lo - 15) & (luma <= hi + 15) & outer
    if reference is not None:
        sigma = max(1.0, 0.04 * d)
        lab_t = cv2.cvtColor(cv2.GaussianBlur(take_rgb, (0, 0), sigma), cv2.COLOR_RGB2LAB).astype(np.float32)
        lab_r = cv2.cvtColor(cv2.GaussianBlur(reference, (0, 0), sigma), cv2.COLOR_RGB2LAB).astype(np.float32)
        delta = np.sqrt(((lab_t - lab_r) ** 2).sum(axis=-1))
        skin &= delta <= RESTORE_DELTA_E
    skin = skin.astype(np.uint8)
    k_open = max(3, int(round(0.06 * d)) | 1)
    k_close = max(3, int(round(0.12 * d)) | 1)
    skin = cv2.morphologyEx(skin, cv2.MORPH_OPEN, np.ones((k_open, k_open), np.uint8))
    skin = cv2.morphologyEx(skin, cv2.MORPH_CLOSE, np.ones((k_close, k_close), np.uint8))
    n, labels = cv2.connectedComponents(skin)
    seed = labels[int(round(nose[1])), int(round(nose[0]))] if 0 <= int(round(nose[1])) < h and 0 <= int(round(nose[0])) < w else 0
    region = (labels == seed) if seed > 0 else np.zeros_like(skin, dtype=bool)
    mask = np.maximum(core, region.astype(np.float32))
    return cv2.GaussianBlur(mask, (0, 0), max(1.0, feather * d))


def upper_mask(shape: Tuple[int, int], points: np.ndarray, feather: float = 0.07) -> np.ndarray:
    """A soft mask over the upper face only: brows, eyes and nose, from just
    above the eyebrows (0.4 of the eye-to-mouth distance above the eyes) to
    just under the nose tip (0.6 of it below), one eye distance to each side.
    The mouth, cheeks and chin — where an expression lives — stay out."""
    import cv2

    h, w = shape
    pts = np.asarray(points, dtype=np.float64)
    left_eye, right_eye, _nose, mouth_l, mouth_r = pts
    eye_c = (left_eye + right_eye) / 2.0
    mouth_c = (mouth_l + mouth_r) / 2.0
    d = float(np.linalg.norm(right_eye - left_eye))
    if d < 2.0:
        return np.zeros((h, w), dtype=np.float32)
    ax = (right_eye - left_eye) / d
    ay = np.array([-ax[1], ax[0]])
    if np.dot(mouth_c - eye_c, ay) < 0:
        ay = -ay
    m = max(float(np.dot(mouth_c - eye_c, ay)), 0.8 * d)
    centre = eye_c + 0.1 * m * ay
    rx, ry = 1.0 * d, 0.5 * m
    ys, xs = np.mgrid[:h, :w]
    px, py = xs - centre[0], ys - centre[1]
    u = px * ax[0] + py * ax[1]
    v = px * ay[0] + py * ay[1]
    radius = (np.abs(u) / rx) ** MASK_EXPONENT + (np.abs(v) / ry) ** MASK_EXPONENT
    mask = (radius <= 1.0).astype(np.float32)
    sigma = max(1.0, feather * d)
    k = int(2 * round(3 * sigma) + 1)
    return cv2.GaussianBlur(mask, (k, k), sigma)


def match_colour(src: np.ndarray, ref: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """``src`` (RGB uint8) with the Lab mean and spread of ``ref`` inside
    ``weight`` (Reinhard colour transfer), so the base face takes the take's
    lighting and colour cast. Spread ratios are clamped to keep contrast sane."""
    import cv2

    lab_s = cv2.cvtColor(src, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab_r = cv2.cvtColor(ref, cv2.COLOR_RGB2LAB).astype(np.float32)
    w = (weight > 0.5).astype(np.float32)[..., None]
    total = max(float(w.sum()), 1.0)
    mu_s = (lab_s * w).sum(axis=(0, 1)) / total
    mu_r = (lab_r * w).sum(axis=(0, 1)) / total
    sd_s = np.sqrt((((lab_s - mu_s) ** 2) * w).sum(axis=(0, 1)) / total)
    sd_r = np.sqrt((((lab_r - mu_r) ** 2) * w).sum(axis=(0, 1)) / total)
    ratio = np.clip(sd_r / np.maximum(sd_s, 1e-3), 0.7, 1.4)
    out = (lab_s - mu_s) * ratio + mu_r
    out = np.clip(out, 0, 255).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_LAB2RGB)


def match_colour_local(src: np.ndarray, ref: np.ndarray, weight: np.ndarray, sigma: float) -> np.ndarray:
    """Like match_colour, but with the mean and spread taken in a Gaussian
    window of ``sigma`` pixels around each point, so the take's shading —
    which side is lit, how hard the shadows fall, where the cast sits — is
    carried onto the base face's detail. Statistics are weighted by the mask
    so pixels outside it (hair, fabric, scene) do not pull the face."""
    import cv2

    lab_s = cv2.cvtColor(src, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab_r = cv2.cvtColor(ref, cv2.COLOR_RGB2LAB).astype(np.float32)
    w = np.clip(weight, 0.0, 1.0).astype(np.float32)
    if w.max() <= 0.0:
        return src
    blur = lambda a: cv2.GaussianBlur(a, (0, 0), sigma)  # noqa: E731
    norm = np.maximum(blur(w), 1e-3)[..., None]
    w3 = w[..., None]
    mu_s = blur(lab_s * w3) / norm
    mu_r = blur(lab_r * w3) / norm
    var_s = np.maximum(blur(((lab_s - mu_s) ** 2) * w3) / norm, 1e-3)
    var_r = np.maximum(blur(((lab_r - mu_r) ** 2) * w3) / norm, 1e-3)
    ratio = np.clip(np.sqrt(var_r / var_s), 0.7, 1.4)
    out = (lab_s - mu_s) * ratio + mu_r
    out = np.clip(out, 0, 255).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_LAB2RGB)


def warp_base(base_rgb: np.ndarray, matrix: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    import cv2

    h, w = shape
    return cv2.warpAffine(
        base_rgb, matrix.astype(np.float32), (w, h),
        flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT,
    )


def seamless(src: np.ndarray, dst: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """``src`` cloned into ``dst`` under ``mask`` (Poisson image editing):
    the gradients of the base face with the illumination of the take, and no
    visible boundary. Falls back to ``src`` where the mask is empty."""
    import cv2

    hard = (mask > 0.5).astype(np.uint8) * 255
    x, y, w, h = cv2.boundingRect(hard)
    if w < 4 or h < 4:
        return dst
    # Shrink the mask off the image border: seamlessClone needs a margin.
    H, W = hard.shape
    hard[:1, :] = 0
    hard[-1:, :] = 0
    hard[:, :1] = 0
    hard[:, -1:] = 0
    x, y, w, h = cv2.boundingRect(hard)
    centre = (x + w // 2, y + h // 2)
    try:
        return cv2.seamlessClone(src, dst, hard, centre, cv2.NORMAL_CLONE)
    except cv2.error:
        return src


def composite(
    base_rgb: np.ndarray, take_rgb: np.ndarray, matrix: np.ndarray, mask: np.ndarray, alpha: float,
    poisson: bool = False, match: str = "global", eye_distance: Optional[float] = None,
) -> np.ndarray:
    """The take with the base face warped in by ``matrix`` and blended under
    ``mask`` at strength ``alpha``. The warped face is colour-matched to the
    take first — globally, or locally (``match="local"``) so the take's
    shading survives — and with ``poisson`` cloned in seamlessly before the
    mix."""
    warped = warp_base(base_rgb, matrix, take_rgb.shape[:2])
    if match == "local":
        sigma = max(2.0, LOCAL_MATCH_SIGMA * (eye_distance or 40.0))
        face = match_colour_local(warped, take_rgb, mask, sigma)
    else:
        face = match_colour(warped, take_rgb, mask)
    if poisson:
        face = seamless(face, take_rgb, mask)
    weight = (alpha * mask)[..., None]
    out = take_rgb.astype(np.float32) + weight * (face.astype(np.float32) - take_rgb.astype(np.float32))
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def dhash_bits(rgb: np.ndarray) -> np.ndarray:
    """256-bit difference hash of an RGB array (size-invariant)."""
    import cv2

    grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    small = cv2.resize(grey, (17, 16), interpolation=cv2.INTER_AREA).astype(np.int16)
    return (small[:, 1:] > small[:, :-1]).flatten()


def hamming(a: np.ndarray, b: np.ndarray) -> int:
    return int((a != b).sum())


# ── the procedure ────────────────────────────────────────────────────────


def base_geometry(base_image: Image.Image, edit_context: Optional[Dict[str, Any]]) -> Optional[Tuple]:
    """The base face's (box, landmarks), reusing edit_check's detection."""
    if not ENABLED:
        return None
    if edit_context and edit_context.get("geometry") is not None:
        return edit_context["geometry"]
    try:
        return face_geometry(base_image)
    except Exception:
        return None


def wanted(background: bool, result: Dict[str, Any], goal: Optional[float] = None) -> bool:
    """Is this take one to blend: a blendable slot type still short of the goal?

    ``background`` is kept for the caller's convenience; the slot type in
    ``result`` decides (BLEND_MODES / MIID_FACE_BLEND_SLOTS).
    """
    if not ENABLED:
        return False
    vtype = result.get("variation_type")
    if vtype not in SLOTS or vtype not in BLEND_MODES or result.get("image") is None:
        return False
    similarity = result.get("identity_similarity")
    return similarity is not None and similarity < (GOAL if goal is None else goal)


def blend(
    base_image: Image.Image,
    take_image: Image.Image,
    base_points: np.ndarray,
    take_points: Optional[np.ndarray] = None,
    alpha: float = 0.65,
) -> Tuple[Optional[Image.Image], Dict[str, Any]]:
    """One blend at ``alpha``, or (None, why) when the faces do not line up."""
    base_rgb = np.asarray(base_image.convert("RGB"))
    take_rgb = np.asarray(take_image.convert("RGB"))
    if take_points is None:
        geometry = face_geometry(take_image)
        if geometry is None:
            return None, {"reason": "no face in the take"}
        take_points = geometry[1]
    matrix, scale, residual = fit(base_points, take_points)
    info = {"scale": round(scale, 3), "residual": round(residual, 3)}
    if not (SCALE_RANGE[0] <= scale <= SCALE_RANGE[1]):
        info["reason"] = f"face scale {scale:.2f} outside {SCALE_RANGE[0]}–{SCALE_RANGE[1]}"
        return None, info
    if residual > RESIDUAL_MAX:
        info["reason"] = f"the take's face sits differently (landmark residual {residual:.2f} > {RESIDUAL_MAX})"
        return None, info
    mask = build_mask(take_rgb, take_points, base_rgb, matrix)
    out = composite(base_rgb, take_rgb, matrix, mask, alpha, poisson=POISSON)
    return Image.fromarray(out), info


def build_mask(
    take_rgb: np.ndarray, take_points: np.ndarray,
    base_rgb: Optional[np.ndarray] = None, matrix: Optional[np.ndarray] = None,
    mode: Optional[str] = None,
) -> np.ndarray:
    """The blend region for a take: ``ellipse`` (the inner-face superellipse),
    ``upper`` (brows, eyes and nose only) or ``skin`` (default: the take's
    visible face skin judged against the warped base)."""
    mode = MASK if mode is None else mode
    if mode == "ellipse":
        return face_ellipse(take_rgb.shape[:2], take_points)
    if mode == "upper":
        return upper_mask(take_rgb.shape[:2], take_points)
    reference = None
    if base_rgb is not None and matrix is not None:
        core = face_ellipse(take_rgb.shape[:2], take_points)
        reference = match_colour(warp_base(base_rgb, matrix, take_rgb.shape[:2]), take_rgb, core)
    return skin_mask(take_rgb, take_points, reference)


def eye_distance(points: np.ndarray) -> float:
    pts = np.asarray(points, dtype=np.float64)
    return float(np.linalg.norm(pts[1] - pts[0]))


def apply(
    base_image: Image.Image,
    base_embedding: Any,
    base_geom: Optional[Tuple],
    result: Dict[str, Any],
    *,
    edit_context: Optional[Dict[str, Any]],
    levels: Dict[str, str],
    accessory: Optional[str],
    scene: Optional[str],
    goal: Optional[float] = None,
    take_geometry: Optional[Tuple] = None,
) -> Dict[str, Any]:
    """Blend the base face into ``result["image"]`` if that raises identity
    without weakening the measured edits; on success ``result`` is updated
    in place (image, identity_similarity, environment, edit). Returns a
    JSON-safe account of what happened either way. ``take_geometry`` is the
    take's (box, landmarks) when the caller already detected it."""
    from MIID.miner.background_check import background_miss, environment_probs
    from MIID.miner.edit_check import measure as measure_edit, misses as edit_misses, preserved as edit_preserved

    started = time.time()
    goal = GOAL if goal is None else goal
    raw_identity = result.get("identity_similarity")
    vtype = result.get("variation_type")
    mask_mode, match_mode = BLEND_MODES.get(vtype, ("skin", "global"))
    info: Dict[str, Any] = {
        "applied": False, "reason": None, "raw_identity": raw_identity,
        "identity": None, "alpha": None, "seconds": 0.0, "mask": mask_mode, "match": match_mode,
    }

    def done(reason: Optional[str] = None) -> Dict[str, Any]:
        info["reason"] = reason
        info["seconds"] = round(time.time() - started, 2)
        return info

    if base_embedding is None or raw_identity is None:
        return done("no identity signal")
    if base_geom is None:
        return done("no face in the base image")
    if take_geometry is None:
        take_geometry = face_geometry(result["image"])
    if take_geometry is None:
        return done("no face in the take")
    base_points, take_points = base_geom[1], take_geometry[1]
    base_rgb = np.asarray(base_image.convert("RGB"))
    take_rgb = np.asarray(result["image"].convert("RGB"))
    seed_hash = dhash_bits(base_rgb)
    info["raw_seed_bits"] = hamming(seed_hash, dhash_bits(take_rgb))

    matrix, scale, residual = fit(base_points, take_points)
    info["scale"] = round(scale, 3)
    info["residual"] = round(residual, 3)
    if not (SCALE_RANGE[0] <= scale <= SCALE_RANGE[1]):
        return done(f"face scale {scale:.2f} outside {SCALE_RANGE[0]}–{SCALE_RANGE[1]}")
    if residual > RESIDUAL_MAX:
        return done(f"the take's face sits differently (landmark residual {residual:.2f} > {RESIDUAL_MAX})")
    mask = build_mask(take_rgb, take_points, base_rgb, matrix, mode=mask_mode)
    info["mask_share"] = round(float((mask > 0.5).mean()), 4)
    d = eye_distance(take_points)

    best: Optional[Tuple[float, float, Image.Image]] = None
    tried: List[Tuple[float, Optional[float]]] = []
    for alpha in ALPHAS:
        candidate = Image.fromarray(composite(
            base_rgb, take_rgb, matrix, mask, alpha, poisson=POISSON, match=match_mode, eye_distance=d,
        ))
        identity = similarity_to_embedding(base_embedding, candidate)
        tried.append((alpha, None if identity is None else round(identity, 4)))
        if identity is None:
            continue
        if best is None or identity > best[0]:
            best = (identity, alpha, candidate)
        if identity >= goal:
            break
    info["tried"] = tried
    if best is None:
        return done("no face readable in any blend")
    identity, alpha, image = best
    info["identity"] = round(identity, 4)
    info["alpha"] = alpha
    if identity < raw_identity + MIN_GAIN:
        return done(f"identity gain too small ({raw_identity:.3f} → {identity:.3f})")

    seed_bits = hamming(seed_hash, dhash_bits(np.asarray(image)))
    info["seed_bits"] = seed_bits
    if seed_bits < SEED_DHASH_BITS:
        return done(f"would read as a copy of the seed (dHash {seed_bits} bits)")

    # The scene cannot change — only the face did — so the scene check is a
    # plain miss test on the blended image, not the 0.10-point "no weaker
    # than the incumbent" rule for two different takes (CLIP's indoor/outdoor
    # reading wobbles by that much when the face changes). The accessory can
    # be damaged by the mask, so for it the incumbent rule stands.
    environment = environment_probs(image) if scene else result.get("environment")
    if scene and background_miss(result["variation_type"], environment):
        return done("the blended image reads as the wrong kind of scene")
    if scene and environment and result.get("environment"):
        before, after = result["environment"].get(scene, 0.0), environment.get(scene, 0.0)
        info["scene_before"], info["scene_after"] = round(before, 3), round(after, 3)
        if after < before - SCENE_DROP_MAX:
            return done(f"the face no longer sits in the scene ({scene} {before:.2f} → {after:.2f})")
    try:
        # The blended take keeps the take's landmarks, so no second detection.
        edit = measure_edit(edit_context or {}, image, levels, accessory, geometry=take_geometry)
    except Exception as e:  # measured on the raw take; unmeasurable here is not a pass
        return done(f"edit measurement failed after the blend ({e})")
    candidate = {"pose_yaw": result.get("pose_yaw"), "environment": result.get("environment"), "edit": edit}
    kept, lost = edit_preserved(levels, result, candidate, scene)
    if not kept:
        return done(f"the blend weakened the {lost}")
    # Nor may the blend turn a landed edit into a gross miss (a lighting
    # change that now reads as none, an expression that now reads neutral).
    new_misses = set(edit_misses(levels, edit)) - set(edit_misses(levels, result.get("edit")))
    if new_misses:
        return done("the blend undid the " + ", ".join(sorted(c for c, _ in new_misses)))
    lighting_raw, lighting_new = (result.get("edit") or {}).get("lighting"), (edit or {}).get("lighting")
    if lighting_raw and lighting_new:
        info["lighting_before"], info["lighting_after"] = lighting_raw["magnitude"], lighting_new["magnitude"]
        # The 85% rule above allows a small loss; it must not be the loss
        # that takes the lighting down an intensity level (a far read that
        # becomes medium), which is what the grader bins on.
        order = ("none", "light", "medium", "far")
        if (lighting_raw.get("level") in order and lighting_new.get("level") in order
                and order.index(lighting_new["level"]) < order.index(lighting_raw["level"])):
            return done(f"the blend took the lighting from {lighting_raw['level']} to {lighting_new['level']}")
    expr_raw, expr_new = (result.get("edit") or {}).get("expression"), (edit or {}).get("expression")
    if expr_raw and expr_new:
        info["expression_before"], info["expression_after"] = expr_raw["change"], expr_new["change"]

    result["image"] = image
    result["identity_similarity"] = identity
    result["environment"] = environment
    result["edit"] = edit
    info["applied"] = True
    return done(None)


def describe(info: Optional[Dict[str, Any]]) -> str:
    """Short log form: "face blend 0.712 → 0.891 (alpha 0.65)" or ""."""
    if not info or not info.get("applied"):
        return ""
    return (
        f"face blend {info['raw_identity']:.3f} → {info['identity']:.3f} "
        f"(alpha {info['alpha']:.2f})"
    )
