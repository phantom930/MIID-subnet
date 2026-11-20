# MIID/miner/face_mesh.py
#
# The face mesh blend: bring the base face's structure back into a take
# whose head turned or whose expression changed.
#
# face_blend.py lines the base face up with a take through one similarity
# transform on five MTCNN landmarks. That fits a background or lighting take
# — same pose, same expression — but not the combo slots the validators now
# send (lighting+expression, lighting+pose, pose+expression): a turned head
# foreshortens one side of the face and a smile moves the mouth and cheeks,
# so a rigidly aligned base face lands beside the take's features and
# AdaFace reads the ghost as a different face (−0.05 to −0.07 on 10 archived
# lighting+expression takes). Those slots were therefore never blended, and
# they were the lowest-identity half of every request: 0.66–0.72 on average
# against 0.80–0.83 for the blended slots (103 requests, 2026-11-19/20).
#
# Here the base face is warped onto the take with MediaPipe's 468-point
# face mesh instead: one affine map per mesh triangle, so every feature of
# the base lands on the same feature of the take however the head turned or
# the face moved. Triangles that fold over or shrink in the take — the far
# side of a turned face — are left out, and so are:
#   - the eyes themselves (lids, iris, sclera): a turned head still looks at
#     the camera, so the base's iris would land on the take's sclera;
#   - on an expression slot, the mouth: the base's closed lips would erase
#     the take's smile or laugh;
#   - the forehead above a band over the brows, the hairline, and anything
#     in the base that is not skin (hair, background at the face's edge).
# The warped face is colour-matched to the take locally (as on lighting
# takes in face_blend) so the take's light, shadows and cast stay, and only
# its structure between DETAIL_SIGMA and the local window is mixed in: the
# take keeps its own skin texture, lighting and expression.
#
# The guards are face_blend.apply's: identity must rise, the result must not
# read as a copy of the seed or carry its skin texture, and every measured
# edit (head turn, lighting, expression) must survive.
#
# Needs the mediapipe package (scripts/miner/setup.sh installs it). Without
# it available() is False and the combo slots are generated as before.

import os
import threading
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from PIL import Image

# The face landmarker task file, shared with the benchmark's model cache
# (MIID/benchmark/analyzers.py) so neither downloads it twice.
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)
MODELS_DIR = Path(
    os.environ.get(
        "MIID_BENCH_MODELS",
        Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "miid_benchmark",
    )
)

# Mesh indices (MediaPipe face mesh topology).
LEFT_EYE_CORNER, RIGHT_EYE_CORNER = 33, 263
MOUTH_LEFT, MOUTH_RIGHT = 61, 291
LIPS_OUTER = (61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185)
LEFT_EYE = (33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246)
RIGHT_EYE = (263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466)
# Iris centre and its four rim points (the landmarker's refined iris model).
IRISES = ((468, (469, 470, 471, 472)), (473, (474, 475, 476, 477)))
BROWS = (
    46, 53, 52, 65, 55, 70, 63, 105, 66, 107,
    276, 283, 282, 295, 285, 300, 293, 334, 296, 336,
)
# Cheek, forehead, cheek, chin: the quadrilateral that is surely face.
INNER = (234, 10, 454, 152)

# A triangle is used only if its area in the take, relative to the base and
# to the face's overall scale, is at least this: below it the triangle has
# folded over or turned away (the far side of a turned head).
MIN_AREA_RATIO = 0.35
# The mask, in eye distances: eroded from the mesh's outline, feathered,
# holes (eyes, mouth) grown and feathered, and the forehead faded out from
# TOP_CUT above the brows over TOP_RAMP.
ERODE = 0.08
FEATHER = 0.10
# What of the eyes stays the take's own: "iris" (a disc IRIS_GROW times the
# iris radius — the lids, lashes and the eye's shape are blended) or "eye"
# (the whole eye opening, grown by EYE_GROW).
EYES = os.environ.get("MIID_FACE_MESH_EYES", "iris").strip().lower()
IRIS_GROW = 1.3
EYE_GROW = 0.06
MOUTH_GROW = 0.12
HOLE_FEATHER = 0.03
TOP_CUT = 0.20
TOP_RAMP = 0.25
# Window of the local colour match, in eye distances (face_blend.LOCAL_MATCH_SIGMA).
LOCAL_SIGMA = 0.2
# Detail finer than this Gaussian sigma (pixels) stays the take's own. Finer
# than face_blend's 2 px: the mesh lines the features up to within a pixel
# or two, so the base's lids, lashes and brow line can come along without
# ghosting, and its skin texture still cannot (see texture_corr).
DETAIL_SIGMA = float(os.environ.get("MIID_FACE_MESH_DETAIL_SIGMA", "1.0"))
# ...but only around the eyes and brows (FEATURES_GROW / FEATURES_FEATHER eye
# distances round them). Elsewhere — cheeks, nose, chin — detail finer than
# SKIN_DETAIL_SIGMA stays the take's own, so its pores, freckles and fine
# shading survive: at 1 px everywhere the face read as airbrushed.
SKIN_DETAIL_SIGMA = float(os.environ.get("MIID_FACE_MESH_SKIN_SIGMA", "2.5"))
FEATURES_GROW = 0.18
FEATURES_FEATHER = 0.08
# Soft limit (tanh, in 8-bit levels) on how far a pixel moves toward the
# base. A changed eyelid or brow line moves pixels by tens of levels; a
# glasses frame, a hard shadow edge or a catchlight the base does not share
# would move them by a hundred or more, and the full move left ghosted
# frames and washed-out shadows. 0 disables the limit.
DIFF_CLIP = float(os.environ.get("MIID_FACE_MESH_CLIP", "40"))
# Blend strengths, strongest first; a weaker one is tried only when the
# stronger one cost a measured edit (face_blend._apply_mesh).
ALPHAS = tuple(
    float(a) for a in os.environ.get("MIID_FACE_MESH_ALPHAS", "0.8,0.5").split(",") if a.strip()
) or (0.8,)

_lock = threading.Lock()
_landmarker = None
_unavailable: Optional[str] = None


def _model_path() -> Path:
    path = MODELS_DIR / "face_landmarker.task"
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(MODEL_URL, tmp)
        tmp.replace(path)
    return path


def _get_landmarker():
    """The MediaPipe face landmarker, loaded once; None when unavailable."""
    global _landmarker, _unavailable
    if _landmarker is not None or _unavailable is not None:
        return _landmarker
    try:
        from mediapipe.tasks.python import BaseOptions, vision

        _landmarker = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(_model_path())),
                num_faces=2,
            )
        )
        # Its close() needs a thread pool that is gone by the time atexit
        # handlers or __del__ run, and fails noisily there: close it while
        # threads are still being shut down (these hooks run in reverse, so
        # before concurrent.futures stops its workers).
        import threading

        register = getattr(threading, "_register_atexit", None)
        if register is not None:
            register(_close)
    except Exception as e:  # noqa: BLE001 — the miner works without it
        _unavailable = f"{type(e).__name__}: {e}"
    return _landmarker


def _close() -> None:
    global _landmarker
    if _landmarker is not None:
        try:
            _landmarker.close()
        except Exception:  # noqa: BLE001
            pass
        _landmarker = None


def available() -> bool:
    with _lock:
        return _get_landmarker() is not None


def unavailable_reason() -> Optional[str]:
    return _unavailable


def landmarks(image: Any) -> Optional[np.ndarray]:
    """(478, 2) pixel coordinates of the largest face's mesh, or None."""
    import mediapipe as mp

    rgb = np.ascontiguousarray(np.asarray(image.convert("RGB") if isinstance(image, Image.Image) else image))
    h, w = rgb.shape[:2]
    with _lock:
        landmarker = _get_landmarker()
        if landmarker is None:
            return None
        faces = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)).face_landmarks
    if not faces:
        return None

    def area(face) -> float:
        xs = [p.x for p in face]
        ys = [p.y for p in face]
        return (max(xs) - min(xs)) * (max(ys) - min(ys))

    best = max(faces, key=area)
    return np.array([[p.x * w, p.y * h] for p in best], dtype=np.float64)


def eye_distance(points: np.ndarray) -> float:
    return float(np.linalg.norm(points[RIGHT_EYE_CORNER] - points[LEFT_EYE_CORNER]))


# ── warp ─────────────────────────────────────────────────────────────────


def _triangles(base_points: np.ndarray) -> np.ndarray:
    from scipy.spatial import Delaunay

    return Delaunay(base_points[:468]).simplices


def _signed_area(tri: np.ndarray) -> np.ndarray:
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    return 0.5 * ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1]))


def warp(
    base_rgb: np.ndarray, base_points: np.ndarray, take_points: np.ndarray,
    shape: Tuple[int, int], extra: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """The base image warped onto the take's mesh, triangle by triangle.

    Returns (warped RGB, usable — 1 where a triangle that kept its
    orientation and most of its area covers the pixel, else 0 — and
    ``extra``, a float image of the base's size, warped the same way)."""
    import cv2

    h, w = shape
    tri = _triangles(base_points)
    src = base_points[tri]
    dst = take_points[tri]
    area_src, area_dst = _signed_area(src), _signed_area(dst)
    scale = np.median(np.abs(area_dst)) / max(float(np.median(np.abs(area_src))), 1e-6)
    ratio = area_dst / np.where(np.abs(area_src) < 1e-6, 1e-6, area_src) / max(scale, 1e-6)
    usable_tri = ratio > MIN_AREA_RATIO

    index = np.full((h, w), -1, dtype=np.int32)
    for i in range(len(tri)):
        cv2.fillConvexPoly(index, np.round(dst[i]).astype(np.int32), int(i))
    maps = np.zeros((len(tri), 2, 3))
    for i in range(len(tri)):
        maps[i] = cv2.getAffineTransform(dst[i].astype(np.float32), src[i].astype(np.float32))
    ys, xs = np.nonzero(index >= 0)
    t = index[ys, xs]
    m = maps[t]
    map_x = np.full((h, w), -1.0, dtype=np.float32)
    map_y = np.full((h, w), -1.0, dtype=np.float32)
    map_x[ys, xs] = m[:, 0, 0] * xs + m[:, 0, 1] * ys + m[:, 0, 2]
    map_y[ys, xs] = m[:, 1, 0] * xs + m[:, 1, 1] * ys + m[:, 1, 2]
    warped = cv2.remap(base_rgb, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    usable = np.zeros((h, w), dtype=np.float32)
    usable[ys, xs] = usable_tri[t].astype(np.float32)
    warped_extra = None
    if extra is not None:
        warped_extra = cv2.remap(
            extra.astype(np.float32), map_x, map_y, interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
    return warped, usable, warped_extra


# ── masks ────────────────────────────────────────────────────────────────


def _poly(shape: Tuple[int, int], points: np.ndarray, hull: bool = False) -> np.ndarray:
    import cv2

    mask = np.zeros(shape, dtype=np.uint8)
    pts = np.round(points).astype(np.int32)
    if hull:
        pts = cv2.convexHull(pts)
    cv2.fillPoly(mask, [pts.reshape(-1, 2)], 1)
    return mask.astype(np.float32)


def base_skin(base_rgb: np.ndarray, base_points: np.ndarray) -> np.ndarray:
    """1 over the base face's skin inside its mesh — chroma close to the inner
    face's and not as dark as hair — and over the whole inner face (brows,
    eyes and lips are not skin-coloured but belong to the face)."""
    import cv2

    h, w = base_rgb.shape[:2]
    hull = _poly((h, w), base_points[:468], hull=True) > 0
    inner = _poly((h, w), base_points[list(INNER)], hull=True) > 0
    if inner.sum() < 100:
        return hull.astype(np.float32)
    ycc = cv2.cvtColor(base_rgb, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    feats = ycc[inner][:, 1:]
    mu = feats.mean(axis=0)
    inv = np.linalg.inv(np.cov(feats.T) + np.eye(2) * 2.0)
    diff = ycc[..., 1:] - mu
    maha = np.einsum("...i,ij,...j->...", diff, inv, diff)
    luma = ycc[..., 0]
    skin = hull & (maha <= 6.0) & (luma >= np.percentile(luma[inner], 5) - 20)
    skin = cv2.morphologyEx(skin.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    return np.maximum(skin.astype(np.float32), inner.astype(np.float32))


def blend_mask(
    shape: Tuple[int, int], take_points: np.ndarray, usable: np.ndarray, keep_mouth: bool,
) -> np.ndarray:
    """Where (and how strongly, 0..1) the warped base is mixed into the take."""
    import cv2

    h, w = shape
    d = eye_distance(take_points)
    mask = _poly(shape, take_points[:468], hull=True) * usable
    k = max(3, int(round(ERODE * d)) | 1)
    mask = cv2.erode(mask, np.ones((k, k), np.uint8))
    mask = cv2.GaussianBlur(mask, (0, 0), max(1.0, FEATHER * d))

    holes = np.zeros(shape, dtype=np.float32)
    if EYES == "eye" or len(take_points) < 478:
        k = max(3, int(round(EYE_GROW * d)) | 1)
        for eye in (LEFT_EYE, RIGHT_EYE):
            holes = np.maximum(holes, cv2.dilate(_poly(shape, take_points[list(eye)], hull=True), np.ones((k, k), np.uint8)))
    else:
        for centre, rim in IRISES:
            c = take_points[centre]
            radius = float(np.mean(np.linalg.norm(take_points[list(rim)] - c, axis=1))) * IRIS_GROW
            disc = np.zeros(shape, dtype=np.uint8)
            cv2.circle(disc, (int(round(c[0])), int(round(c[1]))), max(1, int(round(radius))), 1, -1)
            holes = np.maximum(holes, disc.astype(np.float32))
    if not keep_mouth:
        k = max(3, int(round(MOUTH_GROW * d)) | 1)
        holes = np.maximum(holes, cv2.dilate(_poly(shape, take_points[list(LIPS_OUTER)]), np.ones((k, k), np.uint8)))
    mask *= 1.0 - cv2.GaussianBlur(holes, (0, 0), max(1.0, HOLE_FEATHER * d))

    # Fade the forehead out above the brows: the mesh ends at no edge the
    # take shows, and a hairline, a fringe or headwear sit up there.
    eye_c = (take_points[LEFT_EYE_CORNER] + take_points[RIGHT_EYE_CORNER]) / 2.0
    mouth_c = (take_points[MOUTH_LEFT] + take_points[MOUTH_RIGHT]) / 2.0
    down = mouth_c - eye_c
    down /= max(float(np.linalg.norm(down)), 1e-6)
    ys, xs = np.mgrid[:h, :w]
    v = (xs - eye_c[0]) * down[0] + (ys - eye_c[1]) * down[1]
    brow_top = float(((take_points[list(BROWS)] - eye_c) @ down).min())
    fade = (v - (brow_top - TOP_CUT * d)) / max(TOP_RAMP * d, 1.0)
    return mask * np.clip(fade, 0.0, 1.0).astype(np.float32)


def features_mask(shape: Tuple[int, int], take_points: np.ndarray) -> np.ndarray:
    """A soft mask over the eyes and brows, where the fine detail of the
    base (lids, lash line, brow hairs) is blended."""
    import cv2

    d = eye_distance(take_points)
    mask = np.zeros(shape, dtype=np.float32)
    for eye, brow in ((LEFT_EYE, BROWS[:10]), (RIGHT_EYE, BROWS[10:])):
        mask = np.maximum(mask, _poly(shape, take_points[list(eye) + list(brow)], hull=True))
    k = max(3, int(round(FEATURES_GROW * d)) | 1)
    mask = cv2.dilate(mask, np.ones((k, k), np.uint8))
    return cv2.GaussianBlur(mask, (0, 0), max(1.0, FEATURES_FEATHER * d))


# ── the blend ────────────────────────────────────────────────────────────


def compose(
    base_rgb: np.ndarray, take_rgb: np.ndarray, base_points: np.ndarray, take_points: np.ndarray,
    *, alpha: float, keep_mouth: bool, detail_sigma: float,
) -> Optional[Dict[str, Any]]:
    """The take with the base face's structure mixed in along the mesh.

    Works on a crop around the take's face (the rest of the image is left
    byte for byte). Returns {"image": RGB array, "warped": the warped base
    in the crop, "mask": the mask in the crop, "box": the crop} or None when
    the face is too small to work with."""
    import cv2

    from MIID.miner.face_blend import match_colour_local

    H, W = take_rgb.shape[:2]
    d = eye_distance(take_points)
    if d < 20:
        return None
    x0, y0 = take_points.min(axis=0)
    x1, y1 = take_points.max(axis=0)
    pad = 0.3 * d + 4 * LOCAL_SIGMA * d
    bx0, by0 = max(0, int(x0 - pad)), max(0, int(y0 - pad))
    bx1, by1 = min(W, int(x1 + pad) + 1), min(H, int(y1 + pad) + 1)
    crop = take_rgb[by0:by1, bx0:bx1]
    shape = crop.shape[:2]
    local_take = take_points - np.array([bx0, by0], dtype=np.float64)

    skin = base_skin(base_rgb, base_points)
    warped, usable, skin_warped = warp(base_rgb, base_points, local_take, shape, extra=skin)
    usable = usable * np.clip(skin_warped, 0.0, 1.0)
    mask = blend_mask(shape, local_take, usable, keep_mouth=keep_mouth)
    if float(mask.max()) <= 0.0:
        return None
    face = match_colour_local(warped, crop, mask, max(2.0, LOCAL_SIGMA * d))

    face_f, crop_f = face.astype(np.float32), crop.astype(np.float32)

    def band(sigma: float) -> np.ndarray:
        if sigma > 0:
            return cv2.GaussianBlur(face_f, (0, 0), sigma) - cv2.GaussianBlur(crop_f, (0, 0), sigma)
        return face_f - crop_f

    difference = band(detail_sigma)
    skin_sigma = max(detail_sigma, SKIN_DETAIL_SIGMA)
    if skin_sigma != detail_sigma:
        features = features_mask(shape, local_take)[..., None]
        difference = features * difference + (1.0 - features) * band(skin_sigma)
    if DIFF_CLIP > 0:
        difference = DIFF_CLIP * np.tanh(difference / DIFF_CLIP)
    mixed = np.clip(crop_f + (alpha * mask)[..., None] * difference + 0.5, 0, 255).astype(np.uint8)
    out = take_rgb.copy()
    out[by0:by1, bx0:bx1] = mixed
    return {"image": out, "warped": warped, "mask": mask, "box": (bx0, by0, bx1, by1)}


def texture_corr(warped: np.ndarray, image_crop: np.ndarray, mask: np.ndarray) -> Optional[float]:
    """Correlation of the high-pass detail of the warped base and an image
    over the blended region — face_blend.texture_corr on the mesh's own
    alignment, which lines the seed's texture up far better than the
    benchmark's similarity fit does, so it is the stricter reading."""
    import cv2

    region = mask > 0.5
    if region.sum() < 500:
        return None

    def high_pass(rgb: np.ndarray) -> np.ndarray:
        grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        return grey - cv2.GaussianBlur(grey, (0, 0), 3.0)

    a = high_pass(warped)[region]
    b = high_pass(image_crop)[region]
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0
