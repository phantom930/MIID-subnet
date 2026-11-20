"""Face mesh blend for the combo slots (2026-11-20): warp, mask and composite.

Run with miner_env: ``python -m pytest unittest/test_miner_face_mesh.py``
(skipped when mediapipe is not installed).
"""
import os

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("mediapipe")

from MIID.miner import face_blend, face_mesh  # noqa: E402

SEED = os.path.join(
    os.path.dirname(__file__), "..", "MIID", "miner", "models", "seed_images", "475c5c38e38b_m_doc.png"
)


@pytest.fixture(scope="module")
def seed():
    image = Image.open(SEED).convert("RGB")
    points = face_mesh.landmarks(image)
    if points is None:
        pytest.skip("face landmarker unavailable")
    return np.asarray(image), points


def test_landmarks_cover_the_face(seed):
    rgb, points = seed
    assert points.shape == (478, 2)
    assert face_mesh.eye_distance(points) > 40
    h, w = rgb.shape[:2]
    assert (points[:, 0] > 0).all() and (points[:, 0] < w).all()


def test_identity_warp_reproduces_the_base(seed):
    rgb, points = seed
    warped, usable, _ = face_mesh.warp(rgb, points, points, rgb.shape[:2])
    inside = usable > 0.5
    assert inside.sum() > 1000
    diff = np.abs(warped.astype(np.int16) - rgb.astype(np.int16))[inside]
    assert diff.mean() < 2.0


def test_turned_away_triangles_are_left_out(seed):
    rgb, points = seed
    take = points.copy()
    # Squash the face's right half toward its centre line: those triangles
    # shrink to a sliver, as the far side of a turned head does.
    centre = points[:468, 0].mean()
    right = take[:, 0] > centre
    take[right, 0] = centre + 0.15 * (take[right, 0] - centre)
    _, usable, _ = face_mesh.warp(rgb, points, take, rgb.shape[:2])
    ys = int(points[1, 1])
    left_x = int(points[234, 0] + 0.5 * (centre - points[234, 0]))
    assert usable[ys, left_x] == 1.0
    assert usable[:, int(centre) + 2:].sum() < usable[:, :int(centre) - 2].sum() * 0.2


def test_mask_keeps_irises_mouth_and_forehead_out(seed):
    rgb, points = seed
    shape = rgb.shape[:2]
    usable = np.ones(shape, dtype=np.float32)
    with_mouth = face_mesh.blend_mask(shape, points, usable, keep_mouth=True)
    without_mouth = face_mesh.blend_mask(shape, points, usable, keep_mouth=False)

    def at(mask, idx):
        x, y = points[idx]
        return float(mask[int(round(y)), int(round(x))])

    for iris in (468, 473):
        assert at(with_mouth, iris) < 0.05
    mouth_centre = (points[13] + points[14]) / 2.0
    mx, my = int(round(mouth_centre[0])), int(round(mouth_centre[1]))
    assert with_mouth[my, mx] > 0.5
    assert without_mouth[my, mx] < 0.05
    # Nose bridge and cheeks are blended; the top of the forehead is not.
    assert at(with_mouth, 6) > 0.5
    assert at(with_mouth, 205) > 0.5
    assert at(with_mouth, 10) < 0.05


def test_compose_changes_only_the_face(seed):
    rgb, points = seed
    # A "take": the seed brightened, so the base has something to restore.
    take = np.clip(rgb.astype(np.int16) + 25, 0, 255).astype(np.uint8)
    composed = face_mesh.compose(rgb, take, points, points, alpha=0.8, keep_mouth=False, detail_sigma=2.0)
    assert composed is not None
    out = composed["image"]
    x0, y0, x1, y1 = composed["box"]
    outside = np.ones(rgb.shape[:2], dtype=bool)
    outside[y0:y1, x0:x1] = False
    assert (out[outside] == take[outside]).all()
    assert np.abs(out.astype(np.int16) - take.astype(np.int16))[y0:y1, x0:x1].max() > 0
    # Structure only: the restored face keeps the take's brightness (the
    # colour match is local), not the base's.
    nose = points[4].astype(int)
    patch = (slice(nose[1] - 5, nose[1] + 5), slice(nose[0] - 5, nose[0] + 5))
    assert abs(float(out[patch].mean()) - float(take[patch].mean())) < 8


def test_combo_slots_are_blendable():
    for vtype in ("lighting_edit+expression_edit", "lighting_edit+pose_edit", "pose_edit+expression_edit"):
        result = {"variation_type": vtype, "image": object(), "identity_similarity": 0.7}
        assert face_blend.BLEND_MODES[vtype][0] == "mesh"
        assert face_blend.wanted(False, result) == (vtype in face_blend.SLOTS)
