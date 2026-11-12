"""Accessory-safe face blend (2026-11-12): the fallback mask and the guard.

Run with miner_env: ``python -m pytest unittest/test_miner_face_blend_accessory.py``
"""
import cv2
import numpy as np
import pytest

from MIID.miner import edit_check, face_blend

# Five MTCNN-style landmarks of a frontal face: eyes, nose, mouth corners.
POINTS = np.array([[160.0, 200.0], [240.0, 200.0], [200.0, 240.0], [170.0, 280.0], [230.0, 280.0]])
SKIN = (205, 160, 130)
FABRIC = (40, 60, 160)


def _portrait_with_band():
    """A skin-coloured face whose inner-face ellipse has fabric where the
    facial features end — the jaw and temples, where a hijab wraps. Returns
    the image and the fabric wider than the skin mask's closing (0.12 eye
    distances), which by design fills thinner gaps such as eyes and creases."""
    img = np.full((420, 400, 3), 120, dtype=np.uint8)
    cv2.ellipse(img, (200, 250), (110, 150), 0, 0, 360, SKIN, -1)
    core = face_blend.face_ellipse(img.shape[:2], POINTS, feather=0.0) > 0.5
    feats = face_blend.features_ellipse(img.shape[:2], POINTS) > 0.5
    near_feats = cv2.dilate(feats.astype(np.uint8), np.ones((25, 25), np.uint8)).astype(bool)
    fabric = core & ~near_feats
    img[fabric] = FABRIC
    eye_distance = float(np.linalg.norm(POINTS[1] - POINTS[0]))
    k = 2 * int(round(0.12 * eye_distance)) + 1
    solid = cv2.erode(fabric.astype(np.uint8), np.ones((k, k), np.uint8)).astype(bool)
    return img, solid


def test_features_mask_keeps_headwear_out():
    img, fabric = _portrait_with_band()
    assert fabric.sum() > 200
    old = face_blend.skin_mask(img, POINTS)                      # whole inner face forced
    new = face_blend.skin_mask(img, POINTS, forced="features")   # only the features forced
    assert old[fabric].mean() > 0.9
    assert new[fabric].mean() < 0.1
    # The facial features themselves are blended either way.
    feats = face_blend.features_ellipse(img.shape[:2], POINTS) > 0.5
    inner = cv2.erode(feats.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)
    assert new[inner].mean() > 0.95


def test_build_mask_modes():
    img, fabric = _portrait_with_band()
    assert face_blend.build_mask(img, POINTS, mode="skin_features")[fabric].mean() < 0.1
    assert face_blend.build_mask(img, POINTS, mode="skin")[fabric].mean() > 0.9


def _acc(status, conf):
    return {"edit": {"lighting": None, "expression": None,
                     "accessory": {"requested": "religious_head_covering", "status": status,
                                   "confidence": conf, "top": "religious_head_covering"}}}


@pytest.mark.parametrize("before, after, floor, kept", [
    (("match", 0.94), ("match", 0.68), None, False),   # a drop over 0.10: refused between takes
    (("match", 0.94), ("match", 0.68), 0.6, True),     # still a clear match after a blend: kept
    (("match", 0.93), ("match", 0.55), 0.6, False),    # under the floor
    (("match", 0.82), ("other", 0.45), 0.6, False),    # no longer a match
    (("insufficient", 0.49), ("insufficient", 0.35), 0.6, False),  # was never a match: old rule
    (("match", 0.70), ("match", 0.66), None, True),    # small drops pass either way
])
def test_accessory_floor(before, after, floor, kept):
    ok, lost = edit_check.preserved({}, _acc(*before), _acc(*after), None, accessory_floor=floor)
    assert ok is kept
    if not kept:
        assert lost == "accessory"
