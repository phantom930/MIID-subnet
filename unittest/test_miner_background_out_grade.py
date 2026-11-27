"""background_out scored 0 by the owner's grader (2026-11-27).

The grader's comment on every background_out 0 was "Background environment
mismatch: expected outdoor, CLIP classified indoor."; far takes also drew -1
"Face not dominant". These tests cover the response: the comment becomes a
prompt hint for the re-roll, background_out grades go first in the queue,
and the outdoor scene decks no longer put a wall, fence or door close behind
the subject.

Run with miner_env: ``python -m pytest unittest/test_miner_background_out_grade.py``
"""
import importlib

import pytest

from MIID.miner import grade_check, variety

# MIID.miner re-exports a function named generate_variations over the module.
gv = importlib.import_module("MIID.miner.generate_variations")

INDOOR_ZERO = {
    "breaks": False, "score": 0.0,
    "comment": "Score 0. Background environment mismatch: expected outdoor, CLIP classified indoor.",
}
FACE_SMALL = {
    "breaks": False, "score": -1.0,
    "comment": "Score -1. Face not dominant (face_ratio=0.0946, threshold=0.10).",
}
PLAIN_TWO = {
    "breaks": False, "score": 2.0,
    "comment": "Score 2. Identity 1 (ada_sim=0.836). Variation 1 (background_out light). "
               "Only that variation 0. Our deepfake detector flagged the image.",
}

BG_OUT_REQ = {
    "type": "background_out",
    "intensity": "light",
    "description": "Change the background while keeping the subject unchanged. "
                   "Outdoor background change. Add religious head covering",
    "detail": "Same outdoor context with subtle changes. Additionally, include: "
              "Religious head covering (hijab, turban, kippah, taqiyah, etc.) appropriate to subject.",
}


def test_grade_comment_names_the_hint():
    assert grade_check.hints(INDOOR_ZERO) == {"background": "indoor"}
    assert grade_check.hints(FACE_SMALL) == {"framing": "closer"}
    assert grade_check.hints(PLAIN_TWO) == {}
    assert grade_check.hints(None) == {}
    assert grade_check.hints({"score": 0.0, "comment": ""}) == {}


def test_grade_hints_are_known_prompt_hints():
    for grade in (INDOOR_ZERO, FACE_SMALL):
        for component, hint in grade_check.hints(grade).items():
            table = {"background": gv.BACKGROUND_HINTS, "framing": gv.FRAMING_HINTS}[component]
            assert hint in table


def test_prompt_carries_the_grade_hints():
    prompt = gv._get_prompt_from_request(
        BG_OUT_REQ, "background_out", "light",
        hints={"background": "indoor", "framing": "closer"},
    )
    assert gv.BACKGROUND_HINTS["indoor"] in prompt
    assert gv.FRAMING_HINTS["closer"] in prompt
    assert "open sky" in prompt.lower()


CLOSE_BACKDROPS = ("wall", "fence", "door", "awning", "mural", "stall", "tram", "cathedral", "archway", "hedge")


@pytest.mark.parametrize("level", ["light", "medium", "far"])
def test_outdoor_scenes_avoid_close_built_backdrops(level):
    for scene, climate in variety.SCENES[("outdoor", level)]:
        assert not any(word in scene for word in CLOSE_BACKDROPS), scene
        assert climate in variety.CLOTHING


def test_outdoor_light_look_keeps_the_scene_recognisable():
    for _ in range(len(variety.OUTDOOR_LOOKS["light"]) * 2):
        sentence, _climate = variety.background("outdoor", "light")
        assert "bokeh" not in sentence
        assert any(look in sentence for look in variety.OUTDOOR_LOOKS["light"])
    # Indoor light keeps its own looks.
    sentence, _ = variety.background("indoor", "light")
    assert any(look in sentence for look in variety.LOOKS["light"])


def test_no_wide_angle_far_look():
    assert not any("28 mm" in look for look in variety.LOOKS["far"])


@pytest.fixture
def ig():
    from MIID.miner import image_generator
    return image_generator


def _graded_slot(ig, grade):
    slot = ig._Slot(0, BG_OUT_REQ)
    slot.best = {"identity_similarity": 0.82, "grade": grade, "variation_type": "background_out"}
    slot.identity_ok = slot.gate_ok = True
    slot.generated = 1
    return slot


def test_slot_retries_a_graded_indoor_take_with_the_outdoor_hint(ig):
    slot = _graded_slot(ig, INDOOR_ZERO)
    assert not slot.satisfied
    bias, hints = slot.next_settings()
    assert bias == 0.0
    assert hints == {"background": "indoor"}
    assert slot.retry_priority(attempts_allowed=4) == ig.GRADE_FIX_RETRY_VALUE


def test_slot_retries_a_small_face_with_the_framing_hint(ig):
    slot = _graded_slot(ig, FACE_SMALL)
    assert slot.next_settings()[1] == {"framing": "closer"}


def test_plain_low_grade_keeps_the_old_retry_value(ig):
    slot = _graded_slot(ig, PLAIN_TWO)
    assert slot.next_settings()[1] == {}
    assert slot.retry_priority(attempts_allowed=4) == ig.GRADE_RETRY_VALUE


def test_background_out_grades_go_first(ig):
    assert ig.GRADE_QUEUE_LEAD["background_out"] > 0
    assert ig.GRADE_QUEUE_LEAD.get("background_in", 0.0) == 0.0
