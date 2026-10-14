# MIID/miner/diversity.py
#
# A per-hotkey style profile, so hotkeys run from the same code do not draw
# their scenes, light and colours from one distribution.
#
# Fresh seeds make every render a different image; this makes the *choices*
# differ too: which rooms and streets a hotkey favours on background slots,
# which side its light and head turns prefer, whether its colour casts run
# warm or cool, how its far expressions lean (laughing / surprised / worried)
# and what colours its accessories come in. The profile is derived from the
# hotkey address, so a restart keeps the same character and two hotkeys never
# share one by accident. It reaches the prompt as a short style clause after
# the edit (generate_variations._get_prompt_from_request) and never changes
# the requested edit itself.
#
# This is the complement of the sibling ledger (sibling_registry.py): the
# ledger catches a near-identical take after the fact; the profile makes one
# unlikely in the first place. MIID_DIVERSITY=0 switches it off — a single
# hotkey needs none of this.

import hashlib
import os
import random
from typing import Any, Dict, List, Optional

ENABLED = os.environ.get("MIID_DIVERSITY", "1").strip().lower() not in ("0", "false", "no")

INDOOR_SCENES: List[str] = [
    "a library with full bookshelves",
    "an open-plan office with desks and monitors",
    "a cafe with wooden tables and warm pendant lights",
    "a hotel lobby with armchairs and a reception desk",
    "a living room with a sofa, framed pictures and a plant",
    "a kitchen with cabinets and a window",
    "a classroom with a whiteboard and desks",
    "a museum gallery with paintings on the walls",
    "a modern meeting room with a glass wall",
    "a home study with a desk lamp and shelves",
    "a restaurant interior with set tables",
    "an airport lounge with rows of seats",
]
OUTDOOR_SCENES: List[str] = [
    "a tree-lined park path",
    "a city street with shopfronts",
    "a waterfront promenade",
    "a quiet residential street with houses and hedges",
    "a sunlit garden with flowers",
    "a mountain overlook with a wide view",
    "a beach with the sea behind",
    "a university campus lawn with buildings",
    "a market square with stalls",
    "a forest trail among tall trees",
    "a bridge over a river",
    "a vineyard on a hillside",
]
LIGHT_CASTS: List[str] = ["warm", "cool", "neutral"]
FAR_EXPRESSIONS: List[str] = [
    "laughing with the mouth wide open",
    "visibly surprised, with wide eyes and raised brows",
    "visibly worried, with the brow drawn together and the mouth tense",
]
ACCESSORY_COLOURS: List[str] = [
    "navy blue", "dark green", "burgundy", "mustard yellow", "charcoal grey",
    "teal", "rust orange", "olive", "cream", "plum",
]


def profile_for(hotkey: str) -> Dict[str, Any]:
    """The deterministic style profile of one hotkey."""
    seed = int(hashlib.sha256(hotkey.encode("utf-8")).hexdigest()[:16], 16)
    rng = random.Random(seed)
    return {
        "hotkey": hotkey,
        "side": rng.choice(("left", "right")),
        "indoor": rng.sample(INDOOR_SCENES, 4),
        "outdoor": rng.sample(OUTDOOR_SCENES, 4),
        "cast": rng.choice(LIGHT_CASTS),
        "far_expression": rng.choice(FAR_EXPRESSIONS),
        "colours": rng.sample(ACCESSORY_COLOURS, 3),
    }


def preferred_side(profile: Optional[Dict[str, Any]], rng: random.Random = random) -> str:
    """This hotkey's side three times in four, the other side otherwise."""
    if not profile:
        return rng.choice(("left", "right"))
    side = profile["side"]
    return side if rng.random() < 0.75 else ("right" if side == "left" else "left")


def style_clause(
    profile: Optional[Dict[str, Any]], levels: Dict[str, str],
    accessory: bool = False, rng: random.Random = random,
) -> str:
    """The style sentences for a slot, or "" (never alters the edit itself)."""
    if not profile:
        return ""
    parts: List[str] = []
    if "background_in" in levels:
        parts.append(f"Use {rng.choice(profile['indoor'])} as the setting.")
    if "background_out" in levels:
        parts.append(f"Use {rng.choice(profile['outdoor'])} as the setting.")
    if levels.get("lighting_edit") in ("medium", "far") and profile["cast"] != "neutral":
        parts.append(f"Give the light a {profile['cast']} colour cast.")
    if levels.get("expression_edit") == "far":
        parts.append(f"For the expression, make the subject {profile['far_expression']}.")
    if accessory:
        parts.append(f"The added headwear is {rng.choice(profile['colours'])}.")
    return " ".join(parts)


def describe(profile: Optional[Dict[str, Any]]) -> str:
    if not profile:
        return "diversity profile: off"
    return (
        f"diversity profile for {profile['hotkey'][:8]}: side {profile['side']}, "
        f"{profile['cast']} cast, indoor {[s.split(' with')[0] for s in profile['indoor']]}, "
        f"outdoor {[s for s in profile['outdoor']]}, colours {profile['colours']}"
    )
