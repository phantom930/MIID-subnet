# MIID/miner/variety.py
#
# What keeps one request's background slot from looking like the last one's,
# and like every other miner's.
#
# The subnet owner's guidance (2026-10-28): vary prompts beyond one-word
# tweaks, vary accessories (hats, religious coverings, colours), backgrounds
# (scenery, surfaces, time of day, palette) and clothing where the variation
# allows it — "if your outputs look interchangeable with another miner's,
# expect a penalty". The validator grades that penalty as a duplicate flag
# (validation score -0.2, reputation -0.10), and on the 2026-10-28 stats
# snapshot it fell almost only on background slots: 48 of 269 background_out
# grades across all miners (17.8%), 1 of 149 background_in, none of the 1,078
# pose / lighting / expression ones.
#
# Every miner gets the same seed image per challenge, most run the same FLUX.2
# Klein 4B, and the reference miner relays the validator's own examples (park,
# street, plaza, waterfront; office, cafe, lobby, gallery) as its prompt. Our
# fixed phrases asked for those same places, and the archive shows it: on
# 2026-10-28 nearly every background_out was one tree-lined lakeside
# promenade and nearly every background_in one wood-panelled library, every
# man in the same white crocheted taqiyah, every subject in the seed's grey
# t-shirt. So a background slot now draws its scene, light, palette, lens,
# clothing and accessory colour from the decks below, away from those
# examples.
#
# Each list is dealt from a shuffled deck, so consecutive requests do not
# repeat an item until the deck is used up.

import random
import threading
from typing import Dict, List, Sequence, Tuple


class _Deck:
    """Items dealt in a random order without repeats, reshuffled when used up."""

    def __init__(self, items: Sequence) -> None:
        self._items = list(items)
        self._left: List = []
        self._lock = threading.Lock()

    def draw(self):
        with self._lock:
            if not self._left:
                self._left = random.sample(self._items, len(self._items))
            return self._left.pop()


# Scenes by (environment, level). "cold" / "warm" marks the ones whose
# clothing has to match the weather; the rest take any garment.
#
# light keeps the scene close behind the subject and softly out of focus
# (the validator's light is a minor adjustment); medium is a plausible
# everyday place; far is a distinct place with depth, receding behind them.
SCENES: Dict[Tuple[str, str], Tuple[Tuple[str, str], ...]] = {
    ("outdoor", "light"): (
        ("a sunlit red-brick wall with climbing ivy", "any"),
        ("a row of colourful townhouse front doors", "any"),
        ("a weathered wooden fence with climbing roses", "warm"),
        ("a stone archway in an old town lane", "any"),
        ("tall bamboo leaves in dappled sunlight", "warm"),
        ("a cafe terrace with striped awnings", "warm"),
        ("a hedge of flowering bougainvillea against a white wall", "warm"),
        ("golden autumn maple leaves", "cold"),
        ("a painted mural on a city wall", "any"),
        ("a harbour wall with coiled ropes and fishing nets", "any"),
        ("a sunflower field", "warm"),
        ("pine branches dusted with snow", "cold"),
    ),
    ("outdoor", "medium"): (
        ("a farmers' market with fruit and vegetable stalls", "any"),
        ("a university campus quad with old stone buildings", "any"),
        ("a tram stop on a busy city avenue", "any"),
        ("a small fishing harbour with moored boats", "any"),
        ("a suburban front yard with a white picket fence", "warm"),
        ("a rooftop terrace with potted plants and a city skyline", "warm"),
        ("an open-air railway platform with an arriving train", "any"),
        ("a vineyard with rows of vines on a hillside", "warm"),
        ("a canal bridge lined with bicycles", "any"),
        ("an autumn forest trail covered in fallen leaves", "cold"),
        ("a snowy village street with wooden chalets", "cold"),
        ("a sports field sideline with bleachers", "any"),
        ("a food truck lot with string lights", "any"),
        ("a lavender field with a stone farmhouse", "warm"),
    ),
    ("outdoor", "far"): (
        ("a red-rock desert canyon under a deep blue sky", "warm"),
        ("a snowy alpine valley with jagged peaks", "cold"),
        ("a tropical bay with palm trees and turquoise water", "warm"),
        ("a misty pine forest with tall trunks fading into fog", "cold"),
        ("a hillside town of white houses above a blue sea", "warm"),
        ("terraced green rice fields stepping down a valley", "warm"),
        ("a busy city crossing with tall glass towers", "any"),
        ("an avenue of cherry trees in full pink blossom", "any"),
        ("a golden wheat field under a vast sky", "warm"),
        ("a rocky sea cliff with waves breaking below", "any"),
        ("an old cathedral square with a gothic facade", "any"),
        ("a frozen lake with snow-covered mountains beyond", "cold"),
        ("rolling sand dunes in long evening shadows", "warm"),
        ("a rainforest waterfall surrounded by ferns", "warm"),
    ),
    ("indoor", "light"): (
        ("a sage-green living-room wall with a framed botanical print", "any"),
        ("a kitchen with open wooden shelves of jars and plants", "any"),
        ("a hallway with patterned vintage wallpaper", "any"),
        ("an exposed-brick wall with a hanging plant", "any"),
        ("a bedroom with a linen headboard and a reading lamp", "any"),
        ("a home studio wall of grey acoustic panels", "any"),
        ("a shelf of handmade ceramics against a terracotta wall", "any"),
        ("a navy-blue wall with a row of framed photographs", "any"),
        ("a reading nook with a mustard armchair and a floor lamp", "any"),
        ("a tiled wall in deep green with brass fittings", "any"),
    ),
    ("indoor", "medium"): (
        ("a bakery with bread loaves on wooden racks", "any"),
        ("a bicycle repair workshop with tools on the wall", "any"),
        ("a yoga studio with a wooden floor and plants", "any"),
        ("a classroom with a chalkboard and wooden desks", "any"),
        ("a record shop with crates of vinyl", "any"),
        ("a barber shop with leather chairs and mirrors", "any"),
        ("a pottery studio with wheels and shelves of clay pots", "any"),
        ("a florist's shop full of buckets of flowers", "any"),
        ("a tailor's workshop with rolls of fabric", "any"),
        ("a gym with weight racks and rubber flooring", "any"),
        ("a music rehearsal room with guitars and amplifiers", "any"),
        ("a laundromat with rows of washing machines", "any"),
        ("a small bookshop with crowded shelves and a ladder", "any"),
    ),
    ("indoor", "far"): (
        ("a grand theatre foyer with red velvet and gilded balconies", "any"),
        ("an industrial loft with steel beams and brick walls", "any"),
        ("a neon-lit retro arcade with rows of game cabinets", "any"),
        ("a vintage diner with chrome counters and red booths", "any"),
        ("a Japanese tatami room with paper shoji screens", "any"),
        ("a cathedral nave with stone columns and stained glass", "any"),
        ("a recording studio with a large mixing desk", "any"),
        ("a bowling alley with polished lanes", "any"),
        ("a richly tiled hall with carved arches and lanterns", "any"),
        ("a concert hall with rows of seats and an organ", "any"),
        ("a busy restaurant kitchen with steel counters and copper pans", "any"),
    ),
}

# Outdoor light, never night (identity and the indoor/outdoor check both
# need the face and the scene lit).
OUTDOOR_LIGHT: Tuple[str, ...] = (
    "in soft early-morning light",
    "under a bright overcast sky",
    "in warm late-afternoon sun",
    "at golden hour, warm low sun on the scene behind",
    "in clear midday light",
    "on a hazy, pastel-toned morning",
    "just after a rain shower, the ground still wet and reflective",
)
INDOOR_LIGHT: Tuple[str, ...] = (
    "lit by warm table lamps",
    "in soft daylight from out of view",
    "under bright, even overhead lighting",
    "in warm evening tungsten light",
    "with colourful accent lighting in the background",
    "in cool, clean light",
)

PALETTES: Tuple[str, ...] = (
    "muted earthy tones",
    "rich saturated colours",
    "cool blues and teals",
    "warm ambers and terracotta",
    "soft pastels",
    "deep greens and browns",
    "crisp contrasting colours",
)

# Photographic look: lens and depth of field. Light levels keep the
# background soft; far ones keep its depth readable.
LOOKS: Dict[str, Tuple[str, ...]] = {
    "light": (
        "an 85 mm portrait lens at a wide aperture, the background melting into soft bokeh",
        "a 135 mm lens, the background compressed and softly blurred",
        "a 50 mm lens with gentle background blur",
    ),
    "medium": (
        "a 50 mm lens with moderate depth of field",
        "a 35 mm lens, documentary style, natural colour",
        "an 85 mm lens with the background recognisable but softened",
    ),
    "far": (
        "a 35 mm lens at a mid aperture, the scene behind readable in depth",
        "a 28 mm environmental-portrait lens showing the wide setting",
        "a 50 mm lens with deep focus on the surroundings",
    ),
}

# Garments for background slots, by the scene's climate ("any" for rooms and
# mild weather). Plain fabrics, no text or logos; no scarves, which the score
# sheet would read as a second accessory.
CLOTHING: Dict[str, Tuple[str, ...]] = {
    "any": (
        "a navy crew-neck sweater",
        "a denim jacket over a white t-shirt",
        "a rust-orange knit cardigan over a cream top",
        "an olive utility jacket",
        "a burgundy hoodie",
        "a light-blue button-down shirt",
        "a black leather jacket over a grey top",
        "a navy-and-white striped Breton top",
        "a charcoal blazer over a black t-shirt",
        "a red-and-black flannel shirt",
        "a lavender sweatshirt",
        "a mustard-yellow ribbed sweater",
        "a teal zip-up track jacket",
        "a white linen shirt",
    ),
    "warm": (
        "a white linen shirt",
        "a coral short-sleeved shirt",
        "a sky-blue polo shirt",
        "a sage-green t-shirt",
        "a yellow cotton t-shirt",
        "a light chambray shirt",
        "a navy-and-white striped Breton top",
        "a terracotta short-sleeved shirt",
    ),
    "cold": (
        "a camel wool coat",
        "a quilted navy puffer jacket",
        "a chunky cream fisherman sweater",
        "a dark-green parka",
        "a red wool peacoat",
        "a grey turtleneck under a charcoal coat",
        "a forest-green fleece jacket",
        "a mustard-yellow ribbed sweater",
    ),
}

# The validator's accessories, each in many colours and cuts but always in
# the shape the grader's classifier reads (see generate_variations
# ACCESSORY_RENDERINGS and RELIGIOUS_COVERINGS for why each shape is pinned).
# Contrast with the hair is what carries its confidence.
TAQIYAH_STYLES: Tuple[str, ...] = (
    "a white crocheted taqiyah with a fine open pattern",
    "a cream kufi cap with a gold embroidered band",
    "a light-grey taqiyah with a geometric stitched pattern",
    "a sky-blue knitted kufi",
    "a black kufi cap with bright gold embroidery",
    "a maroon kufi with cream embroidery",
    "a colourfully embroidered kufi in red, green and gold",
    "an olive-green knitted kufi with a white rim",
    "a beige crocheted taqiyah",
)
HIJAB_STYLES: Tuple[str, ...] = (
    "a navy hijab",
    "a dusty-rose hijab",
    "an olive-green hijab",
    "a burgundy hijab",
    "a sand-beige hijab",
    "a soft lavender hijab",
    "a teal hijab",
    "a mustard-yellow hijab",
    "a black hijab",
    "a cream hijab with a subtle floral print",
    "a sky-blue hijab",
    "a charcoal hijab with a light pinned underscarf band",
)
HIJAB_DRAPES: Tuple[str, ...] = (
    "wrapped neatly and pinned at the side",
    "draped softly over the shoulders",
    "wrapped snugly around the face",
    "folded in smooth layers under the chin",
)
BRIM_HATS: Tuple[str, ...] = (
    "a black felt fedora",
    "a tan straw panama hat with a dark band",
    "a wide-brim cream sun hat",
    "a navy felt fedora",
    "an olive wide-brim hat",
    "a burgundy felt hat with a broad brim",
    "a grey fedora with a black ribbon",
    "a natural straw sun hat with a wide floppy brim",
)
KNIT_HATS: Tuple[str, ...] = (
    "a chunky red cable-knit beanie with a thick folded cuff",
    "a mustard ribbed beanie with a pom-pom",
    "a cobalt-blue knit beanie with a folded cuff",
    "an emerald-green cable-knit winter hat",
    "an orange fisherman beanie",
    "a pink ribbed knit hat with a pom-pom",
    "a cream cable-knit beanie with a wide cuff",
    "a purple chunky-knit winter hat",
)
CAPS: Tuple[str, ...] = (
    "a bright red baseball cap",
    "a royal-blue baseball cap",
    "a kelly-green baseball cap",
    "an orange baseball cap",
    "a sunny-yellow baseball cap",
    "a white baseball cap",
    "a teal baseball cap",
    "a maroon baseball cap",
)
BANDANAS: Tuple[str, ...] = (
    "a red paisley bandana",
    "a navy paisley bandana",
    "a black-and-white paisley bandana",
    "an olive bandana with white polka dots",
    "a mustard paisley bandana",
    "a turquoise patterned bandana",
    "a pink paisley bandana",
)

_DECKS: Dict[object, _Deck] = {}
_DECKS_LOCK = threading.Lock()


def deal(key: object, items: Sequence):
    """The next item of the deck named ``key`` (created from ``items``)."""
    with _DECKS_LOCK:
        deck = _DECKS.get(key)
        if deck is None:
            deck = _DECKS[key] = _Deck(items)
    return deck.draw()


def background(environment: str, level: str) -> Tuple[str, str]:
    """(scene sentence, climate) for a background slot: where, in what light,
    with what palette and lens."""
    level = level if level in LOOKS else "medium"
    scene, climate = deal(("scene", environment, level), SCENES[(environment, level)])
    light = deal(("light", environment), OUTDOOR_LIGHT if environment == "outdoor" else INDOOR_LIGHT)
    palette = deal("palette", PALETTES)
    look = deal(("look", level), LOOKS[level])
    return f"{scene}, {light}, in {palette}; photographed with {look}", climate


def clothing(climate: str = "any") -> str:
    """A garment for the subject, matching the scene's weather."""
    return deal(("clothing", climate), CLOTHING.get(climate, CLOTHING["any"]))


def taqiyah() -> str:
    return deal("taqiyah", TAQIYAH_STYLES)


def hijab() -> str:
    return f"{deal('hijab', HIJAB_STYLES)} {deal('drape', HIJAB_DRAPES)}"


def brim_hat() -> str:
    return deal("brim", BRIM_HATS)


def knit_hat() -> str:
    return deal("knit", KNIT_HATS)


def cap() -> str:
    return deal("cap", CAPS)


def bandana() -> str:
    return deal("bandana", BANDANAS)
