# MIID/benchmark/loader.py
#
# Turns request-archive records (miner_requests/<date>/<record>/) into the
# (base image, variation image, request) items the benchmark scores. Only
# records written with MIID_ARCHIVE_IMAGES=1 carry the pixels; the rest are
# reported as skipped rather than silently ignored.

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Mirrors ACCESSORY_TYPES in MIID/validator/image_variations.py. Order
# matters: the brim-hat detail says "(not baseball cap)", so it is matched
# before the baseball cap.
_ACCESSORY_KEYWORDS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("religious_head_covering", ("religious head covering",)),
    ("brim_hat", ("brim hat", "fedora", "wide-brim")),
    ("knit_winter_hat", ("knit hat", "beanie", "winter hat")),
    ("bandana", ("bandana",)),
    ("baseball_cap", ("baseball cap", "sports cap")),
    ("headphones", ("headphones", "headset")),
)

BACKGROUND_TYPES = ("background_in", "background_out", "background_edit")


@dataclass
class Component:
    type: str
    intensity: str


@dataclass
class VariationItem:
    """One generated image and what the validator asked it to be."""

    record_id: str
    record_dir: Optional[Path]
    slot: int
    variation_type: str
    intensity: str
    detail: str
    description: str
    image: Path
    base: Path
    base_filename: Optional[str]
    status: Optional[str] = None  # submitted / dropped, from variation_report
    model: Optional[str] = None
    miner_identity: Optional[float] = None
    components: List[Component] = field(default_factory=list)
    accessory: Optional[str] = None
    environment: Optional[str] = None  # indoor / outdoor for background slots
    submitted_as: Optional[str] = None  # filename stem the miner saved it under

    @property
    def is_background(self) -> bool:
        return self.variation_type in BACKGROUND_TYPES


@dataclass
class Round:
    record_id: str
    record_dir: Optional[Path]
    started_at: Optional[str]
    validator: Optional[str]
    model: Optional[str]
    requested: int
    items: List[VariationItem]
    missing_slots: List[str]


def parse_components(var_type: str, intensity: str) -> List[Component]:
    """'lighting_edit+pose_edit' / 'far+medium' -> two components."""
    types = [t.strip() for t in var_type.split("+") if t.strip()]
    levels = [i.strip() for i in (intensity or "").split("+") if i.strip()]
    if len(levels) < len(types):
        levels += [levels[-1] if levels else "medium"] * (len(types) - len(levels))
    out = []
    for t, level in zip(types, levels):
        if t in BACKGROUND_TYPES:
            t = "background_edit"
        out.append(Component(type=t, intensity=level))
    return out


def parse_accessory(text: str) -> Optional[str]:
    """The accessory a background request asks for, if any."""
    lowered = (text or "").lower()
    if "additionally, include" not in lowered and "add " not in lowered:
        return None
    tail = lowered.split("additionally, include", 1)[-1]
    for key, words in _ACCESSORY_KEYWORDS:
        if any(w in tail for w in words):
            return key
    return None


def parse_environment(var_type: str, text: str) -> Optional[str]:
    """indoor / outdoor for a background request.

    A legacy background_edit carries it in the description ("Indoor
    background change"). Searching the detail for "outdoor" is wrong: the
    indoor far detail ends "...with no outdoor elements".
    """
    if var_type == "background_in":
        return "indoor"
    if var_type == "background_out":
        return "outdoor"
    lowered = (text or "").lower()
    if "indoor background" in lowered:
        return "indoor"
    if "outdoor background" in lowered:
        return "outdoor"
    lowered = lowered.replace("no outdoor", "").replace("no indoor", "")
    if "indoor" in lowered:
        return "indoor"
    if "outdoor" in lowered:
        return "outdoor"
    return None


def _file_names_for(var_type: str) -> Tuple[str, ...]:
    """Filename stems a request's image may be saved under.

    The miner renames a legacy background_edit to background_in/out before
    generating (generate_variations._canonical_background_type).
    """
    if var_type in ("background_edit", "background"):
        return (var_type, "background_in", "background_out")
    return (var_type.replace("+", "_"),)


def _media_index(media: Path) -> Tuple[Optional[Path], List[Tuple[int, str, Path]]]:
    """The base image and every NN_<type>.png variation, in index order."""
    base = next(iter(sorted(media.glob("base_image__*"))), None)
    files: List[Tuple[int, str, Path]] = []
    for path in sorted(media.glob("[0-9][0-9]_*.png")):
        match = re.match(r"(\d\d)_(.+)\.png$", path.name)
        if match:
            files.append((int(match.group(1)), match.group(2), path))
    return base, files


def _pick_file(slot: int, names: Tuple[str, ...], files, used) -> Optional[Tuple[int, str, Path]]:
    """Prefer the file at this slot's own index; otherwise the next unused match."""
    for entry in files:
        if entry[0] == slot and entry[1] in names and entry[0] not in used:
            return entry
    for entry in files:
        if entry[1] in names and entry[0] not in used:
            return entry
    return None


def load_record(record_dir: Path) -> Tuple[Optional[Round], Optional[str]]:
    """A Round, or (None, reason) when the record cannot be benchmarked."""
    record_path = record_dir / "record.json"
    try:
        record = json.loads(record_path.read_text())
    except (OSError, ValueError) as e:
        return None, f"unreadable record.json ({e})"

    request = record.get("request") or {}
    variations = request.get("variation_requests") or []
    if not variations:
        return None, f"no variation requests (outcome={record.get('outcome')})"

    media = record_dir / "media"
    if not media.is_dir():
        return None, "no media/ (archived without MIID_ARCHIVE_IMAGES=1)"
    base, files = _media_index(media)
    if base is None:
        return None, "no base_image in media/"

    # variation_report uses the miner's (canonical) type names, in the order
    # the images were processed — consume per name in that order.
    reports: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for rep in record.get("variation_report") or []:
        reports[rep.get("variation_type", "").replace("+", "_")].append(rep)
    report_used: Dict[str, int] = defaultdict(int)

    generation = record.get("generation") or {}
    record_id = record_dir.name
    used_files: set = set()
    items: List[VariationItem] = []
    missing: List[str] = []

    for slot, req in enumerate(variations):
        var_type = req.get("type", "")
        picked = _pick_file(slot, _file_names_for(var_type), files, used_files)
        if picked is None:
            missing.append(var_type)
            continue
        used_files.add(picked[0])
        file_name, image = picked[1], picked[2]

        rep_list = reports.get(file_name, [])
        rep = rep_list[report_used[file_name]] if report_used[file_name] < len(rep_list) else {}
        report_used[file_name] += 1
        text = f"{req.get('description', '')} {req.get('detail', '')}"
        item = VariationItem(
            record_id=record_id,
            record_dir=record_dir,
            slot=slot,
            variation_type=var_type,
            intensity=req.get("intensity", ""),
            detail=req.get("detail", ""),
            description=req.get("description", ""),
            image=image,
            base=base,
            base_filename=request.get("image_filename"),
            status=rep.get("status"),
            model=rep.get("model") or generation.get("model_key"),
            miner_identity=rep.get("identity_similarity"),
            components=parse_components(var_type, req.get("intensity", "")),
            submitted_as=file_name,
        )
        if item.is_background:
            item.accessory = parse_accessory(text)
            item.environment = parse_environment(var_type, text)
        items.append(item)

    return Round(
        record_id=record_id,
        record_dir=record_dir,
        started_at=record.get("started_at_utc"),
        validator=(record.get("validator") or {}).get("name"),
        model=generation.get("model_key"),
        requested=len(variations),
        items=items,
        missing_slots=missing,
    ), None


def find_record_dirs(paths: List[Path]) -> List[Path]:
    """Every record directory under the given archive roots / date dirs / records."""
    found: List[Path] = []
    for path in paths:
        if (path / "record.json").is_file():
            found.append(path)
            continue
        found.extend(p.parent for p in path.rglob("record.json"))
    return sorted(set(found), key=lambda p: p.name)


def single_item(
    base: Path, image: Path, var_type: str, intensity: str, detail: str = ""
) -> VariationItem:
    """An ad-hoc item for scoring one image outside the archive."""
    item = VariationItem(
        record_id="adhoc",
        record_dir=None,
        slot=0,
        variation_type=var_type,
        intensity=intensity,
        detail=detail,
        description="",
        image=image,
        base=base,
        base_filename=base.name,
        components=parse_components(var_type, intensity),
    )
    if item.is_background:
        item.accessory = parse_accessory(detail)
        item.environment = parse_environment(var_type, detail)
    return item
