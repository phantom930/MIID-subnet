# MIID/benchmark/scoring.py
#
# Maps analyzer measurements onto the grading API's validation_score (-5..5).
#
# Sources, so a rule can be traced and retuned:
#   [sheet]   Face variation score sheet, docs/Face Variation Reward System.pdf
#             (Cycle 4, p.26) — the conditions and the score each one yields.
#   [flow]    Cycle 1 validation flowchart in the same PDF (p.44): identity
#             0.4-0.6 is "acceptable" and recommended a 1.
#   [reward]  MIID/validator/reward.py — how a score becomes KAV reward.
#   [calib]   Our own thresholds, set against archived rounds by eye. These
#             are the guesses: the sheet says *what* is detected, not the
#             numeric cut-offs the grader's detectors use.
#
# Pure functions over plain dicts — no models — so the rules can be changed
# and re-run against cached measurements in a second.

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

INTENSITY_ORDER = ["none", "light", "medium", "far"]

# Δrep per validation score — the manual-review table that drives the UAV
# (reputation) side, which carries 90% of miner emissions. [sheet p.5]
DELTA_REP = {
    6: 0.15, 5: 0.10, 4: 0.04, 3: 0.03, 2: 0.02, 1: 0.01, 0: 0.0,
    -1: -0.05, -2: -0.08, -3: -0.10, -4: -0.25, -5: -0.50,
}

T = {
    # [sheet] / [flow] / [reward]
    "identity_hard_gate": 0.40,     # below: "Face identity below threshold" -> -3
    "identity_full": 0.60,          # 5 needs >= this; also the KAV gate [reward]
    "identity_upper": 0.95,         # partial-match band tops out here
    "aspect_narrower": -0.155,      # face >15.5% narrower than seed -> -4
    "min_width": 512,               # resolution <= 512x672 -> -1
    "min_height": 672,
    "accessory_conf": 0.50,         # background accessory confidence
    # [calib]
    "face_not_dominant": 0.05,      # landmark box / frame area
    "copy_paste_hf_corr": 0.85,     # face skin texture reproduced pixel-for-pixel
    "copy_paste_warn": 0.50,
    "seed_dup_dhash": 12,           # of 256 bits to the base = same image
    "other_dup_dhash": 12,          # to another of our own submissions
    "hijab_side_cloth": 0.35,       # cloth beside the cheeks = wrap-style covering
    "crown_cloth": 0.30,            # cloth above the hairline = something worn
    "env_mismatch_prob": 0.50,
    # Pose: max(|Δyaw|, |Δroll|, 0.6*|Δpitch|) in degrees. The request bins
    # are ±15 / ±30 / >±45; cut-offs sit halfway between them.
    "pose_bins": (7.0, 22.5, 37.5),
    "pose_extra": 12.0,
    # Expression: largest blendshape change (0..1).
    "expr_bins": (0.20, 0.45, 0.70),
    "expr_extra": 0.30,
    # mouthSmile saturates (~0.75 for a closed-mouth smile and a toothy one
    # alike), so a smile is graded by what the mouth does instead: lips
    # apart over the teeth = medium, a laugh = far. Calibrated by eye on 24
    # smiles from 2026-10-07: upper_lip_up is 0–0.03 closed, 0.06–0.19 with
    # teeth, 0.64–0.72 laughing (one laugh had the jaw at only 0.06).
    "smile_teeth_lip": 0.05,
    "smile_laugh_lip": 0.40,
    "smile_laugh_jaw": 0.15,
    # Lighting: max(|ΔL|/25, |Δcontrast|/12, Δasym/0.15, cast/10) on the face.
    "light_bins": (0.60, 1.20, 2.00),
    "light_directional_asym": 0.12,  # a visible light direction is >= medium
    # Background change: histogram distance plus a non-plain scene.
    "bg_hist_changed": 0.50,
    "bg_scene_edges": 0.015,
}

GENDERED_COVERINGS = {"hijab": "f", "turban": "m", "kippah": "m", "taqiyah": "m"}


@dataclass
class Detection:
    component: str
    requested: str
    detected: str
    value: Optional[float]
    match: str  # exact / close / off / none

    def to_dict(self) -> Dict[str, Any]:
        return {
            "component": self.component,
            "requested": self.requested,
            "detected": self.detected,
            "value": None if self.value is None else round(self.value, 3),
            "match": self.match,
        }


@dataclass
class ScoreResult:
    score: int
    path: str
    identity: Optional[float]
    reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    detections: List[Detection] = field(default_factory=list)
    extras: List[str] = field(default_factory=list)
    accessory: Optional[Dict[str, Any]] = None

    @property
    def delta_rep(self) -> float:
        return DELTA_REP.get(self.score, 0.0)

    @property
    def kav_norm(self) -> float:
        return self.score / 5.0

    @property
    def passes_identity_gate(self) -> bool:
        return self.identity is not None and self.identity >= T["identity_full"]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": self.score,
            "path": self.path,
            "identity": self.identity,
            "delta_rep": self.delta_rep,
            "reasons": self.reasons,
            "warnings": self.warnings,
            "detections": [d.to_dict() for d in self.detections],
            "extras": self.extras,
            "accessory": self.accessory,
        }


# ── measurement -> intensity bin ─────────────────────────────────────────


def _bin(value: Optional[float], cuts: Tuple[float, float, float]) -> str:
    if value is None:
        return "none"
    for name, cut in zip(INTENSITY_ORDER, cuts):
        if value < cut:
            return name
    return "far"


def pose_magnitude(m: Dict[str, Any]) -> Optional[float]:
    d = m.get("pose_delta")
    if not d:
        return None
    return max(abs(d["yaw"]), abs(d["roll"]), 0.6 * abs(d["pitch"]))


# Channels that qualify a change rather than being one on their own.
EXPRESSION_HELPER_CHANNELS = ("upper_lip_up",)


def expression_magnitude(m: Dict[str, Any]) -> Tuple[Optional[float], Optional[str]]:
    d = {
        k: v for k, v in (m.get("expression_delta") or {}).items()
        if k not in EXPRESSION_HELPER_CHANNELS
    }
    if not d:
        return None, None
    channel = max(d, key=lambda k: abs(d[k]))
    return abs(d[channel]), channel


def expression_bin(m: Dict[str, Any]) -> str:
    """Expression intensity: the largest blendshape change, but a smile by shape.

    The validator describes light / medium / far as "slight smile" / "smile" /
    "laughing", which is a closed mouth / teeth showing / jaw dropped.
    """
    value, channel = expression_magnitude(m)
    level = _bin(value, T["expr_bins"])
    d = m.get("expression_delta") or {}
    if channel != "smile" or level == "none" or "upper_lip_up" not in d:
        return level
    if d.get("jaw_open", 0.0) >= T["smile_laugh_jaw"] or d["upper_lip_up"] >= T["smile_laugh_lip"]:
        return "far"
    if d["upper_lip_up"] >= T["smile_teeth_lip"]:
        return "medium"
    return "light"


def lighting_magnitude(m: Dict[str, Any]) -> Optional[float]:
    d = m.get("lighting_delta")
    if not d:
        return None
    return max(
        abs(d["brightness"]) / 25.0,
        abs(d["contrast"]) / 12.0,
        d["asym"] / 0.15,
        d["cast"] / 10.0,
    )


def lighting_bin(m: Dict[str, Any]) -> str:
    level = _bin(lighting_magnitude(m), T["light_bins"])
    d = m.get("lighting_delta") or {}
    if d.get("asym", 0.0) >= T["light_directional_asym"] and level in ("none", "light"):
        level = "medium"
    return level


def lighting_is_dramatic(m: Dict[str, Any]) -> bool:
    """An unrequested lighting change a reviewer would call a variation.

    Generators brighten faces by 20-40 L* almost everywhere; that alone reads
    as exposure, not as a lighting edit, so it is not counted as an extra.
    """
    d = m.get("lighting_delta") or {}
    return bool(d) and (
        d["asym"] >= 0.20
        or abs(d["contrast"]) >= 15
        or d["cast"] >= 12
        or abs(d["brightness"]) >= 45
    )


def background_changed(m: Dict[str, Any]) -> bool:
    d = m.get("background_delta") or {}
    if not d.get("valid"):
        return False
    var_bg = (m.get("variation") or {}).get("background") or {}
    return d["hist_distance"] >= T["bg_hist_changed"] and not var_bg.get("plain_white", False)


def background_scene_changed(m: Dict[str, Any]) -> bool:
    """A new scene, not just a darker or tinted backdrop."""
    var_bg = (m.get("variation") or {}).get("background") or {}
    return background_changed(m) and var_bg.get("edge_density", 0.0) >= T["bg_scene_edges"]


def _compare(requested: str, detected: str) -> str:
    if detected == "none":
        return "none"
    gap = abs(INTENSITY_ORDER.index(requested) - INTENSITY_ORDER.index(detected))
    return "exact" if gap == 0 else "close" if gap == 1 else "off"


def detect_component(component: str, requested: str, m: Dict[str, Any]) -> Detection:
    if requested not in INTENSITY_ORDER:
        requested = "medium"
    if component == "pose_edit":
        value = pose_magnitude(m)
        detected = _bin(value, T["pose_bins"])
    elif component == "expression_edit":
        value, _ = expression_magnitude(m)
        detected = expression_bin(m)
    elif component == "lighting_edit":
        value = lighting_magnitude(m)
        detected = lighting_bin(m)
    elif component == "background_edit":
        value = (m.get("background_delta") or {}).get("hist_distance")
        detected = requested if background_changed(m) else "none"
    else:
        return Detection(component, requested, "unknown", None, "none")
    return Detection(component, requested, detected, value, _compare(requested, detected))


def unrequested_variations(requested: Iterable[str], m: Dict[str, Any]) -> List[str]:
    """Variation types present in the image that the request did not ask for."""
    wanted = set(requested)
    extras = []
    pose = pose_magnitude(m)
    if "pose_edit" not in wanted and pose is not None and pose >= T["pose_extra"]:
        extras.append(f"pose_edit ({pose:.0f}°)")
    expr, channel = expression_magnitude(m)
    if "expression_edit" not in wanted and expr is not None and expr >= T["expr_extra"]:
        extras.append(f"expression_edit ({channel} {expr:.2f})")
    if "lighting_edit" not in wanted and lighting_is_dramatic(m):
        extras.append("lighting_edit")
    if "background_edit" not in wanted and background_scene_changed(m):
        extras.append("background_edit")
    return extras


# ── accessory ────────────────────────────────────────────────────────────


def accessory_verdict(requested: str, m: Dict[str, Any]) -> Dict[str, Any]:
    """Is the requested accessory there, something else, or nothing? [calib]

    Presence comes from segmentation (cloth on the crown, or wrapped around
    the cheeks) with CLIP as a fallback; CLIP then names what it is among the
    non-"none" options. Cloth around the cheeks and jaw is a wrap-style
    covering whatever CLIP calls it — a bandana or cap never reaches there.
    """
    groups = m.get("accessory_groups") or {}
    probs = m.get("religious_style_probs") or {}
    head = (m.get("variation") or {}).get("head") or {}
    if not groups:
        return {"requested": requested, "status": "unchecked"}

    side_cloth = head.get("side_cloth", 0.0)
    crown_cloth = head.get("crown_cloth", 0.0)
    present = (
        crown_cloth >= T["crown_cloth"]
        or side_cloth >= T["hijab_side_cloth"]
        or groups.get("none", 0.0) < 0.5
    )
    worn = {k: v for k, v in groups.items() if k != "none"}
    total = sum(worn.values()) or 1.0
    worn = {k: v / total for k, v in worn.items()}
    top_group = max(worn, key=worn.get)
    wrapped = side_cloth >= T["hijab_side_cloth"] and top_group in (
        "religious_head_covering", "bandana",
    )
    # No benefit of the doubt for CLIP's confusions (turban read as bandana,
    # taqiyah as a beanie): the grader's own classifier makes them too. A
    # turban that raw CLIP called 94% bandana graded -1 on 2026-10-07, so a
    # lenient override here only hid the real loss.
    if wrapped:
        top_group = "religious_head_covering"
        worn[top_group] = max(worn[top_group], 0.9)
    conf = worn.get(requested, 0.0) if present else groups.get(requested, 0.0)
    verdict: Dict[str, Any] = {
        "requested": requested,
        "present": present,
        "confidence": round(conf, 3),
        "top": top_group if present else "none",
        "top_confidence": round(worn[top_group] if present else groups.get("none", 0.0), 3),
        "crown_cloth": round(crown_cloth, 3),
    }
    if not present:
        verdict["status"] = "insufficient"
    elif top_group == requested and conf >= T["accessory_conf"]:
        verdict["status"] = "match"
    elif top_group != requested and worn[top_group] >= T["accessory_conf"]:
        verdict["status"] = "other"
    else:
        verdict["status"] = "insufficient"

    # Only name the covering when one is actually there — a beanie drawn for
    # a religious-covering request is a wrong accessory, not a gender issue.
    if present and top_group == "religious_head_covering" and worn[top_group] >= T["accessory_conf"]:
        if side_cloth >= T["hijab_side_cloth"]:
            style = "hijab"
        else:
            # Hijab stays a candidate: a loose one can sit just under the
            # side-cloth cut-off (0.33) while CLIP is sure of it (0.95).
            styles = {k: probs.get(k, 0.0) for k in ("hijab", "turban", "kippah", "taqiyah")}
            style = max(styles, key=styles.get)
        verdict["covering_style"] = style
        verdict["side_cloth"] = round(side_cloth, 3)
        gender = m.get("base_gender")
        expected = GENDERED_COVERINGS.get(style)
        if gender and expected and gender != expected:
            verdict["gender_mismatch"] = f"{style} on a {'male' if gender == 'm' else 'female'} subject"
    return verdict


# ── the score sheet ──────────────────────────────────────────────────────


def score_variation(item, m: Dict[str, Any], duplicates: Optional[Dict[str, Any]] = None) -> ScoreResult:
    """Predicted validation_score for one submitted image. [sheet]"""
    identity = m.get("identity")
    base = m.get("base") or {}
    var = m.get("variation") or {}
    path = "background_accessory" if item.is_background else "standard"
    result = ScoreResult(score=0, path=path, identity=identity)
    penalties: List[Tuple[int, str]] = []

    # ── hard checks, most severe wins ──
    if var.get("sha256") and var.get("sha256") == base.get("sha256"):
        penalties.append((-1, "duplicate of the seed image (hash match)"))
    elif duplicates and duplicates.get("seed_dhash") is not None \
            and duplicates["seed_dhash"] <= T["seed_dup_dhash"]:
        penalties.append((-1, f"near-duplicate of the seed (dHash {duplicates['seed_dhash']})"))

    if duplicates and duplicates.get("same_hash_as"):
        penalties.append((-3, f"identical to another submission: {duplicates['same_hash_as']}"))
    elif duplicates and duplicates.get("near_dup_of"):
        penalties.append((-2, f"near-duplicate of another submission: {duplicates['near_dup_of']}"))

    width, height = var.get("size") or (0, 0)
    if width <= T["min_width"] or height <= T["min_height"]:
        penalties.append((-1, f"resolution {width}x{height} at or below 512x672"))

    faces = var.get("face_count", 0)
    if faces != 1 and not (faces == 0 and var.get("landmark_faces") == 1):
        penalties.append((-3, f"MediaPipe face count {faces}, needs exactly 1"))
    elif faces == 0:
        result.warnings.append("face detector missed the face (landmarker found it)")

    if identity is None:
        penalties.append((-3, "AdaFace found no face to compare"))
    elif identity < T["identity_hard_gate"]:
        penalties.append((-3, f"identity {identity:.3f} below the 0.4 hard gate"))

    wh = m.get("face_wh_delta")
    if wh is not None and wh < T["aspect_narrower"]:
        penalties.append((-4, f"face {abs(wh) * 100:.1f}% narrower than the seed (aspect distortion)"))

    area = var.get("face_area")
    if area is not None and area < T["face_not_dominant"]:
        penalties.append((-1, f"face not dominant ({area * 100:.1f}% of the frame)"))

    cp = m.get("copy_paste") or {}
    hf = cp.get("hf_corr")
    if hf is not None and hf >= T["copy_paste_hf_corr"]:
        penalties.append((-5, f"face pixels copied from the seed (texture corr {hf:.2f})"))
    elif hf is not None and hf >= T["copy_paste_warn"]:
        result.warnings.append(
            f"copy-paste risk: face texture corr {hf:.2f} with the seed — "
            "the edit kept the seed's skin pixels"
        )

    # ── variation / accessory path ──
    if item.is_background:
        score = _score_background(item, m, result, penalties)
    else:
        score = _score_standard(item, m, result)

    if penalties:
        worst = min(p[0] for p in penalties)
        result.reasons = [r for s, r in sorted(penalties)] + result.reasons
        score = min(score, worst)

    if identity is not None and identity > T["identity_upper"] and score >= 3:
        result.warnings.append(
            f"identity {identity:.3f} above 0.95 — the grader may read this as an unchanged image"
        )
    result.score = int(score)
    return result


def _score_standard(item, m, result: ScoreResult) -> int:
    identity = result.identity or 0.0
    requested = [c.type for c in item.components]
    result.detections = [detect_component(c.type, c.intensity, m) for c in item.components]
    result.extras = unrequested_variations(requested, m)

    matches = [d.match for d in result.detections]
    full = bool(matches) and all(x == "exact" for x in matches)
    partial = any(x in ("exact", "close", "off") for x in matches)

    if full and not result.extras:
        score = 5
        result.reasons.append("full variation match")
    elif full:
        score = 3
        result.reasons.append(f"requested variation present plus unrequested {', '.join(result.extras)}")
    elif partial:
        score = 3
        misses = [
            f"{d.component} wanted {d.requested}, got {d.detected}"
            for d in result.detections if d.match != "exact"
        ]
        result.reasons.append("partial match: " + "; ".join(misses))
    elif result.extras:
        score = 0
        result.reasons.append(f"label mismatch: detected {', '.join(result.extras)} instead")
    else:
        score = -2
        result.reasons.append("no variation detected")

    if score > 1 and identity < T["identity_full"]:
        result.reasons.append(f"identity {identity:.3f} in 0.4-0.6 caps the score at 1 [flow]")
        score = 1
    return score


def _score_background(item, m, result: ScoreResult, penalties) -> int:
    identity = result.identity or 0.0
    changed = background_changed(m)
    result.detections = [
        Detection("background_edit", item.intensity or "medium",
                  "changed" if changed else "none",
                  (m.get("background_delta") or {}).get("hist_distance"),
                  "exact" if changed else "none")
    ]

    labelled = {"background_in": "indoor", "background_out": "outdoor"}.get(item.submitted_as or "")
    if item.environment and labelled and labelled != item.environment:
        result.warnings.append(
            f"miner submitted this {item.environment} request as {item.submitted_as} "
            "(see generate_variations._canonical_background_type)"
        )

    env = m.get("environment") or {}
    if item.environment and env:
        other = "outdoor" if item.environment == "indoor" else "indoor"
        if env.get(other, 0.0) >= T["env_mismatch_prob"]:
            result.warnings.append(f"background reads {other} ({env[other]:.2f}), request was {item.environment}")
        elif env.get("studio", 0.0) >= max(env.get(item.environment, 0.0), 0.5):
            result.warnings.append(f"background reads as a plain studio backdrop ({env['studio']:.2f})")

    if not item.accessory:
        if changed:
            result.reasons.append("background changed (no accessory requested)")
            return 5 if identity >= T["identity_full"] else 3
        result.reasons.append("no background change detected")
        return -2

    verdict = accessory_verdict(item.accessory, m)
    result.accessory = verdict
    status = verdict.get("status")
    if verdict.get("gender_mismatch"):
        penalties.append((-3, f"religious head covering does not match the seed's gender: {verdict['gender_mismatch']}"))

    if status == "match":
        if not changed:
            result.reasons.append("accessory matches but no background change detected")
            return 4
        if identity >= T["identity_full"]:
            result.reasons.append(f"accessory {item.accessory} matches ({verdict['confidence']:.2f}) and background changed")
            return 5
        result.reasons.append(f"accessory matches but identity {identity:.3f} < 0.6")
        return 3
    if status == "other":
        result.reasons.append(f"wrong accessory: {verdict['top']} ({verdict['top_confidence']:.2f}) instead of {item.accessory}")
        return 3 if changed else 2
    if status == "unchecked":
        result.reasons.append(
            "background changed, accessory not checked" if changed else "no background change detected"
        )
        result.warnings.append("accessory not checked (--no-clip)")
        return 5 if changed and identity >= T["identity_full"] else 3 if changed else -2
    # The grader's -1 ("insufficient match and no background detected") hit a
    # turban in front of a white-walled gallery that our histogram check calls
    # changed (0.97) but that has almost no scene detail (edges 0.006): judge
    # this branch on a visible scene, not on a colour shift alone.
    result.reasons.append(f"accessory {item.accessory} not found ({verdict.get('confidence', 0):.2f})")
    return 0 if background_scene_changed(m) else -1


# ── duplicates across the benchmarked set ────────────────────────────────


def find_duplicates(entries: List[Tuple[str, Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
    """Seed / cross-submission duplicate checks over every scored image.

    entries: (label, metrics) pairs. Only our own submissions are visible
    here; the grader also checks against every other miner's.
    """
    out: Dict[str, Dict[str, Any]] = {label: {} for label, _ in entries}
    seen_hash: Dict[str, str] = {}
    hashes: List[Tuple[str, int]] = []
    for label, m in entries:
        var = m.get("variation") or {}
        base = m.get("base") or {}
        try:
            vd = int(var.get("dhash", "0"), 16)
            bd = int(base.get("dhash", "0"), 16)
            out[label]["seed_dhash"] = bin(vd ^ bd).count("1")
        except (TypeError, ValueError):
            vd = None
        sha = var.get("sha256")
        if sha in seen_hash:
            out[label]["same_hash_as"] = seen_hash[sha]
        elif sha:
            seen_hash[sha] = label
        if vd is not None:
            for other_label, other in hashes:
                if bin(vd ^ other).count("1") <= T["other_dup_dhash"]:
                    out[label]["near_dup_of"] = other_label
                    break
            hashes.append((label, vd))
    return out


# ── round / run aggregation ──────────────────────────────────────────────


def summarize_round(requested: int, results: List[ScoreResult], missing: int) -> Dict[str, Any]:
    """Expected KAV for one round in the validator's sampled-slot mode. [reward]

    The validator grades ONE random slot per miner per round and gates it on
    that slot's identity >= 0.6, so the expectation is a mean over slots. A
    missing slot is a 0 with identity 0.
    """
    slots = max(requested, len(results) + missing, 1)
    norms = [r.kav_norm for r in results] + [0.0] * missing
    gated = [r.kav_norm if r.passes_identity_gate else 0.0 for r in results] + [0.0] * missing
    return {
        "slots": slots,
        "scores": [r.score for r in results] + [None] * missing,
        "expected_validation_norm": sum(norms) / slots,
        "expected_gated_norm": sum(gated) / slots,
        "identity_gate_pass_rate": sum(1 for r in results if r.passes_identity_gate) / slots,
        "delta_rep": sum(r.delta_rep for r in results),
    }


def summarize_run(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregates over every scored variation (rows from the CLI)."""
    if not rows:
        return {}
    hist = Counter(r["score"] for r in rows)
    by_model: Dict[str, List[int]] = defaultdict(list)
    by_slot: Dict[str, List[int]] = defaultdict(list)
    reasons: Counter = Counter()
    for r in rows:
        by_model[r.get("model") or "?"].append(r["score"])
        by_slot[r["slot_type"]].append(r["score"])
        for reason in r["reasons"]:
            reasons[_reason_key(reason)] += 1

    def stats(scores: List[int]) -> Dict[str, Any]:
        return {
            "n": len(scores),
            "mean_score": sum(scores) / len(scores),
            "mean_delta_rep": sum(DELTA_REP[s] for s in scores) / len(scores),
            "share_5": sum(1 for s in scores if s == 5) / len(scores),
            "share_negative": sum(1 for s in scores if s < 0) / len(scores),
        }

    return {
        "n": len(rows),
        "mean_score": sum(r["score"] for r in rows) / len(rows),
        "mean_delta_rep": sum(DELTA_REP[r["score"]] for r in rows) / len(rows),
        "histogram": {s: hist.get(s, 0) for s in range(-5, 6)},
        "by_model": {k: stats(v) for k, v in sorted(by_model.items())},
        "by_slot": {k: stats(v) for k, v in sorted(by_slot.items())},
        "top_reasons": reasons.most_common(12),
    }


def _reason_key(reason: str) -> str:
    """Collapse numbers so identical failure modes count together."""
    import re

    key = re.sub(r"\(.*?\)", "", reason)
    key = re.sub(r"[-+]?\d+(\.\d+)?%?", "#", key)
    return re.sub(r"\s+", " ", key).strip()
