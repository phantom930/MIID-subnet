# MIID/miner/request_archive.py
#
# Phase 4: on-disk archive of every validator approach the miner handles.
#
# One record per incoming request, written whatever the outcome — a served
# submission, an empty response, a rejected request, or a crash. The record
# holds what the validator asked for and what the miner sent back, so an
# operator can answer "what did I actually submit for challenge X" after the
# fact without re-reading logs.
#
# Nothing here is on the critical path: every entry point swallows its own
# errors, because failing to archive must never fail a response to a validator.
#
# Image bytes are NOT stored by default — a round's variations run to tens of
# megabytes and the miner answers a validator roughly hourly. Set
# MIID_ARCHIVE_IMAGES=1 to keep them too.
#
# Records older than 24 hours are deleted whenever a new one is written
# (MIID_ARCHIVE_RETENTION_HOURS; 0 keeps them until the record cap).

import base64
import hashlib
import json
import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import bittensor as bt


SCHEMA_VERSION = 1

DEFAULT_ARCHIVE_DIRNAME = "miner_requests"

# Records live in the checkout, next to the code that produced them, rather
# than under ~/.bittensor — an operator looking for "what did I submit" should
# find it in the project they are working in. parents[2] is the repo root
# (MIID/miner/request_archive.py -> MIID/miner -> MIID -> root), which holds
# for the editable installs both setup scripts create. Point
# MIID_REQUEST_ARCHIVE elsewhere if you deploy the package some other way.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARCHIVE_DIR = PROJECT_ROOT / DEFAULT_ARCHIVE_DIRNAME

# Keep this many request directories, newest first. 0 disables pruning.
DEFAULT_MAX_RECORDS = 500

# Drop request directories older than this many hours (a rolling window, not
# a calendar day, so the archive is never emptied at midnight UTC). With
# MIID_ARCHIVE_IMAGES=1 a day of rounds is already a few hundred megabytes.
# 0 disables age-based pruning.
DEFAULT_RETENTION_HOURS = 24.0


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def archive_enabled() -> bool:
    """Whether requests are archived at all (MIID_ARCHIVE_ENABLED, default on)."""
    return _env_flag("MIID_ARCHIVE_ENABLED", True)


def archive_images_enabled() -> bool:
    """Whether the actual media is kept alongside the record (default off)."""
    return _env_flag("MIID_ARCHIVE_IMAGES", False)


def max_records() -> int:
    """How many request directories to keep (MIID_ARCHIVE_MAX_RECORDS)."""
    try:
        return max(0, int(os.environ.get("MIID_ARCHIVE_MAX_RECORDS", DEFAULT_MAX_RECORDS)))
    except (TypeError, ValueError):
        return DEFAULT_MAX_RECORDS


def retention_hours() -> float:
    """How long request directories are kept (MIID_ARCHIVE_RETENTION_HOURS)."""
    try:
        return max(0.0, float(
            os.environ.get("MIID_ARCHIVE_RETENTION_HOURS", DEFAULT_RETENTION_HOURS)
        ))
    except (TypeError, ValueError):
        return DEFAULT_RETENTION_HOURS


def _started_at(directory: Path) -> Optional[datetime]:
    """Start time encoded in a request directory's name (see record_dir)."""
    try:
        return datetime.strptime(
            directory.name.split("__", 1)[0], "%Y%m%dT%H%M%SZ"
        ).replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def resolve_archive_dir(override: Optional[str] = None) -> Path:
    """Where records land: ``<repo>/miner_requests`` unless told otherwise.

    An explicit ``override`` (a CLI flag) beats MIID_REQUEST_ARCHIVE, which
    beats the in-project default. Both overrides are used exactly as given —
    no subfolder is appended.
    """
    if override:
        return Path(override).expanduser()
    env_override = os.environ.get("MIID_REQUEST_ARCHIVE", "").strip()
    if env_override:
        return Path(env_override).expanduser()
    return DEFAULT_ARCHIVE_DIR


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _digest_b64(payload: Optional[str]) -> Optional[Dict[str, Any]]:
    """SHA-256 and byte length of a base64 field, without keeping the bytes."""
    if not payload:
        return None
    try:
        raw = base64.b64decode(payload)
    except Exception:
        return {"sha256": None, "bytes": None, "decode_error": True}
    return {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def _variation_requests(image_request: Any) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for var in getattr(image_request, "variation_requests", None) or []:
        out.append({
            "type": getattr(var, "type", None),
            "intensity": getattr(var, "intensity", None),
            "description": getattr(var, "description", ""),
            "detail": getattr(var, "detail", ""),
        })
    return out


def describe_image_request(image_request: Any) -> Dict[str, Any]:
    """Everything the validator asked for, minus the base64 blobs themselves."""
    if image_request is None:
        return {}

    instructions = getattr(image_request, "real_screen_replay_instructions", None)
    return {
        "challenge_id": getattr(image_request, "challenge_id", None),
        "image_filename": getattr(image_request, "image_filename", None),
        "base_image": _digest_b64(getattr(image_request, "base_image", None)),
        "variation_requests": _variation_requests(image_request),
        "variation_count": len(getattr(image_request, "variation_requests", None) or []),
        "target_drand_round": getattr(image_request, "target_drand_round", None),
        "reveal_timestamp": getattr(image_request, "reveal_timestamp", None),
        "daily_seed": {
            "filename": getattr(image_request, "daily_seed_filename", None),
            "date": getattr(image_request, "daily_seed_date", None),
            "image": _digest_b64(getattr(image_request, "daily_seed_image", None)),
        },
        "tomorrow_seed": {
            "filename": getattr(image_request, "tomorrow_seed_filename", None),
            "date": getattr(image_request, "tomorrow_seed_date", None),
            "image": _digest_b64(getattr(image_request, "tomorrow_seed_image", None)),
        },
        "screen_replay_instructions_present": bool(instructions),
        "screen_replay_instructions_chars": len(instructions) if instructions else 0,
    }


def submission_to_dict(submission: Any) -> Dict[str, Any]:
    """Serialize an S3Submission across pydantic versions."""
    if isinstance(submission, dict):
        return submission
    if hasattr(submission, "model_dump"):
        return submission.model_dump()
    if hasattr(submission, "dict"):
        return submission.dict()
    return dict(submission)


def new_record(
    source: str,
    *,
    image_request: Any = None,
    validator_hotkey: Optional[str] = None,
    validator_name: Optional[str] = None,
    validator_uid: Optional[int] = None,
    miner_hotkey: Optional[str] = None,
) -> Dict[str, Any]:
    """Open a record for one approach. ``source`` is "miner" or "dry_run"."""
    started = _utc_now()
    return {
        "schema_version": SCHEMA_VERSION,
        "source": source,
        "record_id": uuid.uuid4().hex[:8],
        "started_at_utc": _iso(started),
        "_started_monotonic": started.timestamp(),
        "outcome": "incomplete",
        "error": None,
        "validator": {
            "hotkey": validator_hotkey,
            "name": validator_name,
            "uid": validator_uid,
        },
        "miner": {"hotkey": miner_hotkey},
        "request": describe_image_request(image_request),
        "generation": {},
        "variation_report": [],
        "submissions": [],
        "submission_count": 0,
        "screen_replay": {"attached": False},
        "archived_media": [],
    }


def record_outcome(
    record: Dict[str, Any],
    *,
    outcome: str,
    submissions: Optional[List[Any]] = None,
    report: Optional[List[Dict[str, Any]]] = None,
    meta: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
    screen_replay: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Close a record. Safe to call on a partially-filled one."""
    if not isinstance(record, dict):
        return record

    finished = _utc_now()
    record["outcome"] = outcome
    record["finished_at_utc"] = _iso(finished)
    started = record.pop("_started_monotonic", None)
    if started is not None:
        record["duration_seconds"] = round(finished.timestamp() - started, 2)
    if error is not None:
        record["error"] = error
    if meta:
        record["generation"] = meta
    if report is not None:
        record["variation_report"] = report
    if submissions is not None:
        serialized = [submission_to_dict(s) for s in submissions]
        record["submissions"] = serialized
        record["submission_count"] = len(serialized)
    if screen_replay is not None:
        record["screen_replay"] = screen_replay
    return record


def record_dir(record: Dict[str, Any], root: Path) -> Path:
    """Per-request directory: ``<root>/<UTC date>/<epoch>__<challenge>__<id>``."""
    started = record.get("started_at_utc", "")
    day = started[:10] or _utc_now().strftime("%Y-%m-%d")
    challenge = record.get("request", {}).get("challenge_id") or "no_challenge"
    safe_challenge = "".join(
        c if (c.isalnum() or c in "-_") else "_" for c in str(challenge)
    )[:64]
    stamp = started.replace(":", "").replace("-", "")
    return root / day / f"{stamp}__{safe_challenge}__{record.get('record_id', 'x')}"


def media_dir(record: Dict[str, Any], root: Path) -> Optional[Path]:
    """Directory for this record's media, or None when image archiving is off."""
    if not (archive_enabled() and archive_images_enabled()):
        return None
    try:
        target = record_dir(record, root) / "media"
        target.mkdir(parents=True, exist_ok=True)
        return target
    except Exception as e:
        bt.logging.debug(f"request_archive: could not create media dir: {e}")
        return None


def save_request_media(record: Dict[str, Any], image_request: Any, root: Path) -> None:
    """Write the base image and both IOTD seeds when image archiving is on."""
    target = media_dir(record, root)
    if target is None or image_request is None:
        return

    wanted = (
        ("base_image", getattr(image_request, "base_image", None),
         getattr(image_request, "image_filename", None) or "base_image.png"),
        ("daily_seed", getattr(image_request, "daily_seed_image", None),
         getattr(image_request, "daily_seed_filename", None) or "daily_seed.png"),
        ("tomorrow_seed", getattr(image_request, "tomorrow_seed_image", None),
         getattr(image_request, "tomorrow_seed_filename", None) or "tomorrow_seed.png"),
    )

    for label, payload, filename in wanted:
        if not payload:
            continue
        try:
            raw = base64.b64decode(payload)
            safe = "".join(
                c if (c.isalnum() or c in "-_.") else "_" for c in str(filename)
            )[:96]
            path = target / f"{label}__{safe}"
            path.write_bytes(raw)
            record.setdefault("archived_media", []).append(
                str(path.relative_to(record_dir(record, root)))
            )
        except Exception as e:
            bt.logging.debug(f"request_archive: could not save {label}: {e}")


def write_record(record: Dict[str, Any], root: Path) -> Optional[Path]:
    """Persist the record. Returns the file path, or None when it was skipped."""
    if not archive_enabled():
        return None
    try:
        target = record_dir(record, root)
        target.mkdir(parents=True, exist_ok=True)

        # Any media written earlier already sits under this directory.
        generated = target / "media"
        if generated.is_dir():
            record["archived_media"] = sorted(
                str(p.relative_to(target)) for p in generated.rglob("*") if p.is_file()
            )

        record.pop("_started_monotonic", None)
        path = target / "record.json"
        path.write_text(json.dumps(record, indent=2, default=str))
        bt.logging.info(
            f"Archived request {record.get('request', {}).get('challenge_id') or '(none)'} "
            f"-> {path} (outcome={record.get('outcome')}, "
            f"submissions={record.get('submission_count', 0)})"
        )
        prune(root)
        return path
    except Exception as e:
        bt.logging.warning(f"request_archive: could not write record: {e}")
        return None


def prune(
    root: Path, keep: Optional[int] = None, max_age_hours: Optional[float] = None,
) -> None:
    """Drop request directories past the age limit or beyond the count limit."""
    keep = max_records() if keep is None else keep
    max_age_hours = retention_hours() if max_age_hours is None else max_age_hours
    if keep <= 0 and max_age_hours <= 0:
        return
    try:
        dirs = sorted(
            (p for p in root.glob("*/*") if p.is_dir()),
            key=lambda p: p.name,
            reverse=True,
        )
        stale = dirs[keep:] if keep > 0 else []
        if max_age_hours > 0:
            cutoff = _utc_now().timestamp() - max_age_hours * 3600
            for directory in dirs[:len(dirs) - len(stale)]:
                started = _started_at(directory)
                # A name that does not parse is not ours to judge by age.
                if started is not None and started.timestamp() < cutoff:
                    stale.append(directory)
        for directory in stale:
            shutil.rmtree(directory, ignore_errors=True)
        # Tidy up day folders left empty by the sweep.
        for day in root.glob("*"):
            if day.is_dir() and not any(day.iterdir()):
                day.rmdir()
    except Exception as e:
        bt.logging.debug(f"request_archive: prune failed: {e}")


def archive_request(
    source: str,
    root: Path,
    *,
    image_request: Any = None,
    outcome: str,
    submissions: Optional[List[Any]] = None,
    report: Optional[List[Dict[str, Any]]] = None,
    meta: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
    validator_hotkey: Optional[str] = None,
    validator_name: Optional[str] = None,
    miner_hotkey: Optional[str] = None,
) -> Optional[Path]:
    """One-shot helper for callers that do not need the two-step API."""
    record = new_record(
        source,
        image_request=image_request,
        validator_hotkey=validator_hotkey,
        validator_name=validator_name,
        miner_hotkey=miner_hotkey,
    )
    record_outcome(
        record,
        outcome=outcome,
        submissions=submissions,
        report=report,
        meta=meta,
        error=error,
    )
    if archive_images_enabled():
        save_request_media(record, image_request, root)
    return write_record(record, root)
