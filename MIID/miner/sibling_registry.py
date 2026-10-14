# MIID/miner/sibling_registry.py
#
# A shared ledger of what sibling hotkeys have generated for a challenge, so
# that no two hotkeys run by the same operator ever submit near-identical
# images.
#
# Why: the Face Variation Reward System (docs/, "Duplicate Submissions and
# Multi-Hotkey Activity") describes a cross-miner duplication analysis in
# which five hotkeys of one operator submitting "nearly identical images"
# were all placed on the Watch tier with their reputation near zero, and
# stricter handling is promised for later cycles. Fresh random seeds already
# make two renders of the same request different images (archived variations
# sit 68–120 of 256 dHash bits from their own seed image), but nothing
# *checked* that before this ledger. Now every attempt is compared with what
# the other hotkeys have already produced for the same challenge and slot,
# and one that lands too close is treated as a miss and re-rolled with a hint
# (image_generator._Slot, generate_variations.DUPLICATE_HINTS).
#
# Mechanism: a directory (MIID_SIBLING_DIR) every sibling process can read
# and write — a local path when the hotkeys share a machine, a network mount
# when they do not. One small JSON file per attempt:
#   <dir>/<challenge_id>/<hotkey>__<variation_type>__<attempt>.json
# holding the attempt's 256-bit dHash (the benchmark's own, from
# MIID/benchmark/analyzers.py) and its CLIP image embedding. Writes are
# atomic (temp file + rename), readers tolerate half-written siblings, and
# challenges older than MIID_SIBLING_TTL_HOURS are pruned on the way through.
#
# "Too close": dHash within MIID_DUP_DHASH_BITS (16; the benchmark treats
# <= 12 as the same image) or CLIP cosine at or above MIID_DUP_CLIP_COSINE
# (0.99: two correct answers to one request share the face and the framing,
# so the bar sits just under identical). Both are estimates of what the
# grader's own analysis calls near-identical, which is not published. Unset
# MIID_SIBLING_DIR disables everything; a single hotkey needs none of this.

import json
import os
import re
import shutil
import tempfile
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

DIR = os.environ.get("MIID_SIBLING_DIR", "").strip()
TTL_HOURS = float(os.environ.get("MIID_SIBLING_TTL_HOURS", "24"))
DUP_DHASH_BITS = int(os.environ.get("MIID_DUP_DHASH_BITS", "16"))
DUP_CLIP_COSINE = float(os.environ.get("MIID_DUP_CLIP_COSINE", "0.99"))

_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")


def enabled() -> bool:
    return bool(DIR)


def dhash(image: Image.Image, size: int = 16) -> int:
    """256-bit difference hash, identical to the benchmark's."""
    gray = image.convert("L").resize((size + 1, size), Image.LANCZOS)
    px = np.asarray(gray, dtype=np.int16)
    bits = (px[:, 1:] > px[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def fingerprint(image: Image.Image) -> Tuple[int, Optional[List[float]]]:
    """(dHash, CLIP embedding or None) of one attempt."""
    try:
        from MIID.miner.background_check import image_embedding
        embedding = image_embedding(image)
    except Exception:
        embedding = None
    return dhash(image), embedding


def cosine(a: Optional[List[float]], b: Optional[List[float]]) -> Optional[float]:
    if not a or not b or len(a) != len(b):
        return None
    va, vb = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    return float(va @ vb / denom) if denom else None


@dataclass
class Match:
    """The closest thing a sibling has produced for the same slot."""
    hotkey: str
    variation_type: str
    attempt: int
    bits: Optional[int]
    cosine: Optional[float]

    @property
    def too_close(self) -> bool:
        if self.bits is not None and self.bits <= DUP_DHASH_BITS:
            return True
        return self.cosine is not None and self.cosine >= DUP_CLIP_COSINE

    def as_dict(self) -> Dict[str, Any]:
        return {
            "hotkey": self.hotkey, "variation_type": self.variation_type,
            "attempt": self.attempt, "bits": self.bits,
            "cosine": None if self.cosine is None else round(self.cosine, 4),
            "too_close": self.too_close,
        }

    def describe(self) -> str:
        parts = []
        if self.bits is not None:
            parts.append(f"dHash {self.bits} bits")
        if self.cosine is not None:
            parts.append(f"CLIP cosine {self.cosine:.3f}")
        return f"sibling {self.hotkey[:8]} attempt {self.attempt} ({', '.join(parts)})"


class Context:
    """This hotkey's view of one challenge in the shared ledger."""

    def __init__(self, root: str, challenge_id: str, hotkey: str):
        self.root = root
        self.hotkey = hotkey
        self.challenge_dir = os.path.join(root, _SAFE.sub("_", challenge_id) or "challenge")

    def _entry_name(self, variation_type: str, attempt: int) -> str:
        return f"{self.hotkey}__{_SAFE.sub('_', variation_type)}__{attempt}.json"

    def register(
        self, variation_type: str, attempt: int,
        fingerprint_value: Tuple[int, Optional[List[float]]],
    ) -> None:
        """Record one attempt so siblings can steer away from it."""
        os.makedirs(self.challenge_dir, exist_ok=True)
        image_hash, embedding = fingerprint_value
        payload = {
            "hotkey": self.hotkey, "variation_type": variation_type,
            "attempt": attempt, "dhash": f"{image_hash:064x}",
            "clip": None if embedding is None else [round(float(x), 5) for x in embedding],
            "ts": time.time(),
        }
        fd, tmp = tempfile.mkstemp(prefix=".tmp-", dir=self.challenge_dir)
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(payload, f)
            os.replace(tmp, os.path.join(self.challenge_dir, self._entry_name(variation_type, attempt)))
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def siblings(self, variation_type: str) -> List[Dict[str, Any]]:
        """Every other hotkey's entries for this slot type."""
        wanted = _SAFE.sub("_", variation_type)
        out = []
        try:
            names = os.listdir(self.challenge_dir)
        except FileNotFoundError:
            return out
        for name in names:
            if name.startswith(".") or not name.endswith(".json"):
                continue
            parts = name[:-5].split("__")
            if len(parts) != 3 or parts[0] == self.hotkey or parts[1] != wanted:
                continue
            try:
                with open(os.path.join(self.challenge_dir, name)) as f:
                    out.append(json.load(f))
            except Exception:
                continue  # half-written or gone: a sibling mid-write
        return out

    def closest(
        self, variation_type: str,
        fingerprint_value: Tuple[int, Optional[List[float]]],
    ) -> Optional[Match]:
        """The sibling attempt nearest to ``fingerprint_value``, or None."""
        image_hash, embedding = fingerprint_value
        best: Optional[Match] = None
        for entry in self.siblings(variation_type):
            try:
                bits = hamming(image_hash, int(entry["dhash"], 16))
            except (KeyError, ValueError, TypeError):
                bits = None
            cos = cosine(embedding, entry.get("clip"))
            match = Match(
                hotkey=str(entry.get("hotkey", "?")), variation_type=variation_type,
                attempt=int(entry.get("attempt", 0)), bits=bits, cosine=cos,
            )
            if best is None or _closer(match, best):
                best = match
        return best


def _closer(a: Match, b: Match) -> bool:
    """Is ``a`` nearer than ``b``? Fewer bits first, then higher cosine."""
    if a.bits is not None and b.bits is not None and a.bits != b.bits:
        return a.bits < b.bits
    if a.cosine is not None and b.cosine is not None:
        return a.cosine > b.cosine
    return a.bits is not None and b.bits is None


def prune(root: str, ttl_hours: float = TTL_HOURS) -> int:
    """Remove challenge directories untouched for longer than the TTL."""
    removed = 0
    cutoff = time.time() - ttl_hours * 3600
    try:
        names = os.listdir(root)
    except FileNotFoundError:
        return 0
    for name in names:
        path = os.path.join(root, name)
        try:
            if os.path.isdir(path) and os.path.getmtime(path) < cutoff:
                shutil.rmtree(path, ignore_errors=True)
                removed += 1
        except OSError:
            continue
    return removed


def context(challenge_id: str, hotkey: str, root: Optional[str] = None) -> Optional[Context]:
    """A Context for this challenge, or None when the ledger is disabled."""
    root = DIR if root is None else root
    if not root:
        return None
    os.makedirs(root, exist_ok=True)
    prune(root)
    return Context(root, challenge_id, hotkey)


def describe() -> str:
    if not DIR:
        return "sibling ledger: off (set MIID_SIBLING_DIR to share one between hotkeys)"
    return (
        f"sibling ledger: {DIR} (near-duplicate at dHash <= {DUP_DHASH_BITS} bits "
        f"or CLIP cosine >= {DUP_CLIP_COSINE})"
    )
