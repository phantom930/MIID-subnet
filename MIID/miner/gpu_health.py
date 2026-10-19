# MIID/miner/gpu_health.py
#
# Two guards against the GPU memory creep that took every hotkey down on
# 2026-10-16 (14:13 UTC) and again from 2026-10-16 23:05 until 2026-10-19:
# each miner process keeps ~10–15 MB more GPU memory after almost every
# request (torch allocated 0.22 → 0.49 GiB, reserved 0.28 → 0.85 GiB per
# process in ~5 h). Six processes share one 16 GB card, so after ~4.7 h the
# active one no longer has room to denoise and every round comes back empty.
#
# Leak trace (MIID_GPU_TRACE, default on): PyTorch records the Python stack
# of every live GPU block ("state" history: no event log, only what is
# allocated now). At the start of each request the live blocks are grouped
# by the code that allocated them, and the largest groups and what grew
# since the previous request are logged, so the leaking call site shows up
# by name after a few hours of rounds.
#
# Recycle (MIID_RECYCLE_RESERVED_GB, default 0.6; 0 turns it off): a watcher
# thread measures this process's idle GPU footprint (torch reserved after
# emptying the cache). Past the threshold, and only while no request is in
# flight or queued here, the process exits and pm2 starts it again with a
# clean card. At most one process on the machine recycles per
# RECYCLE_SPACING_SECONDS (a timestamp in MIID_GPU_LOCK_DIR), so the hotkeys
# are never down together.

import fcntl
import gc
import os
import threading
import time
from typing import Callable, Dict, List, Optional, Tuple

import bittensor as bt

TRACE = os.environ.get("MIID_GPU_TRACE", "1").strip().lower() not in ("0", "false", "no")
# Idle reserved memory per process at which it recycles. A fresh process sits
# at ~0.2–0.3 GiB after its first request; the card ran out at ~0.85 GiB in
# each of six processes. 0.6 leaves every sibling room for the active one.
RECYCLE_RESERVED_GB = float(os.environ.get("MIID_RECYCLE_RESERVED_GB", "0.6"))
RECYCLE_MIN_UPTIME_SECONDS = 30 * 60
RECYCLE_SPACING_SECONDS = 10 * 60
CHECK_SECONDS = 60
TOP_GROUPS = 6

_started = time.time()
_inflight = 0
_inflight_lock = threading.Lock()
# forward() returns before the axon has sent the response; give it this long.
RESPONSE_GRACE_SECONDS = 30
_last_done = 0.0
_previous: Dict[Tuple[str, str], Tuple[int, int]] = {}


# ── leak trace ───────────────────────────────────────────────────────────


def start_trace() -> None:
    """Record allocation stacks for live GPU blocks. Call once, early."""
    if not TRACE:
        return
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.memory._record_memory_history(enabled="state", context="alloc", stacks="python")
            bt.logging.info("GPU leak trace: recording allocation stacks of live GPU blocks (MIID_GPU_TRACE=0 turns it off).")
    except Exception as e:  # noqa: BLE001 — diagnostics must never stop the miner
        bt.logging.warning(f"GPU leak trace unavailable: {e}")


def _site(frames: List[dict]) -> Tuple[str, str]:
    """(nearest non-torch frame, nearest frame in this repo) of an allocation."""
    def short(f: dict) -> str:
        name = f.get("filename", "?")
        for marker in ("site-packages/", "MIID-subnet/"):
            if marker in name:
                name = name.split(marker, 1)[1]
        return f"{name}:{f.get('line')} {f.get('name')}"

    outside = next((f for f in frames if "/torch/" not in f.get("filename", "")), None)
    ours = next((f for f in frames if "MIID-subnet/MIID" in f.get("filename", "")
                 or "MIID-subnet/neurons" in f.get("filename", "")), None)
    return (short(outside) if outside else "?", short(ours) if ours else "-")


def report(stage: str) -> None:
    """Log where this process's live GPU memory was allocated, and what grew."""
    global _previous
    if not TRACE:
        return
    try:
        import torch

        if not torch.cuda.is_available():
            return
        groups: Dict[Tuple[str, str], List[int]] = {}
        for segment in torch.cuda.memory._snapshot().get("segments", []):
            for block in segment.get("blocks", []):
                if block.get("state") != "active_allocated":
                    continue
                key = _site(block.get("frames") or [])
                entry = groups.setdefault(key, [0, 0])
                entry[0] += block.get("size", 0)
                entry[1] += 1
        total = sum(v[0] for v in groups.values())
        mib = 2 ** 20
        largest = sorted(groups.items(), key=lambda kv: -kv[1][0])[:TOP_GROUPS]
        grew = sorted(
            ((k, v[0] - _previous.get(k, (0, 0))[0], v[1] - _previous.get(k, (0, 0))[1])
             for k, v in groups.items()),
            key=lambda x: -x[1],
        )
        grew = [g for g in grew if g[1] > 0][:TOP_GROUPS]
        bt.logging.info(
            f"GPU leak trace [{stage}]: {total / mib:.0f} MiB live in "
            f"{sum(v[1] for v in groups.values())} blocks; largest: "
            + "; ".join(f"{v[0] / mib:.1f} MiB ×{v[1]} at {k[0]} (via {k[1]})" for k, v in largest)
        )
        if _previous and grew:
            bt.logging.info(
                f"GPU leak trace [{stage}]: grew since the last request: "
                + "; ".join(f"+{d / mib:.1f} MiB (+{n} blocks) at {k[0]} (via {k[1]})" for k, d, n in grew)
            )
        _previous = {k: (v[0], v[1]) for k, v in groups.items()}
    except Exception as e:  # noqa: BLE001
        bt.logging.debug(f"GPU leak trace failed: {e}")


# ── recycle ──────────────────────────────────────────────────────────────


class inflight:
    """Context manager marking a request as running in this process."""

    def __enter__(self) -> "inflight":
        global _inflight
        with _inflight_lock:
            _inflight += 1
        return self

    def __exit__(self, *exc) -> None:
        global _inflight, _last_done
        with _inflight_lock:
            _inflight -= 1
            _last_done = time.time()


def _idle_reserved_gb() -> Optional[float]:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        gc.collect()
        torch.cuda.empty_cache()
        return torch.cuda.memory_reserved() / 2 ** 30
    except Exception:  # noqa: BLE001
        return None


def _claim_recycle_slot(lock_dir: Optional[str]) -> bool:
    """True when no sibling recycled within RECYCLE_SPACING_SECONDS (and
    record this one). Without a shared lock dir, every process may recycle."""
    if not lock_dir:
        return True
    path = os.path.join(lock_dir, "recycle.last")
    try:
        os.makedirs(lock_dir, exist_ok=True)
        with open(path + ".lock", "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                last = float(open(path).read().strip())
            except (OSError, ValueError):
                last = 0.0
            if time.time() - last < RECYCLE_SPACING_SECONDS:
                return False
            with open(path, "w") as f:
                f.write(f"{time.time():.0f}")
            return True
    except OSError:
        return True


def start_recycler(busy: Callable[[], bool], lock_dir: Optional[str]) -> None:
    """Watch this process's idle GPU footprint and exit for a clean restart
    once it passes RECYCLE_RESERVED_GB. ``busy`` reports queued GPU work."""
    if RECYCLE_RESERVED_GB <= 0:
        bt.logging.info("GPU recycle: off (MIID_RECYCLE_RESERVED_GB=0).")
        return

    def watch() -> None:
        while True:
            time.sleep(CHECK_SECONDS)
            try:
                if time.time() - _started < RECYCLE_MIN_UPTIME_SECONDS:
                    continue
                with _inflight_lock:
                    running, done = _inflight, _last_done
                if running or busy() or time.time() - done < RESPONSE_GRACE_SECONDS:
                    continue
                reserved = _idle_reserved_gb()
                if reserved is None or reserved < RECYCLE_RESERVED_GB:
                    continue
                if not _claim_recycle_slot(lock_dir):
                    continue
                with _inflight_lock:  # a request may have arrived meanwhile
                    if _inflight or time.time() - _last_done < RESPONSE_GRACE_SECONDS:
                        continue
                    bt.logging.warning(
                        f"GPU recycle: idle GPU footprint {reserved:.2f} GiB ≥ "
                        f"{RECYCLE_RESERVED_GB:.2f} GiB after {(time.time() - _started) / 3600:.1f} h; "
                        "exiting so pm2 restarts this miner with a clean card."
                    )
                    time.sleep(2)  # let the log line reach pm2
                    os._exit(0)
            except Exception as e:  # noqa: BLE001
                bt.logging.debug(f"GPU recycle check failed: {e}")

    threading.Thread(target=watch, name="gpu-recycle", daemon=True).start()
    bt.logging.info(
        f"GPU recycle: exits for a clean restart once idle GPU memory passes "
        f"{RECYCLE_RESERVED_GB:.2f} GiB (one hotkey per {RECYCLE_SPACING_SECONDS // 60} min)."
    )
