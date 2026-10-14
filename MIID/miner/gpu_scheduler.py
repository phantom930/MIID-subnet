# MIID/miner/gpu_scheduler.py
#
# One GPU, many validators: requests take turns, earliest deadline first.
#
# Miner.forward used to run the whole generation synchronously inside the
# axon's event loop, so a second validator's request was not even read until
# the first one finished — and its 1200 s timeout kept running the whole time.
# On 2026-10-07/08, 16 of 151 responses arrived after that timeout and scored
# nothing. forward now runs the GPU work in a worker thread and takes its turn
# here, which (a) lets queued requests be seen at all, (b) serves the one
# closest to its deadline first, and (c) lets a running request see who is
# waiting and stop spending time on optional retries that would make them late.
#
# Several miner processes on one GPU (one per hotkey) share it through a
# directory, MIID_GPU_LOCK_DIR: the GPU itself is an exclusive file lock on
# <dir>/gpu.lock, taken after a request has won its own process's turn, and
# every waiting request announces its deadline and workload in
# <dir>/waiting/ so that waiting() — and with it queue_cutoff and the polish
# phase's "someone is waiting" check — sees the other processes' queues too.
# Order between processes is the kernel's lock order, not strictly earliest
# deadline; validators send to every hotkey at once, so the queues are alike.

import fcntl
import heapq
import itertools
import json
import os
import threading
import time
from contextlib import contextmanager
from typing import Iterator, List, Optional, Tuple

LOCK_DIR = os.environ.get("MIID_GPU_LOCK_DIR", "").strip()


class GpuScheduler:
    """A single GPU slot granted earliest-deadline-first."""

    def __init__(self, lock_dir: Optional[str] = None) -> None:
        self._cond = threading.Condition()
        self._busy = False
        self._seq = itertools.count()
        # (deadline, seq, workload_seconds) for every request still waiting.
        self._waiting: List[Tuple[float, int, float]] = []
        self.lock_dir = LOCK_DIR if lock_dir is None else lock_dir
        if self.lock_dir:
            os.makedirs(os.path.join(self.lock_dir, "waiting"), exist_ok=True)

    # ── cross-process pieces ──

    def _announce(self, seq: int, deadline: float, workload: float) -> Optional[str]:
        if not self.lock_dir:
            return None
        path = os.path.join(self.lock_dir, "waiting", f"{os.getpid()}-{seq}.json")
        try:
            tmp = path + ".tmp"
            with open(tmp, "w") as f:
                json.dump({"pid": os.getpid(), "deadline": deadline, "workload": workload, "ts": time.time()}, f)
            os.replace(tmp, path)
        except OSError:
            return None
        return path

    @staticmethod
    def _withdraw(path: Optional[str]) -> None:
        if path:
            try:
                os.unlink(path)
            except OSError:
                pass

    def _others_waiting(self) -> List[Tuple[float, float]]:
        """(deadline, workload) announced by other processes, stale ones pruned."""
        if not self.lock_dir:
            return []
        out = []
        now = time.time()
        folder = os.path.join(self.lock_dir, "waiting")
        try:
            names = os.listdir(folder)
        except OSError:
            return out
        for name in names:
            if not name.endswith(".json"):
                continue
            path = os.path.join(folder, name)
            try:
                with open(path) as f:
                    entry = json.load(f)
            except (OSError, ValueError):
                continue
            pid = int(entry.get("pid", 0))
            if pid == os.getpid():
                continue
            deadline, workload = float(entry.get("deadline", 0)), float(entry.get("workload", 0))
            # A dead process on this host, or a deadline long gone, is noise.
            dead = False
            if pid > 0:
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    dead = True
                except PermissionError:
                    pass
            if dead or deadline < now - 60:
                self._withdraw(path)
                continue
            out.append((deadline, workload))
        return out

    @contextmanager
    def _gpu_lock(self) -> Iterator[None]:
        if not self.lock_dir:
            yield
            return
        fd = os.open(os.path.join(self.lock_dir, "gpu.lock"), os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)

    # ── the scheduler ──

    @contextmanager
    def acquire(self, deadline: float, workload_seconds: float = 0.0) -> Iterator[float]:
        """Hold the GPU for one request; yields the seconds spent waiting.

        ``deadline`` is a wall-clock time (time.time()). ``workload_seconds``
        is a rough estimate of the request's mandatory work, used only to tell
        the running request how much time the queue needs (see waiting()).
        """
        seq = next(self._seq)
        entry = (deadline, seq, workload_seconds)
        started = time.time()
        announced = self._announce(seq, deadline, workload_seconds)
        with self._cond:
            heapq.heappush(self._waiting, entry)
            try:
                while self._busy or self._waiting[0] is not entry:
                    self._cond.wait()
            except BaseException:
                self._waiting.remove(entry)
                heapq.heapify(self._waiting)
                self._cond.notify_all()
                self._withdraw(announced)
                raise
            heapq.heappop(self._waiting)
            self._busy = True
        try:
            with self._gpu_lock():
                self._withdraw(announced)
                yield time.time() - started
        finally:
            self._withdraw(announced)
            with self._cond:
                self._busy = False
                self._cond.notify_all()

    def waiting(self) -> List[Tuple[float, float]]:
        """(deadline, workload_seconds) of queued requests, earliest first —
        this process's and, with MIID_GPU_LOCK_DIR, the other processes' too."""
        with self._cond:
            local = [(d, w) for d, _, w in sorted(self._waiting)]
        return sorted(local + self._others_waiting())


def queue_cutoff(
    waiting: List[Tuple[float, float]], now: Optional[float] = None,
) -> Optional[float]:
    """Latest time the running request may finish without making a queued one late.

    Queued requests run earliest-deadline-first, each needing its workload;
    the running request must hand over the GPU early enough that every one of
    them still finishes by its own deadline. Requests already past saving
    (their deadline is gone whatever happens) are ignored rather than allowed
    to block everyone else. None when nothing is waiting.
    """
    if not waiting:
        return None
    now = time.time() if now is None else now
    cutoff = None
    needed = 0.0
    for deadline, workload in sorted(waiting):
        if deadline - workload <= now:
            continue
        needed += workload
        latest = deadline - needed
        cutoff = latest if cutoff is None else min(cutoff, latest)
    return cutoff


def describe() -> str:
    if not LOCK_DIR:
        return "GPU sharing: this process only (set MIID_GPU_LOCK_DIR to share the GPU between hotkey processes)"
    return f"GPU sharing: file lock in {LOCK_DIR}"


# The miner process has one GPU and one scheduler.
SCHEDULER = GpuScheduler()
