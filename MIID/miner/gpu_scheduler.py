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
# <dir>/gpu.lock, and every waiting request announces its phase, deadline and
# workload in <dir>/waiting/. A request takes the file lock only when no
# announced request of another process is ahead of it, so the order between
# processes is the same as within one: phase first, then deadline. (The kernel
# hands a contended flock out first-come-first-served; with that order a
# request sent at 14:18 lost its turn to one sent at 14:26 on 2026-10-14 and
# scored nothing.)
#
# Phases: a request queues at phase 0 for its first pass — one attempt per
# slot, the part that decides whether it submits at all. Once that is done it
# hands the GPU back whenever more urgent work is waiting (Turn.yield_turn)
# and queues again for the rest: phase 1 for identity rescues (a slot under
# the 0.6 identity gate, which caps the whole challenge's grade), phase 2 for
# edit retries and polish. Within a phase the earliest deadline goes first,
# so a request's polish yields to another request's identity rescue and to
# an earlier-deadline request's polish, and nothing but a first pass delays
# a first pass. Four hotkeys on one card used to let one request's eleven
# retries run for 300 s while another hotkey's request waited 870 s and then
# had no time left for even one attempt.

import fcntl
import heapq
import itertools
import json
import os
import threading
import time
from contextlib import contextmanager
from typing import Callable, Iterator, List, Optional, Tuple

LOCK_DIR = os.environ.get("MIID_GPU_LOCK_DIR", "").strip()
# How often a request waiting for another process's turn re-checks the lock.
POLL_SECONDS = 0.25


class Turn:
    """One request's hold on the GPU, handed out by GpuScheduler.acquire.

    ``waited`` accumulates the seconds spent queued (initial wait plus every
    later reacquire). ``held`` says whether the GPU is ours right now.
    """

    def __init__(self, scheduler: "GpuScheduler", deadline: float, workload: float):
        self._scheduler = scheduler
        self.deadline = deadline
        self.workload = workload
        self.waited = 0.0
        self.held = False
        self._fd: Optional[int] = None

    def release(self) -> None:
        """Hand the GPU back (idempotent); the request keeps running on the CPU."""
        self._scheduler._release(self)

    def reacquire(self, phase: int = 1, until: Optional[float] = None) -> Optional[float]:
        """Queue for the GPU again at ``phase``; seconds waited, or None when
        ``until`` passed first (the GPU is then not held)."""
        if self.held:
            return 0.0
        return self._scheduler._take(self, phase, until)

    def yield_turn(self, phase: int = 1, until: Optional[float] = None) -> Optional[float]:
        """Give the GPU up and queue again at ``phase`` with the same deadline.

        Returns the seconds waited, or None when ``until`` passed before the
        turn came round again — the GPU is then not held and the caller must
        not use it.
        """
        self.release()
        return self.reacquire(phase, until)


class GpuScheduler:
    """A single GPU slot granted phase first, then earliest deadline."""

    def __init__(self, lock_dir: Optional[str] = None) -> None:
        self._cond = threading.Condition()
        self._busy = False
        self._seq = itertools.count()
        # (phase, deadline, seq, workload_seconds) for every request still waiting.
        self._waiting: List[Tuple[int, float, int, float]] = []
        self.lock_dir = LOCK_DIR if lock_dir is None else lock_dir
        if self.lock_dir:
            os.makedirs(os.path.join(self.lock_dir, "waiting"), exist_ok=True)
        # Run whenever a held turn is given up (release, yield, leaving the
        # acquire block): the image model evicts its resident weights here,
        # so the next process finds the card empty. Hooks must not raise.
        self._release_hooks: List[Callable[[], None]] = []

    def on_release(self, hook: Callable[[], None]) -> None:
        """Call ``hook`` every time this process hands the GPU back."""
        self._release_hooks.append(hook)

    # ── cross-process pieces ──

    def _announce(self, seq: int, deadline: float, workload: float, phase: int) -> Optional[str]:
        if not self.lock_dir:
            return None
        path = os.path.join(self.lock_dir, "waiting", f"{os.getpid()}-{seq}.json")
        try:
            tmp = path + ".tmp"
            with open(tmp, "w") as f:
                json.dump({
                    "pid": os.getpid(), "seq": seq, "phase": phase,
                    "deadline": deadline, "workload": workload, "ts": time.time(),
                }, f)
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

    def _others_waiting(self) -> List[Tuple[int, float, int, int, float]]:
        """(phase, deadline, pid, seq, workload) announced by other processes,
        stale ones pruned."""
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
            phase, seq = int(entry.get("phase", 0)), int(entry.get("seq", 0))
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
            out.append((phase, deadline, pid, seq, workload))
        return out

    def _others_ahead(self, key: Tuple[int, float, int, int]) -> bool:
        return any((p, d, pid, s) < key for p, d, pid, s, _ in self._others_waiting())

    # ── the scheduler ──

    def _take(self, turn: Turn, phase: int, until: Optional[float] = None) -> Optional[float]:
        """Wait for the GPU; seconds waited, or None when ``until`` passed first.

        One decision for every waiter, local or in another process: the
        request with the smallest (phase, deadline, pid, seq) key takes the
        GPU as soon as it is free. Nothing is reserved before that — a
        process must never hold a "local turn" while it defers to another
        process, because that other process's request may itself be queued
        behind this one's turn. (That circular wait idled the GPU for 13
        minutes on 2026-10-14 while four requests sat ready.)
        """
        seq = next(self._seq)
        entry = (phase, turn.deadline, seq, turn.workload)
        key = (phase, turn.deadline, os.getpid(), seq)
        started = time.time()
        announced = self._announce(seq, turn.deadline, turn.workload, phase)
        fd: Optional[int] = None
        if self.lock_dir:
            fd = os.open(os.path.join(self.lock_dir, "gpu.lock"), os.O_RDWR | os.O_CREAT, 0o644)
        with self._cond:
            heapq.heappush(self._waiting, entry)
            try:
                while True:
                    if not self._busy and self._waiting[0] is entry and not self._others_ahead(key):
                        if fd is None:
                            break
                        try:
                            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                            break
                        except BlockingIOError:
                            pass
                    if until is not None and time.time() >= until:
                        raise TimeoutError
                    # Other processes release the lock without waking us, so
                    # poll; a local release notifies the condition directly.
                    timeout = POLL_SECONDS if fd is not None else None
                    if until is not None:
                        remaining = until - time.time()
                        timeout = remaining if timeout is None else min(timeout, remaining)
                    self._cond.wait(timeout=timeout)
                heapq.heappop(self._waiting)
                self._busy = True
            except BaseException as exc:
                self._waiting.remove(entry)
                heapq.heapify(self._waiting)
                self._cond.notify_all()
                self._withdraw(announced)
                if fd is not None:
                    os.close(fd)
                if isinstance(exc, TimeoutError):
                    return None
                raise
        self._withdraw(announced)
        turn._fd = fd
        turn.held = True
        waited = time.time() - started
        turn.waited += waited
        return waited

    def _release(self, turn: Turn) -> None:
        if not turn.held:
            return
        turn.held = False
        for hook in self._release_hooks:
            try:
                hook()
            except Exception:  # noqa: BLE001 — a hook must never keep the GPU locked
                pass
        if turn._fd is not None:
            try:
                fcntl.flock(turn._fd, fcntl.LOCK_UN)
            finally:
                os.close(turn._fd)
                turn._fd = None
        with self._cond:
            self._busy = False
            self._cond.notify_all()

    @contextmanager
    def acquire(
        self, deadline: float, workload_seconds: float = 0.0, phase: int = 0,
    ) -> Iterator[Turn]:
        """Hold the GPU for one request; yields its Turn (see Turn.waited).

        ``deadline`` is a wall-clock time (time.time()). ``workload_seconds``
        is a rough estimate of the request's mandatory GPU work, used only to
        tell the running request how much time the queue needs (see waiting()).
        The turn may be released early (Turn.release) or handed over and taken
        again (Turn.yield_turn); leaving the block releases whatever is held.
        """
        turn = Turn(self, deadline, workload_seconds)
        self._take(turn, phase)
        try:
            yield turn
        finally:
            self._release(turn)

    def waiting(self) -> List[Tuple[float, float]]:
        """(deadline, workload_seconds) of queued requests, earliest first —
        this process's and, with MIID_GPU_LOCK_DIR, the other processes' too."""
        with self._cond:
            local = [(d, w) for _, d, _, w in self._waiting]
        return sorted(local + [(d, w) for _, d, _, _, w in self._others_waiting()])

    def waiting_ahead(self, phase: int, deadline: float) -> bool:
        """True when a queued request — here or in another process — would be
        served before work at ``phase`` for a request due at ``deadline``: a
        lower phase, or the same phase with an earlier deadline."""
        key = (phase, deadline)
        with self._cond:
            if any((p, d) < key for p, d, _, _ in self._waiting):
                return True
        return any((p, d) < key for p, d, _, _, _ in self._others_waiting())

    def first_pass_waiting(self) -> bool:
        """True when a request that has not had its first pass yet is queued."""
        return self.waiting_ahead(1, float("-inf"))


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
    return f"GPU sharing: file lock in {LOCK_DIR}, phase first then earliest deadline across processes"


# The miner process has one GPU and one scheduler.
SCHEDULER = GpuScheduler()
