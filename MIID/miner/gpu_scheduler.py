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
# Phases: a request queues at phase 0 for its first pass — one attempt per
# slot, the part that decides whether it submits at all. Once that is done it
# hands the GPU back whenever more urgent work is waiting (Turn.yield_turn)
# and queues again for the rest: phase 1 for identity rescues (a slot under
# the 0.6 identity gate, which caps the whole challenge's grade), phase 2 for
# edit retries and polish. Within a phase the earliest deadline goes first,
# so a request's polish yields to another request's identity rescue and to
# an earlier-deadline request's polish, and nothing but a first pass delays
# a first pass.
#
# The miner is one process on the GPU: the scheduler only orders this
# process's requests, and handing the GPU over leaves the image model's
# weights where they are, since the next turn uses the same pipeline.

import heapq
import itertools
import threading
import time
from contextlib import contextmanager
from typing import Iterator, List, Optional, Tuple


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

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._busy = False
        self._seq = itertools.count()
        # (phase, deadline, seq, workload_seconds) for every request still waiting.
        self._waiting: List[Tuple[int, float, int, float]] = []

    def _take(self, turn: Turn, phase: int, until: Optional[float] = None) -> Optional[float]:
        """Wait for the GPU; seconds waited, or None when ``until`` passed first.

        The waiter with the smallest (phase, deadline, seq) key takes the GPU
        as soon as it is free.
        """
        entry = (phase, turn.deadline, next(self._seq), turn.workload)
        started = time.time()
        with self._cond:
            heapq.heappush(self._waiting, entry)
            try:
                while self._busy or self._waiting[0] is not entry:
                    if until is not None:
                        remaining = until - time.time()
                        if remaining <= 0:
                            raise TimeoutError
                        self._cond.wait(timeout=remaining)
                    else:
                        self._cond.wait()
                heapq.heappop(self._waiting)
                self._busy = True
            except BaseException as exc:
                self._waiting.remove(entry)
                heapq.heapify(self._waiting)
                self._cond.notify_all()
                if isinstance(exc, TimeoutError):
                    return None
                raise
        turn.held = True
        waited = time.time() - started
        turn.waited += waited
        return waited

    def _release(self, turn: Turn) -> None:
        if not turn.held:
            return
        turn.held = False
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
        """(deadline, workload_seconds) of queued requests, earliest first."""
        with self._cond:
            return sorted((d, w) for _, d, _, w in self._waiting)

    def waiting_ahead(self, phase: int, deadline: float) -> bool:
        """True when a queued request would be served before work at
        ``phase`` for a request due at ``deadline``: a lower phase, or the
        same phase with an earlier deadline."""
        key = (phase, deadline)
        with self._cond:
            return any((p, d) < key for p, d, _, _ in self._waiting)

    def idle(self) -> bool:
        """True when no request holds the GPU or is queued for it."""
        with self._cond:
            return not self._busy and not self._waiting

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


# The miner process has one GPU and one scheduler.
SCHEDULER = GpuScheduler()
