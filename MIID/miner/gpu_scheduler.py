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

import heapq
import itertools
import threading
import time
from contextlib import contextmanager
from typing import Iterator, List, Optional, Tuple


class GpuScheduler:
    """A single GPU slot granted earliest-deadline-first."""

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._busy = False
        self._seq = itertools.count()
        # (deadline, seq, workload_seconds) for every request still waiting.
        self._waiting: List[Tuple[float, int, float]] = []

    @contextmanager
    def acquire(self, deadline: float, workload_seconds: float = 0.0) -> Iterator[float]:
        """Hold the GPU for one request; yields the seconds spent waiting.

        ``deadline`` is a wall-clock time (time.time()). ``workload_seconds``
        is a rough estimate of the request's mandatory work, used only to tell
        the running request how much time the queue needs (see waiting()).
        """
        entry = (deadline, next(self._seq), workload_seconds)
        started = time.time()
        with self._cond:
            heapq.heappush(self._waiting, entry)
            try:
                while self._busy or self._waiting[0] is not entry:
                    self._cond.wait()
            except BaseException:
                self._waiting.remove(entry)
                heapq.heapify(self._waiting)
                self._cond.notify_all()
                raise
            heapq.heappop(self._waiting)
            self._busy = True
        try:
            yield time.time() - started
        finally:
            with self._cond:
                self._busy = False
                self._cond.notify_all()

    def waiting(self) -> List[Tuple[float, float]]:
        """(deadline, workload_seconds) of queued requests, earliest first."""
        with self._cond:
            return [(d, w) for d, _, w in sorted(self._waiting)]


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
