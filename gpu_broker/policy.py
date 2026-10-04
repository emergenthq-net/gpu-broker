"""Queue policies: in which order the GPU thread considers waiting jobs.

A policy only orders; `line.choose` (shared by the live scheduler and the replay) decides which
candidate may start now. `stall_on_blocked` says what happens when the first candidate cannot:
FIFO waits for it (head-of-line blocking, the behaviour before `fair`); `fair` lets others go.
`remove` is called when a job starts (or is dropped), and is where a policy charges for it.

`fair`: interactive jobs before background ones; a background job that has waited `max_wait_s`
counts as interactive (aging, so a stream of interactive work cannot starve it). Jobs re-queued
by a restart come first in their class, in their original order. Otherwise the requester that
has used the least expected GPU time goes first (start-time fair queuing, the virtual-time form
of deficit round-robin), each requester's own jobs in arrival order. A job is charged its
model's expected GPU-seconds: the median measured run, divided by the slots for an LLM.

Virtual time never goes back: it is the lowest start tag waiting when a job starts. A requester
whose line was empty joins at it, so idling banks no credit. An idle requester at or below it is
forgotten, and beyond IDLE_MAX idle requesters the lowest are, so the bookkeeping stays bounded
however many requesters come and go.
"""
from __future__ import annotations

import heapq
import itertools
import statistics
import time
from collections import deque
from collections.abc import Callable, Iterable, Iterator, Mapping
from typing import Any, Protocol

from .constants import Runner

DEFAULT_RUN_S = 60.0     # expected run time of a model with no measured run yet
DEFAULT_MAX_WAIT_S = 3600.0      # background aging, s: above the longest background wait replayed (2,083 s)
DEFAULT_EVICT_WAIT_S = 600.0     # how long a call waiting for a slot holds off a switch
BACKGROUND, INTERACTIVE = 0, 1   # ranks: higher goes first
IDLE_MAX = 1024                  # idle requesters whose charge is remembered above virtual time


class Item(Protocol):
    id: str
    requester: str
    model: str
    interactive: bool
    pooled: bool
    requeued: bool


Cost = Callable[[Any], float]
Clock = Callable[[], float]


class Fifo[T: Item]:
    """Arrival order; the head must start before anything behind it."""
    name = "fifo"
    stall_on_blocked = True

    def __init__(self, cost: Cost | None = None, max_wait_s: float = DEFAULT_MAX_WAIT_S, clock: Clock = time.monotonic) -> None:
        self.clock = clock
        self._q: deque[T] = deque()

    def add(self, item: T) -> None:
        self._q.append(item)

    def candidates(self) -> Iterator[T]:
        return iter(self._q)   # read under the line's lock, never while it changes

    def remove(self, item: T) -> None:
        self._q.remove(item)

    def rank(self, item: T) -> int:
        return INTERACTIVE if item.interactive else BACKGROUND

    def invalidate(self) -> None:
        """The costs changed (nothing to do: arrival order does not use them)."""

    def waited(self, item: T) -> float:
        return 0.0

    def __len__(self) -> int:
        return len(self._q)


class Fair[T: Item]:
    """Interactive (or aged) first, then re-queued jobs, then the least-used requester."""
    name = "fair"
    stall_on_blocked = False

    def __init__(self, cost: Cost, max_wait_s: float = DEFAULT_MAX_WAIT_S, clock: Clock = time.monotonic) -> None:
        self.cost, self.max_wait_s, self.clock = cost, max_wait_s, clock
        self._q: dict[str, deque[tuple[int, T]]] = {}
        self._used: dict[str, float] = {}   # expected GPU-seconds charged per requester
        self._idle: list[tuple[float, str]] = []   # (charge level, requester) of empty lines, to forget
        self._added: dict[str, float] = {}  # job id → when it joined the line
        self._v = 0.0                       # virtual time
        self._seq = itertools.count()
        self._order: list[T] | None = None  # cached candidates
        self._valid_until = float("inf")    # the next background job to age invalidates the cache

    def add(self, item: T) -> None:
        q = self._q.setdefault(item.requester, deque())
        if not q:
            self._used[item.requester] = max(self._used.get(item.requester, 0.0), self._v)
        q.append((next(self._seq), item))
        self._added[item.id] = self.clock()
        self._order = None

    def waited(self, item: T) -> float:
        return self.clock() - self._added.get(item.id, self.clock())

    def invalidate(self) -> None:
        """The costs changed: the start tags (and so the order) must be recomputed."""
        self._order = None

    def rank(self, item: T) -> int:
        return INTERACTIVE if item.interactive or self.waited(item) >= self.max_wait_s else BACKGROUND

    def candidates(self) -> Iterator[T]:
        if self._order is None or self.clock() >= self._valid_until:
            rows = []
            for r, q in self._q.items():
                tag = self._used[r]   # each job's start tag: its requester's level plus its earlier jobs
                for n, it in q:
                    rows.append(((-self.rank(it), not it.requeued, n if it.requeued else tag, n), it))
                    tag += self.cost(it)
            rows.sort(key=lambda x: x[0])
            self._order = [it for _, it in rows]
            ages = [self._added[it.id] + self.max_wait_s for it in self._order if self.rank(it) == BACKGROUND]
            self._valid_until = min(ages, default=float("inf"))
        return iter(self._order)

    def remove(self, item: T) -> None:
        r = item.requester
        q = self._q[r]
        q.remove(next(x for x in q if x[1] is item))
        self._added.pop(item.id, None)
        level = min(self._used[x] for x in self._q)   # the lowest start tag waiting, this one's included
        self._used[r] += self.cost(item)
        if not q:
            del self._q[r]
            heapq.heappush(self._idle, (self._used[r], r))
        self._v = max(self._v, level)
        self._forget()
        self._order = None

    def _forget(self) -> None:
        """Drop idle requesters at or below virtual time (they would rejoin there anyway), and the
        lowest idle ones beyond IDLE_MAX (at most one job's lead each, as their job is likely done)."""
        while self._idle and (self._idle[0][0] <= self._v or len(self._used) - len(self._q) > IDLE_MAX):
            used, x = heapq.heappop(self._idle)
            if x not in self._q and self._used.get(x) == used:
                del self._used[x]
        if len(self._idle) > 2 * (len(self._used) + IDLE_MAX):   # stale entries of requesters that came back
            self._idle = [(u, x) for x, u in self._used.items() if x not in self._q]
            heapq.heapify(self._idle)

    def __len__(self) -> int:
        return sum(len(q) for q in self._q.values())


Policy = Fifo[Any] | Fair[Any]
POLICIES: Mapping[str, Callable[..., Policy]] = {Fifo.name: Fifo, Fair.name: Fair}
DEFAULT_POLICY = Fair.name


def make(name: str, cost: Cost, max_wait_s: float = DEFAULT_MAX_WAIT_S, clock: Clock = time.monotonic) -> Policy:
    if name not in POLICIES:
        raise ValueError(f"scheduler.policy must be one of {sorted(POLICIES)}, not {name!r}")
    return POLICIES[name](cost, max_wait_s, clock)


def gpu_cost(models: Mapping[str, Mapping[str, Any]], run_s: Mapping[str, Iterable[float]]) -> Cost:
    """A job's expected GPU-seconds from measured run times per model (median; LLM calls share
    the card with `slots` others, so they cost a slot's share)."""
    p50 = {k: statistics.median(v) for k, v in ((k, [x for x in xs if x > 0]) for k, xs in run_s.items()) if v}

    def cost(item: Any) -> float:
        m = models.get(item.model, {})
        share = int(m.get("slots", 1)) if m.get("runner") == Runner.LLM_UNIT else 1
        return p50.get(item.model, DEFAULT_RUN_S) / max(1, share)
    return cost
