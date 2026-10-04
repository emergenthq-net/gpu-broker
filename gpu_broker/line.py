"""The GPU thread's line of waiting jobs, in the queue policy's order (policy.py), and the rule
for which of them may start now (`choose`, shared with the replay).

The GPU thread `pick`s a job and only then `take`s it (which is when the policy charges for
it), so a job picked during a quiesce simply stays in line. FIFO returns its head whatever its
state, and the thread waits on it. `fair` skips:
- a call to the resident LLM whose class has no free slot (`Snap.blocked`);
- every later job of a requester with an earlier job skipped, so one requester's jobs never
  overtake each other;
- a job that would evict the resident model while a call to it waits only for a slot, unless
  the job is of a higher class, or the waiting call (or the job itself) has waited `evict_wait_s`
  (priority inversion: a switch must not jump calls the resident model is about to serve).
A job that needs the pool drained is taken at once and holds the GPU thread while it drains, so
calls that arrive meanwhile cannot refill the pool and starve the switch.

`Costs` gives the policy each job's expected GPU-seconds from the store's measured run times.
It is refreshed on its own thread and swapped in whole; the GPU thread only reads it.
"""
from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from .constants import APP_NAME, INTERACTIVE_KEY, SESSION_KEY, JobState, Runner
from .llmpool import limit
from .metrics import job_metrics
from .policy import DEFAULT_EVICT_WAIT_S, DEFAULT_MAX_WAIT_S, Cost, Policy, gpu_cost, make
from .store import Store

log = logging.getLogger(APP_NAME)


@dataclass(eq=False)
class Waiting:
    id: str
    requester: str
    model: str
    interactive: bool
    pooled: bool            # an LLM call that holds a pool slot (not a session)
    requeued: bool = False  # left queued by the previous process


def waiting(models: Mapping[str, Any], jid: str, job: Mapping[str, Any] | None, requeued: bool = False) -> Waiting:
    """A queued job as the policy sees it (its class is the payload's `interactive` flag)."""
    j = job or {}
    key, payload = j.get("resolved") or "", j.get("payload") or {}
    pooled = models.get(key, {}).get("runner") == Runner.LLM_UNIT and not payload.get(SESSION_KEY)
    return Waiting(jid, j.get("requester") or "", key, bool(payload.get(INTERACTIVE_KEY)), pooled, requeued)


@dataclass(frozen=True)
class Snap:
    """The pool as one `choose` sees it, read once: what is resident, how many calls it serves,
    and how many each class may have (background, interactive)."""
    resident: str | None
    inflight: int
    limits: tuple[int, int]

    @classmethod
    def of(cls, resident: str | None, inflight: int, models: Mapping[str, Any]) -> Snap:
        m = models.get(resident or "", {})
        return cls(resident, inflight, (limit(m, False), limit(m, True)))

    def blocked(self, w: Any) -> bool:
        return bool(w.pooled and w.model == self.resident and self.inflight >= self.limits[w.interactive])


def choose(policy: Policy, snap: Snap, evict_wait_s: float) -> Any:
    """The job the GPU thread may start now, or None."""
    if policy.stall_on_blocked:
        return next(policy.candidates(), None)
    cands = list(policy.candidates())
    # The highest class among calls to the resident model that wait only for a slot (and not yet too long).
    guard = max((policy.rank(w) for w in cands if snap.blocked(w) and policy.waited(w) < evict_wait_s), default=None)
    stuck: set[str] = set()
    for w in cands:
        if w.requester in stuck:
            continue
        evicts = w.model != snap.resident
        if snap.blocked(w) or (evicts and guard is not None and policy.rank(w) <= guard
                               and policy.waited(w) < evict_wait_s):
            stuck.add(w.requester)
            continue
        return w
    return None


class Line:
    def __init__(self, policy: str, cost: Cost, max_wait_s: float = DEFAULT_MAX_WAIT_S,
                 evict_wait_s: float = DEFAULT_EVICT_WAIT_S, clock: Callable[[], float] = time.monotonic) -> None:
        self.policy = make(policy, cost, max_wait_s, clock)
        self.evict_wait_s = evict_wait_s
        self._cv = threading.Condition()
        self._gen = 0   # bumped by every change, so a change while reading the pool is never missed

    def put(self, item: Waiting) -> None:
        with self._cv:
            self.policy.add(item)
            self.kick()

    def kick(self) -> None:
        """Something changed (a slot freed, a job ended): look again."""
        with self._cv:
            self._gen += 1
            self._cv.notify_all()

    def order(self) -> list[str]:
        with self._cv:
            return [w.id for w in self.policy.candidates()]

    def __len__(self) -> int:
        with self._cv:
            return len(self.policy)

    def pick(self, snapshot: Callable[[], Snap], timeout: float) -> Waiting | None:
        """The first job the GPU thread may start, waiting up to `timeout` for one; None if none.
        The pool is read once per look, outside the line's lock."""
        end = time.monotonic() + timeout
        while True:
            with self._cv:
                gen = self._gen
            snap = snapshot()
            with self._cv:
                w: Waiting | None = choose(self.policy, snap, self.evict_wait_s)
                if w is not None:
                    return w
                if gen != self._gen:
                    continue
                left = end - time.monotonic()
                if left <= 0 or not self._cv.wait(left):
                    return None   # timed out (the GPU thread asks again at once)

    def take(self, item: Waiting) -> None:
        with self._cv:
            self.policy.remove(item)

    def reprice(self) -> None:
        """New expected costs were swapped in: re-sort, and let the GPU thread look again."""
        with self._cv:
            self.policy.invalidate()
            self.kick()


class Costs:
    """Expected GPU-seconds per job: each model's median measured run time over `lookback_s`."""

    def __init__(self, store: Store, models: Callable[[], Mapping[str, Any]], lookback_s: float, refresh_s: float,
                 now: Callable[[], float] = time.time) -> None:
        self.store, self.models, self.lookback_s, self.refresh_s, self.now = store, models, lookback_s, refresh_s, now
        self._cost: Cost = gpu_cost(models(), {})
        self.changed: Callable[[], None] = lambda: None   # set to Line.reprice

    def refresh(self) -> None:
        run: dict[str, list[float]] = {}
        for r in job_metrics(self.store, self.now() - self.lookback_s, 0):
            if r["state"] == JobState.DONE and r["run_s"] and r["model"]:
                run.setdefault(r["model"], []).append(r["run_s"])
        self._cost = gpu_cost(self.models(), run)   # one assignment: readers see the old or the new
        self.changed()

    def loop(self, stop: threading.Event) -> None:
        while not stop.is_set():
            try:
                self.refresh()
            except Exception as e:  # noqa: BLE001 — keep the last estimate; retried next time
                log.warning("refreshing expected run times failed: %s", e)
            stop.wait(self.refresh_s)

    def __call__(self, item: Any) -> float:
        return self._cost(item)
