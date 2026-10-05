"""The GPU scheduler: one thread that owns residency and runs jobs in the queue policy's order (line.py).

Invariant: only this thread changes what is resident, and only between jobs. Calls to the already-resident
LLM are handed to the pool and overlap up to the model's slot count; any other job first drains the pool,
then switches, then runs on this thread. When the queue has been empty for `idle_restore_s` (or a session
just ended) the default model is made resident again, so the common case pays no load time.
"""
from __future__ import annotations

import contextlib
import threading
import time
from collections.abc import Callable
from typing import Any

from . import templates
from .backends import Backends
from .catalog import Catalog
from .classes import Classes
from .constants import ERR_EVENT, ERR_JOB, INPUTS_KEY, INTERACTIVE_KEY, SESSION_KEY, Event, JobState, Runner
from .drivers import GpuHeld
from .execjob import ExecJobs
from .holds import GpuHold
from .jobline import drop, missing_model, unrunnable_after_restart, with_class
from .line import Costs, Line, Snap, waiting
from .llmpool import LlmPool
from .quiesce import QUIESCED, Quiesced
from .residency import Residency
from .sessions import Sessions
from .settings import Intervals, Scheduling
from .staging import Staging
from .store import Store

OUTPUT_PREFIX = "broker/"   # ComfyUI output subfolder per job: <prefix><job id>


class Scheduler:
    def __init__(self, catalog: Catalog, store: Store, residency: Residency, backends: Backends,
                 sessions: Sessions, intervals: Intervals, staging: Staging, exec_jobs: ExecJobs,
                 scheduling: Scheduling | None = None, clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self.catalog, self.store, self.res, self.backends = catalog, store, residency, backends
        self.sessions, self.i, self.clock, self.sleep, self.staging, self.exec_jobs = sessions, intervals, clock, sleep, staging, exec_jobs
        sch = scheduling or Scheduling()
        self.classes = Classes.of(sch.policy, sch.may_claim_interactive)
        self.costs = Costs(store, lambda: catalog.models, sch.cost_lookback_s, sch.cost_refresh_s)   # broker refreshes it
        self.line = Line(sch.policy, self.costs, sch.max_wait_s, sch.evict_wait_s, clock)   # in the policy's order
        self.costs.changed = self.line.reprice
        self.pool = LlmPool(store, backends, self.touch)
        self._snap = lambda: Snap.of(self.res.current, self.pool.inflight(), self.catalog.models)   # read once per look
        self.paused = threading.Event()        # set by a quiesce: queued jobs wait, in-flight calls finish
        self.admission = threading.Lock()      # a quiesce sets `paused` under it; submit checks under it
        self.hold = GpuHold(store, residency.driver.clean_recipe, intervals.held_retry_s, clock)   # persisted (holds.py)
        self._lock = threading.Lock()
        self._running: str | None = None
        self._last_activity = clock()
        self._restore_now = False              # a session ended: give the GPU back without the idle wait

    def touch(self) -> None:
        self._last_activity = self.clock()
        self.line.kick()   # a pool slot may have freed: a skipped call may start now

    def admit(self) -> None:
        """Raise Quiesced while quiesced: new work is refused, never queued (quiesce.py)."""
        if self.paused.is_set():
            raise Quiesced(self.i.quiesced_retry_s)

    def submit(self, jid: str) -> int:
        """Queue a job; returns its 1-based position (0 = running)."""
        with self.admission, self._lock:
            if self.paused.is_set():   # quiesced after the caller's admit(): refuse it, recorded
                self.store.update_job(jid, state=JobState.REJECTED, error=QUIESCED)
                self.staging.discard(jid)
                raise Quiesced(self.i.quiesced_retry_s)
            self.store.update_job(jid, state=JobState.QUEUED)
            self.line.put(waiting(self.catalog.models, jid, self.store.job(jid)))
            pos = self._pos(jid, self.line.order())   # under the lock: the GPU thread cannot take it before this
        return pos if pos is not None else 0

    def requeue(self, jids: list[str]) -> None:
        """At startup: jobs the previous process queued but never started go back in line, in order."""
        for jid in jids:
            if why := unrunnable_after_restart(job := self.store.job(jid), self.catalog.models):
                drop(self.store, self.staging, jid, why)
                continue
            job = with_class(self.store, self.classes, self.catalog, job)   # first in its class, in order:
            self.line.put(waiting(self.catalog.models, jid, job, requeued=True))
            self.store.event(Event.JOB_REQUEUED, jid)

    def _pos(self, jid: str, order: list[str]) -> int | None:
        return order.index(jid) + 1 + (1 if self._running else 0) if jid in order else None

    def position(self, jid: str) -> int | None:
        with self._lock:
            return 0 if jid == self._running or jid in self.pool.ids() else self._pos(jid, self.line.order())

    def snapshot(self) -> tuple[list[str], str | None, list[str]]:
        """(queued ids in the order they will be considered, id running on the GPU thread, ids running in the pool)."""
        with self._lock:
            running = self._running
        return self.line.order(), running, sorted(self.pool.ids())

    def has_queued(self) -> bool:
        return bool(self.snapshot()[0])

    def loop(self, stop: threading.Event) -> None:
        while not stop.is_set():
            if self.paused.is_set():
                self.sleep(self.i.paused_s)
                continue
            if self.hold.held():
                self.hold.wait()   # woken at once by a clear or shutdown
                continue
            w = self.line.pick(self._snap, self.i.worker_poll_s)
            if w is None or self.paused.is_set():   # quiesced while picking: it stays in line, in its place
                if w is None and not len(self.line) and not self.pool.busy():
                    self.maybe_restore()
                continue
            with self._lock:   # taken and running at once, for position()
                self.line.take(w)
                self._running = jid = w.id
            try:
                self.run(jid)
            except Exception as e:  # noqa: BLE001 — the GPU thread must never die
                self.store.event(Event.WORKER_ERROR, jid, error=str(e)[:ERR_EVENT])
            finally:
                with self._lock:
                    self._running = None
                self.touch()

    def run(self, jid: str) -> None:
        job = self.store.job(jid)
        if job is None:
            raise RuntimeError(f"job {jid} vanished from the store")
        key, payload = job["resolved"], job["payload"]
        if gone := missing_model(key, self.catalog.models):   # never a KeyError that leaves it queued
            drop(self.store, self.staging, jid, gone)
            return
        m = self.catalog.models[key]
        session, interactive = bool(payload.get(SESSION_KEY)), bool(payload.get(INTERACTIVE_KEY))
        pooled = m["runner"] == Runner.LLM_UNIT and not session   # every LLM call holds a pool slot
        if pooled and self.res.current == key and self.res.healthy(key):
            self.pool.dispatch(jid, key, m, payload, interactive)   # returns once a slot is taken
            return
        try:
            if m["runner"] == Runner.EXEC:   # before anything is evicted: a misconfigured recipe evicts nothing
                self.exec_jobs.check(key, m)
            self.pool.close_and_drain()     # nothing may be mid-call (queued or direct) while residency changes
            self.store.update_job(jid, state=JobState.SWITCHING)
            self.res.ensure(key, jid)
            self.pool.reopen(self.res.current)   # direct chat may use the new resident now
            if pooled:
                self.pool.dispatch(jid, key, m, payload, interactive)
                return
            self.store.update_job(jid, state=JobState.RUNNING)
            result = self._execute(jid, key, payload, session)
            self.store.update_job(jid, state=JobState.DONE, result=result)
        except Exception as e:  # noqa: BLE001 — any failure is reported to the requester
            if isinstance(e, GpuHeld):   # before the job ends: no next job may take the GPU
                self.hold.set(jid, m.get("exec", {}).get("recipe", ""), str(e))
            # Reopen first: a caller that sees the failure and retries must find the pool open.
            self.pool.reopen(self.res.current)   # a failed switch must not leave the pool closed
            self.store.update_job(jid, state=JobState.FAILED, error=str(e)[:ERR_JOB])
        finally:
            # After the terminal state, never changing it; a file left behind is cleared at next start.
            with contextlib.suppress(OSError):
                self.staging.discard(jid)

    def _execute(self, jid: str, key: str, payload: dict[str, Any], session: bool) -> dict[str, Any]:
        m = self.catalog.models[key]
        if session:
            out = self.sessions.hold(jid, key, payload)
            self._restore_now = True
            return out
        if "template" not in m and m["runner"] != Runner.EXEC:
            raise RuntimeError("session-only model: open it from the dashboard (POST /v1/sessions)")
        self.staging.check(jid, payload.get(INPUTS_KEY, {}))   # a re-queued job's files may be gone
        if m["runner"] == Runner.EXEC:
            return self.exec_jobs.run(jid, key, m, payload)
        uploaded = self.staging.upload(jid, self.backends.comfy_upload)   # {slot: ComfyUI input file}
        graph = templates.build(m, {**payload, **uploaded}, OUTPUT_PREFIX + jid)
        return self.backends.comfy_run(key, graph, jid)

    def maybe_restore(self) -> None:
        """Bring the default model back once the GPU has been idle long enough."""
        want, idle_s = self.catalog.defaults["resident"], self.catalog.defaults["idle_restore_s"]
        due = self._restore_now or self.clock() - self._last_activity > idle_s
        if not due or len(self.line) or (self.res.current == want and self.res.healthy(want)):
            return
        self._restore_now = False
        self.pool.close_and_drain()
        try:
            self.res.release_comfy()
            self.res.ensure(want)
            self.store.event(Event.RES_IDLE_RESTORE, model=want)
        except Exception as e:  # noqa: BLE001 — retried on the next idle tick
            self.store.event(Event.RES_RESTORE_FAILED, model=want, error=str(e)[:ERR_EVENT])
        finally:
            self.pool.reopen(self.res.current)
        self.touch()
