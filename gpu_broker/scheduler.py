"""The GPU scheduler: one thread owns residency; queue ordering is a pure policy.

Invariant: only this thread changes what is resident, and only between jobs. Calls to the
already-resident LLM are handed to the pool and overlap up to the model's slot count; any
other job first drains the pool, then switches, then runs on this thread.

Pending work is ordered by policy.py. The default balanced policy combines declared
priority, starvation-safe aging and residency locality. Set scheduling.policy=fifo for
strict submission order.
"""
from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any

from .backends import Backends
from .catalog import Catalog
from .constants import CHAT_PATH, ERR_EVENT, ERR_JOB, INTERACTIVE_KEY, OPENAI_PATH_KEY, SESSION_KEY, Event, JobState, Runner
from .llmpool import LlmPool
from .policy import Candidate, normalize_priority, order
from .residency import Residency
from .sessions import Sessions
from .settings import Intervals, Scheduling
from .store import Store
from .templates import TEMPLATES

OUTPUT_PREFIX = "broker/"   # ComfyUI output subfolder per job: <prefix><job id>


class Scheduler:
    def __init__(self, catalog: Catalog, store: Store, residency: Residency, backends: Backends,
                 sessions: Sessions, intervals: Intervals, scheduling: Scheduling,
                 clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self.catalog, self.store, self.res, self.backends = catalog, store, residency, backends
        self.sessions, self.i, self.scheduling = sessions, intervals, scheduling
        self.clock, self.sleep = clock, sleep
        self.pool = LlmPool(store, backends, self.touch)
        self.paused = threading.Event()        # set by a quiesce: queued jobs wait, in-flight calls finish
        self._cv = threading.Condition()
        self._pending: dict[str, Candidate] = {}
        self._running: str | None = None
        self._sequence = 0
        self._last_activity = clock()
        self._restore_now = False              # a session ended: give the GPU back without the idle wait
        order([], scheduling.policy, None, clock(), scheduling.aging_s)  # validate config at startup

    def touch(self) -> None:
        self._last_activity = self.clock()

    def _ordered_locked(self) -> list[Candidate]:
        return order(list(self._pending.values()), self.scheduling.policy, self.res.current,
                     self.clock(), self.scheduling.aging_s)

    def submit(self, jid: str, priority: str | None = None, interactive: bool = False) -> int:
        """Queue a job; returns its 1-based effective position (0 = running)."""
        p = normalize_priority(priority, interactive)
        self.store.update_job(jid, state=JobState.QUEUED)
        with self._cv:
            self._sequence += 1
            self._pending[jid] = Candidate(jid, self._sequence, p, self.clock(),
                                           (self.store.job(jid) or {}).get("resolved"))
            ordered = self._ordered_locked()
            pos = next(i for i, c in enumerate(ordered, 1) if c.jid == jid) + (1 if self._running else 0)
            self._cv.notify()
        return pos

    def position(self, jid: str) -> int | None:
        if jid in self.pool.ids():
            return 0
        with self._cv:
            if jid == self._running:
                return 0
            ordered = self._ordered_locked()
            for i, c in enumerate(ordered, 1):
                if c.jid == jid:
                    return i + (1 if self._running else 0)
        return None

    def snapshot(self) -> tuple[list[str], str | None, list[str]]:
        """(queued ids in effective order, id running on the GPU thread, pool ids)."""
        with self._cv:
            queued, running = [c.jid for c in self._ordered_locked()], self._running
        return queued, running, sorted(self.pool.ids())

    def has_queued(self) -> bool:
        with self._cv:
            return bool(self._pending)

    def _take_next(self) -> str | None:
        with self._cv:
            if not self._pending:
                self._cv.wait(timeout=self.i.worker_poll_s)
            if self.paused.is_set() or not self._pending:
                return None
            jid = self._ordered_locked()[0].jid
            self._pending.pop(jid, None)
            self._running = jid
            return jid

    def loop(self, stop: threading.Event) -> None:
        while not stop.is_set():
            if self.paused.is_set():
                self.sleep(self.i.paused_s)
                continue
            jid = self._take_next()
            if jid is None:
                if not self.paused.is_set() and not self.pool.busy():
                    self.maybe_restore()
                continue
            try:
                self.run(jid)
            except Exception as e:  # noqa: BLE001 — the GPU thread must never die
                self.store.event(Event.WORKER_ERROR, jid, error=str(e)[:ERR_EVENT])
            finally:
                with self._cv:
                    self._running = None
                    self._cv.notify_all()
                self.touch()

    def run(self, jid: str) -> None:
        job = self.store.job(jid)
        if job is None:
            raise RuntimeError(f"job {jid} vanished from the store")
        key, payload = job["resolved"], job["payload"]
        m = self.catalog.models[key]
        session = bool(payload.get(SESSION_KEY))
        interactive = bool(payload.get(INTERACTIVE_KEY))
        pooled = m["runner"] == Runner.LLM_UNIT and not session   # every LLM call holds a pool slot
        if pooled and self.res.current == key and self.res.healthy(key):
            self.pool.dispatch(jid, key, m, payload, interactive)
            return
        self.pool.close_and_drain()     # nothing may be mid-call (queued or direct) while residency changes
        try:
            self.store.update_job(jid, state=JobState.SWITCHING)
            self.res.ensure(key, jid)
            self.pool.reopen(self.res.current)   # direct chat may use the new resident now
            if pooled:
                self.pool.dispatch(jid, key, m, payload, interactive)
                return
            self.store.update_job(jid, state=JobState.RUNNING)
            self.store.update_job(jid, state=JobState.DONE, result=self._execute(jid, key, payload, session))
        except Exception as e:  # noqa: BLE001 — any failure is reported to the requester
            self.pool.reopen(self.res.current)
            self.store.update_job(jid, state=JobState.FAILED, error=str(e)[:ERR_JOB])

    def _execute(self, jid: str, key: str, payload: dict[str, Any], session: bool) -> dict[str, Any]:
        m = self.catalog.models[key]
        if session:
            out = self.sessions.hold(jid, key, payload)
            self._restore_now = True
            return out
        if m["runner"] == Runner.LLM_UNIT:
            return self.backends.llm_request(m, payload.get(OPENAI_PATH_KEY, CHAT_PATH), payload)
        if "template" not in m:
            raise RuntimeError("session-only model: open it from the dashboard (POST /v1/sessions)")
        graph = TEMPLATES[m["template"]](payload, m.get("params", {}), OUTPUT_PREFIX + jid)
        return self.backends.comfy_run(key, graph, jid)

    def maybe_restore(self) -> None:
        """Bring the default model back once the GPU has been idle long enough."""
        want = self.catalog.defaults["resident"]
        idle_s = self.catalog.defaults["idle_restore_s"]
        if self.has_queued() or (self.res.current == want and self.res.healthy(want)):
            return
        if not (self._restore_now or self.clock() - self._last_activity > idle_s):
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
