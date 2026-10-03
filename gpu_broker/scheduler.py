"""The GPU scheduler: one thread that owns residency and runs jobs in FIFO order.

Invariant: only this thread changes what is resident, and only between jobs. Calls to the
already-resident LLM are handed to the pool and overlap up to the model's slot count; any
other job first drains the pool, then switches, then runs on this thread. When the queue
has been empty for `idle_restore_s` (or an interactive session just ended) the default
model is made resident again, so the common case pays no load time.
"""
from __future__ import annotations

import contextlib
import queue
import threading
import time
from collections.abc import Callable
from typing import Any

from . import templates
from .backends import Backends
from .catalog import Catalog
from .constants import ERR_EVENT, ERR_JOB, INPUTS_KEY, INTERACTIVE_KEY, SESSION_KEY, Event, JobState, Runner
from .drivers import GpuHeld
from .execjob import ExecJobs
from .holds import GpuHold
from .llmpool import LlmPool
from .residency import Residency
from .sessions import Sessions
from .settings import Intervals
from .staging import Staging
from .store import Store

OUTPUT_PREFIX = "broker/"   # ComfyUI output subfolder per job: <prefix><job id>


class Scheduler:
    def __init__(self, catalog: Catalog, store: Store, residency: Residency, backends: Backends,
                 sessions: Sessions, intervals: Intervals, staging: Staging, exec_jobs: ExecJobs, clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self.catalog, self.store, self.res, self.backends = catalog, store, residency, backends
        self.sessions, self.i, self.clock, self.sleep = sessions, intervals, clock, sleep
        self.staging, self.exec_jobs = staging, exec_jobs
        self.pool = LlmPool(store, backends, self.touch)
        self.paused = threading.Event()        # set by a quiesce: queued jobs wait, in-flight calls finish
        # Persisted: set when a recipe may still hold the GPU; its clean retries in the background.
        self.hold = GpuHold(store, residency.driver.clean_recipe, intervals.held_retry_s, clock)
        self._q: queue.Queue[str] = queue.Queue()
        self._lock = threading.Lock()
        self._order: list[str] = []            # queued job ids, for queue positions
        self._running: str | None = None
        self._last_activity = clock()
        self._restore_now = False              # a session ended: give the GPU back without the idle wait

    def touch(self) -> None:
        self._last_activity = self.clock()

    def submit(self, jid: str) -> int:
        """Queue a job; returns its 1-based position (0 = running)."""
        with self._lock:
            self._order.append(jid)
            pos = len(self._order) + (1 if self._running else 0)
        self.store.update_job(jid, state=JobState.QUEUED)
        self._q.put(jid)
        return pos

    def position(self, jid: str) -> int | None:
        with self._lock:
            if jid == self._running or jid in self.pool.ids():
                return 0
            if jid not in self._order:
                return None
            return self._order.index(jid) + 1 + (1 if self._running else 0)

    def snapshot(self) -> tuple[list[str], str | None, list[str]]:
        """(queued ids, id running on the GPU thread, ids running in the pool)."""
        with self._lock:
            queued, running = list(self._order), self._running
        return queued, running, sorted(self.pool.ids())

    def has_queued(self) -> bool:
        with self._lock:
            return bool(self._order)

    def loop(self, stop: threading.Event) -> None:
        while not stop.is_set():
            if self.paused.is_set():
                self.sleep(self.i.paused_s)
                continue
            if self.hold.held():
                self.hold.wait()   # woken at once by a clear or shutdown
                continue
            try:
                jid = self._q.get(timeout=self.i.worker_poll_s)
            except queue.Empty:
                if not self.pool.busy():
                    self.maybe_restore()
                continue
            if self.paused.is_set():  # quiesced while waiting in get(): hand the job back
                self._q.put(jid)
                continue
            with self._lock:
                self._order.remove(jid)
                self._running = jid
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
        m = self.catalog.models[key]
        session = bool(payload.get(SESSION_KEY))
        interactive = bool(payload.get(INTERACTIVE_KEY))
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
            # After the terminal state, and never able to change it: the files were uploaded or
            # handed to the recipe, and a file left behind is cleared at the next start.
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
        self._check_staged(jid, payload)
        if m["runner"] == Runner.EXEC:
            return self.exec_jobs.run(jid, key, m, payload)
        uploaded = self.staging.upload(jid, self.backends.comfy_upload)   # {slot: ComfyUI input file}
        graph = templates.build(m, {**payload, **uploaded}, OUTPUT_PREFIX + jid)
        return self.backends.comfy_run(key, graph, jid)

    def _check_staged(self, jid: str, payload: dict[str, Any]) -> None:
        """The files on disk must be the ones accepted at submit; a vanished file fails the job."""
        recorded = {slot: len(v) if isinstance(v, list) else 1 for slot, v in payload.get(INPUTS_KEY, {}).items()}
        if (staged := self.staging.received(jid)) != recorded:
            raise RuntimeError(f"input files missing: expected {recorded}, found {staged}")

    def maybe_restore(self) -> None:
        """Bring the default model back once the GPU has been idle long enough."""
        want = self.catalog.defaults["resident"]
        idle_s = self.catalog.defaults["idle_restore_s"]
        if not self._q.empty() or (self.res.current == want and self.res.healthy(want)):
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
