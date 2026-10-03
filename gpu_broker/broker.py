"""The broker: the object every route calls. It owns the parts (catalog, store, driver,
backends, worker threads), starts and stops them, and offers the three operations routes are
built from: submit a job (admission.py), view it, wait for it.
"""
from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable, Mapping
from typing import Any

from . import admission, drivers, execjob, netguard
from .admission import STAGING_FAILED as STAGING_FAILED
from .admission import StagingError as StagingError
from .admission import validate_request as validate_request
from .backends import Backends, HttpBackends
from .catalog import Catalog
from .chat import DirectChat
from .constants import TERMINAL, UPSTREAM_TOKEN_PREFIX, Event, JobState
from .downloads import Downloader
from .metrics import GpuSampler
from .residency import Residency
from .scheduler import Scheduler
from .sessions import Sessions
from .settings import Settings
from .staging import Staging
from .store import Row, Store

STOP_JOIN_S = 5.0   # how long shutdown waits for the sampler thread after closing its stream


class Broker:
    def __init__(self, settings: Settings, env: Mapping[str, str] | None = None,
                 driver: drivers.Driver | None = None, backends: Backends | None = None,
                 clock: Callable[[], float] = time.monotonic) -> None:
        env = os.environ if env is None else env
        s = self.settings = settings
        self.clock = clock
        self.catalog = Catalog(s.catalog)
        self.store = Store(s.db, s.events_jsonl)
        self.driver = driver or drivers.build(s, self.catalog.units())
        tokens = {k: v for k, v in env.items() if k.startswith(UPSTREAM_TOKEN_PREFIX)}
        self.backends = backends or HttpBackends(s.comfy, s.timeouts, s.intervals, tokens)
        self.residency = Residency(self.catalog, self.driver, self.backends, self.store, s.timeouts, s.intervals,
                                   s.comfy.unit)
        self.staging = Staging(s.inputs.staging_dir)
        # <slot>_url may always reach the broker's own ComfyUI: chained jobs pass output URLs on.
        self.fetch_policy = netguard.Policy(s.inputs.url_allow_networks,
                                            netguard.endpoints(s.comfy.url, s.comfy.browser_url))
        self.sessions = Sessions(self.catalog.defaults, self.backends.comfy_queue_len, s.intervals.session_poll_s)
        self.exec_jobs = execjob.ExecJobs(self.driver, self.staging, s.comfy, s.timeouts)
        self.scheduler = Scheduler(self.catalog, self.store, self.residency, self.backends, self.sessions, s.intervals,
                                   self.staging, self.exec_jobs)
        execjob.check_timeouts(self.catalog, self.exec_jobs, self.store)
        self.sessions.queued_jobs = self.scheduler.has_queued
        self.chat = DirectChat(self.catalog, self.store, self.scheduler.pool, self.backends)
        self.downloads = Downloader(self.store, self.driver, self.catalog, s.intervals.worker_poll_s)
        self.sampler = GpuSampler(self.driver.gpu_stream, s.limits.gpu_samples, s.intervals.sampler_retry_s)
        self.residency.cached_gpu = lambda after: self.sampler.latest(s.intervals.gpu_cache_s, after)
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []

    def start(self) -> None:
        """Recover from the previous process, adopt the running model, start the threads. The GPU
        is not read here: which reader to use is resolved on first use, so a GPU that cannot be
        read yet never stops the broker serving; /v1/gpu and the dashboard say why while the
        sampler keeps retrying."""
        for job in self.store.fail_orphans():
            if job["state"] == JobState.RUNNING:
                self.scheduler.hold.orphan(job, self.catalog.models)
        self.staging.clear()   # images of jobs the previous process never ran
        self.scheduler.pool.reopen(self.residency.detect())
        self.store.event(Event.BROKER_STARTED, resident=self.residency.current, driver=self.settings.driver.kind,
                         config=self.settings.source)
        loops = [self.scheduler.loop, self.downloads.loop] + ([self.sampler.loop] if self.settings.gpu_stream else [])
        for loop in loops:
            t = threading.Thread(target=loop, args=(self._stop,), daemon=True, name=loop.__qualname__)
            t.start()
            self._threads.append(t)

    def stop(self, join_s: float = STOP_JOIN_S) -> None:
        """Shutdown. The GPU sample stream is closed (on Proxmox that kills its SSH session, which
        would otherwise outlive us and hold the service's stop); the sampler then exits at once.
        The GPU and download workers exit after their current step and are not waited for: a
        running job may take minutes, and its record is failed on the next start."""
        self._stop.set()
        self.scheduler.hold.changed.set()   # a GPU thread waiting out a hold wakes and sees the stop
        self.driver.close()
        for t in self._threads:
            if t.name == self.sampler.loop.__qualname__:
                t.join(join_s)

    def submit(self, body: Mapping[str, Any], requester: str, requested: str | None = None,
               note: str | None = None) -> tuple[str, dict[str, Any]]:
        """Admit a request (admission.py); returns (job id, what the caller needs to know)."""
        return admission.submit(self, body, requester, requested, note)

    def view(self, jid: str) -> Row | None:
        """A job as callers see it: without the request payload, with its queue position."""
        j = self.store.job(jid)
        if j is None:
            return None
        j.pop("payload", None)
        j["queue_position"] = self.scheduler.position(jid)
        j["using"] = j.get("resolved")
        return j

    def wait(self, jid: str, timeout_s: float) -> Row | None:
        """Block until the job is terminal or `timeout_s` passes; returns the job either way.
        Woken by the store on each job update, not by polling. The job is read while holding the
        condition, so an update between the read and the wait cannot be missed."""
        end = self.clock() + timeout_s
        with self.store.job_changed:
            while (j := self.store.job(jid)) is not None and j["state"] not in TERMINAL:
                if (left := end - self.clock()) <= 0:
                    break
                self.store.job_changed.wait(left)
        return j
