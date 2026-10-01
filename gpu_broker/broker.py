"""The broker: wires configuration, catalog, store, driver, backends and the worker threads
together, and implements the three operations every route is built from — submit a job,
view it, wait for it.
"""
from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable, Mapping
from typing import Any

from . import drivers
from .backends import Backends, HttpBackends
from .catalog import Catalog
from .chat import DirectChat
from .constants import INTERACTIVE_KEY, PRIORITY_KEY, SESSION_KEY, TERMINAL, UPSTREAM_TOKEN_PREFIX, Event, JobState
from .downloads import Downloader
from .metrics import GpuSampler
from .residency import Residency
from .resolve import resolve
from .scheduler import Scheduler
from .sessions import Sessions
from .settings import Settings
from .store import Row, Store

REQUESTER_MAX = 120   # caller labels are stored and shown; keep them short
STRING_FIELDS = ("model", "kind", "requester", PRIORITY_KEY)


def validate_request(body: Mapping[str, Any]) -> None:
    """Shape checks on the fields the broker itself interprets; the rest is the model's business."""
    for f in STRING_FIELDS:
        if f in body and body[f] is not None and not isinstance(body[f], str):
            raise ValueError(f"`{f}` must be a string")
    caps = body.get("caps")
    if caps is not None and not (isinstance(caps, list) and all(isinstance(c, str) for c in caps)):
        raise ValueError("`caps` must be a list of strings")


STOP_JOIN_S = 5.0   # how long shutdown waits for the sampler thread after closing its stream

class Broker:
    def __init__(self, settings: Settings, env: Mapping[str, str] | None = None,
                 driver: drivers.Driver | None = None, backends: Backends | None = None,
                 clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep) -> None:
        env = os.environ if env is None else env
        s = self.settings = settings
        self.clock, self.sleep = clock, sleep
        self.catalog = Catalog(s.catalog)
        self.store = Store(s.db, s.events_jsonl)
        self.driver = driver or drivers.build(s, self.catalog.units())
        tokens = {k: v for k, v in env.items() if k.startswith(UPSTREAM_TOKEN_PREFIX)}
        self.backends = backends or HttpBackends(s.comfy, s.timeouts, s.intervals, tokens)
        self.residency = Residency(self.catalog, self.driver, self.backends, self.store, s.timeouts, s.intervals,
                                   s.comfy.unit)
        self.sessions = Sessions(self.catalog.defaults, self.backends.comfy_queue_len, s.intervals.session_poll_s)
        self.scheduler = Scheduler(self.catalog, self.store, self.residency, self.backends, self.sessions, s.intervals, s.scheduling)
        self.sessions.queued_jobs = self.scheduler.has_queued
        self.chat = DirectChat(self.catalog, self.store, self.scheduler.pool, self.backends)
        self.downloads = Downloader(self.store, self.driver, self.catalog, s.intervals.worker_poll_s)
        self.sampler = GpuSampler(self.driver.gpu_stream, s.limits.gpu_samples, s.intervals.sampler_retry_s)
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []

    def start(self) -> None:
        """Recover from the previous process, adopt the running model, start the threads."""
        self.store.fail_orphans()
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
        self.driver.close()
        for t in self._threads:
            if t.name == self.sampler.loop.__qualname__:
                t.join(join_s)

    def submit(self, body: Mapping[str, Any], requester: str) -> tuple[str, dict[str, Any]]:
        """Resolve, record and queue a request; returns (job id, what the caller needs to know)."""
        validate_request(body)
        name = body.get("model") or self.catalog.defaults["resident"]
        r = resolve(self.catalog.data, name, body.get("kind"), body.get("caps"), session=bool(body.get(SESSION_KEY)))
        jid = self.store.create_job(requester[:REQUESTER_MAX], name, dict(body))
        info: dict[str, Any] = {"requested": name, "resolved": r.resolved, "substitution": r.substitution, "notes": r.notes}
        if r.download:
            key: str | None
            if r.register:
                key = r.download.slug
                self.catalog.register(key, r.register)
            else:
                key = self._key_for_slug(r.download.slug)
            self.downloads.request(r.download, key)
            info["download"] = {**r.download.as_dict(), "state": (self.store.download(r.download.slug) or {}).get("state")}
            self.store.update_job(jid, download=info["download"])
        if r.resolved is None:
            self.store.update_job(jid, state=JobState.REJECTED, error=r.error)
            info["error"] = r.error
            return jid, info
        self.store.update_job(jid, resolved=r.resolved, substitution=r.substitution)
        if r.substitution:
            self.store.event(Event.JOB_SUBSTITUTED, jid, requested=name, resolved=r.resolved, reason=r.substitution)
        info["queue_position"] = self.scheduler.submit(
            jid, body.get(PRIORITY_KEY), interactive=bool(body.get(INTERACTIVE_KEY)))
        return jid, info

    def _key_for_slug(self, slug: str) -> str | None:
        return next((k for k, m in self.catalog.models.items() if m.get("source", {}).get("slug", k) == slug), None)

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
        """Block until the job is terminal or `timeout_s` passes; returns the job either way."""
        start = self.clock()
        while self.clock() - start < timeout_s:
            j = self.store.job(jid)
            if j is None or j["state"] in TERMINAL:
                return j
            self.sleep(self.settings.intervals.job_wait_poll_s)
        return self.store.job(jid)
