"""The broker: wires configuration, catalog, store, driver, backends and the worker threads
together, and implements the three operations every route is built from — submit a job,
view it, wait for it.
"""
from __future__ import annotations

import contextlib
import logging
import os
import threading
import time
from collections.abc import Callable, Mapping
from typing import Any

from . import drivers, execjob, inputs, media, netguard, schema
from .backends import Backends, HttpBackends
from .catalog import Catalog
from .chat import DirectChat
from .constants import APP_NAME, INPUTS_KEY, SESSION_KEY, TERMINAL, UPSTREAM_TOKEN_PREFIX, Event, JobState, Runner
from .downloads import Downloader
from .metrics import GpuSampler
from .modelmap import joined
from .residency import Residency
from .resolve import resolve
from .scheduler import Scheduler
from .sessions import Sessions
from .settings import Settings
from .staging import Staging
from .store import Row, Store

REQUESTER_MAX = 120   # caller labels are stored and shown; keep them short
STRING_FIELDS = ("model", "kind", "requester")


def validate_request(body: Mapping[str, Any]) -> None:
    """Shape checks on the fields the broker itself interprets; the rest is the model's business."""
    for f in STRING_FIELDS:
        if f in body and body[f] is not None and not isinstance(body[f], str):
            raise ValueError(f"`{f}` must be a string")
    caps = body.get("caps")
    if caps is not None and not (isinstance(caps, list) and all(isinstance(c, str) for c in caps)):
        raise ValueError("`caps` must be a list of strings")


STOP_JOIN_S = 5.0   # how long shutdown waits for the sampler thread after closing its stream

log = logging.getLogger(APP_NAME)
STAGING_FAILED = "could not store the job's input files"


class StagingError(Exception):
    """Submit recorded the job but could not store its input files; the job is FAILED."""

    def __init__(self, jid: str) -> None:
        super().__init__(STAGING_FAILED)
        self.jid = jid


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
        """Resolve, record and queue a request; returns (job id, what the caller needs to know).
        A malformed request, or input files the model cannot take, raise ValueError (HTTP 400)
        before anything is recorded. `requested`/`note`: the name the caller really sent and why
        `body["model"]` differs (model_map), recorded on the job and its substitution event."""
        validate_request(body)
        given = inputs.slots(body)
        name = body.get("model") or self.catalog.defaults["resident"]
        r = resolve(self.catalog.data, name, body.get("kind"), body.get("caps"), session=bool(body.get(SESSION_KEY)),
                    images=given)
        files: list[media.InputFile] = []
        recipe = schema.NOT_EXEC
        if r.resolved is not None:   # a rejected job never decodes or fetches its files
            m = self.catalog.models[r.resolved]
            inputs.check(r.resolved, m, given)
            inputs.check_counts(r.resolved, m, body)
            if m.get("runner") == Runner.EXEC:
                execjob.params(m, body)
                recipe = m["exec"]["recipe"]
            files = media.read(body, self.settings.inputs, self.fetch_policy)
        payload = media.strip(body) | ({INPUTS_KEY: media.summarize(files)} if files else {})
        # Its own column, never the caller's payload: a restart cleans with it (GpuHold.orphan).
        asked = requested or name
        substitution = joined(note, r.substitution)
        jid = self.store.create_job(requester[:REQUESTER_MAX], asked, payload, exec_recipe=recipe)
        info: dict[str, Any] = {"requested": asked, "resolved": r.resolved, "substitution": substitution, "notes": r.notes}
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
        self.store.update_job(jid, resolved=r.resolved, substitution=substitution)
        if substitution:
            self.store.event(Event.JOB_SUBSTITUTED, jid, requested=asked, resolved=r.resolved, reason=substitution)
        try:
            self.staging.put(jid, files)
        except OSError as e:   # disk full, permissions: fail the job rather than leave it queued-but-never-run
            self.store.update_job(jid, state=JobState.FAILED, error=STAGING_FAILED)
            log.error("job %s: staging input files failed: %s", jid, e)   # paths stay server-side
            with contextlib.suppress(OSError):
                self.staging.discard(jid)
            raise StagingError(jid) from e
        info["queue_position"] = self.scheduler.submit(jid)
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
