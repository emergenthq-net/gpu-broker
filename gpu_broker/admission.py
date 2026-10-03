"""Job admission: check a request, resolve its model, record the job, stage its files, queue it.

A malformed request, or input files the resolved model cannot take, raise ValueError (HTTP
400) before anything is recorded; a rejected job is recorded but never decodes or fetches its
files. Everything is read through the broker at call time, so the broker's current settings
and parts apply.
"""
from __future__ import annotations

import contextlib
import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from . import execjob, inputs, media, schema
from .constants import APP_NAME, INPUTS_KEY, SESSION_KEY, Event, JobState, Runner
from .modelmap import joined
from .resolve import resolve

if TYPE_CHECKING:
    from .broker import Broker

log = logging.getLogger(APP_NAME)

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


STAGING_FAILED = "could not store the job's input files"


class StagingError(Exception):
    """Submit recorded the job but could not store its input files; the job is FAILED."""

    def __init__(self, jid: str) -> None:
        super().__init__(STAGING_FAILED)
        self.jid = jid


def submit(b: Broker, body: Mapping[str, Any], requester: str, requested: str | None = None,
           note: str | None = None) -> tuple[str, dict[str, Any]]:
    """Resolve, record and queue a request; returns (job id, what the caller needs to know).
    A malformed request, or input files the model cannot take, raise ValueError (HTTP 400)
    before anything is recorded. `requested`/`note`: the name the caller really sent and why
    `body["model"]` differs (model_map), recorded on the job and its substitution event."""
    validate_request(body)
    given = inputs.slots(body)
    name = body.get("model") or b.catalog.defaults["resident"]
    r = resolve(b.catalog.data, name, body.get("kind"), body.get("caps"), session=bool(body.get(SESSION_KEY)),
                images=given)
    files: list[media.InputFile] = []
    recipe = schema.NOT_EXEC
    if r.resolved is not None:   # a rejected job never decodes or fetches its files
        m = b.catalog.models[r.resolved]
        inputs.check(r.resolved, m, given)
        inputs.check_counts(r.resolved, m, body)
        if m.get("runner") == Runner.EXEC:
            execjob.params(m, body)
            recipe = m["exec"]["recipe"]
        files = media.read(body, b.settings.inputs, b.fetch_policy)
    payload = media.strip(body) | ({INPUTS_KEY: media.summarize(files)} if files else {})
    # Its own column, never the caller's payload: a restart cleans with it (GpuHold.orphan).
    asked = requested or name
    substitution = joined(note, r.substitution)
    jid = b.store.create_job(requester[:REQUESTER_MAX], asked, payload, exec_recipe=recipe)
    info: dict[str, Any] = {"requested": asked, "resolved": r.resolved, "substitution": substitution, "notes": r.notes}
    if r.download:
        key: str | None
        if r.register:
            key = r.download.slug
            b.catalog.register(key, r.register)
        else:
            key = _key_for_slug(b, r.download.slug)
        b.downloads.request(r.download, key)
        info["download"] = {**r.download.as_dict(), "state": (b.store.download(r.download.slug) or {}).get("state")}
        b.store.update_job(jid, download=info["download"])
    if r.resolved is None:
        b.store.update_job(jid, state=JobState.REJECTED, error=r.error)
        info["error"] = r.error
        return jid, info
    b.store.update_job(jid, resolved=r.resolved, substitution=substitution)
    if substitution:
        b.store.event(Event.JOB_SUBSTITUTED, jid, requested=asked, resolved=r.resolved, reason=substitution)
    try:
        b.staging.put(jid, files)
    except OSError as e:   # disk full, permissions: fail the job rather than leave it queued-but-never-run
        b.store.update_job(jid, state=JobState.FAILED, error=STAGING_FAILED)
        log.error("job %s: staging input files failed: %s", jid, e)   # paths stay server-side
        with contextlib.suppress(OSError):
            b.staging.discard(jid)
        raise StagingError(jid) from e
    info["queue_position"] = b.scheduler.submit(jid)
    return jid, info

def _key_for_slug(b: Broker, slug: str) -> str | None:
    return next((k for k, m in b.catalog.models.items() if m.get("source", {}).get("slug", k) == slug), None)
