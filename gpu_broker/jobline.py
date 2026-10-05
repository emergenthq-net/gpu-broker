"""Which jobs from a previous process can still run, and dropping one that cannot.

A job re-queued at startup may no longer be runnable: its model left the catalog in the deploy,
or it was an interactive session whose watcher is gone (holding the GPU for nobody)."""
from __future__ import annotations

import contextlib
from collections.abc import Mapping
from typing import Any

from .catalog import Catalog
from .classes import Classes
from .constants import ERR_JOB, INTERACTIVE_KEY, SESSION_KEY, JobState
from .staging import Staging
from .store import Store

MODEL_GONE = "model '{key}' no longer in catalog"
VANISHED = "job vanished from the store"
SESSION_EXPIRED = "session expired by restart"


def missing_model(key: str, models: Mapping[str, Any]) -> str | None:
    """Why a job on model `key` cannot run (its model is gone), or None."""
    return None if key in models else MODEL_GONE.format(key=key)


def unrunnable_after_restart(job: Mapping[str, Any] | None, models: Mapping[str, Any]) -> str | None:
    """Why a job the previous process left queued must fail instead of running again, or None."""
    if job is None:
        return VANISHED
    if gone := missing_model(job["resolved"], models):
        return gone
    return SESSION_EXPIRED if (job.get("payload") or {}).get(SESSION_KEY) else None


def drop(store: Store, staging: Staging, jid: str, error: str) -> None:
    """Fail a job that cannot run, and remove its staged files (they would otherwise leak)."""
    store.update_job(jid, state=JobState.FAILED, error=error[:ERR_JOB])
    with contextlib.suppress(OSError):
        staging.discard(jid)


def with_class(store: Store, classes: Classes, catalog: Catalog, job: dict[str, Any] | None) -> dict[str, Any] | None:
    """A re-queued job queued before it had a class gets one now (by its requester), recorded."""
    if job is not None and classes.strict and INTERACTIVE_KEY not in job["payload"]:
        job["payload"] = classes.classify(catalog, job["payload"], "", job["requester"])
        store.update_job(job["id"], payload=job["payload"])
    return job
