"""Interactive chat: a person in a chat UI is served by the resident LLM without queueing.

When the requested LLM is already resident and the pool is open, an interactive request
takes a slot directly (it may use the slots reserved for interactive work) and is answered
or streamed straight from the model server. Background callers, a model that is not
resident, or a residency switch in progress all fall back to the job queue, so the GPU
thread still decides every switch.

Priority: header `x-priority: interactive|background` wins; otherwise callers listed in the
catalog's `defaults.background_requesters` are background and everyone else is interactive.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from . import schema
from .backends import Backends
from .catalog import Catalog, Model
from .constants import Event, JobState, Kind, Priority, Runner
from .llmpool import LlmPool
from .modelmap import joined
from .resolve import resolve
from .store import Store

SWITCHED = "model switched before a slot opened; retry"


@dataclass(frozen=True)
class Lease:
    """A slot held on the resident LLM for one direct call."""
    jid: str
    key: str
    model: Model
    substitution: str | None = None   # why `key` stands in for the name asked for


def apply_variant(catalog: Catalog, body: Mapping[str, Any]) -> dict[str, Any]:
    """A variant id (e.g. `my-model-fast`) becomes its parent model plus the variant's overrides."""
    found = catalog.variant(body["model"]) if isinstance(body.get("model"), str) else None
    if found is None:
        return dict(body)
    key, overrides = found
    return {**body, **overrides, "model": key}


def is_interactive(catalog: Catalog, priority: str, requester: str) -> bool:
    p = priority.strip().lower()
    if p in set(Priority):
        return p == Priority.INTERACTIVE
    return requester not in catalog.defaults.get("background_requesters", [])


class DirectChat:
    def __init__(self, catalog: Catalog, store: Store, pool: LlmPool, backends: Backends) -> None:
        self.catalog, self.store, self.pool, self.backends = catalog, store, pool, backends

    def open(self, body: Mapping[str, Any], requester: str, requested: str | None = None,
             note: str | None = None) -> Lease | None:
        """A slot on the resident model, or None to fall back to the job queue. `requested`/`note`
        as for Broker.submit: recorded on the job when it is created, never patched in later."""
        name = body.get("model") or self.catalog.defaults["resident"]
        r = resolve(self.catalog.data, name, body.get("kind") or Kind.LLM, body.get("caps"))
        key = r.resolved
        m = self.catalog.models.get(key) if key else None
        if key is None or m is None or m.get("runner") != Runner.LLM_UNIT or self.pool.resident != key:
            return None
        substitution = joined(note, r.substitution)
        jid = self.store.create_job(requester, requested or name, dict(body), exec_recipe=schema.NOT_EXEC)   # never exec
        self.store.update_job(jid, resolved=key, substitution=substitution)
        if not self.pool.acquire(jid, key, m, interactive=True, direct=True):
            self.store.update_job(jid, state=JobState.FAILED, error=SWITCHED)
            return None
        self.store.event(Event.JOB_DIRECT, jid, model=key)
        return Lease(jid, key, m, substitution)

    def finish(self, lease: Lease, state: JobState, **fields: Any) -> None:
        self.store.update_job(lease.jid, state=state, **fields)
        self.pool.release(lease.jid)
