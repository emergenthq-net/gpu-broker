"""The GPU hold: an exec recipe may still be running (holding the GPU) after its job ended.

Set by the scheduler when a driver raises GpuHeld, and at startup for an exec job the previous
process left running. While it is set the GPU thread takes no job and restores no model. It is
persisted, so neither a restart nor POST /v1/admin/resume clears it; only a successful clean
of the held job or an operator's POST /v1/admin/gpu-held/clear does.

The clean is retried off the GPU thread (one worker, at most one clean in flight, the next
`retry_s` after a failure ends). The GPU thread sleeps on `changed`, which a clear (by either
route) and shutdown set, so both take effect at once.
"""
from __future__ import annotations

import threading
import time
from collections.abc import Callable, Mapping
from typing import Any

from . import schema
from .catalog import RECIPE
from .constants import ERR_EVENT, Event, Kind, Runner
from .store import Row, Store

FLAG = "gpu_held"
BY_CLEAN, BY_OPERATOR = "exec-clean", "operator"
ORPHAN_HELD = "the broker restarted while this exec job ran; its program may still hold the GPU"
ORPHAN_UNKNOWN = ("the broker restarted while job {jid} ran on model {model}; its recipe is unknown (no valid one "
                  "recorded, none in the catalog): it may still hold the GPU. Check the host, then POST /v1/admin/gpu-held/clear")
UNKNOWN = ""   # orphan_recipe: may be exec, recipe unknown (an operator-only hold)


def orphan_recipe(job: Row, models: Mapping[str, Any]) -> str | None:
    """The recipe a job the previous process left running may still be running: a name, UNKNOWN,
    or None when it cannot have been an exec job. A direct chat never is. A stored value is used
    only if it matches the recipe grammar in full; NOT_EXEC means not exec, except that a model
    the catalog runs as exec is always held (no stored value un-holds it). NULL (unknown writer:
    an older or rolled-back broker) or an invalid value: the catalog decides if it has the model,
    else only an LLM-shaped request rules exec out."""
    if job.get("direct"):
        return None
    rec = job.get("exec_recipe")
    if isinstance(rec, str) and RECIPE.fullmatch(rec):
        return rec
    m = models.get(job.get("resolved") or "") or models.get(job.get("requested") or "")
    if m is not None:
        return m["exec"]["recipe"] if m["runner"] == Runner.EXEC else None
    if rec == schema.NOT_EXEC:
        return None
    payload = job.get("payload") or {}
    if "messages" in payload or payload.get("kind") == Kind.LLM:
        return None
    return UNKNOWN

Clean = Callable[[str, str], None]   # (recipe, job id): returns once the job is confirmed gone, else raises


class GpuHold:
    def __init__(self, store: Store, clean: Clean, retry_s: float,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.store, self.clean, self.retry_s, self.clock = store, clean, retry_s, clock
        self.changed = threading.Event()
        self._lock = threading.Lock()
        self._worker: threading.Thread | None = None
        self._next = 0.0   # earliest next clean attempt

    def get(self) -> Row | None:
        """{since, job, recipe, reason} while held, else None."""
        return self.store.flag(FLAG)

    def set(self, jid: str, recipe: str, reason: str, retry_now: bool = False) -> None:
        self.store.set_flag(FLAG, {"job": jid, "recipe": recipe, "reason": reason[:ERR_EVENT]})
        self.store.event(Event.EXEC_GPU_HELD, jid, recipe=recipe, error=reason[:ERR_EVENT])
        with self._lock:   # the driver's own clean just failed: wait before the next one
            self._next = 0.0 if retry_now else self.clock() + self.retry_s

    def orphan(self, job: Row, models: Mapping[str, Any]) -> None:
        """A job the previous process left running: hold the GPU before the GPU thread starts if it
        may have been an exec job. A known recipe is cleaned at once, in the background (even if
        the model left the catalog); an unknown one only an operator can clear."""
        recipe = orphan_recipe(job, models)
        if recipe == UNKNOWN:
            self.set(job["id"], UNKNOWN, ORPHAN_UNKNOWN.format(jid=job["id"], model=job.get("resolved") or job.get("requested")))
        elif recipe is not None:
            self.set(job["id"], recipe, ORPHAN_HELD, retry_now=True)

    def clear(self, by: str) -> Row | None:
        """Clear the hold; returns what was held (None if nothing was)."""
        held = self.get()
        if held is not None:
            self.store.set_flag(FLAG, None)
            self.store.event(Event.GPU_HELD_CLEARED, held.get("job"), by=by)
        self.changed.set()
        return held

    def held(self) -> bool:
        """True while held; starts the held job's clean in the background when one is due."""
        if (h := self.get()) is None:
            return False
        with self._lock:
            idle = self._worker is None or not self._worker.is_alive()
            if idle and h.get("recipe") and self.clock() >= self._next:   # no recipe: operator only
                self._worker = threading.Thread(target=self._clean, args=(h,), daemon=True, name="gpu-hold-clean")
                self._worker.start()
        return True

    def wait(self) -> None:
        """Sleep until a clear, shutdown (`changed`), or the next clean is due."""
        with self._lock:
            left = max(self._next - self.clock(), 0.0) if self._worker is None or not self._worker.is_alive() \
                else self.retry_s
        self.changed.wait(left or self.retry_s)
        self.changed.clear()

    def _clean(self, h: Row) -> None:
        try:
            self.clean(h["recipe"], h["job"])
        except Exception:  # noqa: BLE001 — any failure keeps the hold
            with self._lock:
                self._next = self.clock() + self.retry_s
            self.changed.set()   # let the GPU thread re-arm the retry timer
            return
        if (now := self.get()) is not None and now.get("job") == h["job"]:   # not replaced meanwhile
            self.clear(BY_CLEAN)
