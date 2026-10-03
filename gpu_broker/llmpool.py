"""Concurrent calls to the resident LLM, and who may make them.

An LLM server started with N parallel slots (llama.cpp `-np N`, vLLM batching) answers N
requests at once, and batched decoding gives far more aggregate tokens/s than one stream.

Residency: the GPU thread closes the pool and drains it before any switch, and reopens it
naming the model that is now resident. Nothing is mid-call when a server stops, and a
caller that bypasses the job queue (interactive chat) can only ever reach the model the
pool says is resident.

Priority: background calls may hold `slots - reserved_interactive` slots; interactive calls
may use every slot, so a person in a chat UI never waits behind batch work.
"""
from __future__ import annotations

import threading
from collections.abc import Callable, Mapping
from typing import Any

from .backends import Backends
from .catalog import Model
from .constants import EMBED_KEY, ERR_JOB, JobState
from .store import Store

DEFAULT_SLOTS = 1
MIN_SLOTS = 1


def slots(model: Model) -> int:
    return int(model.get("slots", DEFAULT_SLOTS))


def limit(model: Model, interactive: bool) -> int:
    """Slots a caller of this priority may fill."""
    reserved = 0 if interactive else int(model.get("reserved_interactive", 0))
    return max(MIN_SLOTS, slots(model) - reserved)


class LlmPool:
    def __init__(self, store: Store, backends: Backends, on_finish: Callable[[], None]) -> None:
        self.store, self.backends, self.on_finish = store, backends, on_finish
        self._cv = threading.Condition()
        self._inflight: set[str] = set()
        self._resident: str | None = None   # the model direct callers may reach; None while switching

    @property
    def resident(self) -> str | None:
        with self._cv:
            return self._resident

    def busy(self) -> bool:
        with self._cv:
            return bool(self._inflight)

    def ids(self) -> set[str]:
        with self._cv:
            return set(self._inflight)

    def close_and_drain(self, timeout: float | None = None) -> bool:
        """Stop admitting direct calls, then wait until no call is in flight; True if drained."""
        with self._cv:
            self._resident = None
            self._cv.notify_all()
            return self._cv.wait_for(lambda: not self._inflight, timeout)

    def reopen(self, key: str | None) -> None:
        with self._cv:
            self._resident = key
            self._cv.notify_all()

    def acquire(self, jid: str, key: str, model: Model, interactive: bool, direct: bool) -> bool:
        """Take a slot. A `direct` caller gets False if `key` is not resident, or stops being
        resident while it waits; the GPU thread (direct=False) always gets its slot."""
        cap = limit(model, interactive)
        with self._cv:
            def gone() -> bool:
                return direct and self._resident != key
            self._cv.wait_for(lambda: gone() or len(self._inflight) < cap)
            if gone():
                return False
            self._inflight.add(jid)
        self.store.update_job(jid, state=JobState.RUNNING)
        return True

    def release(self, jid: str) -> None:
        with self._cv:
            self._inflight.discard(jid)
            self._cv.notify_all()
        self.on_finish()

    def dispatch(self, jid: str, key: str, model: Model, payload: Mapping[str, Any], interactive: bool) -> None:
        """Start one queued call. Blocks the GPU thread while its slots are busy, which keeps
        the queue's FIFO order."""
        self.acquire(jid, key, model, interactive, direct=False)
        threading.Thread(target=self._call, args=(jid, model, payload), daemon=True).start()

    def _call(self, jid: str, model: Model, payload: Mapping[str, Any]) -> None:
        try:
            call = self.backends.llm_embed if payload.get(EMBED_KEY) else self.backends.llm_chat
            self.store.update_job(jid, state=JobState.DONE, result=call(model, payload))
        except Exception as e:  # noqa: BLE001 — any failure is reported to the requester, never raised
            self.store.update_job(jid, state=JobState.FAILED, error=str(e)[:ERR_JOB])
        finally:
            self.release(jid)
