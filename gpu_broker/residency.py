"""GPU residency: make exactly the model a job needs resident, evicting whatever is in the way.

Two kinds of tenant share the card. LLM servers are units the driver starts and stops
whole; at most one is resident. ComfyUI stays running (its idle footprint is small) and
loads models per job, releasing them on POST /free.

Only the scheduler's GPU thread calls `ensure`, and only between jobs, so there is no
locking here. State is re-checked rather than trusted: other operators may stop an LLM or
ComfyUI behind the broker's back, and `last_comfy` does not survive a restart.
"""
from __future__ import annotations

import time
from collections.abc import Callable

from .backends import Backends
from .catalog import Catalog
from .constants import ERR_SHORT, Event, Runner, Verb
from .drivers import DRIVER_ERRORS, Driver
from .settings import Intervals, Timeouts
from .store import Store
from .units import UnitRef, unit_ref

DECIMALS = 1


class Residency:
    def __init__(self, catalog: Catalog, driver: Driver, backends: Backends, store: Store, timeouts: Timeouts,
                 intervals: Intervals, comfy_unit: UnitRef | None = None,
                 clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep) -> None:
        self.catalog, self.driver, self.backends, self.store = catalog, driver, backends, store
        self.t, self.i, self.comfy_unit = timeouts, intervals, comfy_unit
        self.clock, self.sleep = clock, sleep
        self.current: str | None = None      # catalog key of the resident LLM, if any
        self.last_comfy: str | None = None   # last ComfyUI model that ran (its weights may still be loaded)

    def detect(self) -> str | None:
        """At startup: adopt whichever LLM unit is already running. A broken driver is logged,
        not raised, so the API still comes up."""
        self.current = None
        for key, m in self.catalog.llm_units():
            try:
                if self.driver.unit(m["unit"], Verb.IS_ACTIVE):
                    self.current = key
                    break
            except DRIVER_ERRORS as e:
                self.store.event(Event.RES_DETECT_FAILED, model=key, error=str(e)[:ERR_SHORT])
        return self.current

    def healthy(self, key: str) -> bool:
        return self.backends.llm_healthy(self.catalog.models[key])

    def _wait(self, ready: Callable[[], bool], timeout_s: float) -> float | None:
        """Poll `ready` until it holds; seconds waited, or None on timeout."""
        start = self.clock()
        while self.clock() - start < timeout_s:
            if ready():
                return self.clock() - start
            self.sleep(self.i.health_poll_s)
        return None

    def _stop_llm(self, key: str, jid: str | None) -> None:
        self.store.event(Event.RES_STOP, jid, model=key)
        self.driver.unit(self.catalog.models[key]["unit"], Verb.STOP)

    def _start_llm(self, key: str, jid: str | None) -> None:
        unit = self.catalog.models[key]["unit"]
        self.store.event(Event.RES_START, jid, model=key)
        if not self.driver.unit(unit, Verb.START):
            raise RuntimeError(f"could not start {unit_ref(unit).name}")
        waited = self._wait(lambda: self.healthy(key), self.t.llm_start_s)
        if waited is None:
            raise RuntimeError(f"{key} did not become healthy within {self.t.llm_start_s}s")
        self.store.event(Event.RES_READY, jid, model=key, load_s=round(waited, DECIMALS))

    def _comfy_up(self, jid: str | None) -> None:
        """ComfyUI is meant to stay up, but it can be stopped from outside; start it if we may."""
        if self.backends.comfy_alive():
            return
        self.store.event(Event.RES_COMFY_DOWN, jid)
        if self.comfy_unit is None:
            raise RuntimeError("ComfyUI is down and no comfy.unit is configured to start it")
        if not self.driver.unit(self.comfy_unit, Verb.START):
            raise RuntimeError("ComfyUI is down and could not be started")
        if self._wait(self.backends.comfy_alive, self.t.comfy_start_s) is None:
            raise RuntimeError(f"ComfyUI did not come up within {self.t.comfy_start_s}s")
        self.store.event(Event.RES_COMFY_STARTED, jid)

    def ensure(self, key: str, jid: str | None = None) -> None:
        """Make `key` runnable now."""
        if self.catalog.models[key]["runner"] == Runner.LLM_UNIT:
            if self.current == key and self.healthy(key):
                return
            if self.current == key:  # stopped from outside since we last looked
                self.store.event(Event.RES_LOST, jid, model=key)
                self.current = None
            # Free ComfyUI unconditionally: it may have been used directly, outside the broker.
            self.backends.comfy_free()
            self.last_comfy = None
            if self.current:
                self._stop_llm(self.current, jid)
            self._start_llm(key, jid)
            self.current = key
        else:
            if self.current:
                self._stop_llm(self.current, jid)
                self.current = None
            self._comfy_up(jid)
            if self.last_comfy and self.last_comfy != key:
                self.backends.comfy_free()
            self.last_comfy = key
        self.store.event(Event.RES_RESIDENT, jid, model=key)

    def release_comfy(self) -> None:
        """Unload ComfyUI's weights after an image/video run so the default LLM fits again."""
        if self.last_comfy:
            self.backends.comfy_free()
            self.last_comfy = None
