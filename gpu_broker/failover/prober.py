"""The background prober: wakes when the earliest open breaker's probe falls due (or a breaker
changes), probes each due provider on its own thread with a short timeout, so one hung provider
holds up no other, and records the failover windows that have ended.

Only providers with an operator key (`key_env`) are probed: client keys are never kept for it.
Without one, the provider's next real request is its half-open trial once the probe is due.
"""
from __future__ import annotations

import dataclasses
import threading
from typing import TYPE_CHECKING

from . import transport
from .classify import Kind, classify, failed
from .config import Api
from .creds import DEFAULT_ANTHROPIC_VERSION, Cred, auth_headers

if TYPE_CHECKING:
    from .router import Router

MODELS_PATH = "/v1/models"


class Prober:
    def __init__(self, router: Router) -> None:
        self.r = router
        self.wake, self.stopped = threading.Event(), threading.Event()
        self.timeouts = dataclasses.replace(router.cfg.timeouts, response_s=router.cfg.timeouts.probe_s)

    def probeable(self) -> set[str]:
        return {p.name for p in self.r.cfg.providers.values() if p.configured_key(self.r.env)}

    def probe_once(self, block: bool = True) -> list[threading.Thread]:
        """Start a probe for every provider that is due; with `block`, wait for them all."""
        threads = [threading.Thread(target=self._probe, args=(name,), name=f"upstream-probe-{name}", daemon=True)
                   for name in self.r.board.due(self.r.clock(), self.probeable())]
        for t in threads:
            t.start()
        if block:
            for t in threads:
                t.join()
        return threads

    def _probe(self, name: str) -> None:
        p = self.r.cfg.providers[name]
        hdrs = auth_headers(p.api, Cred(p.configured_key(self.r.env)))
        if p.api is Api.ANTHROPIC:
            hdrs["anthropic-version"] = DEFAULT_ANTHROPIC_VERSION
        try:
            a = self.r.send(p.url + MODELS_PATH, "GET", hdrs, None, False, self.timeouts)
        except transport.Unreachable as e:
            self.r.board.failure(name, self.r.clock(), failed(name, str(e)).reason)
            return
        if classify(name, a.status, a.headers, a.body, self.r.wall).kind is Kind.FAIL:
            self.r.board.failure(name, self.r.clock(), f"{name} probe {a.status}")
        else:   # any other answer, a 401 included, means the provider is back
            self.r.board.reachable(name)

    def delay(self) -> float | None:
        """Seconds until there is something to do (None: nothing until a breaker changes)."""
        due = [t for t in (self.r.board.next_due(self.probeable()), self.r.tally.next_flush()) if t is not None]
        return max(0.0, min(due) - self.r.clock()) if due else None

    def run(self) -> None:
        while not self.stopped.is_set():
            self.wake.wait(self.delay())
            self.wake.clear()
            if not self.stopped.is_set():
                self.probe_once(block=False)
                self.r.tally.flush()

    def stop(self) -> None:
        self.stopped.set()
        self.wake.set()
