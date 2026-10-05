"""Circuit breakers: one per provider for "is it reachable", one per (provider, credential) for
"does this key still have quota". In memory only; a restart starts every breaker closed.

closed --N consecutive failures--> open --probe due--> half-open --success--> closed
                                     ^-------------------- failure (backoff doubles) --'
While a breaker is open, requests go straight to the next entry of their chain: nothing waits
on a timeout. When the probe is due, one caller (the background prober, or else the next real
request) is the half-open trial; everyone else keeps going local until it reports back.
A quota breaker opens at once, for `quota_probe_s` or as long as the provider's Retry-After or
reset headers say, whichever is longer; its trial is the next real request with that key,
because a model list answers fine with an empty balance and so proves nothing.
Credentials are keyed by a SHA-256 fingerprint and never kept past their request. Only quota
breakers that are not closed are kept, at most KEYS_MAX of them (least recently used dropped).
Breaker changes are reported (Notify: the store's sqlite and JSONL) after the lock is released.
"""
from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from .config import BreakerCfg

Notify = Callable[[str, str, str], None]   # (provider, new state, reason)
KEYS_MAX = 1024   # quota breakers kept, one per (provider, credential) out of quota


class State(StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half-open"


@dataclass
class Breaker:
    cfg: BreakerCfg
    threshold: int
    state: State = State.CLOSED
    failures: int = 0
    next_probe: float = 0.0
    backoff: float = 0.0
    trial_at: float = 0.0
    reason: str = ""
    quota: bool = False

    def ready(self, now: float) -> bool:
        """Would `allow` let a call through (without making it the trial)?"""
        stale = self.state is State.HALF_OPEN and now - self.trial_at >= self.cfg.trial_s
        return self.state is State.CLOSED or (self.state is State.OPEN and now >= self.next_probe) or stale

    def allow(self, now: float) -> bool:
        """May this call go to the provider? Due while open: this call becomes the trial."""
        if not self.ready(now):
            return False
        if self.state is not State.CLOSED:
            self.state, self.trial_at = State.HALF_OPEN, now
        return True

    def probe_due(self, now: float) -> bool:
        return self.state is State.OPEN and not self.quota and now >= self.next_probe

    def success(self) -> bool:
        """Closes it; True if that is a change."""
        changed = self.state is not State.CLOSED
        self.state, self.failures, self.backoff, self.reason, self.quota = State.CLOSED, 0, 0.0, "", False
        return changed

    def failure(self, now: float, reason: str) -> bool:
        """One more failure; True if the breaker is now open and was not before."""
        self.failures += 1
        self.reason = reason
        if self.state is State.HALF_OPEN:
            self.backoff = min(self.cfg.probe_max_s, max(self.cfg.probe_s, self.backoff * 2))
        elif self.state is State.CLOSED and self.failures >= self.threshold:
            self.backoff = self.cfg.probe_s
        else:
            return False
        was = self.state
        self.state, self.next_probe = State.OPEN, now + self.backoff
        return was is State.CLOSED

    def exhausted(self, now: float, reason: str, wait_s: float) -> bool:
        """A quota error: open at once for the longer of quota_probe_s and the provider's wait."""
        was = self.state
        self.backoff = max(self.cfg.quota_probe_s, wait_s)
        self.state, self.next_probe, self.reason, self.quota = State.OPEN, now + self.backoff, reason, True
        return was is State.CLOSED

    def view(self, now: float) -> dict[str, Any]:
        out: dict[str, Any] = {"state": self.state.value, "failures": self.failures}
        if self.state is not State.CLOSED:
            out |= {"reason": self.reason, "quota": self.quota, "retry_in_s": round(max(0.0, self.next_probe - now), 1)}
        return out


def fingerprint(credential: str) -> str:
    return hashlib.sha256(credential.encode()).hexdigest()


class Board:
    """Every breaker, behind one lock; what changed is reported once the lock is released."""

    def __init__(self, cfg: BreakerCfg, providers: list[str], notify: Notify | None = None) -> None:
        self.cfg, self.notify = cfg, notify or (lambda *_: None)
        self.lock = threading.Lock()
        self.links = {p: Breaker(cfg, cfg.failures) for p in providers}
        self.keys: OrderedDict[tuple[str, str], Breaker] = OrderedDict()

    def _report(self, changes: list[tuple[str, str, str]]) -> None:
        for change in changes:
            self.notify(*change)

    def allow(self, provider: str, cred: str, now: float) -> str:
        """"" if the call may go to the provider, else why not."""
        with self.lock:
            k = (provider, fingerprint(cred))
            key, link = self.keys.get(k), self.links[provider]
            if key is not None:
                self.keys.move_to_end(k)
                if not key.ready(now):
                    return f"{provider} quota exhausted ({key.reason})"
            if not link.allow(now):
                return f"{provider} circuit open ({link.reason})"
            if key is not None:
                key.allow(now)   # a due quota breaker: this call is its trial
            return ""

    def success(self, provider: str, cred: str) -> None:
        with self.lock:
            key = self.keys.pop((provider, fingerprint(cred)), None)   # closed: nothing to keep
            changed = self.links[provider].success() or key is not None
        self._report([(provider, State.CLOSED, "")] if changed else [])

    def reachable(self, provider: str) -> None:
        """It answered (a probe, or a client error): close the provider's breaker, not any key's."""
        with self.lock:
            changed = self.links[provider].success()
        self._report([(provider, State.CLOSED, "")] if changed else [])

    def failure(self, provider: str, now: float, reason: str) -> None:
        with self.lock:
            opened = self.links[provider].failure(now, reason)
        self._report([(provider, State.OPEN, reason)] if opened else [])

    def exhausted(self, provider: str, cred: str, now: float, reason: str, wait_s: float) -> None:
        with self.lock:
            self.links[provider].success()   # it answered: the provider itself is reachable
            k = (provider, fingerprint(cred))
            key = self.keys.pop(k, None) or Breaker(self.cfg, 1)
            self.keys[k] = key
            while len(self.keys) > KEYS_MAX:
                self.keys.popitem(last=False)
            opened = key.exhausted(now, reason, wait_s)
        self._report([(provider, "quota", reason)] if opened else [])

    def due(self, now: float, probeable: set[str]) -> list[str]:
        """Providers in `probeable` whose probe is due, each now marked half-open."""
        with self.lock:
            return [n for n, link in self.links.items() if n in probeable and link.probe_due(now) and link.allow(now)]

    def next_due(self, probeable: set[str]) -> float | None:
        """When the earliest probe of a provider in `probeable` falls due (None: nothing open)."""
        with self.lock:
            return min((link.next_probe for n, link in self.links.items()
                        if n in probeable and link.state is State.OPEN and not link.quota), default=None)

    def view(self, now: float) -> dict[str, Any]:
        with self.lock:
            return {name: link.view(now) | {"keys_out_of_quota": sum(
                1 for (p, _), b in self.keys.items() if p == name and b.state is not State.CLOSED)}
                for name, link in self.links.items()}
