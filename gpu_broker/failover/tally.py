"""Failover events, aggregated: one `upstream.failover` per (provider, local model) per window.

The first failover of a window is recorded at once; the rest are counted and recorded as one
event (with their count) when the window ends, so an outage costs one row a minute rather than
one per request. Emitting happens outside the lock.
"""
from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ..constants import Event

WINDOW_S = 60.0


@dataclass
class Window:
    start: float
    model: str
    reason: str
    count: int = 0   # failovers since the one recorded when the window opened


class Tally:
    def __init__(self, emit: Callable[..., None], clock: Callable[[], float], window_s: float = WINDOW_S) -> None:
        self.emit, self.clock, self.window_s = emit, clock, window_s
        self.lock = threading.Lock()
        self.windows: dict[tuple[str, str], Window] = {}

    def add(self, provider: str, to: str, model: str, reason: str) -> None:
        now, out = self.clock(), []
        with self.lock:
            w = self.windows.get((provider, to))
            if w is None or now - w.start >= self.window_s:
                if w is not None and w.count:
                    out.append(self._fields(provider, to, w))
                self.windows[(provider, to)] = Window(now, model, reason)
                out.append({"provider": provider, "to": to, "model": model, "reason": reason, "count": 1})
            else:
                w.count, w.model, w.reason = w.count + 1, model, reason
        for fields in out:
            self.emit(Event.UPSTREAM_FAILOVER, **fields)

    def flush(self) -> None:
        """Record every window that has ended."""
        now, out = self.clock(), []
        with self.lock:
            for key, w in list(self.windows.items()):
                if now - w.start >= self.window_s:
                    del self.windows[key]
                    if w.count:
                        out.append(self._fields(*key, w))
        for fields in out:
            self.emit(Event.UPSTREAM_FAILOVER, **fields)

    def next_flush(self) -> float | None:
        with self.lock:
            return min((w.start + self.window_s for w in self.windows.values()), default=None)

    @staticmethod
    def _fields(provider: str, to: str, w: Window) -> dict[str, Any]:
        return {"provider": provider, "to": to, "model": w.model, "reason": w.reason, "count": w.count}
