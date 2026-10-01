"""Interactive ComfyUI sessions: hold the GPU for a person using ComfyUI directly.

A session is an ordinary GPU job, so it queues like any other. While it runs, the LLM is
stopped and ComfyUI has the whole card. It ends when ComfyUI has had nothing queued for
`session_idle_s` (shortened to `session_yield_s` while another job is waiting), when the
user ends it, or at `session_max_s`. A render in progress, or a ComfyUI that cannot be
asked, counts as activity, so a session never ends mid-render.
"""
from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

from .catalog import Defaults

SECONDS_PER_MINUTE = 60
IDLE_MINUTES_KEY = "idle_min"   # request field overriding the idle limit


class End:
    MAX, USER, IDLE, YIELDED = "max", "ended by user", "idle", "idle, yielded to queued job"


@dataclass
class Active:
    job: str
    model: str
    started: float
    last_busy: float
    idle_s: float
    limit_s: float
    end: bool = False


class Sessions:
    def __init__(self, defaults: Defaults, queue_len: Callable[[], int | None], poll_s: float,
                 clock: Callable[[], float] = time.time, sleep: Callable[[float], None] = time.sleep) -> None:
        self.d, self.queue_len, self.poll_s = defaults, queue_len, poll_s
        self.clock, self.sleep = clock, sleep
        self.queued_jobs: Callable[[], bool] = lambda: False   # wired to the scheduler by the broker
        self._active: Active | None = None
        self._lock = threading.Lock()

    def hold(self, jid: str, model: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Block until the session ends; returns how long it held the GPU and why it ended."""
        idle_s = float(payload.get(IDLE_MINUTES_KEY, 0)) * SECONDS_PER_MINUTE or self.d["session_idle_s"]
        start = self.clock()
        with self._lock:
            self._active = a = Active(jid, model, start, start, idle_s, idle_s)
        reason = End.MAX
        try:
            while self.clock() - start < self.d["session_max_s"]:
                n, now = self.queue_len(), self.clock()
                limit = self.d["session_yield_s"] if self.queued_jobs() else idle_s
                with self._lock:
                    if n != 0:
                        a.last_busy = now
                    a.limit_s = limit
                    if a.end:
                        reason = End.USER
                        break
                if now - a.last_busy >= limit:
                    reason = End.IDLE if limit == idle_s else End.YIELDED
                    break
                self.sleep(self.poll_s)
        finally:
            with self._lock:
                self._active = None
        return {"session": True, "model": model, "held_s": round(self.clock() - start), "ended": reason}

    def view(self) -> dict[str, Any] | None:
        with self._lock:
            if self._active is None:
                return None
            out = asdict(self._active)
        out["idle_for_s"] = round(self.clock() - out["last_busy"])
        out["ends_in_s"] = max(0, round(out["limit_s"] - out["idle_for_s"]))
        return out

    def end(self) -> bool:
        with self._lock:
            if self._active is None:
                return False
            self._active.end = True
            return True
