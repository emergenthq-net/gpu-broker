"""The demo's traffic: a small team chatting with the resident model while, now and then,
someone asks for an image or a video. Requests go through `Broker.submit`, the same path
`POST /v1/jobs` takes, so the queue, switches and idle restore are the real ones.

The script (tuning.SCENARIO) repeats until the demo stops. A submit only queues the request,
so chat requests sent a second apart overlap on the model server as they would in a team.
"""
from __future__ import annotations

import logging
import random
import threading
from typing import Any

from ..broker import Broker
from ..constants import APP_NAME, TERMINAL
from . import content
from .tuning import Step, Traffic

log = logging.getLogger(APP_NAME)


class TrafficGenerator:
    def __init__(self, broker: Broker, traffic: Traffic, rng: random.Random) -> None:
        self.broker, self.t, self.rng = broker, traffic, rng
        self.stop = threading.Event()
        self.submitted: list[str] = []   # job ids, in order (tests read them)
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self.loop, daemon=True, name="demo-traffic")
        self._thread.start()

    def close(self) -> None:
        """Stop the script and wait (briefly) for it, so nothing is submitted once the broker stops."""
        self.stop.set()
        if self._thread:
            self._thread.join(self.t.close_s)

    def loop(self) -> None:
        if self.stop.wait(self.t.start_s):
            return
        while not self.stop.is_set():
            for step in self.t.steps:
                if self.stop.is_set():
                    return
                self.run(step)

    def run(self, step: Step) -> None:
        jid = self.submit({"model": step.job, "prompt": step.prompt}, content.JOB_REQUESTER) if step.job else None
        self.chat_for(step.chat_s)
        if step.quiet_s:
            if jid:   # the quiet starts when the job is done, not when it was queued
                self.wait_done(jid)
            self.stop.wait(step.quiet_s)

    def wait_done(self, jid: str) -> None:
        """Wait for a job to end, up to job_wait_s, in slices that notice a stop."""
        left = self.t.job_wait_s
        while left > 0 and not self.stop.is_set():
            j = self.broker.wait(jid, min(self.t.wait_slice_s, left))
            if j is None or j["state"] in TERMINAL:
                return
            left -= self.t.wait_slice_s

    def chat_for(self, seconds: float) -> None:
        """Send chat requests at random gaps for `seconds` (or until stopped)."""
        left = seconds
        while left > 0 and not self.stop.is_set():
            body = {"model": content.CHAT_MODEL, "max_tokens": self.t.max_tokens,
                    "messages": [{"role": "user", "content": self.rng.choice(content.CHAT_PROMPTS)}]}
            self.submit(body, self.rng.choice(content.REQUESTERS))
            gap = self.rng.uniform(*self.t.chat_gap_s)
            self.stop.wait(min(gap, left))
            left -= gap

    def submit(self, body: dict[str, Any], requester: str) -> str | None:
        try:
            jid, _ = self.broker.submit(body, requester)
        except Exception as e:  # noqa: BLE001 — a failed simulated request must not end the demo
            log.warning("demo traffic: %s", e)
            return None
        self.submitted.append(jid)
        return jid
