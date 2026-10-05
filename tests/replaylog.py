"""Builds small events.jsonl logs for the replay tests, in the broker's own event shapes."""
from __future__ import annotations

import json
import pathlib
from typing import Any

from gpu_broker.catalog import Catalog

FIX = pathlib.Path(__file__).parent / "fixtures"
CAT = Catalog(str(FIX / "catalog.yaml"))


class Log:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def ev(self, ts: float, kind: str, jid: str | None = None, **data: Any) -> Log:
        self.events.append({"ts": ts, "job_id": jid, "kind": kind, **data})
        return self

    def job(self, jid: str, at: float, model: str, run_s: float, requester: str = "a",
            wait_s: float = 0.0, switch_s: float = 0.0, end: str = "done") -> Log:
        """A job that queued at `at`, started `wait_s` later (after `switch_s` of switching) and ran `run_s`."""
        self.ev(at, "job.received", jid, requester=requester, requested=model).ev(at, "job.queued", jid)
        t = at + wait_s
        if switch_s:
            self.ev(t, "job.switching", jid)
        self.ev(t + switch_s, "job.running", jid)
        return self.ev(t + switch_s + run_s, f"job.{end}", jid)

    def sorted(self) -> list[dict[str, Any]]:
        return sorted(self.events, key=lambda e: e["ts"])

    def write(self, path: pathlib.Path) -> pathlib.Path:
        path.write_text("".join(json.dumps(e) + "\n" for e in self.sorted()))
        return path
