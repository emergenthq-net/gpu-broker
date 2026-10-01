"""Deploy support: quiesce the broker so a restart cuts no LLM call mid-flight.

A broker with steady LLM traffic is never idle, so a deploy cannot wait for a gap.
POST /v1/admin/quiesce stops the GPU thread taking new jobs, closes the direct chat path,
and waits (up to `wait_s`) for in-flight calls to finish. Jobs still queued at the restart are failed at startup as
orphans; clients retry them. POST /v1/admin/resume undoes a quiesce without restarting.
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from ..broker import Broker
from ..constants import Event


def router(broker: Broker) -> APIRouter:
    r = APIRouter()
    sched, max_wait = broker.scheduler, broker.settings.timeouts.quiesce_wait_s

    @r.post("/v1/admin/quiesce")
    def quiesce(wait_s: float = max_wait) -> dict[str, Any]:
        sched.paused.set()
        broker.store.event(Event.QUIESCE, inflight=len(sched.pool.ids()))
        drained = sched.pool.close_and_drain(timeout=max(0.0, min(wait_s, max_wait)))   # direct chat too
        queued, running, inflight = sched.snapshot()
        return {"drained": drained, "inflight": inflight, "running": running, "queued": len(queued)}

    @r.post("/v1/admin/resume")
    def resume() -> dict[str, bool]:
        sched.pool.reopen(broker.residency.current)
        sched.paused.clear()
        broker.store.event(Event.RESUME)
        return {"paused": False}

    return r
