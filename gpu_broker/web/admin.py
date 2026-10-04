"""Deploy support: quiesce the broker so a restart cuts no LLM call mid-flight.

A broker with steady LLM traffic is never idle, so a deploy cannot wait for a gap.
POST /v1/admin/quiesce stops the GPU thread taking new jobs, closes the direct chat path,
and waits (up to `wait_s`) for in-flight calls to finish. While quiesced, new jobs and chats get
HTTP 503 with Retry-After (quiesce.py) instead of being queued. Jobs already queued wait, and
the next start re-queues them in order (Broker.start); only work that had started is orphaned.
POST /v1/admin/resume undoes a quiesce without restarting.

POST /v1/admin/gpu-held/clear acknowledges a GPU hold (holds.py) after an operator has made
sure the held job's processes are gone; resume and restarts leave a hold in place.
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from ..broker import Broker
from ..constants import Event
from ..holds import BY_OPERATOR


def router(broker: Broker) -> APIRouter:
    r = APIRouter()
    sched, max_wait = broker.scheduler, broker.settings.timeouts.quiesce_wait_s

    @r.post("/v1/admin/quiesce")
    def quiesce(wait_s: float = max_wait) -> dict[str, Any]:
        with sched.admission:   # from here every new job or chat is refused with 503 (quiesce.py)
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
        return {"paused": False, "gpu_held": sched.hold.get() is not None}

    @r.post("/v1/admin/gpu-held/clear")
    def clear_gpu_held() -> dict[str, Any]:
        return {"cleared": sched.hold.clear(BY_OPERATOR)}

    return r
