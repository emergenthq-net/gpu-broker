"""Job routes: the native job API and read-only views (chat is in chat.py).

    POST /v1/jobs              {model, kind?, caps?, ...request, wait?, wait_s?}
    GET  /v1/jobs/{id}         state, what is being used, outputs
    GET  /v1/status            residency, queue, downloads, recent jobs
    GET  /v1/events?since=N    event log
    GET  /v1/models            OpenAI-style list of ready LLMs
    GET  /v1/catalog           the catalog's models (dashboard)
"""
from __future__ import annotations

from http import HTTPStatus
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from ..broker import Broker
from ..store import Row
from .openai import model_list

UNKNOWN_CLIENT = "unknown"


def client(request: Request) -> str:
    return request.client.host if request.client else UNKNOWN_CLIENT


def router(broker: Broker) -> APIRouter:
    r = APIRouter()
    limits, timeouts = broker.settings.limits, broker.settings.timeouts

    def view(jid: str) -> Row:
        j = broker.view(jid)
        if j is None:
            raise HTTPException(HTTPStatus.NOT_FOUND, "no such job")
        return j

    @r.post("/v1/jobs")
    def post_job(body: dict[str, Any], request: Request) -> dict[str, Any]:
        jid, info = broker.submit(body, body.get("requester") or client(request))
        if body.get("wait"):
            broker.wait(jid, min(float(body.get("wait_s", timeouts.job_wait_s)), timeouts.job_wait_s))
        return {"id": jid, **info, "job": view(jid)}

    @r.get("/v1/jobs/{jid}")
    def get_job(jid: str) -> Row:
        return view(jid)

    @r.get("/v1/status")
    def status() -> dict[str, Any]:
        queued, running, inflight = broker.scheduler.snapshot()
        current = running if running and running not in inflight else (inflight[0] if inflight else None)
        return {"resident_llm": broker.residency.current, "last_comfy": broker.residency.last_comfy,
                "session": broker.sessions.view(), "running": current and broker.view(current),
                "inflight": [broker.view(j) for j in inflight], "queue": [broker.view(j) for j in queued],
                "downloads": broker.store.downloads(limits.status_downloads), "recent": broker.store.jobs(limits.status_recent)}

    @r.get("/v1/events")
    def events(since: int = 0, limit: int = limits.events_page) -> list[Row]:
        return broker.store.events(since, min(limit, limits.events_page))

    @r.get("/v1/models")
    def models() -> dict[str, Any]:
        return model_list(broker.catalog)

    @r.get("/v1/catalog")
    def catalog() -> dict[str, Any]:
        return dict(broker.catalog.models)

    return r
