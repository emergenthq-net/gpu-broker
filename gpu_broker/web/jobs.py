"""Job routes: the native job API and read-only views (chat is in chat.py).

    POST /v1/jobs              {model, kind?, caps?, ...request, wait?, wait_s?}
    GET  /v1/jobs/{id}         state, what is being used, outputs
    GET  /v1/status            residency, queue, downloads, recent jobs
    GET  /v1/events?since=N    event log
    GET  /v1/catalog           the catalog's models (dashboard)
    GET  /v1/models            ready LLMs: OpenAI's list shape, Anthropic's with an anthropic-version header
                               (models_router: the one read route client keys may use)
"""
from __future__ import annotations

from http import HTTPStatus
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from ..broker import Broker
from ..constants import ANTHROPIC_VERSION_HEADER, PRIORITY_HEADER, REQUESTER_HEADER
from ..store import Row
from .anthropic_resp import models as anthropic_models
from .openai import model_list

UNKNOWN_CLIENT = "unknown"
CLIENT_STATE = "client"   # request.state attribute set by auth.py: a client key's name, None for the main token
CLIENT_ID_STATE = "client_id"   # ... and that key's id (names repeat; ownership compares ids)


def client(request: Request) -> str:
    return request.client.host if request.client else UNKNOWN_CLIENT


def requester(request: Request, claimed: str | None = None) -> str:
    """Who is asking: a client key's own name (what it claims is ignored, so one key cannot pose
    as another), else what a main-token caller says, else its address."""
    return getattr(request.state, CLIENT_STATE, None) or claimed or client(request)


def identity(request: Request) -> str:
    """Who owns what a request leaves behind (a stored response): the client key, or the main
    token plus the x-requester it names. Never the address: callers behind one proxy differ."""
    kid = getattr(request.state, CLIENT_ID_STATE, None)
    return f"key:{kid}" if kid else f"main:{request.headers.get(REQUESTER_HEADER) or ''}"


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
        jid, info = broker.submit(body, requester(request, body.get("requester")),
                                  priority=request.headers.get(PRIORITY_HEADER, ""))
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
        return {"gpu_held": broker.scheduler.hold.get(), "resident_llm": broker.residency.current, "last_comfy": broker.residency.last_comfy,
                "session": broker.sessions.view(), "running": current and broker.view(current),
                "inflight": [broker.view(j) for j in inflight], "queue": [broker.view(j) for j in queued],
                "downloads": broker.store.downloads(limits.status_downloads), "recent": broker.store.jobs(limits.status_recent)}

    @r.get("/v1/events")
    def events(since: int = 0, limit: int = limits.events_page) -> list[Row]:
        return broker.store.events(since, min(limit, limits.events_page))

    @r.get("/v1/catalog")
    def catalog() -> dict[str, Any]:
        return dict(broker.catalog.models)

    return r


def models_router(broker: Broker) -> APIRouter:
    r = APIRouter()

    @r.get("/v1/models")
    def models(request: Request) -> dict[str, Any]:
        if ANTHROPIC_VERSION_HEADER in request.headers:
            return anthropic_models(broker.catalog)
        return model_list(broker.catalog)

    return r
