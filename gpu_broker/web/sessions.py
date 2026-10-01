"""Interactive-session routes: borrow the GPU for ComfyUI from the dashboard, and give it back."""
from __future__ import annotations

from http import HTTPStatus
from typing import Any

from fastapi import APIRouter, HTTPException

from ..broker import Broker
from ..constants import SESSION_KEY, ModelStatus, Runner
from ..sessions import IDLE_MINUTES_KEY

DEFAULT_REQUESTER = "dashboard"


def router(broker: Broker) -> APIRouter:
    r = APIRouter()

    @r.post("/v1/sessions")
    def start(body: dict[str, Any]) -> dict[str, Any]:
        key = body.get("model")
        m = broker.catalog.models.get(key) if isinstance(key, str) else None
        if m is None or m.get("runner") != Runner.COMFY or m.get("status") != ModelStatus.READY:
            raise HTTPException(HTTPStatus.BAD_REQUEST, f"{key!r} is not a ready ComfyUI model")
        idle = body.get(IDLE_MINUTES_KEY, 0)
        if not isinstance(idle, (int, float)) or idle < 0:
            raise HTTPException(HTTPStatus.BAD_REQUEST, f"`{IDLE_MINUTES_KEY}` must be a non-negative number")
        jid, info = broker.submit({"model": key, SESSION_KEY: True, IDLE_MINUTES_KEY: idle},
                                  str(body.get("requester") or DEFAULT_REQUESTER))
        return {"id": jid, **info, "comfy_template": m.get("comfy_template"), "open_url": m.get("open_url")}

    @r.post("/v1/sessions/end")
    def end() -> dict[str, bool]:
        return {"ended": broker.sessions.end()}

    return r
