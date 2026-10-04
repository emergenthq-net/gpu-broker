"""The routes' side of cloud -> local failover (gpu_broker/failover): pick out the client's own
provider key, send the request down its chain, and turn the outcome into a response.

`attempt` returns a Response (the provider answered, or every provider failed and the chain has
no local model: 503 in the request's API shape), a Fallback (run the broker path with its
local model and add its headers), or None (no route matches the model: unchanged behaviour).

The client's provider key is whichever of `x-api-key` / `Authorization: Bearer` did not carry
a broker credential (auth.py records which did), and it goes upstream in the same header. A client that passes its own key through sends
its broker credential as `x-gpu-broker-key`. Every answer says who served it:
`x-gpu-broker-served-by: anthropic` or `local:<model>`, and `x-gpu-broker-fallback: <why>`
when the chain moved on. Credentials are never logged, stored, or put in an event.
"""
from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from http import HTTPStatus
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response, StreamingResponse

from ..connect.core import KEY_PREFIX
from ..constants import API_KEY_HEADER, PASSTHROUGH_PATH, SERVED_BY_HEADER
from ..failover.config import Api
from ..failover.creds import Cred
from ..failover.outcome import Exhausted, Local, Served
from ..failover.router import Router
from ..resolve import lookup
from .auth import AUTH_HEADER, BROKER_CREDS, MAIN_CHECK, presented

SERVED_BY = "x-gpu-broker-served-by"
FALLBACK = "x-gpu-broker-fallback"
HEADER_MAX = 300
UNSAFE = re.compile(r"[^\x20-\x7e]")
SSE = "text/event-stream"
HOSTED = "hosted"   # the older x-broker-served-by value for a provider's answer
NONE_LEFT = "no upstream could answer and this model has no local fallback: {why}"
NOT_LOCAL = "no upstream could answer, and only the provider can continue this conversation: {why}"
RELAYED = ("content-type", "retry-after", "request-id", "x-request-id")   # provider headers passed back


def header_value(text: str) -> str:
    return UNSAFE.sub("?", text)[:HEADER_MAX]


@dataclass
class Fallback:
    model: str
    headers: dict[str, str] = field(default_factory=dict)


def broker_shaped(request: Request, value: str) -> bool:
    """Never forwarded, whatever auth made of it: the main token, or anything that looks like a
    broker client key (revoked and unknown ones included)."""
    is_main: Callable[[str], bool] = getattr(request.state, MAIN_CHECK, lambda _: False)
    return value.lower().startswith(KEY_PREFIX) or is_main(value)


def client_key(request: Request, api: Api) -> Cred | None:
    """The client's own provider key, if it sent one that is not a broker credential, with the
    header it came in (Anthropic takes x-api-key or Bearer; OpenAI-style APIs take Bearer)."""
    broker: set[str] = getattr(request.state, BROKER_CREDS, set())
    sent = {h: v for h, v in presented(request).items() if h not in broker and not broker_shaped(request, v)}
    if api is Api.ANTHROPIC and (key := sent.get(API_KEY_HEADER)):
        return Cred(key, bearer=False)
    return Cred(value, bearer=True) if (value := sent.get(AUTH_HEADER)) else None


def attempt(router: Router | None, api: Api, path: str, body: Mapping[str, Any], request: Request,
            stream: bool, local: bool = True) -> Response | Fallback | None:
    """`local=False`: the request continues a conversation held by the provider, so it never goes local."""
    if router is None or not router.cfg.enabled:
        return None
    got = router.route(api, path, str(body.get("model") or ""), body, client_key(request, api), request.headers,
                       stream, local)
    if got is None:
        return None
    if isinstance(got, Local):
        hdrs = {SERVED_BY: f"local:{got.model}"} | ({FALLBACK: header_value(got.reason)} if got.reason else {})
        return Fallback(got.model, hdrs)
    if isinstance(got, Exhausted):
        if got.last is not None:   # the provider's own quota answer says the most
            return _response(got.last, got.reason)
        raise HTTPException(HTTPStatus.SERVICE_UNAVAILABLE, (NONE_LEFT if got.local else NOT_LOCAL).format(why=got.reason))
    return _response(got, "; ".join(got.skipped))


def _response(s: Served, why: str) -> Response:
    hdrs = {k: v for k in RELAYED if k != "content-type" and (v := s.headers.get(k))}
    hdrs |= {SERVED_BY: s.provider, SERVED_BY_HEADER: HOSTED} | ({FALLBACK: header_value(why)} if why else {})
    media = s.headers.get("content-type", SSE if s.chunks is not None else "application/json")
    if s.chunks is not None:
        return StreamingResponse(s.chunks, status_code=s.status, media_type=media, headers=hdrs)
    return Response(s.body, status_code=s.status, media_type=media, headers=hdrs)


def make_router(broker: Any, env: Mapping[str, str] | None = None) -> Router:
    """The broker's Router, its local fallbacks checked against the catalog."""
    cfg, cat = broker.settings.upstreams, broker.catalog
    unknown = sorted(n for n in cfg.local_names() if lookup(cat.data, n) is None and cat.variant(n) is None)
    if unknown:
        raise ValueError(f"upstreams.routes: {unknown} are neither providers nor catalog models")
    return Router(cfg, broker.store.event, env)


def status_router(cloud: Router) -> APIRouter:
    r = APIRouter()

    @r.get("/v1/upstreams")
    def upstreams() -> dict[str, Any]:
        """Each provider's breaker, and the routes (no credentials: there are none to show)."""
        return cloud.view()

    return r


def passthrough_router(cloud: Router) -> APIRouter:
    """For client keys (model scope): only which APIs pass a client's own key through, so `connect`
    run with nothing but its own key can tell a tool to keep its provider key. No breakers, routes,
    URLs or provider names: those stay operator-only on /v1/upstreams."""
    r = APIRouter()

    @r.get(PASSTHROUGH_PATH)
    def passthrough() -> dict[str, Any]:
        return {"enabled": cloud.cfg.enabled,
                "apis": sorted({p.api.value for p in cloud.cfg.providers.values() if p.pass_client_key})}

    return r
