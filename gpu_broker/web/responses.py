"""POST /v1/responses — the OpenAI Responses API, for Codex and the Responses SDK.

The request is translated to an OpenAI chat request (responses_req.py) and goes through the
same broker path as /v1/chat/completions (completion.py: name mapping, direct lease or
queued job). The answer comes back as a `response` object, or with `stream: true` as the
Responses event sequence (responses_out.py). `previous_response_id` continues a conversation
kept in memory for a while (responses_store.py), for the identity that created it (the client
key, or the main token plus its x-requester). The `model` field always echoes the name the
caller asked for; what really ran is in `x_broker`. Errors use OpenAI's error shape; a body
over `limits.responses_body_bytes` is a 413 before any of it is parsed. With `upstreams:` routes,
a matching name goes to its cloud provider first and falls back to a local model (failover.py);
a conversation continued by `previous_response_id` is local-only and is never sent upstream.
"""
from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from http import HTTPStatus
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from ..broker import Broker
from ..failover.config import Api as UpApi
from ..failover.router import Router
from . import completion, failover
from .anthropic_resp import new_id
from .jobs import identity
from .openai import SSE_MEDIA_TYPE, sse
from .responses_obj import RESP, base
from .responses_out import Translator, events, finish
from .responses_req import messages, to_openai
from .responses_store import ResponseStore
from .responses_tools import dropped_tools

PATH = "/v1/responses"
NOT_FOUND = "previous response {id!r} not found (it expired, belongs to another caller, or was not stored)"
TOO_LARGE = "request body is over {n} bytes (limits.responses_body_bytes)"


def capped_body(limit: int) -> Callable[[Request], Awaitable[dict[str, Any]]]:
    """The JSON body, read up to `limit` bytes: a larger one is a 413, whatever Content-Length says."""
    async def read(request: Request) -> dict[str, Any]:
        declared = request.headers.get("content-length", "")
        if declared.isdigit() and int(declared) > limit:
            raise HTTPException(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, TOO_LARGE.format(n=limit))
        raw = bytearray()
        async for piece in request.stream():
            raw += piece
            if len(raw) > limit:
                raise HTTPException(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, TOO_LARGE.format(n=limit))
        try:
            body = json.loads(raw)
        except ValueError:
            raise ValueError("the request body is not valid JSON") from None
        if not isinstance(body, dict):
            raise ValueError("the request body must be a JSON object")
        return body
    return read


def router(broker: Broker, store: ResponseStore | None = None, cloud: Router | None = None) -> APIRouter:
    r = APIRouter()
    lim = broker.settings.limits
    kept = store or ResponseStore(entry_bytes=lim.response_store_entry_bytes, total_bytes=lim.response_store_bytes)
    body_dep = Depends(capped_body(lim.responses_body_bytes))

    def remembering(who: str, conversation: list[dict[str, Any]]) -> Callable[[Translator], None]:
        def remember(t: Translator) -> None:   # runs before the final event goes out
            if t.response["store"] and t.response["status"] != "failed":
                kept.put(t.response["id"], who, conversation + messages(t.items))
        return remember

    @r.post(PATH, response_model=None)
    def responses(request: Request, body: dict[str, Any] = body_dep) -> Any:
        stream, who = bool(body.get("stream")), identity(request)
        if body.get("previous_response_id") is None:   # null is the same as leaving it out
            body.pop("previous_response_id", None)
        prev = body.get("previous_response_id")
        found = kept.get(prev, who) if isinstance(prev, str) else None
        # A turn this broker served continues here; any other is the provider's to continue (never local).
        fo = failover.attempt(cloud, UpApi.OPENAI, PATH, body, request, stream, local=prev is None) if found is None else None
        if isinstance(fo, Response):
            return fo
        if fo is not None:
            body = {**body, "model": fo.model}
        elif found is not None and cloud is not None and (here := cloud.cfg.local_for(str(body.get("model") or ""))):
            body = {**body, "model": here}   # a turn the local fallback served: continue it there
        history: list[dict[str, Any]] = found or []
        if prev is not None and found is None:
            raise HTTPException(HTTPStatus.BAD_REQUEST, {"error": NOT_FOUND.format(id=prev), "code": "previous_response_not_found"})
        chat, turn, custom = to_openai(body, history)
        try:
            out = completion.run(broker, chat, request, stream)
        except completion.ChatFailed as e:
            raise HTTPException(e.status, {"error": e.message, "x_broker": e.meta}) from None
        meta = out.meta | ({"dropped_tools": dropped} if (dropped := dropped_tools(body)) else {})
        t = Translator(base(new_id(RESP), body, out.requested, meta), custom, remembering(who, history + turn))
        lines = out.chunks if out.chunks is not None else sse(out.result or {})
        hdrs = fo.headers if fo else {}
        if not stream:
            return JSONResponse(finish(t, lines), headers=hdrs)
        return StreamingResponse(events(t, lines, out.failure), media_type=SSE_MEDIA_TYPE, headers=hdrs)

    return r
