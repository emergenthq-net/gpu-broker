"""POST /v1/messages — Anthropic-compatible (the Messages API), for the Anthropic SDKs.

The request is translated to an OpenAI chat request (anthropic_req.py) and goes through the
same broker path as /v1/chat/completions (completion.py: name mapping, direct lease or
queued job). The answer comes back as an Anthropic message, or with `stream: true` as
Anthropic's SSE event sequence (anthropic_sse.py). Errors use Anthropic's error shape
(errors.py). With the hosted fallback on (upstream.py), a hosted model name that the local
side cannot serve — busy GPU, failed call, or a hosted-only feature — goes to Anthropic.
With `upstreams:` routes, a matching name goes to its cloud provider first and falls back to a
local model through the same translation (failover.py).
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from ..broker import Broker
from ..failover.config import Api as UpApi
from ..failover.router import Router
from ..modelmap import map_name
from ..resolve import lookup
from . import completion, failover
from .anthropic_req import thinking_enabled, to_openai
from .anthropic_resp import message
from .anthropic_sse import events
from .openai import SSE_MEDIA_TYPE, sse
from .upstream import LOCAL, Api, Upstream, served

PATH = "/v1/messages"


def router(broker: Broker, upstream: Upstream, cloud: Router | None = None) -> APIRouter:
    r = APIRouter()
    cat = broker.catalog

    def hosted_name(body: dict[str, Any]) -> bool:
        name = str(body.get("model") or "")
        known = lookup(cat.data, name) is not None or cat.variant(name) is not None
        return map_name(broker.settings.model_map, known, name, cat.defaults["resident"]) is not None

    @r.post(PATH, response_model=None)
    def messages(body: dict[str, Any], request: Request) -> Any:
        stream = bool(body.get("stream"))
        fo = failover.attempt(cloud, UpApi.ANTHROPIC, PATH, body, request, stream)
        if isinstance(fo, Response):
            return fo
        if fo is not None:
            body = {**body, "model": fo.model}
        hdrs = served(LOCAL) | (fo.headers if fo else {})
        thinking = thinking_enabled(body)
        may_forward = upstream.available(Api.ANTHROPIC) and fo is None
        try:
            translated = to_openai(body)
        except ValueError as e:   # a feature only the hosted API has (server tools, documents, ...)
            if may_forward and hosted_name(body):
                return upstream.forward(Api.ANTHROPIC, PATH, body, request, str(e))
            raise
        try:
            out = completion.run(broker, translated, request, stream, may_forward)
        except completion.GoHosted as e:
            return upstream.forward(Api.ANTHROPIC, PATH, body, request, str(e))
        except completion.ChatFailed as e:
            if e.hosted and may_forward:
                return upstream.forward(Api.ANTHROPIC, PATH, body, request, e.message)
            raise HTTPException(e.status, {"error": e.message, "x_broker": e.meta}) from None
        result = out.result or {}   # run() returns either live chunks or a finished result
        if not stream:
            return JSONResponse(message(result, out.requested, thinking, out.meta), headers=hdrs)
        lines = out.chunks if out.chunks is not None else sse(result)
        return StreamingResponse(events(lines, out.requested, thinking, out.meta, out.failure),
                                 media_type=SSE_MEDIA_TYPE, headers=hdrs)

    return r
