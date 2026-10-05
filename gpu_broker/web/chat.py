"""POST /v1/chat/completions — OpenAI-compatible.

The broker path (direct lease or queued job, name mapping, `x_broker`) is in completion.py.
Every field of the request other than the broker's own (`gpu_broker.constants.BROKER_FIELDS`)
reaches the model server untouched: `tools`, `tool_choice`, `response_format`, `stop`,
`seed`, `n`, image content parts. `max_completion_tokens` is sent as `max_tokens`.
With the hosted fallback on (upstream.py), a hosted model name the local side cannot serve
right now is answered by the real provider instead. With `upstreams:` routes, a matching name
goes to its cloud provider first and falls back to a local model (failover.py).
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from ..broker import Broker
from ..failover.config import Api as UpApi
from ..failover.router import Router
from . import completion, failover
from .openai import SSE_MEDIA_TYPE, sse
from .upstream import LOCAL, Api, Upstream, served

PATH = "/v1/chat/completions"


def router(broker: Broker, upstream: Upstream, cloud: Router | None = None) -> APIRouter:
    r = APIRouter()

    @r.post(PATH, response_model=None)
    def chat(body: dict[str, Any], request: Request) -> Any:
        stream = bool(body.get("stream"))
        fo = failover.attempt(cloud, UpApi.OPENAI, PATH, body, request, stream)
        if isinstance(fo, Response):
            return fo
        if fo is not None:
            body = {**body, "model": fo.model}
        hdrs = served(LOCAL) | (fo.headers if fo else {})
        may_forward = upstream.available(Api.OPENAI) and fo is None
        try:
            out = completion.run(broker, body, request, stream, may_forward)
        except completion.GoHosted as e:
            return upstream.forward(Api.OPENAI, PATH, body, request, str(e))
        except completion.ChatFailed as e:
            if e.hosted and may_forward:
                return upstream.forward(Api.OPENAI, PATH, body, request, e.message)
            raise HTTPException(e.status, {"error": e.message, "x_broker": e.meta}) from None
        if fo is not None:
            out.mapped = True   # echo the local model's catalog name (run() was given it), not its served name
        if out.chunks is not None:
            lines = (completion.echo_model(line, out.requested) for line in out.chunks) if out.mapped else out.chunks
            return StreamingResponse(lines, media_type=SSE_MEDIA_TYPE, headers=hdrs)
        result = out.result or {}   # run() returns either live chunks or a finished result
        if fo is not None:
            result = {**result, "model": fo.model}
        if stream:
            return StreamingResponse(sse(result), media_type=SSE_MEDIA_TYPE, headers=hdrs)
        return JSONResponse(result, headers=hdrs)

    return r
