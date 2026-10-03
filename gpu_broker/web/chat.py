"""POST /v1/chat/completions — OpenAI-compatible.

The broker path (direct lease or queued job, name mapping, `x_broker`) is in completion.py.
Every field of the request other than the broker's own (`gpu_broker.constants.BROKER_FIELDS`)
reaches the model server untouched: `tools`, `tool_choice`, `response_format`, `stop`,
`seed`, `n`, image content parts. `max_completion_tokens` is sent as `max_tokens`.
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from ..broker import Broker
from . import completion
from .openai import SSE_MEDIA_TYPE, sse


def router(broker: Broker) -> APIRouter:
    r = APIRouter()

    @r.post("/v1/chat/completions", response_model=None)
    def chat(body: dict[str, Any], request: Request) -> Any:
        stream = bool(body.get("stream"))
        try:
            out = completion.run(broker, body, request, stream)
        except completion.ChatFailed as e:
            raise HTTPException(e.status, {"error": e.message, "x_broker": e.meta}) from None
        if out.chunks is not None:
            lines = (completion.echo_model(line, out.requested) for line in out.chunks) if out.mapped else out.chunks
            return StreamingResponse(lines, media_type=SSE_MEDIA_TYPE)
        result = out.result or {}   # run() returns either live chunks or a finished result
        return StreamingResponse(sse(result), media_type=SSE_MEDIA_TYPE) if stream else result

    return r
