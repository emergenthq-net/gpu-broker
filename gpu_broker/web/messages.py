"""POST /v1/messages — Anthropic-compatible (the Messages API), for the Anthropic SDKs.

The request is translated to an OpenAI chat request (anthropic_req.py) and goes through the
same broker path as /v1/chat/completions (completion.py: name mapping, direct lease or
queued job). The answer comes back as an Anthropic message, or with `stream: true` as
Anthropic's SSE event sequence (anthropic_sse.py). Errors use Anthropic's error shape
(errors.py).
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from ..broker import Broker
from . import completion
from .anthropic_req import thinking_enabled, to_openai
from .anthropic_resp import message
from .anthropic_sse import events
from .openai import SSE_MEDIA_TYPE, sse


def router(broker: Broker) -> APIRouter:
    r = APIRouter()

    @r.post("/v1/messages", response_model=None)
    def messages(body: dict[str, Any], request: Request) -> Any:
        stream = bool(body.get("stream"))
        thinking = thinking_enabled(body)
        try:
            out = completion.run(broker, to_openai(body), request, stream)
        except completion.ChatFailed as e:
            raise HTTPException(e.status, {"error": e.message, "x_broker": e.meta}) from None
        result = out.result or {}   # run() returns either live chunks or a finished result
        if not stream:
            return message(result, out.requested, thinking, out.meta)
        lines = out.chunks if out.chunks is not None else sse(result)
        return StreamingResponse(events(lines, out.requested, thinking, out.meta, out.failure), media_type=SSE_MEDIA_TYPE)

    return r
