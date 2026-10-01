"""POST /v1/chat/completions — OpenAI-compatible.

Interactive callers on the resident model are served directly (see `gpu_broker.chat`):
`stream: true` relays the server's tokens as they are generated. Everything else is a
queued job that this request waits for (up to `timeouts.chat_wait_s`); broker details are
in the response's `x_broker` field.
"""
from __future__ import annotations

from collections.abc import Iterator
from http import HTTPStatus
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from ..broker import Broker, validate_request
from ..chat import Lease, apply_variant, request_priority
from ..constants import ERR_EVENT, ERR_JOB, INTERACTIVE_KEY, PRIORITY_HEADER, PRIORITY_KEY, REQUESTER_HEADER, TERMINAL, JobState, Kind, Priority
from .jobs import client
from .openai import sse

SSE_MEDIA_TYPE = "text/event-stream"


def router(broker: Broker) -> APIRouter:
    r = APIRouter()
    chat_wait_s = broker.settings.timeouts.chat_wait_s

    def direct(lease: Lease, body: dict[str, Any], stream: bool, requested: str) -> Any:
        meta = {"job": lease.jid, "requested": requested, "used": lease.key, "direct": True}
        if not stream:
            try:
                out = broker.backends.llm_chat(lease.model, body)
            except Exception as e:  # noqa: BLE001 — reported to the caller and recorded on the job
                broker.chat.finish(lease, JobState.FAILED, error=str(e)[:ERR_JOB])
                raise HTTPException(HTTPStatus.BAD_GATEWAY, {"error": str(e)[:ERR_EVENT], "x_broker": meta}) from None
            broker.chat.finish(lease, JobState.DONE, result=out)
            return {**out, "x_broker": meta}

        def relay() -> Iterator[str]:
            summary: dict[str, Any] = {}
            state, error = JobState.DONE, None
            try:
                yield from broker.backends.llm_stream(lease.model, body, summary)
            except Exception as e:  # noqa: BLE001 — the client sees a truncated stream; the job records why
                state, error = JobState.FAILED, str(e)[:ERR_JOB]
            finally:
                broker.chat.finish(lease, state, result={"streamed": True, **summary}, error=error)
        return StreamingResponse(relay(), media_type=SSE_MEDIA_TYPE)

    @r.post("/v1/chat/completions", response_model=None)
    def chat(body: dict[str, Any], request: Request) -> Any:
        validate_request(body)
        requester = request.headers.get(REQUESTER_HEADER) or client(request)
        stream = bool(body.get("stream"))
        requested = body.get("model") or broker.catalog.defaults["resident"]
        body = {**apply_variant(broker.catalog, body), "kind": body.get("kind", Kind.LLM)}
        priority = request_priority(broker.catalog, request.headers.get(PRIORITY_HEADER, ""), requester)
        interactive = priority == Priority.INTERACTIVE
        lease = broker.chat.open(body, requester) if interactive else None
        if lease is not None:
            return direct(lease, body, stream, requested)

        jid, info = broker.submit({**body, INTERACTIVE_KEY: interactive, PRIORITY_KEY: priority.value}, requester)
        meta = {"job": jid, **info}
        if info.get("error"):
            raise HTTPException(HTTPStatus.SERVICE_UNAVAILABLE, {"error": info["error"], "x_broker": meta})
        j = broker.wait(jid, chat_wait_s) or {}
        state = j.get("state")
        if state != JobState.DONE:
            code = HTTPStatus.BAD_GATEWAY if state in TERMINAL else HTTPStatus.GATEWAY_TIMEOUT
            raise HTTPException(code, {"error": j.get("error") or f"still {state}", "x_broker": meta})
        out = {**j["result"], "x_broker": {"job": jid, "requested": requested, "used": j["resolved"],
                                           "substitution": j.get("substitution")}}
        return StreamingResponse(sse(out), media_type=SSE_MEDIA_TYPE) if stream else out

    return r
