"""Queued JSON compatibility routes for OpenAI-style inference servers.

Chat keeps its optimized direct/streaming path in web/chat.py. These endpoints share the
same resolution, priority, residency and substitution machinery, but intentionally support
non-streamed JSON first so the broker remains a narrow allowlisted proxy.
"""
from __future__ import annotations

from http import HTTPStatus
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from ..broker import Broker, validate_request
from ..chat import apply_variant, request_priority
from ..constants import OPENAI_PATH_KEY, PRIORITY_HEADER, PRIORITY_KEY, REQUESTER_HEADER, TERMINAL, JobState
from .jobs import client


def router(broker: Broker) -> APIRouter:
    r = APIRouter()
    wait_s = broker.settings.timeouts.chat_wait_s

    def call(path: str, body: dict[str, Any], request: Request) -> dict[str, Any]:
        validate_request(body)
        if body.get("stream"):
            raise HTTPException(HTTPStatus.BAD_REQUEST, "streaming is currently supported on /v1/chat/completions")
        if not isinstance(body.get("model"), str) or not body["model"]:
            raise HTTPException(HTTPStatus.BAD_REQUEST, "model is required")
        requester = request.headers.get(REQUESTER_HEADER) or client(request)
        priority = request_priority(broker.catalog, request.headers.get(PRIORITY_HEADER, ""), requester)
        payload = {**apply_variant(broker.catalog, body), "kind": body.get("kind", "llm"),
                   OPENAI_PATH_KEY: path, PRIORITY_KEY: priority.value}
        jid, info = broker.submit(payload, requester)
        meta = {"job": jid, **info}
        if info.get("error"):
            raise HTTPException(HTTPStatus.SERVICE_UNAVAILABLE, {"error": info["error"], "x_broker": meta})
        job = broker.wait(jid, wait_s) or {}
        state = job.get("state")
        if state != JobState.DONE:
            code = HTTPStatus.BAD_GATEWAY if state in TERMINAL else HTTPStatus.GATEWAY_TIMEOUT
            raise HTTPException(code, {"error": job.get("error") or f"still {state}", "x_broker": meta})
        result = job.get("result")
        if not isinstance(result, dict):
            raise HTTPException(HTTPStatus.BAD_GATEWAY, {"error": "upstream returned non-object JSON", "x_broker": meta})
        return {**result, "x_broker": {"job": jid, "requested": body["model"], "used": job["resolved"],
                                       "substitution": job.get("substitution")}}

    @r.post("/v1/completions")
    def completions(body: dict[str, Any], request: Request) -> dict[str, Any]:
        return call("/v1/completions", body, request)

    @r.post("/v1/responses")
    def responses(body: dict[str, Any], request: Request) -> dict[str, Any]:
        return call("/v1/responses", body, request)

    @r.post("/v1/embeddings")
    def embeddings(body: dict[str, Any], request: Request) -> dict[str, Any]:
        return call("/v1/embeddings", body, request)

    @r.post("/v1/rerank")
    def rerank(body: dict[str, Any], request: Request) -> dict[str, Any]:
        return call("/v1/rerank", body, request)

    @r.post("/v1/score")
    def score(body: dict[str, Any], request: Request) -> dict[str, Any]:
        return call("/v1/score", body, request)

    return r
