"""Error bodies in the shape each client expects, by route.

The Anthropic route (/v1/messages, and /v1/models when the caller sends `anthropic-version`)
answers `{"type": "error", "error": {"type", "message"}}`. The OpenAI routes answer
`{"error": {"message", "type", "code"}}` and keep the broker's `detail` alongside it, so
existing callers that read `detail` still work; for a malformed body that `detail` stays
FastAPI's list of field errors. Every other route is unchanged.
"""
from __future__ import annotations

from collections.abc import Mapping
from http import HTTPStatus
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exception_handlers import http_exception_handler, request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response

from ..constants import ANTHROPIC_VERSION_HEADER
from ..quiesce import Quiesced
from . import anthropic_resp

ANTHROPIC_PATHS = frozenset({"/v1/messages"})
OPENAI_PATHS = frozenset({"/v1/chat/completions", "/v1/responses", "/v1/embeddings", "/v1/models"})
MODELS_PATH = "/v1/models"
OPENAI_TYPES: dict[int, str] = {HTTPStatus.UNAUTHORIZED: "authentication_error", HTTPStatus.NOT_FOUND: "not_found_error",
                                HTTPStatus.TOO_MANY_REQUESTS: "rate_limit_error"}
CLIENT_ERROR, SERVER_ERROR = "invalid_request_error", "server_error"


def anthropic_client(request: Request) -> bool:
    path = request.url.path
    return path in ANTHROPIC_PATHS or (path == MODELS_PATH and ANTHROPIC_VERSION_HEADER in request.headers)


INVALID_BODY = "invalid request body: "


def message(detail: Any) -> str:
    if isinstance(detail, list):   # FastAPI's validation errors
        return INVALID_BODY + "; ".join(f"{'.'.join(map(str, e.get('loc', ())))}: {e.get('msg', '')}"
                                        for e in detail if isinstance(e, Mapping))
    if isinstance(detail, Mapping):
        return str(detail.get("error") or detail)
    return str(detail)


def openai_error(status: int, detail: Any) -> dict[str, Any]:
    kind = OPENAI_TYPES.get(status) or (SERVER_ERROR if status >= HTTPStatus.INTERNAL_SERVER_ERROR else CLIENT_ERROR)
    code = detail.get("code") if isinstance(detail, Mapping) else None
    extra = {"x_broker": detail["x_broker"]} if isinstance(detail, Mapping) and "x_broker" in detail else {}
    return {"error": {"message": message(detail), "type": kind, "code": code}, "detail": detail, **extra}


def shaped(request: Request, status: int, detail: Any) -> JSONResponse | None:
    """The response for a compatibility route, or None for the broker's own routes."""
    if anthropic_client(request):
        body = anthropic_resp.error(status, message(detail))
        if isinstance(detail, Mapping) and "x_broker" in detail:
            body["x_broker"] = detail["x_broker"]
        return JSONResponse(body, status_code=status)
    if request.url.path in OPENAI_PATHS:
        return JSONResponse(openai_error(status, detail), status_code=status)
    return None


def install(app: FastAPI) -> None:
    @app.exception_handler(HTTPException)
    async def http_error(request: Request, e: HTTPException) -> Response:
        return shaped(request, e.status_code, e.detail) or await http_exception_handler(request, e)

    @app.exception_handler(RequestValidationError)
    async def invalid(request: Request, e: RequestValidationError) -> Response:
        return (shaped(request, HTTPStatus.UNPROCESSABLE_ENTITY, jsonable_encoder(e.errors()))
                or await request_validation_exception_handler(request, e))

    @app.exception_handler(Quiesced)
    async def quiesced(request: Request, e: Quiesced) -> JSONResponse:
        r = shaped(request, HTTPStatus.SERVICE_UNAVAILABLE, str(e)) or JSONResponse({"detail": str(e)}, status_code=HTTPStatus.SERVICE_UNAVAILABLE)
        r.headers.update({"Retry-After": str(e.retry_after_s), "x-should-retry": "true"})   # both SDKs honour these
        return r

    @app.exception_handler(ValueError)
    async def bad_request(request: Request, e: ValueError) -> JSONResponse:
        return shaped(request, HTTPStatus.BAD_REQUEST, str(e)) or JSONResponse({"detail": str(e)}, status_code=HTTPStatus.BAD_REQUEST)
