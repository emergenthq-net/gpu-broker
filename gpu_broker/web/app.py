"""Application factory: authentication, security headers, routers, and the broker lifecycle.

Every route except /health and the static dashboard shell needs the token from $BROKER_TOKEN,
sent as `Authorization: Bearer <token>` (OpenAI SDKs) or `x-api-key: <token>` (Anthropic
SDKs). With no token configured every call is refused rather than left open.
"""
from __future__ import annotations

import hmac
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from http import HTTPStatus

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response

from ..broker import Broker, StagingError
from ..constants import API_KEY_HEADER, APP_NAME, AUTH_SCHEME
from . import admin, chat, dash, embeddings, errors, jobs, messages, sessions

Auth = Callable[..., None]

SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "X-Frame-Options": "DENY",
    "Content-Security-Policy": ("default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
                                "img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; "
                                "base-uri 'none'; form-action 'none'"),
}


def make_auth(token: str, bearer_only: bool = False) -> Auth:
    """A dependency that accepts exactly `Authorization: Bearer <token>` or (unless
    `bearer_only`, as on the admin routes) `x-api-key: <token>`. Both are compared in constant
    time, every time (no short-circuit on the first)."""
    bearer, key = f"{AUTH_SCHEME} {token}".encode(), token.encode()

    def require_token(request: Request) -> None:
        by_bearer = hmac.compare_digest(request.headers.get("authorization", "").encode(), bearer)
        by_key = hmac.compare_digest(request.headers.get(API_KEY_HEADER, "").encode(), key) and not bearer_only
        if not token or not (by_bearer | by_key):
            raise HTTPException(HTTPStatus.UNAUTHORIZED, "bad token")
    return require_token


def create_app(broker: Broker, token: str, start: bool = True) -> FastAPI:
    """Build the API. `start=False` leaves the worker threads off (for tests and `check`)."""
    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        if start:
            broker.start()
        yield
        broker.stop()

    app = FastAPI(title=APP_NAME, lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    auth, admin_auth = Depends(make_auth(token)), Depends(make_auth(token, bearer_only=True))

    @app.middleware("http")
    async def security_headers(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        response = await call_next(request)
        response.headers.update(SECURITY_HEADERS)
        return response

    errors.install(app)   # ValueError -> 400, and per-API error shapes

    @app.exception_handler(StagingError)
    async def staging_failed(_: Request, e: StagingError) -> JSONResponse:
        return JSONResponse({"detail": str(e), "id": e.jid}, status_code=HTTPStatus.INTERNAL_SERVER_ERROR)

    @app.get("/health")
    def health() -> dict[str, bool]:
        return {"ok": True}

    app.include_router(jobs.router(broker), dependencies=[auth])
    app.include_router(chat.router(broker), dependencies=[auth])
    app.include_router(messages.router(broker), dependencies=[auth])
    app.include_router(embeddings.router(broker), dependencies=[auth])
    app.include_router(sessions.router(broker), dependencies=[auth])
    app.include_router(admin.router(broker), dependencies=[admin_auth])   # Bearer only: x-api-key is for SDKs
    app.include_router(dash.data_router(broker), dependencies=[auth])
    app.include_router(dash.page_router())
    return app
