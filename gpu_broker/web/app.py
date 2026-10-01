"""Application factory: authentication, security headers, routers, and the broker lifecycle.

Every route except /health and the static dashboard shell needs the bearer token from
$BROKER_TOKEN. With no token configured every call is refused rather than left open.
"""
from __future__ import annotations

import hmac
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from http import HTTPStatus

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response

from ..broker import Broker
from ..constants import APP_NAME, AUTH_SCHEME
from . import admin, chat, compat, dash, jobs, sessions

Auth = Callable[..., None]

SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "X-Frame-Options": "DENY",
    "Content-Security-Policy": ("default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
                                "img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; "
                                "base-uri 'none'; form-action 'none'"),
}


def make_auth(token: str) -> Auth:
    """A dependency that accepts exactly `Authorization: Bearer <token>`, compared in constant time."""
    expected = f"{AUTH_SCHEME} {token}".encode()

    def require_token(request: Request) -> None:
        given = request.headers.get("authorization", "").encode()
        if not token or not hmac.compare_digest(given, expected):
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
    auth = Depends(make_auth(token))

    @app.middleware("http")
    async def security_headers(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        response = await call_next(request)
        response.headers.update(SECURITY_HEADERS)
        return response

    @app.exception_handler(ValueError)
    async def bad_request(_: Request, e: ValueError) -> JSONResponse:
        return JSONResponse({"detail": str(e)}, status_code=HTTPStatus.BAD_REQUEST)

    @app.get("/health")
    def health() -> dict[str, bool]:
        return {"ok": True}

    app.include_router(jobs.router(broker), dependencies=[auth])
    app.include_router(chat.router(broker), dependencies=[auth])
    app.include_router(compat.router(broker), dependencies=[auth])
    app.include_router(sessions.router(broker), dependencies=[auth])
    app.include_router(admin.router(broker), dependencies=[auth])
    app.include_router(dash.data_router(broker), dependencies=[auth])
    app.include_router(dash.page_router())
    return app
