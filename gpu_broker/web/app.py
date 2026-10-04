"""Application factory: authentication, security headers, routers, and the broker lifecycle.

Every route except /health, the static dashboard shell and the connect installer needs a
credential (web/auth.py): the main token from $BROKER_TOKEN, or a per-client key issued by
`gpu-broker connect`, as `Authorization: Bearer` (OpenAI SDKs) or `x-api-key` (Anthropic
SDKs). Client keys reach the model routes only; see auth.Scope for what each route needs.
With no token configured every call is refused rather than left open.
"""
from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AsyncExitStack, asynccontextmanager
from http import HTTPStatus
from typing import Any

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse, Response

from .. import mcp_server
from ..broker import Broker, StagingError
from ..constants import APP_NAME
from ..failover.router import Router
from ..keys import KeyStore
from . import admin, chat, connect, dash, embeddings, errors, failover, jobs, keys, messages, responses, sessions
from .auth import Auth as Auth
from .auth import Scope
from .auth import make_auth as make_auth
from .upstream import Upstream

SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "X-Frame-Options": "DENY",
    "Content-Security-Policy": ("default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
                                "img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; "
                                "base-uri 'none'; form-action 'none'"),
}


def create_app(broker: Broker, token: str, start: bool = True, keystore: KeyStore | None = None,
               upstream: Upstream | None = None, cloud: Router | None = None) -> FastAPI:
    """Build the API. `start=False` leaves the worker threads off (for tests and `check`)."""
    store = keystore or KeyStore(broker.settings.db)
    cloud = cloud or failover.make_router(broker)
    mcp: list[Any] = []   # the MCP endpoint, when /mcp is served (mounted below)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        try:   # a failing MCP start still stops the broker, the prober and the key store
            if start:
                broker.start()
                cloud.start()
            async with AsyncExitStack() as stack:
                for endpoint in mcp:
                    await stack.enter_async_context(endpoint.run())   # a new session manager each lifespan
                yield
        finally:
            cloud.stop()
            broker.stop()
            store.close()

    app = FastAPI(title=APP_NAME, lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    model_auth = Depends(make_auth(token, store, Scope.MODEL))
    auth = Depends(make_auth(token, store, Scope.OPERATOR))
    admin_auth = Depends(make_auth(token, store, Scope.ADMIN))

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
        return {"ok": True, "mcp": bool(mcp)}   # `mcp`: connect registers MCP clients only when /mcp is served

    app.include_router(jobs.router(broker), dependencies=[auth])
    up = upstream or Upstream(broker.settings.fallback)
    app.include_router(jobs.models_router(broker), dependencies=[model_auth])
    app.include_router(chat.router(broker, up, cloud), dependencies=[model_auth])
    app.include_router(messages.router(broker, up, cloud), dependencies=[model_auth])
    app.include_router(embeddings.router(broker, up), dependencies=[model_auth])
    app.include_router(responses.router(broker, cloud=cloud), dependencies=[model_auth])
    app.include_router(sessions.router(broker), dependencies=[auth])
    app.include_router(admin.router(broker), dependencies=[admin_auth])   # Bearer only: x-api-key is for SDKs
    app.include_router(keys.router(store), dependencies=[admin_auth])
    connect_admin, connect_public = connect.routers(broker, store)
    app.include_router(connect_admin, dependencies=[admin_auth])
    app.include_router(connect_public)   # connect.sh: its invite code is the credential
    app.include_router(dash.data_router(broker), dependencies=[auth])
    app.include_router(failover.status_router(cloud), dependencies=[auth])
    app.include_router(failover.passthrough_router(cloud), dependencies=[model_auth])
    app.include_router(dash.page_router())
    if broker.settings.mcp.enabled and mcp_server.available():
        from ..mcp_server import http as mcp_http  # the optional SDK: imported only when installed
        mcp.append(mcp_http.mount(app, broker, token, store))
    return app
