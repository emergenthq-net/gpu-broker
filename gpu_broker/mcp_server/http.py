"""Streamable HTTP at /mcp on the broker itself, for remote MCP clients and ChatGPT connectors.

The same credentials as the model routes (web/auth.py, scope MODEL): the main token or a
client key, as `Authorization: Bearer` or `x-api-key`. A request without one gets 401 before
the MCP SDK sees it, as a JSON-RPC error saying why (an MCP client shows that, not a bare
status). The key lookup (sqlite, under a lock) runs off the event loop. The guard then states
who is calling in CALLER_HEADER (`key:<id>:<name>`, or `main`), removing any copy the client
sent, and the tools read only that (core.caller). The session manager is made per app
lifespan (Endpoint.run): a manager runs once, an app may start more than once.

Stateless: each POST is answered on its own, so nothing is kept between calls and a restart
loses no sessions. Requests up to `mcp.max_body_bytes` (base64 inputs); larger files go by URL.
DNS-rebinding checks are off: a browser page cannot add the bearer token the guard requires.
"""
from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from http import HTTPStatus

from fastapi import FastAPI, HTTPException
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.server.transport_security import TransportSecuritySettings
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.routing import Route
from starlette.types import Receive, Scope, Send

from ..broker import Broker
from ..keys import KeyStore
from ..web.auth import Auth, make_auth
from ..web.auth import Scope as AuthScope
from ..web.jobs import CLIENT_ID_STATE, CLIENT_STATE
from . import core, server

PATH = "/mcp"
METHODS = ["POST"]   # stateless: no server-initiated stream (GET) and no session to end (DELETE); both get 405
AUTH_ERROR = -32001   # JSON-RPC server-error range: the credential was refused
NOT_SERVING = "the MCP endpoint is not running (the broker is starting or stopping)"
WHY: dict[int, str] = {HTTPStatus.UNAUTHORIZED: "key revoked or invalid"}


class Endpoint:
    """The tools, and the session manager of the app lifespan that is running now."""

    def __init__(self, broker: Broker) -> None:
        self.broker, self.lowlevel = broker, server.build(broker)._lowlevel_server
        self.manager: StreamableHTTPSessionManager | None = None

    @asynccontextmanager
    async def run(self) -> AsyncIterator[None]:
        self.manager = StreamableHTTPSessionManager(
            self.lowlevel, stateless=True, max_request_body_size=self.broker.settings.mcp.max_body_bytes,
            security_settings=TransportSecuritySettings(enable_dns_rebinding_protection=False))
        try:
            async with self.manager.run():
                yield
        finally:
            self.manager = None


class Guard:
    """Authenticate, then hand the request to the MCP SDK with the caller stated."""

    def __init__(self, auth: Auth, endpoint: Endpoint) -> None:
        self.auth, self.endpoint = auth, endpoint

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        request = Request(scope)
        try:
            await run_in_threadpool(self.auth, request)
        except HTTPException as e:
            await _refuse(send, e.status_code, WHY.get(e.status_code, str(e.detail)))
            return
        if (manager := self.endpoint.manager) is None:
            await _refuse(send, HTTPStatus.SERVICE_UNAVAILABLE, NOT_SERVING)
            return
        kid = getattr(request.state, CLIENT_ID_STATE, None)
        who = (f"{core.KEY_PREFIX}{kid}:{getattr(request.state, CLIENT_STATE, '')}" if kid else core.MAIN).encode()
        name = server.CALLER_HEADER.encode()
        headers = [(k, v) for k, v in scope["headers"] if k.lower() != name] + [(name, who)]
        await manager.handle_request({**scope, "headers": headers}, receive, send)


async def _refuse(send: Send, status: int, detail: str) -> None:
    """A JSON-RPC error (id null: the request is not read) with the HTTP status in the message."""
    error = {"code": AUTH_ERROR, "message": f"{detail} (HTTP {status})", "data": {"http_status": status}}
    body = json.dumps({"jsonrpc": "2.0", "id": None, "error": error}).encode()
    await send({"type": "http.response.start", "status": status,
                "headers": [(b"content-type", b"application/json"), (b"www-authenticate", b"Bearer")]})
    await send({"type": "http.response.body", "body": body})


def mount(app: FastAPI, broker: Broker, token: str, keys: KeyStore) -> Endpoint:
    """Add /mcp to `app`; the caller runs `endpoint.run()` in each app lifespan."""
    endpoint = Endpoint(broker)
    app.router.routes.append(Route(PATH, endpoint=Guard(make_auth(token, keys, AuthScope.MODEL), endpoint), methods=METHODS))
    return endpoint

