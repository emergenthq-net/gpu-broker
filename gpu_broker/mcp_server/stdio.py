"""`gpu-broker mcp`: MCP over stdio for apps that launch a local command (Claude Desktop, Claude
Code, Codex), relayed message by message to the broker's /mcp. One implementation of the tools,
on the broker; this side only carries messages.

The broker is `--url`, else $GPU_BROKER_URL, else the one `gpu-broker connect` last issued a
key for, else http://127.0.0.1:8095. The credential comes from the environment only (a command
line shows in `ps`): $GPU_BROKER_API_KEY, else the key connect issued for that broker, else
$BROKER_TOKEN, and the main token only for the broker on this machine that the broker config
names (loopback, its `server.port`): it is never sent anywhere else. No credential goes over
plain http to a host that is not loopback or private.

Every request gets an answer: when the broker cannot be reached or refuses, the app gets a
JSON-RPC error with the reason and HTTP status. A refused credential (401) also ends the relay
(exit 2). Errors go to stderr; stdout carries only MCP messages.
"""
from __future__ import annotations

import argparse
import ipaddress
import json
import os
import socket
import sys
from collections.abc import Mapping
from http import HTTPStatus
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import anyio
import httpx2
from mcp.client.streamable_http import streamable_http_client
from mcp.server.stdio import stdio_server
from mcp.shared.message import SessionMessage
from mcp.types import INTERNAL_ERROR, ErrorData, JSONRPCError, JSONRPCRequest, JSONRPCResponse

from .. import settings
from ..connect import engine
from ..connect.cli import DEFAULT_URL, TOKEN_ENV, URL_ENV
from ..connect.core import KEY_ENV, checked
from .http import PATH
from .pinning import LOOPBACK_NAMES, PinnedBackend, Resolver, literal, vetted

CONNECT_S, READ_S = 15, 3600   # a tool call may wait as long as the broker's mcp.wait_s allows, and then some
EXIT_REFUSED = 2               # the broker refused the credential
FLUSH_S = 0.5                  # for the answer to reach stdout before a refused relay exits
REFUSED = "key revoked or invalid"


def _loopback(host: str) -> bool:
    return host.lower() in LOOPBACK_NAMES or (literal(host) and ipaddress.ip_address(host.split("%")[0]).is_loopback)


def local_port(env: Mapping[str, str]) -> int:
    """The port the broker config on this machine names (settings default when there is none)."""
    try:
        return settings.load(env=env).server.port
    except (OSError, ValueError):
        return settings.Server().port


def target(url: str | None, env: Mapping[str, str], home: Path, resolve: Resolver = socket.getaddrinfo) -> tuple[str, str, list[str]]:
    """(the broker's /mcp URL, the credential, the addresses plain http may connect to: [] for any);
    ValueError when there is no credential it may use there."""
    known = engine.load_manifest(home).get("key") or {}
    base = (url or env.get(URL_ENV) or known.get("broker") or DEFAULT_URL).rstrip("/")
    checked(base, "", "")
    parts = urlsplit(base)
    host, port = parts.hostname or "", parts.port or (443 if parts.scheme == "https" else 80)
    main_ok = _loopback(host) and port == local_port(env)
    cred = env.get(KEY_ENV) or (known.get("key") if known.get("broker") == base else None) or (env.get(TOKEN_ENV) if main_ok else None)
    if not cred:
        why = "" if not env.get(TOKEN_ENV) or main_ok else f" (${TOKEN_ENV} is the main token: it goes only to the broker on this machine)"
        raise ValueError(f"no credential for {base}{why}: set ${KEY_ENV} (a client key), or run `gpu-broker connect`")
    pins = vetted(host, resolve) if parts.scheme == "http" else []
    if pins is None:
        raise ValueError(f"refusing to send a credential over plain http to {host}, which is not loopback or private; use https")
    return base + PATH, cred, pins


class Answering(httpx2.AsyncHTTPTransport):
    """The broker could not be reached, or stopped answering: a JSON-RPC error for that POST's
    request id (502, or 504 for a timeout), so the MCP client answers it instead of failing the
    whole transport and leaving the app waiting."""

    async def handle_async_request(self, request: httpx2.Request) -> httpx2.Response:
        try:
            return await super().handle_async_request(request)
        except httpx2.TransportError as e:
            status = HTTPStatus.GATEWAY_TIMEOUT if isinstance(e, httpx2.TimeoutException) else HTTPStatus.BAD_GATEWAY
            try:
                rid = json.loads(request.content).get("id") if request.method == "POST" else None
            except (ValueError, AttributeError, httpx2.RequestNotRead):
                rid = None
            error = {"code": INTERNAL_ERROR, "message": f"broker unreachable: {e or type(e).__name__}"}
            return httpx2.Response(status, json={"jsonrpc": "2.0", "id": rid, "error": error}, request=request)


class Relay:
    """Carries messages both ways and answers whatever the broker leaves unanswered."""

    def __init__(self) -> None:
        self.pending: dict[Any, None] = {}       # ids of requests sent and not yet answered
        self.status: dict[Any, int] = {}         # HTTP status of a failed POST, by request id
        self.refused = False

    async def on_response(self, response: httpx2.Response) -> None:
        if response.status_code < HTTPStatus.BAD_REQUEST:
            return
        try:
            rid = json.loads(response.request.content).get("id")
        except (ValueError, AttributeError, httpx2.RequestNotRead):
            return
        if rid is not None:
            self.status[rid] = response.status_code
        self.refused |= response.status_code == HTTPStatus.UNAUTHORIZED

    def error(self, rid: Any, message: str, status: int | None) -> SessionMessage:
        why = REFUSED if status == HTTPStatus.UNAUTHORIZED else message
        text = f"gpu-broker: {why}" + (f" (HTTP {status})" if status and f"HTTP {status}" not in why else "")
        return SessionMessage(JSONRPCError(jsonrpc="2.0", id=rid, error=ErrorData(code=INTERNAL_ERROR, message=text,
                                                                                    data={"http_status": status} if status else None)))

    def answered(self, message: SessionMessage) -> SessionMessage:
        """A reply from the broker; an error from a failed POST says its status and why."""
        m = message.message
        if isinstance(m, JSONRPCResponse | JSONRPCError):
            self.pending.pop(m.id, None)
            if isinstance(m, JSONRPCError) and (status := self.status.pop(m.id, None)):
                return self.error(m.id, m.error.message, status)
        return message

    async def to_broker(self, src: Any, dst: Any) -> None:
        async for message in src:
            if isinstance(message, Exception):
                print(f"gpu-broker mcp: client: {message}", file=sys.stderr)
                continue
            if isinstance(message.message, JSONRPCRequest):
                self.pending[message.message.id] = None
            await dst.send(message)

    async def to_app(self, src: Any, dst: Any) -> None:
        async for message in src:
            if isinstance(message, Exception):   # the transport failed: answer everything in flight
                print(f"gpu-broker mcp: broker: {message}", file=sys.stderr)
                for rid in list(self.pending):
                    await dst.send(self.error(rid, f"broker unreachable or failed: {message}", self.status.pop(rid, None)))
                self.pending.clear()
            else:
                await dst.send(self.answered(message))
            if self.refused:
                print(f"gpu-broker mcp: {REFUSED}; run `gpu-broker connect` for a new key", file=sys.stderr)
                sys.stderr.flush()
                await anyio.sleep(FLUSH_S)   # the stdout writer sends the answer first
                os._exit(EXIT_REFUSED)       # stdin's reader thread cannot be cancelled while the app keeps it open


def transport(url: str, pins: list[str]) -> Answering:
    """Connections go only to `pins` when there are any (plain http to a name: see pinning.py)."""
    t = Answering()
    if pins:
        t._pool._network_backend = PinnedBackend(urlsplit(url).hostname or "", pins)
    return t


async def relay(url: str, credential: str, pins: list[str]) -> int:
    r = Relay()
    headers = {"Authorization": f"Bearer {credential}"}
    async with httpx2.AsyncClient(headers=headers, timeout=httpx2.Timeout(CONNECT_S, read=READ_S), transport=transport(url, pins),
                                  event_hooks={"response": [r.on_response]}) as http, \
            stdio_server() as (stdin, stdout), streamable_http_client(url, http_client=http) as (broker_in, broker_out), \
            stdout, anyio.create_task_group() as tg:   # stdout closed last: stdio_server's writer then ends
        tg.start_soon(r.to_app, broker_in, stdout)
        await r.to_broker(stdin, broker_out)   # the app closed stdin: done
        tg.cancel_scope.cancel()
    return 0


def main(argv: list[str], env: Mapping[str, str], home: Path | None = None) -> int:
    ap = argparse.ArgumentParser(prog="gpu-broker mcp", description="MCP over stdio, relayed to a broker's /mcp")
    ap.add_argument("--url", help=f"the broker (default: ${URL_ENV}, the broker connect used, or {DEFAULT_URL})")
    a = ap.parse_args(argv)
    try:
        url, cred, pins = target(a.url, env, home or Path(env.get("HOME") or Path.home()))
    except ValueError as e:
        print(f"gpu-broker mcp: {e}", file=sys.stderr)
        return 1
    return anyio.run(relay, url, cred, pins)
