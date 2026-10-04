"""Fakes for the failover tests: a real HTTP server standing in for Anthropic and OpenAI that
can answer, refuse, drop the connection, hang, or break a stream part way; a fake clock; and a
broker wired to both with `upstreams:` routes for claude-*, gpt-* and a cloud-only name."""
from __future__ import annotations

import dataclasses
import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from fastapi.testclient import TestClient

from gpu_broker.broker import Broker
from gpu_broker.failover.config import BreakerCfg, Upstreams, UpTimeouts
from gpu_broker.failover.router import Router
from gpu_broker.web.app import create_app
from tests import cloud_shapes
from tests.cloud_shapes import sse
from tests.dropin_fakes import ToolBackends, catalog_with_embedder
from tests.helpers import TOKEN, FakeDriver, make_settings

LOCAL = "llama-8b"
PLANTED = "sk-ant-api03-PLANTED-0f9e8d7c6b5a"   # the client's own provider key: must never be kept
OPENAI_PLANTED = "sk-proj-PLANTED-1a2b3c4d5e6f"
ANT = {"x-gpu-broker-key": TOKEN, "x-api-key": PLANTED, "anthropic-version": "2023-06-01"}
OAI = {"x-gpu-broker-key": TOKEN, "Authorization": f"Bearer {OPENAI_PLANTED}"}
HANG_S = 3.0
FAST = UpTimeouts(connect_s=1, probe_s=0.3, first_byte_s=0.4, response_s=0.4, idle_s=0.4)
BREAKER = BreakerCfg(failures=3, probe_s=10, probe_max_s=80, quota_probe_s=900, trial_s=60)

ERRORS: dict[str, tuple[int, dict[str, Any], dict[str, str]]] = {
    "529": (529, {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}, {}),
    "500": (500, {"error": {"type": "server_error", "message": "boom"}}, {}),
    "quota429": (429, {"error": {"type": "insufficient_quota", "code": "insufficient_quota",
                                 "message": "You exceeded your current quota"}}, {"retry-after": "1200"}),
    "credit402": (402, {"type": "error", "error": {"type": "billing_error",
                                                   "message": "Your credit balance is too low to access the API"}}, {}),
    "rate429": (429, {"type": "error", "error": {"type": "rate_limit_error", "message": "slow down"}}, {"retry-after": "3"}),
    "400": (400, {"type": "error", "error": {"type": "invalid_request_error", "message": "max_tokens: required"}}, {}),
    "401": (401, {"type": "error", "error": {"type": "authentication_error", "message": "invalid x-api-key"}}, {}),
}


def answer(path: str, stream: bool, variant: str = "text") -> list[bytes] | dict[str, Any]:
    """What the cloud says when all is well, per API (tests/cloud_shapes.py)."""
    return cloud_shapes.stream(path, variant) if stream else cloud_shapes.whole(path, variant)


class FakeCloud:
    """`mode`: ok | one of ERRORS | drop | hang | redirect | stream_die | stream_stall | stream_error |
    stream_error_openai | stream_ping_error (a comment and a ping, then the error) |
    stream_cut (ends cleanly before its finishing event) | stream_late_error (an error event after
    the answer began) | slow_stream (a pause before each event) | no_stream (400 to any request that
    asks for a stream) | ignore_stream (a whole JSON answer even when a stream is asked for).
    `variant` picks the answer (cloud_shapes). `probe_mode` overrides `mode` for GET /v1/models.
    `seen` records each request."""

    def __init__(self) -> None:
        self.mode = "ok"
        self.variant = "text"
        self.pause_s = 0.0   # slow_stream: before each event
        self.probe_mode: str | None = None
        self.seen: list[dict[str, Any]] = []
        cloud = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a: Any) -> None:
                pass

            def do_GET(self) -> None:
                cloud.seen.append({"method": "GET", "path": self.path, "headers": {k.lower(): v for k, v in self.headers.items()}})
                self.reply()

            def do_POST(self) -> None:
                raw = self.rfile.read(int(self.headers.get("content-length", 0)))
                cloud.seen.append({"method": "POST", "path": self.path, "headers": {k.lower(): v for k, v in self.headers.items()}, "body": json.loads(raw)})
                self.reply(json.loads(raw))

            def reply(self, body: dict[str, Any] | None = None) -> None:
                mode = cloud.probe_mode if body is None and cloud.probe_mode else cloud.mode
                if mode == "drop":
                    self.connection.shutdown(socket.SHUT_RDWR)
                    return None
                if mode == "hang":
                    time.sleep(HANG_S)
                    return None
                if mode == "redirect":
                    return self.send(302, b"moved", "text/plain", {"location": "https://elsewhere.invalid/"})
                if mode in ERRORS:
                    status, data, hdrs = ERRORS[mode]
                    return self.send(status, json.dumps(data).encode(), "application/json", hdrs)
                stream = bool(body and body.get("stream"))
                if body is None:   # GET /v1/models
                    return self.send(200, b'{"data": []}', "application/json")
                if mode == "no_stream" and stream:
                    return self.send(400, json.dumps({"error": {"type": "invalid_request_error",
                                                                "message": "stream is not supported"}}).encode(), "application/json")
                got = answer(self.path, stream and mode != "ignore_stream", cloud.variant)
                if isinstance(got, dict):
                    return self.send(200, json.dumps(got).encode(), "application/json")
                if mode == "stream_error":
                    got = [sse("error", {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}})]
                if mode == "stream_ping_error":
                    got = [b": keep-alive\n\n", sse("ping", {"type": "ping"}),
                           sse("error", {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}})]
                if mode == "stream_error_openai":
                    got = [sse(None, {"error": {"type": "server_error", "message": "boom"}})]
                if mode == "stream_cut":
                    got = got[:-1]
                if mode == "stream_late_error":
                    late = ({"error": {"type": "server_error", "message": "boom"}} if self.path == cloud_shapes.CHAT
                            else {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}})
                    got = [*got[:3], sse(None if self.path == cloud_shapes.CHAT else "error", late)]
                self.send_response(200)
                self.send_header("content-type", "text/event-stream")
                self.send_header("transfer-encoding", "chunked")
                self.end_headers()
                for i, piece in enumerate(got):
                    if mode == "slow_stream":
                        time.sleep(cloud.pause_s)
                    self.wfile.write(f"{len(piece):x}\r\n".encode() + piece + b"\r\n")
                    self.wfile.flush()
                    if mode == "stream_stall" and i == 0:
                        time.sleep(HANG_S)
                        return None
                    if mode == "stream_die" and i == 0:
                        time.sleep(0.05)
                        self.connection.shutdown(socket.SHUT_RDWR)   # mid-stream, no terminating chunk
                        return None
                self.wfile.write(b"0\r\n\r\n")
                return None

            def send(self, status: int, data: bytes, ctype: str, extra: dict[str, str] | None = None) -> None:
                self.send_response(status)
                self.send_header("content-type", ctype)
                self.send_header("content-length", str(len(data)))
                for k, v in (extra or {}).items():
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(data)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def posts(self) -> list[dict[str, Any]]:
        return [s for s in self.seen if s["method"] == "POST"]

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


class Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


def upstreams(url: str, **over: Any) -> Upstreams:
    base: dict[str, Any] = {
        "providers": {"anthropic": {"url": url, "api": "anthropic", "pass_client_key": True},
                      "openai": {"url": url, "api": "openai", "pass_client_key": True}},
        "routes": {"claude-*": ["anthropic", LOCAL], "gpt-*": ["openai", LOCAL], "cloud-only-*": ["openai"]},
        "breaker": BREAKER, "timeouts": FAST}
    return Upstreams(**(base | over))


def build(tmp_path: Any, cloud: FakeCloud, env: dict[str, str] | None = None, **over: Any) -> tuple[TestClient, Broker, Router, Clock]:
    driver = FakeDriver({LOCAL})
    s = make_settings(tmp_path, catalog=catalog_with_embedder(tmp_path, embed=False))
    s = dataclasses.replace(s, upstreams=upstreams(cloud.url, **over))
    b = Broker(s, env={}, driver=driver, backends=ToolBackends(driver))
    b.start()
    clock = Clock()
    router = Router(s.upstreams, b.store.event, env or {}, clock=clock, wall=lambda: 0.0)
    return TestClient(create_app(b, TOKEN, start=False, cloud=router)), b, router, clock
