"""Fixtures for the MCP tests: a broker on the fake driver whose ComfyUI outputs carry a /view
URL (so finished images can come back inline), a gate that holds a ComfyUI job, and a helper
that calls the tools through the SDK's own client, in-process."""
from __future__ import annotations

import base64
import dataclasses
import json
import threading
from typing import Any

import anyio
import pytest
from mcp import Client

from gpu_broker.backends import view_url
from gpu_broker.broker import Broker
from gpu_broker.mcp_server import server
from tests.helpers import WAIT_S, FakeBackends, FakeDriver, make_settings, wait_idle

COMFY = "http://gpu-host.example:8188"   # ComfyUI as browsers reach it (comfy.public_url)
COMFY_INTERNAL = "http://127.0.0.1:8188"  # as the broker does (comfy.url)
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
PNG_B64 = base64.b64encode(PNG).decode()


class ViewBackends(FakeBackends):
    """ComfyUI outputs with a URL, as HttpBackends reports them; `gate` holds a run until set."""

    def __init__(self, driver: FakeDriver) -> None:
        super().__init__(driver)
        self.gate: threading.Event | None = None

    def comfy_run(self, key, graph, jid) -> dict[str, Any]:
        self.graphs.append(graph)
        if self.gate is not None:
            assert self.gate.wait(WAIT_S)
        name = f"{jid}.png"
        return {"model": key, "outputs": [{"file": f"broker/{name}", "url": view_url(COMFY, name, "broker")}]}


@pytest.fixture
def mcp_broker(tmp_path):
    driver = FakeDriver({"llama-8b"})
    s = make_settings(tmp_path, comfy=dataclasses.replace(
        make_settings(tmp_path).comfy, url=COMFY_INTERNAL, public_url=COMFY))
    b = Broker(s, env={}, driver=driver, backends=ViewBackends(driver))
    b.start()
    yield b
    if b.backends.gate is not None:
        b.backends.gate.set()
    assert wait_idle(b), "test left a broker job in flight"
    b.stop()


def call(broker: Broker, tool: str, args: dict[str, Any] | None = None, caller: str = "main") -> Any:
    """One tool call through the SDK client, connected in-process to the broker's MCP server."""
    async def go() -> Any:
        async with Client(server.build(broker, default_caller=caller)) as c:
            return await c.call_tool(tool, args or {})
    return anyio.run(go)


def report(result: Any) -> dict[str, Any]:
    """The JSON a tool returned in its first text block."""
    assert not result.is_error, result.content
    return json.loads(result.content[0].text)
