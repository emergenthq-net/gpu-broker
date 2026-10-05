"""End to end with the SDK's own MCP client and a real HTTP server: the broker's app (fake
driver) under uvicorn on a free loopback port, reached over Streamable HTTP at /mcp and over
stdio through `gpu-broker mcp` run as a subprocess. Also the guard: no credential is a 401,
a client key cannot pose as the main token through the caller header."""
import importlib.metadata
import json
import os
import socket
import sys
import threading
import time

import anyio
import httpx
import httpx2
import pytest
import uvicorn
from mcp import Client
from mcp.client.stdio import StdioServerParameters
from mcp.client.streamable_http import streamable_http_client

from gpu_broker.keys import KeyStore
from gpu_broker.mcp_server import core, server
from gpu_broker.web.app import create_app
from tests.helpers import TOKEN
from tests.mcp_fakes import PNG, mcp_broker

__all__ = ["mcp_broker"]
STARTUP_S = 10


@pytest.fixture
def live(mcp_broker, monkeypatch):
    """(base URL, a client key) for the broker's app served on a free port."""
    mcp_broker.backends.view_data = PNG
    keys = KeyStore(mcp_broker.settings.db)
    key = keys.issue("laptop")["key"]
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    srv = uvicorn.Server(uvicorn.Config(create_app(mcp_broker, TOKEN, start=False, keystore=keys), host="127.0.0.1",
                                        port=port, log_level="warning"))
    t = threading.Thread(target=srv.run, daemon=True)
    t.start()
    end = time.monotonic() + STARTUP_S
    while not srv.started:
        assert time.monotonic() < end, "uvicorn did not start"
        time.sleep(0.02)
    yield f"http://127.0.0.1:{port}", key
    srv.should_exit = True
    t.join(STARTUP_S)


def over_http(url, credential, calls, extra_headers=None):
    async def go():
        headers = {"Authorization": f"Bearer {credential}", **(extra_headers or {})}
        async with httpx2.AsyncClient(headers=headers) as http, Client(streamable_http_client(url + "/mcp", http_client=http)) as c:
            return [await c.call_tool(name, args) for name, args in calls]
    return anyio.run(go)


def over_stdio(url, env, calls):
    async def go():
        params = StdioServerParameters(command=sys.executable, args=["-m", "gpu_broker", "mcp", "--url", url],
                                       env={"PATH": os.environ.get("PATH", ""), "HOME": env.pop("HOME"), **env})
        with anyio.fail_after(60):
            async with Client(params) as c:
                listed = [t.name for t in (await c.list_tools()).tools]
                return listed, [await c.call_tool(name, args) for name, args in calls]
    return anyio.run(go)


def text(result):
    assert not result.is_error, result.content
    return json.loads(result.content[0].text)


def test_streamable_http_runs_a_job_and_returns_the_image(live):
    url, _ = live
    gen, models = over_http(url, TOKEN, [("generate_image", {"prompt": "a fox"}), ("list_models", {})])
    out = text(gen)
    assert out["state"] == "done" and gen.content[1].type == "image"
    assert any(m["name"] == "sdxl-base" for m in text(models))


def test_stdio_relays_to_the_broker_with_a_client_key(live, tmp_path):
    url, key = live
    listed, (gen, status) = over_stdio(url, {"HOME": str(tmp_path), "GPU_BROKER_API_KEY": key},
                                       [("generate_image", {"prompt": "a fox"}), ("gpu_status", {})])
    assert set(listed) == {"list_models", "generate_image", "generate_video", "edit_image", "image_to_video", "make_3d",
                           "job_status", "job_result", "chat_local", "gpu_status"}
    out = text(gen)
    assert out["state"] == "done" and gen.content[1].type == "image"
    assert "vram" in text(status)


def test_stdio_without_a_credential_says_so_and_exits(tmp_path):
    import subprocess
    r = subprocess.run([sys.executable, "-m", "gpu_broker", "mcp", "--url", "http://127.0.0.1:9"], capture_output=True, text=True,
                       env={"PATH": os.environ.get("PATH", ""), "HOME": str(tmp_path)}, timeout=60, check=False)
    assert r.returncode == 1 and "GPU_BROKER_API_KEY" in r.stderr and r.stdout == ""


def test_no_credential_is_a_401_before_the_sdk(live):
    url, _ = live
    body = {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
    r = httpx.post(url + "/mcp", json=body, headers={"Accept": "application/json, text/event-stream"})
    assert r.status_code == 401 and r.headers["www-authenticate"] == "Bearer"
    r = httpx.post(url + "/mcp", json=body, headers={"Authorization": "Bearer nope", server.CALLER_HEADER: core.MAIN})
    assert r.status_code == 401
    for method in ("GET", "DELETE"):   # stateless: no open-ended server stream, no session to end
        assert httpx.request(method, url + "/mcp", headers={"Authorization": f"Bearer {TOKEN}"}, timeout=5).status_code == 405


def test_a_client_key_cannot_pose_as_the_main_token(live):
    url, key = live
    (theirs,) = over_http(url, TOKEN, [("generate_image", {"prompt": "main's"})])
    jid = text(theirs)["job_id"]
    (seen,) = over_http(url, key, [("job_status", {"job_id": jid})], extra_headers={server.CALLER_HEADER: core.MAIN})
    assert seen.is_error and core.HIDDEN in seen.content[0].text
    (mine,) = over_http(url, key, [("generate_image", {"prompt": "mine"})], extra_headers={server.CALLER_HEADER: core.MAIN})
    assert text(mine)["state"] == "done"
    assert live_requester(url, text(mine)["job_id"]) == "laptop"   # the key's own name, not the main token's


def live_requester(url, jid):
    return httpx.get(f"{url}/v1/jobs/{jid}", headers={"Authorization": f"Bearer {TOKEN}"}).json()["requester"]


def test_health_says_mcp_is_served(live):
    url, _ = live
    assert httpx.get(url + "/health").json() == {"ok": True, "mcp": True}


def test_what_connect_registers_reaches_the_tools(live, tmp_path):
    """The entries connect writes, used exactly as the apps use them: Claude Code's and Codex's
    URL and header over Streamable HTTP, Claude Desktop's command, args and env over stdio."""
    import tomllib

    from gpu_broker.connect import CLIENTS, engine, mcpinfo
    from gpu_broker.connect.core import Target
    url, key = live
    home = tmp_path / "home"
    for d in (".claude", ".codex"):
        (home / d).mkdir(parents=True)
    desktop = CLIENTS["claude-desktop"].folder(Target(url, key, "llama-8b", home, {"HOME": str(home)}, {}))
    desktop.mkdir(parents=True)
    t = Target(url, key, "llama-8b", home, {"HOME": str(home)}, mcpinfo.options(mcpinfo.serves_mcp(url, http_call)))
    engine.connect(home, [CLIENTS[n].plan(t) for n in ("claude-code-mcp", "claude-desktop", "codex-mcp")])

    code = json.loads((home / ".claude.json").read_text())["mcpServers"]["gpu-broker"]
    codex = tomllib.loads((home / ".codex/config.toml").read_text())["mcp_servers"]["gpu-broker"]
    for entry_url, headers in ((code["url"], code["headers"]), (codex["url"], codex["http_headers"])):
        (status,) = over_http(entry_url.removesuffix("/mcp"), headers["Authorization"].removeprefix("Bearer "),
                              [("gpu_status", {})])
        assert "vram" in text(status)
    app = json.loads((desktop / "claude_desktop_config.json").read_text())["mcpServers"]["gpu-broker"]
    assert app["command"] == sys.executable

    async def go():
        params = StdioServerParameters(command=app["command"], args=app["args"],
                                       env={"PATH": os.environ.get("PATH", ""), "HOME": str(home), **app["env"]})
        with anyio.fail_after(60):
            async with Client(params) as c:
                return await c.call_tool("gpu_status", {})
    assert "vram" in text(anyio.run(go))


def http_call(method, url, credential, body):
    """connect's broker call, on httpx (the real one, cli.api, uses urllib, which the test guard blocks)."""
    return httpx.request(method, url, headers={"Authorization": f"Bearer {credential}"} if credential else {}, json=body).json()


def test_initialize_reports_the_package_version_and_no_session(live):
    """serverInfo carries the installed version; /mcp is stateless, so no mcp-session-id comes back
    (docs/mcp.md says so for client authors)."""
    url, key = live
    init = {"jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t", "version": "1"}}}
    r = httpx2.post(url + "/mcp", json=init, headers={"Authorization": f"Bearer {key}", "Accept": "application/json, text/event-stream"})
    assert r.status_code == 200 and "mcp-session-id" not in r.headers
    body = r.text if r.headers["content-type"].startswith("application/json") else r.text.split("data: ", 1)[1]
    info = json.loads(body)["result"]["serverInfo"]
    assert info == {**info, "name": "gpu-broker", "version": importlib.metadata.version("gpu-broker")} and info["version"]
