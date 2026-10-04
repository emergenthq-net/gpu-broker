"""Review of the MCP PR, the HTTP side: ComfyUI reads (its own type, its auth), the app
lifespan around the MCP session manager, and the /mcp guard (key lookup off the event loop,
a refusal as a JSON-RPC error)."""
import threading

import pytest
from fastapi.testclient import TestClient

from gpu_broker import settings
from gpu_broker.backends import HttpBackends
from gpu_broker.keys import KeyStore
from gpu_broker.mcp_server import http as mcp_http
from gpu_broker.web.app import create_app
from tests.helpers import TOKEN
from tests.mcp_fakes import mcp_broker

__all__ = ["mcp_broker"]
LIST = {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
ACCEPT = {"Accept": "application/json, text/event-stream"}


class Body:
    status = 200

    def __init__(self, data):
        self.data = data

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self, n=-1):
        return self.data if n < 0 else self.data[:n]


@pytest.mark.parametrize("auth_env, header", [("", None), ("UPSTREAM_TOKEN_COMFY", "Bearer s3cret")])
def test_comfy_view_reads_the_outputs_type_over_comfy_url_with_its_auth(monkeypatch, auth_env, header):
    seen = []
    monkeypatch.setattr("urllib.request.urlopen", lambda req, timeout: seen.append(req) or Body(b"x" * 10))
    b = HttpBackends(settings.Comfy(url="http://127.0.0.1:8188", public_url="http://gpu:8188", auth_env=auth_env),
                     settings.Timeouts(), settings.Intervals(), {"UPSTREAM_TOKEN_COMFY": "s3cret"})
    assert b.comfy_view("a b.png", "broker", "temp", 10) == b"x" * 10
    assert b.comfy_view("a.png", "", "output", 9) is None   # larger than the cap
    assert seen[0].full_url == "http://127.0.0.1:8188/view?filename=a+b.png&subfolder=broker&type=temp"
    assert seen[0].get_header("Authorization") == header
    b.comfy_alive()                                         # every ComfyUI call carries it, not just /view
    assert seen[-1].get_header("Authorization") == header


def test_the_lifespan_runs_twice_with_a_new_session_manager(mcp_broker):
    app = create_app(mcp_broker, TOKEN, start=False)
    endpoint = next(r.endpoint.endpoint for r in app.router.routes if getattr(r, "path", "") == mcp_http.PATH)
    managers = []
    for _ in range(2):
        with TestClient(app) as c:
            r = c.post("/mcp", json=LIST, headers={"Authorization": f"Bearer {TOKEN}", **ACCEPT})
            assert r.status_code == 200 and "list_models" in r.text
            managers.append(endpoint.manager)
    assert None not in managers and managers[0] is not managers[1] and endpoint.manager is None


def test_a_failing_mcp_start_still_stops_the_broker_and_closes_the_keys(mcp_broker, monkeypatch):
    keys = KeyStore(mcp_broker.settings.db)
    stopped, closed = [], []
    monkeypatch.setattr(mcp_broker, "stop", lambda *a: stopped.append(1))
    monkeypatch.setattr(keys, "close", lambda: closed.append(1))

    class Broken:
        def __init__(self, *a, **k):
            raise RuntimeError("no MCP today")
    monkeypatch.setattr(mcp_http, "StreamableHTTPSessionManager", Broken)
    with pytest.raises(RuntimeError, match="no MCP today"), TestClient(create_app(mcp_broker, TOKEN, start=False, keystore=keys)):
        pass
    assert stopped == [1] and closed == [1]


def test_the_guard_looks_keys_up_off_the_event_loop_and_refuses_in_json_rpc(mcp_broker, monkeypatch):
    keys = KeyStore(mcp_broker.settings.db)
    key = keys.issue("laptop")["key"]
    threads, identify = [], keys.identify
    monkeypatch.setattr(keys, "identify", lambda v: threads.append(threading.current_thread().name) or identify(v))
    with TestClient(create_app(mcp_broker, TOKEN, start=False, keystore=keys)) as c:
        assert c.post("/mcp", json=LIST, headers={"Authorization": f"Bearer {key}", **ACCEPT}).status_code == 200
        loop_thread = c.portal.call(lambda: threading.current_thread().name)
        r = c.post("/mcp", json=LIST, headers={"Authorization": "Bearer gbk_revoked", **ACCEPT})
    assert threads and loop_thread not in threads
    assert r.status_code == 401 and r.headers["www-authenticate"] == "Bearer"
    assert r.json() == {"jsonrpc": "2.0", "id": None,
                        "error": {"code": mcp_http.AUTH_ERROR, "message": "key revoked or invalid (HTTP 401)", "data": {"http_status": 401}}}
