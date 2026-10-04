"""`gpu-broker mcp` (stdio): which credential goes where, and that plain http reaches only the
addresses that were checked (a name that rebinds to a public address later never gets the key)."""
import http.server
import ipaddress
import socket
import threading
from typing import ClassVar

import anyio
import httpx2
import pytest

from gpu_broker.mcp_server import pinning, stdio
from gpu_broker.netguard import embedded

KEY = {"GPU_BROKER_API_KEY": "k"}


def answers(table):
    def resolve(host, port):
        if host not in table:
            raise OSError("no such host")
        return [(None, None, None, "", (a, 0)) for a in table[host]]
    return resolve


def test_plain_http_is_pinned_to_the_checked_private_addresses(tmp_path):
    r = answers({"gpu": ["192.0.2.26", "192.0.2.26", "fd00::5"]})
    assert stdio.target("http://gpu:8095", KEY, tmp_path, r) == ("http://gpu:8095/mcp", "k", ["192.0.2.26", "fd00::5"])
    assert stdio.target("https://gpu:8095", KEY, tmp_path, r)[2] == []           # TLS checks the name
    assert stdio.target("http://192.0.2.7:8095", KEY, tmp_path, r)[2] == []      # a literal needs no DNS


@pytest.mark.parametrize("answer", [["8.8.8.8"], ["192.0.2.1", "8.8.8.8"], ["2002:808:808::1"], ["2001:0:4136:e378:8000:63bf:f7f7:f7f7"]])
def test_no_key_over_plain_http_to_a_public_answer(tmp_path, answer):
    with pytest.raises(ValueError, match="plain http"):
        stdio.target("http://gpu:8095", KEY, tmp_path, answers({"gpu": answer}))


def test_the_main_token_needs_the_local_broker(tmp_path, monkeypatch):
    (tmp_path / "c.yaml").write_text("server: {port: 9000}\n")
    env = {"BROKER_TOKEN": "t", "BROKER_CONFIG": str(tmp_path / "c.yaml")}
    monkeypatch.setenv("BROKER_CONFIG", env["BROKER_CONFIG"])
    assert stdio.target("http://127.0.0.1:9000", env, tmp_path, answers({}))[1] == "t"
    for elsewhere in ("http://127.0.0.1:8095", "http://192.0.2.26:9000"):
        with pytest.raises(ValueError, match="main token"):
            stdio.target(elsewhere, env, tmp_path, answers({}))


def test_6to4_and_teredo_are_judged_by_the_ipv4_they_carry():
    assert str(embedded(ipaddress.ip_address("2002:808:808::1"))) == "8.8.8.8"
    teredo = ipaddress.ip_address("2001:0:4136:e378:8000:63bf:f7f7:f7f7")
    assert str(embedded(teredo)) == "8.8.8.8"                                    # the client, not the server
    assert not pinning.private("2002:808:808::1") and not pinning.private("2001:0:4136:e378:8000:63bf:f7f7:f7f7")
    assert pinning.private("2002:c000:21a::1")                                    # 6to4 of 192.0.2.26


class Seen(http.server.BaseHTTPRequestHandler):
    seen: ClassVar[list[tuple[str, str]]] = []

    def do_GET(self):
        Seen.seen.append((self.headers["Host"], self.headers["Authorization"]))
        self.send_response(200)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, *a):
        pass


def test_a_name_that_rebinds_later_still_reaches_only_the_checked_address(monkeypatch):
    """Checked at 127.0.0.1; afterwards DNS answers a public address. Every new connection over the
    relay's lifetime still goes to 127.0.0.1, with the original name as Host."""
    srv = http.server.HTTPServer(("127.0.0.1", 0), Seen)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    port = srv.server_address[1]
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("203.0.113.9", port))])
    Seen.seen.clear()

    async def go():
        url = f"http://rebind.test:{port}/mcp"
        async with httpx2.AsyncClient(transport=stdio.transport(url, ["127.0.0.1"]), headers={"Authorization": "Bearer k"}, timeout=2,
                                      limits=httpx2.Limits(max_keepalive_connections=0)) as c:
            for _ in range(3):   # no keep-alive: three separate connections
                assert (await c.get(url)).status_code == 200
            assert (await c.get(f"http://elsewhere.test:{port}/mcp")).status_code == 502   # any other host is refused
    try:
        anyio.run(go)
    finally:
        srv.shutdown()
    assert Seen.seen == [(f"rebind.test:{port}", "Bearer k")] * 3
