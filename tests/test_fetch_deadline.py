"""`images.fetch_s` bounds the whole <slot>_url fetch however the server stalls: dripping the
headers, dripping the body, accepting and never answering, or spreading it over redirect hops.
Network: loopback servers only, allowlisted for the test."""
import socket
import threading
import time
import urllib.request

import pytest

from gpu_broker import media
from gpu_broker.settings import Inputs

REAL_OPEN = urllib.request.OpenerDirector.open   # captured before conftest blocks the network
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
DEADLINE_S = 1.0
SLACK_S = 0.5           # thread wake-up and teardown
DRIP_S = 0.1            # one byte per DRIP_S: each read is quick, the total is not
HEAD = b"HTTP/1.0 200 OK\r\nContent-Type: image/png\r\n\r\n"


def drip_head(conn, port):
    for b in HEAD + PNG:
        conn.sendall(bytes([b]))
        time.sleep(DRIP_S)


def drip_body(conn, port):
    conn.sendall(HEAD + PNG[:4])
    for b in PNG[4:]:
        conn.sendall(bytes([b]))
        time.sleep(DRIP_S)


def silent(conn, port):
    time.sleep(DEADLINE_S * 4)


def redirect_then_drip(conn, port):
    """Each hop alone stays under the deadline; together they do not."""
    req = conn.recv(1024)
    if b"/next" not in req:
        time.sleep(DEADLINE_S * 0.6)
        conn.sendall(b"HTTP/1.0 302 Found\r\nLocation: http://127.0.0.1:%d/next\r\n\r\n" % port)
        return
    time.sleep(DEADLINE_S * 0.6)
    conn.sendall(HEAD + PNG)


@pytest.fixture
def serve(monkeypatch):
    monkeypatch.setattr(urllib.request.OpenerDirector, "open", REAL_OPEN)
    socks = []

    def start(behaviour):
        srv = socket.create_server(("127.0.0.1", 0))
        socks.append(srv)
        port = srv.getsockname()[1]

        def loop():
            while True:
                try:
                    conn, _ = srv.accept()
                except OSError:
                    return
                threading.Thread(target=handle, args=(conn,), daemon=True).start()

        def handle(conn):
            try:
                behaviour(conn, port)
            except OSError:
                pass   # the broker hung up: that is the point
            finally:
                conn.close()
        threading.Thread(target=loop, daemon=True).start()
        return f"http://127.0.0.1:{port}/a.png"
    yield start
    for s in socks:
        s.close()


CFG = Inputs(max_bytes=64, allow_urls=True, fetch_s=DEADLINE_S, url_allow_networks=("127.0.0.1/32",))


@pytest.mark.parametrize("behaviour", [drip_head, drip_body, silent, redirect_then_drip])
def test_the_deadline_is_total(serve, behaviour):
    url = serve(behaviour)
    t0 = time.monotonic()
    with pytest.raises(ValueError, match=r"^`image_url` could not be fetched$"):
        media.fetch("image", url, CFG)
    assert time.monotonic() - t0 < DEADLINE_S + SLACK_S


def test_a_prompt_server_is_fetched_in_full(serve):
    url = serve(lambda conn, port: (conn.recv(1024), conn.sendall(HEAD + PNG)))
    assert media.fetch("image", url, CFG).data == PNG


@pytest.mark.parametrize("sent", [PNG, b""])
def test_a_body_shorter_than_its_content_length_is_not_a_file(serve, sent):
    """read1 returns b"" when the connection closes early; the missing bytes must fail the fetch."""
    head = b"HTTP/1.0 200 OK\r\nContent-Type: image/png\r\nContent-Length: 60\r\n\r\n"
    url = serve(lambda conn, port: (conn.recv(1024), conn.sendall(head + sent)))
    with pytest.raises(ValueError, match=r"^`image_url` could not be fetched$"):
        media.fetch("image", url, CFG)
