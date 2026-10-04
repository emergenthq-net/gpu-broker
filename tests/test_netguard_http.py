"""`<slot>_url` fetches against a loopback HTTP server: the default refusal, the allowlist,
every redirect hop checked, redirect bodies never read, and the broker's own ComfyUI reachable
for output views only. Network: a loopback server only."""
import contextlib
import dataclasses
import http.server
import threading
import time
import urllib.request

import pytest

from gpu_broker import media
from gpu_broker.broker import Broker
from gpu_broker.settings import Inputs
from tests.helpers import FakeBackends, FakeDriver, make_settings

REAL_OPEN = urllib.request.OpenerDirector.open   # captured before conftest blocks the network
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32


@pytest.fixture
def server(monkeypatch):
    """A loopback HTTP server: /a.png is an image, /hop redirects to an address given in ?to=,
    /huge and /endless redirect to /a.png with a body that never ends."""
    release = threading.Event()

    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.startswith(("/huge", "/endless")):
                self.send_response(int(self.path.split("code=")[1]) if "code=" in self.path else 302)
                self.send_header("Location", "/a.png")
                if self.path.startswith("/huge"):
                    self.send_header("Content-Length", str(10 ** 12))
                self.end_headers()
                with contextlib.suppress(OSError):   # the client hangs up without reading
                    while not release.is_set():
                        self.wfile.write(b"x" * 65536)
                        if self.path.startswith("/huge"):
                            release.wait(30)
                return
            if self.path.startswith("/hop"):
                self.send_response(302)
                self.send_header("Location", self.path.split("to=", 1)[1])
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.end_headers()
            self.wfile.write(PNG)

        def log_message(self, *a):
            pass
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setattr(urllib.request.OpenerDirector, "open", REAL_OPEN)
    monkeypatch.setenv("http_proxy", "http://203.0.113.1:3128")   # must be ignored
    yield f"http://127.0.0.1:{srv.server_port}"
    release.set()
    srv.shutdown()


def cfg(*nets):
    return Inputs(max_bytes=64, allow_urls=True, fetch_s=5, url_allow_networks=nets)


def test_loopback_is_refused_by_default_with_the_uniform_error(server):
    with pytest.raises(ValueError, match=r"^`image_url` could not be fetched$"):
        media.fetch("image", server + "/a.png", cfg())


def test_an_allowlisted_network_is_fetched_directly_despite_a_proxy_setting(server):
    assert media.fetch("image", server + "/a.png", cfg("127.0.0.1/32")).data == PNG


@pytest.mark.parametrize("target", ["http://10.9.0.1/a.png", "http://169.254.169.254/latest/meta-data",
                                    "http://[::1]:9/a.png", "file:///etc/passwd"])
def test_every_redirect_hop_is_checked(server, target):
    with pytest.raises(ValueError, match=r"^`image_url` could not be fetched$"):
        media.fetch("image", f"{server}/hop?to={target}", cfg("127.0.0.1/32"))


@pytest.mark.parametrize("path", ["/huge", "/endless", *(f"/huge?code={c}" for c in (301, 303, 307, 308))])
def test_a_redirect_body_is_never_read(server, path):
    """Following a 30x reads only its headers: a body that never ends costs nothing."""
    t0 = time.monotonic()
    assert media.fetch("image", server + path, cfg("127.0.0.1/32")).data == PNG
    assert time.monotonic() - t0 < 2   # fetch_s is 5: reading the body would hit it


def test_the_broker_allows_its_own_comfy_from_comfy_url(server, tmp_path):
    """Chained jobs: a job's output URL (the broker's ComfyUI) is fed to the next job."""
    s = make_settings(tmp_path)
    s = dataclasses.replace(s, comfy=dataclasses.replace(s.comfy, url=server, public_url=""))
    driver = FakeDriver()
    b = Broker(s, env={}, driver=driver, backends=FakeBackends(driver))
    view = "/view?filename=a.png&type=output"
    assert media.fetch("image", server + view, cfg(), b.fetch_policy).data == PNG
    for url in (server + "/a.png",                                          # not an output view
                server + "/view?filename=broker-0123456789ab-image.png&type=input",   # another job's input
                "http://127.0.0.1:9" + view):                              # another port on the host
        with pytest.raises(ValueError, match="could not be fetched"):
            media.fetch("image", url, cfg(), b.fetch_policy)


def test_video_urls_are_guarded_too(server):
    with pytest.raises(ValueError, match=r"^`video_url` could not be fetched$"):
        media.fetch("video", server + "/a.png", cfg())
