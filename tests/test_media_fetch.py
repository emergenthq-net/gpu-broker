"""`<slot>_url` fetching: bounded reads, the time left of the fetch's deadline, and one uniform
error for every failure. The opener is faked; tests/test_fetch_deadline.py uses real sockets."""
import base64
import email.message
import http.client
import io
import urllib.request

import pytest

from gpu_broker import media, netguard
from gpu_broker.settings import Inputs

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 32
GIF = b"GIF89a" + b"\x00" * 32
MP4 = b"\x00\x00\x00\x18ftypisom" + b"\x00" * 32
CFG = Inputs(max_bytes=64, video_max_bytes=96, max_frames=3, allow_urls=True)
b64 = lambda data: base64.b64encode(data).decode()  # noqa: E731


class Resp(io.BytesIO):
    def __init__(self, body, ctype):
        super().__init__(body)
        self.headers = email.message.Message()
        self.headers["Content-Type"] = ctype
        self.reads = []

    read = None               # read() waits for the full size: the broker must use read1()

    def read1(self, size=-1):
        assert size > 0           # the broker must never read an unbounded body
        self.reads.append(size)
        return super().read1(size)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


@pytest.fixture
def web(monkeypatch):
    seen = []

    def serve(body, ctype="image/png"):
        def opener_open(self, req, *a, timeout=None, **k):
            resp = Resp(body, ctype)
            seen.append((req.full_url, timeout, resp.reads))
            return resp
        monkeypatch.setattr(urllib.request.OpenerDirector, "open", opener_open)
    return serve, seen


def test_fetch_reads_an_http_image_with_the_fetch_timeout(web):
    serve, seen = web
    serve(PNG)
    ticks = iter(range(100))                  # a clock that moves 1 s per read
    im = media.fetch("image", "https://h/a.png", CFG, clock=lambda: next(ticks))
    assert (im.data, im.kind, im.source) == (PNG, "png", "url")
    ((url, timeout, reads),) = seen
    assert (url, reads) == ("https://h/a.png", [CFG.max_bytes + 1, CFG.max_bytes + 1 - len(PNG)])
    assert 0 < timeout < CFG.fetch_s         # the time left of the fetch's deadline, not all of it


def test_fetch_reads_in_chunks_and_stops_at_the_cap(web, monkeypatch):
    serve, seen = web
    monkeypatch.setattr(media, "READ_CHUNK", 10)
    serve(PNG + b"\x00" * 500)
    with pytest.raises(ValueError, match=r"^`image_url` could not be fetched$"):
        media.fetch("image", "http://h/a.png", CFG)
    assert seen[0][2] == [10] * 6 + [5]       # 65 bytes read (cap + 1), never the whole body


@pytest.mark.parametrize(("slot", "body", "ctype"), [("image", PNG, "image/png"), ("video", MP4, "video/mp4")])
def test_fetch_enforces_a_total_deadline_across_reads(web, monkeypatch, slot, body, ctype):
    serve, seen = web
    monkeypatch.setattr(media, "READ_CHUNK", 8)
    serve(body + b"\x00" * 40, ctype)
    ticks = iter(range(0, 1000, 10))           # every clock read is 10 s later; fetch_s is 30
    with pytest.raises(ValueError, match=rf"^`{slot}_url` could not be fetched$"):
        media.fetch(slot, f"http://h/{slot}", CFG, clock=lambda: next(ticks))
    assert len(seen[0][2]) == 1                # stops reading as soon as the time is up
    serve(body, ctype)
    late = iter([0.0])                          # start at 0, then every read sees 29.9 s: still in time
    assert media.fetch(slot, f"http://h/{slot}", CFG, clock=lambda: next(late, 29.9)).data == body


@pytest.mark.parametrize(("slot", "body", "ctype"), [("image", PNG + b"\x00" * 64, "image/png"),
                                                     ("image", PNG, "image/webp"), ("image", GIF, "image/gif"),
                                                     ("image", b"", "image/png"), ("video", MP4 + b"\x00" * 96, "video/mp4"),
                                                     ("video", PNG, "image/png")])
def test_url_content_errors_are_the_uniform_fetch_error(web, slot, body, ctype):
    """Too large, wrong type, not accepted, empty: each would reveal that something answered."""
    serve, _ = web
    serve(body, ctype)
    with pytest.raises(ValueError, match=rf"^`{slot}_url` could not be fetched$"):
        media.fetch(slot, "http://h/x", CFG)


def test_fetch_checks_the_scheme_and_that_urls_are_enabled():
    for url in ("file:///etc/passwd", "ftp://h/a.png"):
        with pytest.raises(ValueError, match="must be an http"):
            media.fetch("image", url, CFG)
    with pytest.raises(ValueError, match="disabled on this broker"):
        media.fetch("end_image", "http://h/a.png", Inputs())


def test_redirects_may_not_leave_http():
    h = netguard._HttpOnlyRedirects()
    req = urllib.request.Request("http://h/a.png")
    with pytest.raises(netguard.Refused, match="non-HTTP"):
        h.redirect_request(req, None, 302, "Found", {}, "file:///etc/passwd")
    assert h.redirect_request(req, None, 302, "Found", {}, "https://cdn/a.png").full_url == "https://cdn/a.png"


@pytest.mark.parametrize("error", [OSError("connection refused"), netguard.Refused("10.1.2.1 is not public"),
                                   http.client.RemoteDisconnected("bye"), http.client.IncompleteRead(b""),
                                   TimeoutError("slow"), ValueError("redirect to file:")])
@pytest.mark.parametrize("slot", ["image", "video"])
def test_every_fetch_failure_is_the_same_400(monkeypatch, error, slot):
    def fail(self, req, *a, **k):
        raise error
    monkeypatch.setattr(urllib.request.OpenerDirector, "open", fail)
    with pytest.raises(ValueError, match=rf"^`{slot}_url` could not be fetched$"):   # no detail: no network map
        media.fetch(slot, "http://h/x", CFG)


def test_read_and_strip(web):
    serve, _ = web
    serve(JPEG, "image/jpeg")
    body = {"prompt": "p", "image": b64(PNG), "end_image_url": "http://h/e.jpg"}
    assert [(i.slot, i.kind, i.source) for i in media.read(body, CFG)] == [("image", "png", "inline"),
                                                                           ("end_image", "jpeg", "url")]
    assert media.strip(body) == {"prompt": "p"}
    assert media.strip({"frames": ["x"], "video_url": "u", "video": "v", "frame_stride": 2}) == {"frame_stride": 2}


def test_the_deadline_is_closed_even_when_building_the_opener_fails(monkeypatch):
    closed = []

    class Tracked(media.Deadline):
        def close(self):
            closed.append(True)
            super().close()
    monkeypatch.setattr(media, "Deadline", Tracked)
    monkeypatch.setattr(netguard, "opener", lambda *a, **k: (_ for _ in ()).throw(OSError("no handler")))
    with pytest.raises(ValueError, match=r"^`image_url` could not be fetched$"):
        media.fetch("image", "http://h/a.png", CFG)
    assert closed == [True]
