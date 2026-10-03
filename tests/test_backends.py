"""HTTP backends against a fake urlopen: request shapes, auth, ComfyUI polling and URL escaping."""
import io
import json
import urllib.error
import urllib.request

import pytest

from gpu_broker import settings
from gpu_broker.backends import HttpBackends

MODEL = {"endpoint": "http://llm:8081", "served_name": "served", "auth_env": "UPSTREAM_TOKEN_A"}
COMFY = settings.Comfy(url="http://comfy:8188", public_url="https://comfy.example")


class Resp(io.BytesIO):
    def __init__(self, body, status=200):
        super().__init__(json.dumps(body).encode())
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


@pytest.fixture
def http(monkeypatch):
    """Route by URL path; record every request."""
    seen, routes = [], {}

    def urlopen(req, timeout):
        seen.append(req)
        for suffix, answer in routes.items():
            if req.full_url.endswith(suffix) or suffix in req.full_url:
                return answer(req) if callable(answer) else Resp(answer)
        raise urllib.error.URLError("refused")
    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    return seen, routes


def backends(**kw):
    return HttpBackends(COMFY, settings.Timeouts(comfy_run_s=1), settings.Intervals(comfy_poll_s=0),
                        {"UPSTREAM_TOKEN_A": "s3cret"}, sleep=lambda _: None, **kw)


def test_chat_sends_the_served_name_and_never_streams(http):
    seen, routes = http
    routes["/v1/chat/completions"] = {"choices": []}
    backends().llm_chat(MODEL, {"model": "alias", "stream": True, "messages": [], "temperature": 0.2})
    body = json.loads(seen[0].data)
    assert body == {"messages": [], "temperature": 0.2, "model": "served", "stream": False}
    assert seen[0].headers["Authorization"] == "Bearer s3cret"


def test_health_is_false_on_any_failure(http):
    _, routes = http
    assert backends().llm_healthy(MODEL) is False
    routes["/health"] = {}
    assert backends().llm_healthy(MODEL) is True
    assert backends().comfy_queue_len() is None
    routes["/prompt"] = {"exec_info": {"queue_remaining": 2}}
    assert backends().comfy_queue_len() == 2


def test_comfy_run_polls_until_complete_and_escapes_urls(http):
    seen, routes = http
    routes["/prompt"] = {"prompt_id": "a/b?c"}
    polls = iter([{}, {"a/b?c": {"status": {"completed": True}, "outputs": {"9": {"images": [
        {"filename": "x y&z.png", "subfolder": "broker/j1", "type": "output"}]}}}}])
    routes["/history/"] = lambda req: Resp(next(polls))
    out = backends().comfy_run("m", {"1": {}}, "j1")
    assert seen[1].full_url == "http://comfy:8188/history/a%2Fb%3Fc"
    assert out["outputs"] == [{"file": "broker/j1/x y&z.png", "url":
                               "https://comfy.example/view?filename=x+y%26z.png&subfolder=broker%2Fj1&type=output"}]


def test_comfy_run_reports_execution_errors_and_timeouts(http):
    _, routes = http
    routes["/prompt"] = {"prompt_id": "p"}
    routes["/history/"] = {"p": {"status": {"status_str": "error", "messages": [["execution_error", {"exception_message": "OOM"}]]}}}
    with pytest.raises(RuntimeError, match="OOM"):
        backends().comfy_run("m", {}, "j")
    routes["/history/"] = {}
    ticks = iter(range(100))
    with pytest.raises(RuntimeError, match="timed out"):
        backends(clock=lambda: next(ticks)).comfy_run("m", {}, "j")


def test_free_is_best_effort(http):
    backends().comfy_free()   # ComfyUI down: no exception


def test_only_http_urls_are_ever_opened(http):
    for url in ("file:///etc/passwd", "ftp://x/y", "gopher://x"):
        with pytest.raises(ValueError, match="non-HTTP"):
            backends().llm_chat({**MODEL, "endpoint": url}, {})
    assert http[0] == []


def test_stream_relays_lines_and_keeps_usage_and_timings(http):
    seen, routes = http

    class Lines(Resp):
        def __iter__(self):
            return iter([b'data: {"choices":[{"delta":{"content":"a"}}]}\n', b"\n",
                         b'data: {"choices":[],"usage":{"total_tokens":3},"timings":{"predicted_n":1}}\n',
                         b"data: not json {\n", b"data: [DONE]\n"])
    routes["/v1/chat/completions"] = lambda req: Lines({})
    summary = {}
    lines = list(backends().llm_stream(MODEL, {"model": "alias", "messages": [], "interactive": True, "caps": ["x"]}, summary))
    assert len(lines) == 5 and lines[-1] == "data: [DONE]\n"
    assert summary == {"usage": {"total_tokens": 3}, "timings": {"predicted_n": 1}}
    assert json.loads(seen[0].data) == {"messages": [], "model": "served", "stream": True}   # broker fields stripped


def test_comfy_upload_posts_multipart_into_the_input_folder(http):
    seen, routes = http
    routes["/upload/image"] = {"name": "broker-j1-image.png", "subfolder": "", "type": "input"}
    assert backends().comfy_upload("broker-j1-image.png", b"\x89PNG-data", "png") == "broker-j1-image.png"
    req = seen[-1]
    assert req.full_url == "http://comfy:8188/upload/image" and req.get_method() == "POST"
    boundary = req.get_header("Content-type").split("boundary=")[1]
    body = req.data
    assert body.endswith(f"--{boundary}--\r\n".encode())
    assert b'name="type"\r\n\r\ninput\r\n' in body and b'name="overwrite"\r\n\r\ntrue\r\n' in body
    assert (b'name="image"; filename="broker-j1-image.png"\r\nContent-Type: image/png\r\n\r\n\x89PNG-data\r\n'
            in body)


def test_comfy_upload_returns_the_subfolder_path_and_reports_rejections(http):
    _, routes = http
    routes["/upload/image"] = {"name": "x.png", "subfolder": "in", "type": "input"}
    assert backends().comfy_upload("x.png", b"d", "webp") == "in/x.png"

    def reject(req):
        raise urllib.error.HTTPError(req.full_url, 400, "bad", {}, io.BytesIO(b"Invalid image file"))
    routes["/upload/image"] = reject
    with pytest.raises(RuntimeError, match="rejected the input image: Invalid image file"):
        backends().comfy_upload("x.png", b"d", "png")
