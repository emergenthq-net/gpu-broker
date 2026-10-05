"""Hosted fallback (off by default): forwards hosted names the local side cannot serve, with the
operator's upstream key, and every answer says which side served it."""
from __future__ import annotations

import dataclasses
import io
import json
import urllib.error

import pytest
from fastapi.testclient import TestClient

from gpu_broker.broker import Broker
from gpu_broker.settings import Fallback
from gpu_broker.web.app import create_app
from gpu_broker.web.upstream import Upstream
from tests.dropin_fakes import ToolBackends, catalog_with_embedder
from tests.helpers import TOKEN, FakeDriver, make_settings, wait_idle

MAIN = {"Authorization": f"Bearer {TOKEN}"}
HI = [{"role": "user", "content": "hi"}]
UPSTREAM_KEY = "sk-upstream-secret"
ENV = {"UPSTREAM_OPENAI_API_KEY": UPSTREAM_KEY, "UPSTREAM_ANTHROPIC_API_KEY": UPSTREAM_KEY}


class FakeProvider:
    def __init__(self, fail: int | None = None) -> None:
        self.requests: list[tuple[str, dict, dict]] = []
        self.fail = fail

    def __call__(self, req, timeout):
        body = json.loads(req.data)
        self.requests.append((req.full_url, dict(req.header_items()), body))
        if self.fail:
            raise urllib.error.HTTPError(req.full_url, self.fail, "x", {}, io.BytesIO(b'{"error": {"type": "rate_limit"}}'))
        if body.get("stream"):
            return io.BytesIO(b'data: {"choices": [{"delta": {"content": "hosted"}}]}\n\ndata: [DONE]\n\n')
        return io.BytesIO(json.dumps({"id": "x", "choices": [{"message": {"content": "hosted"}}]}).encode())


@pytest.fixture
def make(tmp_path):
    made = []

    def build(enabled=True, provider=None, resident="llama-8b", env=ENV):
        driver = FakeDriver({resident})
        s = make_settings(tmp_path, catalog=catalog_with_embedder(tmp_path, embed=False))
        s = dataclasses.replace(s, fallback=Fallback(enabled=enabled))
        b = Broker(s, env={}, driver=driver, backends=ToolBackends(driver))
        b.start()
        made.append(b)
        up = Upstream(s.fallback, env, provider)
        return TestClient(create_app(b, TOKEN, start=False, upstream=up)), b
    yield build
    for b in made:
        assert wait_idle(b)
        b.stop()


def test_local_answers_are_marked_local(make):
    c, _ = make(provider=FakeProvider())
    r = c.post("/v1/chat/completions", headers=MAIN, json={"model": "gpt-4o", "messages": HI})
    assert r.headers["x-broker-served-by"] == "local" and r.json()["x_broker"]["served_by"] == "local"


def test_busy_gpu_sends_a_hosted_name_upstream(make):
    p = FakeProvider()
    c, b = make(provider=p)
    b.scheduler.pool.reopen(None)   # what a residency switch (say, to a video model) looks like
    r = c.post("/v1/chat/completions", headers=MAIN, json={"model": "gpt-4o", "messages": HI})
    assert r.headers["x-broker-served-by"] == "hosted" and r.json()["x_broker"]["served_by"] == "hosted"
    url, headers, body = p.requests[0]
    assert url == "https://api.openai.com/v1/chat/completions" and headers["Authorization"] == f"Bearer {UPSTREAM_KEY}"
    assert body["model"] == "gpt-4o"                                       # the hosted name, not the local mapping
    assert UPSTREAM_KEY not in r.text and TOKEN not in json.dumps(p.requests)   # neither credential crosses over


def test_catalog_names_never_go_upstream(make, monkeypatch):
    p = FakeProvider()
    c, b = make(provider=p)
    b.scheduler.pool.reopen(None)
    r = c.post("/v1/chat/completions", headers=MAIN, json={"model": "llama", "messages": HI})
    assert r.status_code == 200 and r.headers["x-broker-served-by"] == "local" and p.requests == []
    body = {"model": "llama", "max_tokens": 8, "messages": HI, "tools": [{"type": "web_search_20250305", "name": "w"}]}
    assert c.post("/v1/messages", headers=MAIN, json=body).status_code == 400 and p.requests == []
    b.scheduler.pool.reopen("llama-8b")

    def broken(*a):
        raise OSError("server down")
    monkeypatch.setattr(b.backends, "llm_chat", broken)
    assert c.post("/v1/chat/completions", headers=MAIN, json={"model": "llama", "messages": HI}).status_code == 502
    assert p.requests == []


def test_a_failed_local_call_on_a_hosted_name_goes_upstream(make, monkeypatch):
    p = FakeProvider()
    c, b = make(provider=p)

    def broken(*a):
        raise OSError("server down")
    monkeypatch.setattr(b.backends, "llm_chat", broken)
    r = c.post("/v1/chat/completions", headers=MAIN, json={"model": "gpt-4o", "messages": HI})
    assert r.headers["x-broker-served-by"] == "hosted" and r.json()["x_broker"]["reason"] == "server down"


def test_resident_model_is_never_bypassed(make, monkeypatch):
    """No direct slot free, but the model is loaded: wait in the queue, do not go upstream."""
    p = FakeProvider()
    c, b = make(provider=p)
    monkeypatch.setattr(b.chat, "open", lambda *a: None)
    r = c.post("/v1/chat/completions", headers=MAIN, json={"model": "gpt-4o", "messages": HI})
    assert r.headers["x-broker-served-by"] == "local" and p.requests == []


def test_off_by_default_and_without_a_key(make):
    for kw in ({"enabled": False}, {"env": {}}):
        p = FakeProvider()
        c, b = make(provider=p, **kw)
        b.scheduler.pool.reopen(None)
        r = c.post("/v1/chat/completions", headers={**MAIN, "x-priority": "background"}, json={"model": "gpt-4o", "messages": HI})
        assert r.headers["x-broker-served-by"] == "local" and p.requests == []
        r = c.post("/v1/embeddings", headers=MAIN, json={"model": "text-embedding-3-small", "input": "x"})
        assert r.status_code == 404 and p.requests == []


def test_hosted_only_anthropic_feature_goes_upstream(make):
    p = FakeProvider()
    c, _ = make(provider=p)
    body = {"model": "claude-sonnet-4-5", "max_tokens": 8, "messages": HI, "tools": [{"type": "web_search_20250305", "name": "web_search"}]}
    r = c.post("/v1/messages", headers={**MAIN, "anthropic-version": "2023-06-01", "anthropic-beta": "b1"}, json=body)
    assert r.headers["x-broker-served-by"] == "hosted"
    url, headers, sent = p.requests[0]
    assert url == "https://api.anthropic.com/v1/messages" and headers["X-api-key"] == UPSTREAM_KEY
    assert headers["Anthropic-version"] == "2023-06-01" and headers["Anthropic-beta"] == "b1" and sent == body


def test_provider_errors_and_streams_pass_through(make):
    c, b = make(provider=FakeProvider(fail=429))
    b.scheduler.pool.reopen(None)
    r = c.post("/v1/chat/completions", headers=MAIN, json={"model": "gpt-4o", "messages": HI})
    assert r.status_code == 429 and r.json() == {"error": {"type": "rate_limit"}} and r.headers["x-broker-served-by"] == "hosted"
    c, b = make(provider=FakeProvider())
    b.scheduler.pool.reopen(None)
    r = c.post("/v1/chat/completions", headers=MAIN, json={"model": "gpt-4o", "messages": HI, "stream": True})
    assert "hosted" in r.text and r.headers["x-broker-served-by"] == "hosted"


def test_embeddings_without_a_local_model_go_upstream(make):
    p = FakeProvider()
    c, _ = make(provider=p)
    r = c.post("/v1/embeddings", headers=MAIN, json={"model": "text-embedding-3-small", "input": "x"})
    assert r.headers["x-broker-served-by"] == "hosted" and p.requests[0][0].endswith("/v1/embeddings")


def test_broker_only_fields_are_not_sent_upstream(make):
    p = FakeProvider()
    c, b = make(provider=p)
    b.scheduler.pool.reopen(None)
    extra = {"requester": "me", "wait": True, "wait_s": 5, "caps": ["chat"], "kind": "llm", "session": True,
             "temperature": 0.5}
    c.post("/v1/chat/completions", headers=MAIN, json={"model": "gpt-4o", "messages": HI, **extra})
    _, _, body = p.requests[0]
    assert set(body) == {"model", "messages", "temperature"}
