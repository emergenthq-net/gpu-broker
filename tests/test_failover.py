"""Cloud first, local on failure, through the real routes and a real (fake) provider over HTTP.

What fails over (connection dropped, hang, 5xx, 529, quota/credit), what is returned as-is (400,
401, a plain 429), the breaker skipping a provider that is down and coming back after a probe,
streams failing before vs after their first byte, and the Responses API."""
from __future__ import annotations

import json
import time

import pytest

from gpu_broker.constants import Event
from gpu_broker.web.failover import FALLBACK, SERVED_BY
from tests.failover_fakes import ANT, BREAKER, ERRORS, LOCAL, OAI, OPENAI_PLANTED, PLANTED, FakeCloud, build
from tests.helpers import TOKEN, wait_idle

HI = [{"role": "user", "content": "hi"}]
MSG = {"model": "claude-sonnet-4-5", "max_tokens": 50, "messages": HI}
CHAT = {"model": "gpt-4o", "messages": HI}


@pytest.fixture
def cloud():
    c = FakeCloud()
    yield c
    c.close()


@pytest.fixture
def setup(tmp_path, cloud):
    made = []

    def make(**kw):
        client, b, router, clock = build(tmp_path, cloud, **kw)
        made.append(b)
        return client, b, router, clock
    yield make
    for b in made:
        assert wait_idle(b)
        b.stop()


OPERATOR = {"env": {"UP_ANT": "sk-ant-operator"}}   # an operator key: what the prober uses


def with_operator_key(cloud):
    return OPERATOR | {"providers": {
        "anthropic": {"url": cloud.url, "api": "anthropic", "key_env": "UP_ANT", "pass_client_key": True},
        "openai": {"url": cloud.url, "api": "openai", "pass_client_key": True}}}


def events(b, kind):
    return [e["data"] for e in b.store.events(0, 1000) if e["kind"] == kind]


# ---- the happy path: forwarded with the client's own key ------------------------------

def test_anthropic_request_goes_to_the_cloud_with_the_clients_key(setup, cloud):
    client, *_ = setup()
    r = client.post("/v1/messages", json=MSG | {"requester": "me", "wait": 5}, headers=ANT)
    assert r.status_code == 200 and r.json()["content"][0]["text"] == "from the cloud"
    assert r.headers[SERVED_BY] == "anthropic" and FALLBACK not in r.headers
    sent = cloud.posts()[-1]
    assert sent["path"] == "/v1/messages" and sent["headers"]["x-api-key"] == PLANTED
    assert "x-gpu-broker-key" not in {k.lower() for k in sent["headers"]}
    assert TOKEN not in json.dumps(sent["headers"]) and TOKEN not in json.dumps(sent["body"])
    assert "requester" not in sent["body"] and "wait" not in sent["body"]   # the broker's own fields stay here


def test_openai_stream_is_relayed_from_the_cloud(setup, cloud):
    client, *_ = setup()
    r = client.post("/v1/chat/completions", json=CHAT | {"stream": True}, headers=OAI)
    assert r.headers[SERVED_BY] == "openai" and '"content": "from th' in r.text and r.text.endswith("[DONE]\n\n")
    assert cloud.posts()[-1]["headers"]["authorization"] == f"Bearer {OPENAI_PLANTED}"


def test_an_unrouted_model_never_touches_the_cloud(setup, cloud):
    client, *_ = setup()
    r = client.post("/v1/chat/completions", json={"model": LOCAL, "messages": HI}, headers=OAI)
    assert r.status_code == 200 and SERVED_BY not in r.headers and cloud.seen == []


# ---- fail over ----------------------------------------------------------------------

@pytest.mark.parametrize(("mode", "why"), [("529", "529 overloaded_error"), ("500", "500 server_error"),
                                           ("drop", "connection dropped"), ("credit402", "402 billing_error")])
def test_failures_go_to_the_local_model(setup, cloud, mode, why):
    client, b, *_ = setup()
    cloud.mode = mode
    r = client.post("/v1/messages", json=MSG, headers=ANT)
    assert r.status_code == 200 and r.json()["type"] == "message"
    assert r.headers[SERVED_BY] == f"local:{LOCAL}" and why in r.headers[FALLBACK]
    assert r.json()["model"] == LOCAL   # the local model that answered, named as such
    assert events(b, Event.UPSTREAM_FAILOVER)[-1]["to"] == f"local:{LOCAL}"


def test_a_hung_provider_fails_over_after_the_first_byte_timeout(setup, cloud):
    client, *_ = setup()
    cloud.mode = "hang"
    start = time.monotonic()
    r = client.post("/v1/chat/completions", json=CHAT, headers=OAI)
    assert r.status_code == 200 and r.headers[SERVED_BY] == f"local:{LOCAL}" and "timed out" in r.headers[FALLBACK]
    assert time.monotonic() - start < 2.5 and r.json()["model"] == LOCAL


def test_a_quota_error_fails_over_and_that_key_is_left_alone(setup, cloud):
    client, b, router, _ = setup()
    cloud.mode = "quota429"
    r = client.post("/v1/chat/completions", json=CHAT, headers=OAI)
    assert r.headers[SERVED_BY] == f"local:{LOCAL}" and "insufficient_quota" in r.headers[FALLBACK]
    assert events(b, Event.UPSTREAM_QUOTA)
    n = len(cloud.posts())
    r = client.post("/v1/chat/completions", json=CHAT, headers=OAI)   # same key: straight to local
    assert len(cloud.posts()) == n and "quota exhausted" in r.headers[FALLBACK]
    cloud.mode = "ok"
    other = OAI | {"Authorization": "Bearer sk-proj-another-key"}
    r = client.post("/v1/chat/completions", json=CHAT, headers=other)   # another key still goes to the cloud
    assert r.headers[SERVED_BY] == "openai"
    assert router.view()["providers"]["openai"]["keys_out_of_quota"] == 1


# ---- returned as-is -------------------------------------------------------------------

@pytest.mark.parametrize(("mode", "status"), [("400", 400), ("401", 401), ("rate429", 429)])
def test_client_errors_come_back_as_the_provider_sent_them(setup, cloud, mode, status):
    client, b, router, _ = setup()
    cloud.mode = mode
    r = client.post("/v1/messages", json=MSG, headers=ANT)
    assert r.status_code == status and r.headers[SERVED_BY] == "anthropic"
    assert r.json() == ERRORS[mode][1]
    assert not events(b, Event.UPSTREAM_FAILOVER) and router.view()["providers"]["anthropic"]["state"] == "closed"
    if mode == "rate429":
        assert r.headers["retry-after"] == "3"   # so the SDK backs off itself


# ---- the breaker ----------------------------------------------------------------------

def test_the_breaker_opens_skips_the_provider_and_closes_after_a_probe(setup, cloud):
    client, b, router, clock = setup(**with_operator_key(cloud))
    cloud.mode = "500"
    for _ in range(BREAKER.failures):
        client.post("/v1/messages", json=MSG, headers=ANT)
    assert router.view()["providers"]["anthropic"]["state"] == "open" and events(b, Event.UPSTREAM_OPEN)
    n = len(cloud.seen)
    start = time.monotonic()
    r = client.post("/v1/messages", json=MSG, headers=ANT)
    assert len(cloud.seen) == n and "circuit open" in r.headers[FALLBACK]   # no wait, no call
    assert time.monotonic() - start < 1
    router.probe_once()
    assert len(cloud.seen) == n   # not due yet
    cloud.mode = "ok"
    clock.t += BREAKER.probe_s
    router.probe_once()
    probe = cloud.seen[-1]
    assert probe["method"] == "GET" and probe["path"] == "/v1/models" and probe["headers"]["x-api-key"] == "sk-ant-operator"
    assert router.view()["providers"]["anthropic"]["state"] == "closed" and events(b, Event.UPSTREAM_CLOSED)
    r = client.post("/v1/messages", json=MSG, headers=ANT)
    assert r.headers[SERVED_BY] == "anthropic"


def test_a_failed_probe_backs_off(setup, cloud):
    client, _, router, clock = setup(**with_operator_key(cloud))
    cloud.mode = "drop"
    for _ in range(BREAKER.failures):
        client.post("/v1/messages", json=MSG, headers=ANT)
    clock.t += BREAKER.probe_s
    router.probe_once()
    assert router.view()["providers"]["anthropic"]["retry_in_s"] == 2 * BREAKER.probe_s


# ---- streams ----------------------------------------------------------------------------

def test_a_stream_that_fails_before_its_first_byte_goes_local(setup, cloud):
    client, *_ = setup()
    cloud.mode = "529"
    r = client.post("/v1/messages", json=MSG | {"stream": True}, headers=ANT)
    assert r.headers[SERVED_BY] == f"local:{LOCAL}" and "event: message_start" in r.text and "message_stop" in r.text


def test_a_stream_that_opens_with_an_error_event_goes_local(setup, cloud):
    client, *_ = setup()
    cloud.mode = "stream_error"
    r = client.post("/v1/messages", json=MSG | {"stream": True}, headers=ANT)
    assert r.headers[SERVED_BY] == f"local:{LOCAL}" and "stream error overloaded_error" in r.headers[FALLBACK]


@pytest.mark.parametrize(("path", "body", "headers", "marker"), [
    ("/v1/messages", MSG, ANT, "event: error"),
    ("/v1/chat/completions", CHAT, OAI, '"type": "upstream_error"')])
def test_a_stream_that_breaks_after_its_first_byte_ends_with_an_error_not_another_model(setup, cloud, path, body, headers, marker):
    client, b, router, _ = setup()
    cloud.mode = "stream_die"
    r = client.post(path, json=body | {"stream": True}, headers=headers)
    assert r.headers[SERVED_BY] in ("anthropic", "openai")
    assert "chatcmpl-cloud" in r.text or "message_start" in r.text   # what had already been sent
    assert marker in r.text and "no other model was substituted" in r.text
    assert "[DONE]" not in r.text and not events(b, Event.UPSTREAM_FAILOVER)
    provider = r.headers[SERVED_BY]
    assert router.view()["providers"][provider]["failures"] == 1


# ---- chains without a local model, no credential, Responses ---------------------------

def test_no_local_model_and_the_provider_down_is_a_503(setup, cloud):
    client, *_ = setup()
    cloud.mode = "500"
    r = client.post("/v1/chat/completions", json={"model": "cloud-only-1", "messages": HI}, headers=OAI)
    assert r.status_code == 503 and "no local fallback" in r.json()["error"]["message"]


def test_no_local_model_and_no_quota_relays_the_providers_answer(setup, cloud):
    client, *_ = setup()
    cloud.mode = "quota429"
    r = client.post("/v1/chat/completions", json={"model": "cloud-only-1", "messages": HI}, headers=OAI)
    assert r.status_code == 429 and r.json()["error"]["code"] == "insufficient_quota" and r.headers[SERVED_BY] == "openai"


def test_without_a_provider_key_the_request_goes_local(setup, cloud):
    client, *_ = setup()
    r = client.post("/v1/messages", json=MSG, headers={"x-api-key": TOKEN})   # only the broker credential
    assert r.headers[SERVED_BY] == f"local:{LOCAL}" and "no credential" in r.headers[FALLBACK] and cloud.seen == []


def test_a_configured_key_is_used_when_the_client_sends_none(setup, cloud):
    client, *_ = setup(env={"UP_ANT": "sk-ant-operator"},
                       providers={"anthropic": {"url": cloud.url, "api": "anthropic", "key_env": "UP_ANT"},
                                  "openai": {"url": cloud.url, "api": "openai"}})
    r = client.post("/v1/messages", json=MSG, headers={"Authorization": f"Bearer {TOKEN}"})
    assert r.headers[SERVED_BY] == "anthropic" and cloud.posts()[-1]["headers"]["x-api-key"] == "sk-ant-operator"


def test_a_stream_opening_with_an_openai_error_goes_local(setup, cloud):
    client, *_ = setup()
    cloud.mode = "stream_error_openai"
    r = client.post("/v1/chat/completions", json=CHAT | {"stream": True}, headers=OAI)
    assert r.headers[SERVED_BY] == f"local:{LOCAL}" and '"model": "llama-8b"' in r.text


def test_a_provider_of_the_other_api_is_skipped(setup, cloud):
    client, *_ = setup(routes={"mixed-*": ["anthropic", "openai", LOCAL]})
    r = client.post("/v1/chat/completions", json={"model": "mixed-1", "messages": HI}, headers=OAI)
    assert r.headers[SERVED_BY] == "openai" and "anthropic speaks the anthropic API" in r.headers[FALLBACK]
    assert all(s["path"] == "/v1/chat/completions" for s in cloud.posts())


def test_any_probe_answer_but_a_server_error_closes_the_breaker(setup, cloud):
    client, _, router, clock = setup(**with_operator_key(cloud))
    cloud.mode = "500"
    for _ in range(BREAKER.failures):
        client.post("/v1/messages", json=MSG, headers=ANT)
    cloud.probe_mode = "401"   # the key is wrong, but Anthropic is back
    clock.t += BREAKER.probe_s
    router.probe_once()
    assert router.view()["providers"]["anthropic"]["state"] == "closed"


def test_the_probe_prefers_the_configured_key(setup, cloud):
    client, _, router, clock = setup(env={"UP_ANT": "sk-ant-operator"},
                                     providers={"anthropic": {"url": cloud.url, "api": "anthropic", "key_env": "UP_ANT"},
                                                "openai": {"url": cloud.url, "api": "openai"}})
    cloud.mode = "500"
    for _ in range(BREAKER.failures):
        client.post("/v1/messages", json=MSG, headers=ANT)
    clock.t += BREAKER.probe_s
    router.probe_once()
    assert cloud.seen[-1]["method"] == "GET" and cloud.seen[-1]["headers"]["x-api-key"] == "sk-ant-operator"


def test_a_client_key_is_a_broker_credential_never_a_provider_key(setup, cloud):
    client, *_ = setup()
    key = client.post("/v1/keys", json={"name": "laptop"}, headers={"Authorization": f"Bearer {TOKEN}"}).json()["key"]
    r = client.post("/v1/messages", json=MSG, headers={"x-api-key": key})   # connect's setup: the key is the broker's
    assert r.headers[SERVED_BY] == f"local:{LOCAL}" and "no credential" in r.headers[FALLBACK] and cloud.seen == []
    r = client.post("/v1/messages", json=MSG, headers={"x-gpu-broker-key": key, "x-api-key": PLANTED})
    assert r.headers[SERVED_BY] == "anthropic" and cloud.posts()[-1]["headers"]["x-api-key"] == PLANTED
    assert key not in json.dumps(cloud.seen)


def test_responses_api_goes_to_the_cloud_then_falls_back(setup, cloud):
    client, *_ = setup()
    body = {"model": "gpt-5", "input": "hi"}
    r = client.post("/v1/responses", json=body, headers=OAI)
    assert r.headers[SERVED_BY] == "openai" and cloud.posts()[-1]["path"] == "/v1/responses"
    cloud.mode = "529"
    r = client.post("/v1/responses", json=body, headers=OAI)
    assert r.status_code == 200 and r.json()["object"] == "response" and r.json()["model"] == LOCAL
    assert r.headers[SERVED_BY] == f"local:{LOCAL}"


def test_the_dashboard_sees_upstream_health(setup, cloud):
    client, *_ = setup()
    cloud.mode = "500"
    for _ in range(BREAKER.failures):
        client.post("/v1/messages", json=MSG, headers=ANT)
    view = client.get("/v1/upstreams", headers={"Authorization": f"Bearer {TOKEN}"}).json()
    assert view["providers"]["anthropic"]["state"] == "open" and view["routes"]["claude-*"] == ["anthropic", LOCAL]
    assert client.get("/v1/upstreams", headers={"x-api-key": "nope"}).status_code == 401
