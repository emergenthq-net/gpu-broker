"""Unstreamed requests go upstream as streams and come back as the provider's own unstreamed answer.

Checked with the official SDKs against the fake provider, whose streams are built from its whole
answers the way each provider builds them (tests/cloud_shapes.py): what the SDK parses through
the broker equals what it parses from the provider's unstreamed answer, field for field (usage,
stop reason, tool calls, thinking, ids, logprobs), and the raw JSON equals that answer. The
request changes only by `stream` (and `stream_options.include_usage` for chat). A provider that
refuses streams is asked again plainly, not counted against it; a stream with no whole answer
fails over before the caller sees anything; a long, slow answer is not cut off.
"""
from __future__ import annotations

import anthropic
import openai
import pytest

from gpu_broker.constants import Event
from gpu_broker.web.failover import SERVED_BY
from tests import cloud_shapes
from tests.failover_fakes import ANT, LOCAL, OAI, OPENAI_PLANTED, PLANTED, FakeCloud, build
from tests.helpers import TOKEN, wait_idle

HI = [{"role": "user", "content": "weather in Rome?"}]
MSG = {"model": "claude-sonnet-4-5", "max_tokens": 50, "messages": HI}
CHAT = {"model": "gpt-4o", "messages": HI}
RESP = {"model": "gpt-4o", "input": "weather in Rome?"}
BROKER = {"x-gpu-broker-key": TOKEN}
VARIANTS = ("text", "rich")


@pytest.fixture
def env(tmp_path):
    cloud = FakeCloud()
    client, b, router, _ = build(tmp_path, cloud)
    yield cloud, client, b, router
    assert wait_idle(b)
    b.stop()
    cloud.close()


def via_broker_anthropic(client):
    return anthropic.Anthropic(base_url="http://testserver", api_key=PLANTED, default_headers=BROKER,
                               http_client=client, max_retries=0)


def via_broker_openai(client):
    return openai.OpenAI(base_url="http://testserver/v1", api_key=OPENAI_PLANTED, default_headers=BROKER,
                         http_client=client, max_retries=0)


def direct_anthropic(cloud):
    return anthropic.Anthropic(base_url=cloud.url, api_key=PLANTED, max_retries=0)


def direct_openai(cloud):
    return openai.OpenAI(base_url=cloud.url + "/v1", api_key=OPENAI_PLANTED, max_retries=0)


def failures(router, name):
    return router.view()["providers"][name]["failures"]


@pytest.mark.parametrize("variant", VARIANTS)
def test_anthropic_unstreamed_is_streamed_and_reassembled_exactly(env, variant):
    cloud, client, _, router = env
    cloud.variant = variant
    raw = via_broker_anthropic(client).messages.with_raw_response.create(**MSG)
    sent = cloud.posts()[-1]["body"]
    direct = direct_anthropic(cloud).messages.create(**MSG)
    assert raw.headers[SERVED_BY] == "anthropic" and raw.headers["content-type"].startswith("application/json")
    assert raw.parse().model_dump() == direct.model_dump()
    assert client.post("/v1/messages", json=MSG, headers=ANT).json() == cloud_shapes.whole(cloud_shapes.ANTHROPIC, variant)
    assert sent == MSG | {"stream": True}   # nothing else about the request changes
    assert failures(router, "anthropic") == 0


@pytest.mark.parametrize("variant", VARIANTS)
def test_chat_unstreamed_is_streamed_with_usage_and_reassembled_exactly(env, variant):
    cloud, client, *_ = env
    cloud.variant = variant
    raw = via_broker_openai(client).chat.completions.with_raw_response.create(**CHAT)
    sent = cloud.posts()[-1]["body"]
    direct = direct_openai(cloud).chat.completions.create(**CHAT)
    assert raw.headers[SERVED_BY] == "openai"
    assert raw.parse().model_dump() == direct.model_dump()
    assert client.post("/v1/chat/completions", json=CHAT, headers=OAI).json() == cloud_shapes.whole(cloud_shapes.CHAT, variant)
    assert sent == CHAT | {"stream": True, "stream_options": {"include_usage": True}}


@pytest.mark.parametrize("variant", VARIANTS)
def test_responses_unstreamed_is_streamed_and_reassembled_exactly(env, variant):
    cloud, client, *_ = env
    cloud.variant = variant
    got = via_broker_openai(client).responses.create(**RESP)
    sent = cloud.posts()[-1]["body"]
    assert got.model_dump() == direct_openai(cloud).responses.create(**RESP).model_dump()
    assert client.post("/v1/responses", json=RESP, headers=OAI).json() == cloud_shapes.whole(cloud_shapes.RESPONSES, variant)
    assert sent == RESP | {"stream": True}


def test_a_streamed_request_is_relayed_not_reassembled(env):
    cloud, client, *_ = env
    r = client.post("/v1/messages", json=MSG | {"stream": True}, headers=ANT)
    assert r.headers["content-type"].startswith("text/event-stream") and "message_stop" in r.text
    assert cloud.posts()[-1]["body"] == MSG | {"stream": True}


@pytest.mark.parametrize(("path", "body", "headers"), [("/v1/messages", MSG, ANT), ("/v1/chat/completions", CHAT, OAI)])
def test_a_provider_that_refuses_streams_is_asked_again_plainly_and_not_counted(env, path, body, headers):
    cloud, client, _, router = env
    cloud.mode = "no_stream"
    r = client.post(path, json=body, headers=headers)
    first, second = cloud.posts()[-2:]
    assert first["body"]["stream"] is True and second["body"] == body   # the retry is the request as sent
    assert r.status_code == 200 and r.headers[SERVED_BY] in ("anthropic", "openai")
    assert r.json() == cloud_shapes.whole(path)
    assert failures(router, r.headers[SERVED_BY]) == 0


def test_a_streamed_request_refused_is_not_retried(env):
    cloud, client, *_ = env
    cloud.mode = "no_stream"
    r = client.post("/v1/messages", json=MSG | {"stream": True}, headers=ANT)
    assert r.status_code == 400 and len(cloud.posts()) == 1


def test_a_bad_request_is_returned_after_the_one_plain_retry(env):
    cloud, client, _, router = env
    cloud.mode = "400"
    r = client.post("/v1/messages", json=MSG, headers=ANT)
    assert r.status_code == 400 and "max_tokens" in r.text and len(cloud.posts()) == 2
    assert failures(router, "anthropic") == 0


def test_a_provider_that_ignores_stream_is_relayed_whole(env):
    cloud, client, *_ = env
    cloud.mode = "ignore_stream"
    r = client.post("/v1/chat/completions", json=CHAT, headers=OAI)
    assert r.headers[SERVED_BY] == "openai" and r.json() == cloud_shapes.whole(cloud_shapes.CHAT)


@pytest.mark.parametrize("mode", ["stream_cut", "stream_late_error", "stream_die"])
@pytest.mark.parametrize(("path", "body", "headers", "provider"), [
    ("/v1/messages", MSG, ANT, "anthropic"), ("/v1/chat/completions", CHAT, OAI, "openai"),
    ("/v1/responses", RESP, OAI, "openai")])
def test_a_stream_with_no_whole_answer_fails_over_unseen(env, mode, path, body, headers, provider):
    cloud, client, b, router = env
    cloud.mode = mode
    r = client.post(path, json=body, headers=headers)
    assert r.status_code == 200 and r.headers[SERVED_BY] == f"local:{LOCAL}"
    assert failures(router, provider) == 1
    reasons = [e["data"].get("reason", "") for e in b.store.events(0, 1000) if e["kind"] == Event.UPSTREAM_FAILOVER]
    assert reasons and "boom" not in reasons[-1] and "Overloaded" not in reasons[-1]   # fixed text, never the provider's


def test_a_long_slow_answer_is_not_cut_off(env):
    """Each event comes inside idle_s, but the whole answer takes longer than response_s: an
    unstreamed call would have timed out; streamed, it is the provider's whole answer."""
    cloud, client, _, router = env
    cloud.mode, cloud.pause_s = "slow_stream", 0.1   # FAST: idle_s 0.4, response_s 0.4
    events = len(cloud_shapes.stream(cloud_shapes.ANTHROPIC))
    assert events * cloud.pause_s > router.cfg.timeouts.response_s
    r = client.post("/v1/messages", json=MSG, headers=ANT)
    assert r.headers[SERVED_BY] == "anthropic" and r.json() == cloud_shapes.whole(cloud_shapes.ANTHROPIC)
