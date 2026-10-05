"""The official SDKs through failover, set up the way docs/failover.md says: the client's own
provider key as usual, the broker credential in `x-gpu-broker-key`. The same calls work whether
the cloud answers or the local model does, streamed or not, and a client error stays an error."""
from __future__ import annotations

import anthropic
import openai
import pytest

from tests.failover_fakes import LOCAL, OPENAI_PLANTED, PLANTED, FakeCloud, build
from tests.helpers import TOKEN, wait_idle

BROKER = {"x-gpu-broker-key": TOKEN}
HI = [{"role": "user", "content": "hi"}]


@pytest.fixture
def env(tmp_path):
    cloud = FakeCloud()
    client, b, *_ = build(tmp_path, cloud)
    yield cloud, client
    assert wait_idle(b)
    b.stop()
    cloud.close()


def claude(client):
    return anthropic.Anthropic(base_url="http://testserver", api_key=PLANTED, default_headers=BROKER,
                               http_client=client, max_retries=0)


def gpt(client):
    return openai.OpenAI(base_url="http://testserver/v1", api_key=OPENAI_PLANTED, default_headers=BROKER,
                         http_client=client, max_retries=0)


@pytest.mark.parametrize(("mode", "text"), [("ok", "from the cloud"), ("529", None), ("drop", None)])
def test_anthropic_sdk(env, mode, text):
    cloud, client = env
    cloud.mode = mode
    raw = claude(client).messages.with_raw_response.create(model="claude-sonnet-4-5", max_tokens=50, messages=HI)
    m = raw.parse()
    assert m.content[0].type == "text" and (text is None or m.content[0].text == text)
    assert raw.headers["x-gpu-broker-served-by"] == ("anthropic" if mode == "ok" else f"local:{LOCAL}")


def test_anthropic_sdk_stream_falls_back(env):
    cloud, client = env
    cloud.mode = "529"
    with claude(client).messages.stream(model="claude-sonnet-4-5", max_tokens=50, messages=HI) as s:
        final = s.get_final_message()
    assert final.model == LOCAL and final.content


def test_anthropic_sdk_sees_a_bad_request_as_one(env):
    cloud, client = env
    cloud.mode = "400"
    with pytest.raises(anthropic.BadRequestError, match="max_tokens"):
        claude(client).messages.create(model="claude-sonnet-4-5", max_tokens=50, messages=HI)


@pytest.mark.parametrize("mode", ["ok", "quota429", "hang"])
def test_openai_sdk_stream(env, mode):
    cloud, client = env
    cloud.mode = mode
    chunks = list(gpt(client).chat.completions.create(model="gpt-4o", messages=HI, stream=True))
    text = "".join(c.choices[0].delta.content or "" for c in chunks if c.choices)
    assert text.strip()
    assert ("from the cloud" in text) == (mode == "ok")


def test_openai_sdk_mid_stream_break_is_an_error(env):
    cloud, client = env
    cloud.mode = "stream_die"
    with pytest.raises(openai.APIError, match="no other model was substituted"):
        list(gpt(client).chat.completions.create(model="gpt-4o", messages=HI, stream=True))


def test_responses_sdk_falls_back(env):
    cloud, client = env
    cloud.mode = "500"
    r = gpt(client).responses.create(model="gpt-5", input="hi")
    assert r.model == LOCAL and r.output_text
