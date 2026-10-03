"""The official Anthropic Python SDK against the broker, changing only base_url and api_key:
plain message, streaming event sequence, tool_use and tool_result, thinking, errors, models."""
from __future__ import annotations

import anthropic
import pytest

from tests.dropin_fakes import ANSWER, CALL_ID, REASONING, TOOL, USAGE, app_client, dropin_broker  # noqa: F401
from tests.helpers import TOKEN

TOOLS = [{"name": TOOL, "description": "weather", "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}}}]
HI = [{"role": "user", "content": "hi"}]
MAX = 64
EVENT_ORDER = ["message_start", "content_block_start", "content_block_delta", "content_block_stop", "message_delta", "message_stop"]


@pytest.fixture
def ant(app_client):
    return anthropic.Anthropic(base_url="http://testserver", api_key=TOKEN, http_client=app_client, max_retries=0)


def test_plain_message(ant, dropin_broker):
    m = ant.messages.create(model="claude-sonnet-4-5", max_tokens=MAX, system="be brief", messages=HI)
    assert [b.type for b in m.content] == ["text"] and m.content[0].text == ANSWER   # reasoning dropped: thinking off
    assert m.model == "claude-sonnet-4-5" and m.stop_reason == "end_turn"
    assert (m.usage.input_tokens, m.usage.output_tokens) == (USAGE["prompt_tokens"], USAGE["completion_tokens"])
    sent = dropin_broker.backends.sent[-1]
    assert sent["messages"][0] == {"role": "system", "content": "be brief"} and sent["max_tokens"] == MAX
    assert m.model_extra["x_broker"]["used"] == "llama-8b" and "claude-*" in m.model_extra["x_broker"]["substitution"]


def test_streaming_follows_anthropic_event_order(ant):
    with ant.messages.stream(model="claude-haiku-4-5", max_tokens=MAX, messages=HI) as s:
        kinds = [e.type for e in s if e.type in EVENT_ORDER]
        final = s.get_final_message()
    collapsed = [k for i, k in enumerate(kinds) if i == 0 or k != kinds[i - 1]]
    assert collapsed == EVENT_ORDER
    assert final.content[0].text.strip() == ANSWER and final.stop_reason == "end_turn"
    assert final.usage.output_tokens == USAGE["completion_tokens"] and final.model == "claude-haiku-4-5"


def test_tool_use_then_tool_result(ant, dropin_broker):
    m = ant.messages.create(model="claude-opus-4-1", max_tokens=MAX, messages=HI, tools=TOOLS,
                            tool_choice={"type": "tool", "name": TOOL})
    assert m.stop_reason == "tool_use"
    use = m.content[0]
    assert (use.type, use.id, use.name, use.input) == ("tool_use", CALL_ID, TOOL, {"city": "Paris"})
    sent = dropin_broker.backends.sent[-1]
    assert sent["tools"][0]["function"]["name"] == TOOL and sent["tools"][0]["function"]["parameters"] == TOOLS[0]["input_schema"]
    assert sent["tool_choice"] == {"type": "function", "function": {"name": TOOL}}
    follow = [*HI, {"role": "assistant", "content": [b.model_dump() for b in m.content]},
              {"role": "user", "content": [{"type": "tool_result", "tool_use_id": use.id, "content": "sunny"}]}]
    m2 = ant.messages.create(model="claude-opus-4-1", max_tokens=MAX, messages=follow, tools=TOOLS)
    assert m2.content[0].text == "It is sunny" and m2.stop_reason == "end_turn"
    msgs = dropin_broker.backends.sent[-1]["messages"]
    assert msgs[1]["tool_calls"][0]["id"] == CALL_ID and msgs[2] == {"role": "tool", "tool_call_id": CALL_ID, "content": "sunny"}


def test_streamed_tool_use_is_reassembled(ant):
    with ant.messages.stream(model="claude-sonnet-4-5", max_tokens=MAX, messages=HI, tools=TOOLS) as s:
        final = s.get_final_message()
    assert final.stop_reason == "tool_use" and final.content[0].input == {"city": "Paris"}


def test_thinking_only_when_enabled(ant):
    on = {"type": "enabled", "budget_tokens": 1024}
    m = ant.messages.create(model="claude-sonnet-4-5", max_tokens=2048, messages=HI, thinking=on)
    assert [b.type for b in m.content] == ["thinking", "text"] and m.content[0].thinking == REASONING
    with ant.messages.stream(model="claude-sonnet-4-5", max_tokens=2048, messages=HI, thinking=on) as s:
        final = s.get_final_message()
    assert final.content[0].type == "thinking" and final.content[0].thinking == REASONING


def test_queued_stream_is_translated_too(ant):
    with ant.messages.stream(model="claude-sonnet-4-5", max_tokens=MAX, messages=HI,
                             extra_headers={"x-priority": "background"}) as s:
        final = s.get_final_message()
    assert final.content[0].text == ANSWER and final.model_extra["x_broker"]["job"]


def test_image_block_becomes_a_data_uri(ant, dropin_broker):
    img = {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}}
    ant.messages.create(model="claude-sonnet-4-5", max_tokens=MAX, messages=[{"role": "user", "content": [img, {"type": "text", "text": "what?"}]}],
                        stop_sequences=["END"], extra_body={"temperature": 0.3})   # newer SDKs dropped the kwarg
    sent = dropin_broker.backends.sent[-1]
    assert sent["messages"][0]["content"][0] == {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
    assert sent["stop"] == ["END"] and sent["temperature"] == 0.3


def test_hosted_tools_are_refused_in_anthropic_shape(ant):
    with pytest.raises(anthropic.BadRequestError) as e:
        ant.messages.create(model="claude-sonnet-4-5", max_tokens=MAX, messages=HI,
                            tools=[{"type": "web_search_20250305", "name": "web_search"}])
    assert e.value.body == {"type": "error", "error": {"type": "invalid_request_error",
                                                       "message": "hosted tool 'web_search_20250305' is not available on a local model"}}


def test_wrong_key_is_an_anthropic_auth_error(app_client):
    bad = anthropic.Anthropic(base_url="http://testserver", api_key="nope", http_client=app_client, max_retries=0)
    with pytest.raises(anthropic.AuthenticationError) as e:
        bad.messages.create(model="claude-sonnet-4-5", max_tokens=MAX, messages=HI)
    assert e.value.body["error"]["type"] == "authentication_error"


def test_models_list_in_anthropic_shape(ant):
    page = ant.models.list()
    assert "llama-8b" in [m.id for m in page.data] and page.data[0].type == "model"
