"""The official OpenAI Python SDK against the broker, changing only base_url and api_key:
plain chat, streaming, a tool call and its follow-up, name mapping, pass-through, embeddings."""
from __future__ import annotations

import json

import openai
import pytest

from tests.dropin_fakes import ANSWER, CALL_ID, EMBED_MODEL, TOOL, TOOL_ARGS, VECTOR, app_client, dropin_broker  # noqa: F401
from tests.helpers import TOKEN

TOOLS = [{"type": "function", "function": {"name": TOOL, "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}]
HI = [{"role": "user", "content": "hi"}]


@pytest.fixture
def oai(app_client):
    return openai.OpenAI(base_url="http://testserver/v1", api_key=TOKEN, http_client=app_client, max_retries=0)


def test_plain_chat_echoes_the_hosted_name_and_reports_what_ran(oai, dropin_broker):
    r = oai.chat.completions.create(model="gpt-4o", messages=HI)
    assert r.choices[0].message.content == ANSWER
    assert r.model == "gpt-4o"                                     # clients that check `model` keep working
    x = r.model_extra["x_broker"]
    assert x["used"] == "llama-8b" and "model_map pattern 'gpt-*'" in x["substitution"]
    assert dropin_broker.store.job(x["job"])["substitution"] == x["substitution"]   # recorded on the job


def test_streaming_relays_tokens_with_the_requested_model(oai):
    chunks = list(oai.chat.completions.create(model="o1-preview", messages=HI, stream=True))
    text = "".join(c.choices[0].delta.content or "" for c in chunks if c.choices)
    assert text.strip() == ANSWER
    assert {c.model for c in chunks} == {"o1-preview"}


def test_tool_call_and_tool_result_round_trip(oai, dropin_broker):
    first = oai.chat.completions.create(model="gpt-4.1", messages=HI, tools=TOOLS, tool_choice="auto")
    choice = first.choices[0]
    assert choice.finish_reason == "tool_calls"
    call = choice.message.tool_calls[0]
    assert (call.id, call.function.name, call.function.arguments) == (CALL_ID, TOOL, TOOL_ARGS)
    sent = dropin_broker.backends.sent[-1]
    assert sent["tools"] == TOOLS and sent["tool_choice"] == "auto"   # reached the server untouched
    follow = [*HI, choice.message.model_dump(exclude_none=True), {"role": "tool", "tool_call_id": call.id, "content": "sunny"}]
    second = oai.chat.completions.create(model="gpt-4.1", messages=follow, tools=TOOLS)
    assert second.choices[0].message.content == "It is sunny"
    assert dropin_broker.backends.sent[-1]["messages"][-1] == {"role": "tool", "tool_call_id": CALL_ID, "content": "sunny"}


def test_streamed_tool_call_arrives_whole(oai):
    chunks = list(oai.chat.completions.create(model="gpt-4o", messages=HI, tools=TOOLS, stream=True))
    calls = [tc for c in chunks if c.choices for tc in c.choices[0].delta.tool_calls or []]
    assert calls[0].id == CALL_ID and calls[0].function.name == TOOL
    assert "".join(tc.function.arguments or "" for tc in calls) == TOOL_ARGS
    assert [c.choices[0].finish_reason for c in chunks if c.choices and c.choices[0].finish_reason] == ["tool_calls"]


def test_queued_stream_keeps_tool_calls(oai):
    """A background caller is answered from the queue as one SSE chunk: tool calls included."""
    chunks = list(oai.chat.completions.create(model="gpt-4o", messages=HI, tools=TOOLS, stream=True,
                                              extra_headers={"x-priority": "background"}))
    delta = chunks[0].choices[0].delta
    assert delta.tool_calls[0].function.arguments == TOOL_ARGS and chunks[0].choices[0].finish_reason == "tool_calls"


def test_openai_fields_pass_through(oai, dropin_broker):
    image = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
    fmt = {"type": "json_schema", "json_schema": {"name": "x", "schema": {"type": "object"}}}
    oai.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": [{"type": "text", "text": "hi"}, image]}],
                                response_format=fmt, stop=["END"], seed=7, n=1, max_completion_tokens=33)
    sent = dropin_broker.backends.sent[-1]
    assert sent["response_format"] == fmt and sent["stop"] == ["END"] and sent["seed"] == 7 and sent["n"] == 1
    assert sent["max_tokens"] == 33 and "max_completion_tokens" not in sent
    assert sent["messages"][0]["content"][1] == image


def test_override_pattern_wins_and_reaches_a_variant(oai, dropin_broker):
    r = oai.chat.completions.create(model="gpt-4o-mini", messages=HI)
    assert r.model == "gpt-4o-mini" and "pattern 'gpt-4o-mini'" in r.model_extra["x_broker"]["substitution"]
    assert dropin_broker.backends.sent[-1]["temperature"] == 0.1    # llama-8b-precise's override


def test_catalog_names_are_never_mapped(oai):
    r = oai.chat.completions.create(model="llama", messages=HI)
    assert r.model_extra["x_broker"]["substitution"] is None


def test_embeddings_use_the_embedding_model(oai, dropin_broker):
    r = oai.embeddings.create(model="text-embedding-3-small", input="hello")
    assert r.data[0].embedding == VECTOR and r.model == "text-embedding-3-small"
    assert r.model_extra["x_broker"]["used"] == EMBED_MODEL
    assert dropin_broker.backends.embeds[-1]["input"] == "hello"


def test_wrong_token_is_an_openai_error(app_client):
    bad = openai.OpenAI(base_url="http://testserver/v1", api_key="nope", http_client=app_client, max_retries=0)
    with pytest.raises(openai.AuthenticationError) as e:
        bad.models.list()
    assert e.value.body["type"] == "authentication_error"


def test_models_list(oai):
    assert "llama-8b" in [m.id for m in oai.models.list()]


def test_tool_arguments_are_valid_json():
    assert json.loads(TOOL_ARGS) == {"city": "Paris"}
