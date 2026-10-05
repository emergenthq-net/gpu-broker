"""The official OpenAI Python SDK's Responses API against the broker, changing only base_url
and api_key: plain answers, streaming, a function call and its output, previous_response_id,
and name mapping."""
from __future__ import annotations

import openai
import pytest

from tests.dropin_fakes import ANSWER, CALL_ID, REASONING, TOOL, TOOL_ARGS, USAGE, app_client, dropin_broker  # noqa: F401
from tests.helpers import TOKEN

TOOLS = [{"type": "function", "name": TOOL, "description": "weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}]


@pytest.fixture
def oai(app_client):
    return openai.OpenAI(base_url="http://testserver/v1", api_key=TOKEN, http_client=app_client, max_retries=0)


def test_plain_answer_maps_the_name_and_reports_what_ran(oai, dropin_broker):
    r = oai.responses.create(model="gpt-5", instructions="be brief", input="hi")
    assert r.output_text == ANSWER and r.status == "completed" and r.model == "gpt-5"
    assert [i.type for i in r.output] == ["reasoning", "message"]
    assert r.output[0].summary[0].text == REASONING
    assert (r.usage.input_tokens, r.usage.output_tokens, r.usage.total_tokens) == (7, 3, 10)
    x = r.model_extra["x_broker"]
    assert x["used"] == "llama-8b" and "model_map pattern 'gpt-*'" in x["substitution"]
    sent = dropin_broker.backends.sent[-1]
    assert sent["messages"] == [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"}]


def test_streaming_events_and_text(oai):
    events = list(oai.responses.create(model="gpt-5", input="hi", stream=True))
    kinds = [e.type for e in events]
    assert kinds[:2] == ["response.created", "response.in_progress"] and kinds[-1] == "response.completed"
    assert "".join(e.delta for e in events if e.type == "response.output_text.delta").strip() == ANSWER
    assert "".join(e.delta for e in events if e.type == "response.reasoning_summary_text.delta") == REASONING
    assert [e.sequence_number for e in events] == list(range(len(events)))
    final = events[-1].response
    assert final.output_text.strip() == ANSWER and final.usage.total_tokens == USAGE["total_tokens"]


def test_function_call_and_output_round_trip(oai, dropin_broker):
    first = oai.responses.create(model="gpt-5", input="weather?", tools=TOOLS, tool_choice="auto")
    call = next(i for i in first.output if i.type == "function_call")
    assert (call.call_id, call.name, call.arguments) == (CALL_ID, TOOL, TOOL_ARGS)
    sent = dropin_broker.backends.sent[-1]
    assert sent["tools"][0]["function"]["name"] == TOOL and sent["tool_choice"] == "auto"
    follow = [{"role": "user", "content": "weather?"}, call.model_dump(exclude_none=True),
              {"type": "function_call_output", "call_id": call.call_id, "output": "sunny"}]
    second = oai.responses.create(model="gpt-5", input=follow, tools=TOOLS)
    assert second.output_text == "It is sunny"
    msgs = dropin_broker.backends.sent[-1]["messages"]
    assert msgs[-2]["tool_calls"][0]["id"] == CALL_ID and msgs[-1] == {"role": "tool", "tool_call_id": CALL_ID, "content": "sunny"}


def test_streamed_function_call_arrives_whole(oai):
    events = list(oai.responses.create(model="gpt-5", input="weather?", tools=TOOLS, stream=True))
    assert "".join(e.delta for e in events if e.type == "response.function_call_arguments.delta") == TOOL_ARGS
    done = next(e for e in events if e.type == "response.function_call_arguments.done")
    assert (done.arguments, done.name) == (TOOL_ARGS, TOOL)
    call = next(i for i in events[-1].response.output if i.type == "function_call")
    assert (call.call_id, call.status) == (CALL_ID, "completed")


def test_previous_response_id_continues_the_conversation(oai, dropin_broker):
    first = oai.responses.create(model="gpt-5", instructions="first rules", input="hi")
    oai.responses.create(model="gpt-5", previous_response_id=first.id, input="again")
    msgs = dropin_broker.backends.sent[-1]["messages"]
    assert msgs == [{"role": "user", "content": "hi"}, {"role": "assistant", "content": ANSWER}, {"role": "user", "content": "again"}]


def test_previous_response_id_after_a_stream_and_a_tool_call(oai, dropin_broker):
    events = list(oai.responses.create(model="gpt-5", input="weather?", tools=TOOLS, stream=True))
    rid = events[-1].response.id
    oai.responses.create(model="gpt-5", previous_response_id=rid, tools=TOOLS,
                         input=[{"type": "function_call_output", "call_id": CALL_ID, "output": "rain"}])
    msgs = dropin_broker.backends.sent[-1]["messages"]
    assert msgs[1]["tool_calls"][0]["function"] == {"name": TOOL, "arguments": TOOL_ARGS}
    assert msgs[-1] == {"role": "tool", "tool_call_id": CALL_ID, "content": "rain"}


def test_unknown_unstored_or_foreign_previous_response_is_refused(oai, app_client):
    kept = oai.responses.create(model="gpt-5", input="hi")
    unstored = oai.responses.create(model="gpt-5", input="hi", store=False)
    for rid in ("resp_missing", unstored.id):
        with pytest.raises(openai.BadRequestError, match="not found"):
            oai.responses.create(model="gpt-5", previous_response_id=rid, input="x")
    other = openai.OpenAI(base_url="http://testserver/v1", api_key=TOKEN, http_client=app_client, max_retries=0,
                          default_headers={"x-requester": "someone-else"})
    with pytest.raises(openai.BadRequestError) as e:
        other.responses.create(model="gpt-5", previous_response_id=kept.id, input="x")
    assert e.value.body["code"] == "previous_response_not_found"
