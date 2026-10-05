"""/v1/responses below the SDKs: the request translation and what it refuses, the event
translator's edge cases (interleaved text, cut-off and broken streams), and the store."""
from __future__ import annotations

import json
from typing import Any

import pytest

from gpu_broker.web.responses_obj import base
from gpu_broker.web.responses_out import Translator, events, finish
from gpu_broker.web.responses_req import messages, to_openai
from gpu_broker.web.responses_store import ResponseStore
from gpu_broker.web.responses_tools import dropped_tools
from tests.dropin_fakes import app_client, dropin_broker  # noqa: F401
from tests.helpers import TOKEN

AUTH = {"authorization": f"Bearer {TOKEN}"}


def line(delta: dict[str, Any], finish_reason: str | None = None) -> str:
    return "data: " + json.dumps({"choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}]}) + "\n\n"


def run(lines: list[str], failure: dict[str, str] | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    t = Translator(base("resp_x", {}, "m", {}))
    evs = [json.loads(e.split("data: ", 1)[1]) for e in events(t, lines, failure or {})]
    return evs, t.response


def test_text_between_argument_deltas_does_not_split_the_call():   # calls come whole, after the text
    _, r = run([line({"content": "Let me check. "}),
                line({"tool_calls": [{"index": 0, "id": "a", "function": {"name": "f", "arguments": '{"x"'}}]}),
                line({"content": "one moment"}),
                line({"tool_calls": [{"index": 0, "function": {"arguments": ": 1}"}}]}),
                line({"tool_calls": [{"index": 1, "id": "b", "function": {"name": "g", "arguments": "{}"}}]}),
                line({}, "tool_calls")])
    out = r["output"]
    assert [i["type"] for i in out] == ["message", "function_call", "function_call"]
    assert (out[1]["arguments"], out[2]["call_id"], out[0]["content"][0]["text"]) == ('{"x": 1}', "b", "Let me check. one moment")


def test_deltas_without_an_index_follow_the_call_id():
    _, r = run([line({"tool_calls": [{"id": "a", "function": {"name": "f", "arguments": '{"x"'}}]}),
                line({"tool_calls": [{"function": {"arguments": ": 1}"}}]}),
                line({"tool_calls": [{"id": "b", "function": {"name": "g", "arguments": "{}"}}]})])
    assert [(i["call_id"], i["arguments"]) for i in r["output"]] == [("a", '{"x": 1}'), ("b", "{}")]


@pytest.mark.parametrize(("finish_reason", "reason"), [("length", "max_output_tokens"), ("content_filter", "content_filter")])
def test_a_cut_off_answer_is_incomplete(finish_reason, reason):
    evs, r = run([line({"content": "partial"}, finish_reason)])
    assert evs[-1]["type"] == "response.incomplete"
    assert r["status"] == "incomplete" and r["incomplete_details"] == {"reason": reason}


def test_a_broken_stream_ends_in_response_failed():
    evs, r = run([line({"content": "par"})], {"message": "server went away"})
    assert evs[-1]["type"] == "response.failed" and r["error"] == {"code": "server_error", "message": "server went away"}
    assert not any(e["type"] == "response.completed" for e in evs)


def test_usage_falls_back_to_llama_timings():
    t = Translator(base("resp_x", {}, "m", {}))
    r = finish(t, ["data: " + json.dumps({"choices": [], "timings": {"prompt_n": 4, "predicted_n": 2}}) + "\n\n"])
    assert (r["usage"]["input_tokens"], r["usage"]["output_tokens"], r["usage"]["total_tokens"]) == (4, 2, 6)


def test_input_items_become_chat_messages():
    msgs = messages([
        {"role": "developer", "content": "rules"},
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "look"},
                                                         {"type": "input_image", "image_url": "data:image/png;base64,AA"}]},
        {"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": None},
        {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "calling"}]},
        {"type": "function_call", "call_id": "c1", "name": "f", "arguments": "{}"},
        {"type": "function_call", "call_id": "c2", "name": "g", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "c1", "output": [{"type": "input_text", "text": "ok"},
                                                                     {"type": "input_image", "image_url": "https://x/i.png"}]},
    ])
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "tool", "user"]
    assert msgs[1]["content"][1] == {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}}
    assert msgs[2]["content"] == "calling" and [c["id"] for c in msgs[2]["tool_calls"]] == ["c1", "c2"]
    assert msgs[3] == {"role": "tool", "tool_call_id": "c1", "content": "ok"}
    assert msgs[4]["content"][1] == {"type": "image_url", "image_url": {"url": "https://x/i.png"}}


def test_scalars_format_and_tool_choice_are_mapped():
    tools = [{"type": "function", "name": "f", "parameters": {"type": "object"}, "strict": True}]
    out, _, _ = to_openai({"input": "x", "max_output_tokens": 50, "temperature": 0.2, "tools": tools, "parallel_tool_calls": False,
                        "tool_choice": {"type": "function", "name": "f"},
                        "text": {"format": {"type": "json_schema", "name": "s", "schema": {"type": "object"}, "strict": True}}}, [])
    assert out["max_tokens"] == 50 and out["temperature"] == 0.2 and out["parallel_tool_calls"] is False
    assert out["tools"][0]["function"]["strict"] is True
    assert out["tool_choice"] == {"type": "function", "function": {"name": "f"}}
    assert out["response_format"] == {"type": "json_schema", "json_schema": {"name": "s", "schema": {"type": "object"}, "strict": True}}
    allowed, _, _ = to_openai({"input": "x", "tools": tools, "tool_choice": {"type": "allowed_tools", "mode": "required", "tools": [{"type": "function", "name": "f"}]}}, [])
    assert allowed["tool_choice"] == "required"
    plain, _, _ = to_openai({"input": "x", "tools": [{"type": "web_search"}], "tool_choice": "required"}, [])
    assert "tools" not in plain and "tool_choice" not in plain and "parallel_tool_calls" not in plain


@pytest.mark.parametrize(("body", "msg"), [
    ({}, "`input` is required"),
    ({"input": 5}, "string or a list"),
    ({"input": [{"type": "item_reference", "id": "x"}]}, "item_reference"),
    ({"input": [{"role": "robot", "content": "x"}]}, "role must be"),
    ({"input": [{"role": "user", "content": [{"type": "input_file", "file_id": "f"}]}]}, "input_file"),
    ({"input": [{"role": "assistant", "content": [{"type": "input_image", "image_url": "u"}]}]}, "only accepted in user"),
    ({"input": [{"type": "function_call", "name": "f"}]}, "call_id"),
    ({"input": "x", "instructions": 3}, "`instructions` must be a string"),
    ({"input": "x", "background": True}, "background"),
    ({"input": "x", "prompt": {"id": "p"}}, "prompt template"),
    ({"input": "x", "max_output_tokens": "9"}, "must be an integer"),
    ({"input": "x", "temperature": True}, "must be a number"),
    ({"input": "x", "tools": [{"type": "function", "name": "f"}], "tool_choice": "sometimes"}, "tool_choice"),
    ({"input": "x", "text": {"format": {"type": "xml"}}}, "text.format.type"),
])
def test_malformed_requests_are_an_openai_400(app_client, body, msg):
    r = app_client.post("/v1/responses", json={"model": "gpt-5", **body}, headers=AUTH)
    assert r.status_code == 400 and msg in r.json()["error"]["message"]
    assert r.json()["error"]["type"] == "invalid_request_error"


def test_store_expires_caps_and_binds_to_the_owner(monkeypatch):
    now = [100.0]
    monkeypatch.setattr("gpu_broker.web.responses_store.time.monotonic", lambda: now[0])
    s = ResponseStore(ttl_s=10, cap=2)
    s.put("a", "me", [{"role": "user", "content": "1"}])
    assert s.get("a", "me") == [{"role": "user", "content": "1"}] and s.get("a", "you") is None
    now[0] = 109.9
    assert s.get("a", "me") is not None
    now[0] = 110.0
    assert s.get("a", "me") is None
    for rid in ("b", "c", "d"):
        s.put(rid, "me", [])
    assert [s.get(r, "me") is not None for r in ("b", "c", "d")] == [False, True, True]


def test_custom_and_function_tools_are_never_reported_as_dropped():
    tools = [{"type": "custom", "name": "apply_patch"}, {"type": "function", "name": "exec_command", "parameters": {}},
             {"type": "web_search"}]
    assert dropped_tools({"tools": tools}) == ["web_search"]
