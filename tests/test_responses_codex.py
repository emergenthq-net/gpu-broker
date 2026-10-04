"""The Codex CLI's wire format on /v1/responses. The fixture is a real Codex 0.145 request
(second turn of `codex exec "run echo hello"`, captured with a recording server), with every
free text, id and path replaced: `instructions`, developer and user message items, a
function_call and its function_call_output, function tools plus a `namespace` group and the
hosted `web_search` tool, `store: false`, `stream: true`, `include`, `reasoning`,
`client_metadata`. Codex ignores a delta whose item was not announced, so the event order is
checked as Codex reads it."""
from __future__ import annotations

import json
from typing import Any

from gpu_broker.web.anthropic_req import BLOCK_JOIN
from tests.dropin_fakes import ANSWER, CALL_ID, TOOL, app_client, dropin_broker  # noqa: F401
from tests.helpers import FIX, TOKEN

REQUEST = json.loads((FIX / "codex_responses_request.json").read_text())
HEADERS = {"authorization": f"Bearer {TOKEN}", "accept": "text/event-stream", "originator": "codex_exec"}


def parse(text: str) -> list[dict[str, Any]]:
    out = []
    for block in text.split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines() if ": " in line)
        if "data" in lines:
            data = json.loads(lines["data"])
            assert lines["event"] == data["type"]
            out.append(data)
    return out


def checked(events: list[dict[str, Any]]) -> None:
    """The order Codex relies on: every delta and done names an item that is open."""
    open_items: set[str] = set()
    for e in events:
        if e["type"] == "response.output_item.added":
            open_items.add(e["item"]["id"])
        elif e["type"] == "response.output_item.done":
            open_items.remove(e["item"]["id"])
        elif "item_id" in e:
            assert e["item_id"] in open_items, e
    assert not open_items
    assert [e["sequence_number"] for e in events] == list(range(len(events)))


def test_codex_turn_with_a_tool_result(app_client, dropin_broker):
    r = app_client.post("/v1/responses", json=REQUEST, headers=HEADERS)
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    events = parse(r.text)
    checked(events)
    final = events[-1]
    assert final["type"] == "response.completed" and final["response"]["model"] == "gpt-5-codex"
    text = final["response"]["output"][-1]["content"][0]["text"]
    assert text.startswith("It is Chunk ID")                     # the model answered the tool output
    x = final["response"]["x_broker"]
    assert x["used"] == "llama-8b" and x["dropped_tools"] == ["multi_agent_v1", "web_search"]
    sent = dropin_broker.backends.streamed[-1]
    roles = [m["role"] for m in sent["messages"]]
    assert roles == ["system", "user", "user", "assistant", "tool"]   # instructions + developer item: one system message
    developer = REQUEST["input"][0]["content"][0]["text"]
    assert sent["messages"][0]["content"] == REQUEST["instructions"] + BLOCK_JOIN + developer
    assert sent["messages"][3]["tool_calls"][0] == {"id": "call_1", "type": "function", "function": {
        "name": "exec_command", "arguments": REQUEST["input"][3]["arguments"]}}
    assert [t["function"]["name"] for t in sent["tools"]] == ["exec_command", "update_plan"]
    assert sent["parallel_tool_calls"] is False and sent["tool_choice"] == "auto"
    for field in ("client_metadata", "include", "reasoning", "prompt_cache_key", "store", "input", "instructions"):
        assert field not in sent


def test_codex_first_turn_gets_a_function_call(app_client, dropin_broker):
    first = {**REQUEST, "input": REQUEST["input"][:3],
             "tools": [{"type": "function", "name": TOOL, "parameters": {"type": "object", "properties": {}}}, *REQUEST["tools"][2:]]}
    events = parse(app_client.post("/v1/responses", json=first, headers=HEADERS).text)
    checked(events)
    done = [e["item"] for e in events if e["type"] == "response.output_item.done"]
    assert done[-1]["type"] == "function_call" and done[-1]["call_id"] == CALL_ID and done[-1]["status"] == "completed"


def test_codex_without_stream_gets_a_response_object(app_client):
    r = app_client.post("/v1/responses", json={**REQUEST, "stream": False, "input": "hi", "tools": []}, headers=HEADERS)
    body = r.json()
    assert body["object"] == "response" and body["store"] is False
    assert body["output"][-1]["content"][0]["text"] == ANSWER
