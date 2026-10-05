"""Review round on /v1/responses (PR #18): one or more tests per finding.

1/10 the store owner is an identity (client key, or main token + x-requester); one requester helper
2 a streamed response is stored before response.completed goes out
3 byte budgets per stored entry and in total; a body over the cap is a 413
4 instructions + every developer message: ONE leading system message, also when replaying
5 custom (freeform) tools round-trip; only hosted tools are dropped; local_shell is refused
6 no private field on the wire; a broken stream closes its open item as incomplete
7 a list where a type/role/mode tag belongs is a 400, not a crash
8 allowed_tools sends only the allowed subset
9 interleaved tool-call deltas (index 0, 1, 0) land on the right calls
"""
from __future__ import annotations

import dataclasses
import json
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from gpu_broker.broker import Broker
from gpu_broker.constants import REQUESTER_HEADER
from gpu_broker.settings import Limits
from gpu_broker.web.app import create_app
from gpu_broker.web.jobs import CLIENT_ID_STATE, CLIENT_STATE, identity, requester
from gpu_broker.web.responses_obj import base
from gpu_broker.web.responses_out import Translator, events
from gpu_broker.web.responses_req import messages, to_openai
from gpu_broker.web.responses_store import ResponseStore, size
from tests.dropin_fakes import MODEL_MAP, ToolBackends, app_client, catalog_with_embedder, dropin_broker  # noqa: F401
from tests.helpers import TOKEN, FakeDriver, make_settings, wait_idle
from tests.test_responses_units import line

AUTH = {"authorization": f"Bearer {TOKEN}"}
EP = "/v1/responses"


def fake_request(key: str | None = None, claimed: str | None = None, host: str = "192.0.2.9") -> Any:
    kid = f"id-{key}" if key else None   # the key's id; names may repeat
    return SimpleNamespace(state=SimpleNamespace(**{CLIENT_STATE: key, CLIENT_ID_STATE: kid}), client=SimpleNamespace(host=host),
                           headers={REQUESTER_HEADER: claimed} if claimed else {})


def stream_of(t: Translator, lines: list[str], failure: dict[str, str] | None = None) -> list[dict[str, Any]]:
    return [json.loads(e.split("data: ", 1)[1]) for e in events(t, lines, failure or {})]


# ---- 1 / 10 ------------------------------------------------------------------
def test_identity_is_the_key_or_the_main_token_plus_requester_never_the_address():
    assert identity(fake_request("laptop", claimed="someone-else")) == "key:id-laptop"
    assert identity(fake_request(claimed="ci")) == "main:ci"
    assert identity(fake_request(host="198.51.100.1")) == identity(fake_request(host="198.51.100.2")) == "main:"
    assert identity(fake_request("ci")) != identity(fake_request(claimed="ci"))   # a key cannot pose as a main caller


def test_requester_is_the_key_name_whatever_is_claimed():
    assert requester(fake_request("laptop"), "x") == "laptop"
    assert requester(fake_request(), "x") == "x"
    assert requester(fake_request(host="192.0.2.1")) == "192.0.2.1"


def test_chat_jobs_use_the_shared_requester(app_client, dropin_broker):
    import gpu_broker.web.completion as completion
    seen = []
    real = completion.requester_of
    completion.requester_of = lambda r, claimed=None: seen.append(real(r, claimed)) or "via-helper"
    try:
        r = app_client.post(EP, json={"model": "gpt-4o", "input": "hi"}, headers={**AUTH, REQUESTER_HEADER: "ci"})
    finally:
        completion.requester_of = real
    assert r.status_code == 200 and seen == ["ci"]


def test_a_stored_response_belongs_to_its_requester(app_client):
    first = app_client.post(EP, json={"model": "gpt-4o", "input": "hi"}, headers={**AUTH, REQUESTER_HEADER: "a"}).json()
    other = app_client.post(EP, json={"model": "gpt-4o", "input": "again", "previous_response_id": first["id"]},
                            headers={**AUTH, REQUESTER_HEADER: "b"})
    assert other.status_code == 400 and other.json()["error"]["code"] == "previous_response_not_found"
    same = app_client.post(EP, json={"model": "gpt-4o", "input": "again", "previous_response_id": first["id"]},
                           headers={**AUTH, REQUESTER_HEADER: "a"})
    assert same.status_code == 200


# ---- 2 -----------------------------------------------------------------------
def test_a_streamed_response_is_stored_before_completed_is_sent():
    store = ResponseStore()
    t = Translator(base("resp_s", {}, "m", {}), on_final=lambda t: store.put(t.response["id"], "me", messages(t.items)))
    for e in events(t, [line({"content": "hi"}), line({}, "stop")], {}):
        if "response.completed" in e:
            assert store.get("resp_s", "me") is not None, "completed went out before the response was stored"
            break
    else:
        pytest.fail("no response.completed")


def test_an_immediate_follow_up_after_a_stream_finds_it(app_client, dropin_broker):
    r = app_client.post(EP, json={"model": "gpt-4o", "input": "hi", "stream": True}, headers=AUTH)
    rid = next(json.loads(e.split("data: ", 1)[1]) for e in r.text.split("\n\n") if "response.completed" in e)["response"]["id"]
    again = app_client.post(EP, json={"model": "gpt-4o", "input": "more", "previous_response_id": rid}, headers=AUTH)
    assert again.status_code == 200


# ---- 3 -----------------------------------------------------------------------
def test_an_entry_over_its_budget_is_not_kept():
    convo = [{"role": "user", "content": "x" * 100}]
    s = ResponseStore(entry_bytes=size(convo) - 1)
    assert s.put("r", "me", convo) is False and s.get("r", "me") is None and s.bytes == 0


def test_the_total_budget_evicts_the_oldest_first():
    one = [{"role": "user", "content": "x" * 100}]
    s = ResponseStore(total_bytes=2 * size(one))
    for rid in ("a", "b", "c"):
        assert s.put(rid, "me", one)
    assert [s.get(r, "me") is not None for r in ("a", "b", "c")] == [False, True, True]
    assert s.bytes == 2 * size(one)
    s.put("c", "me", one)   # replacing an entry does not count it twice
    assert s.bytes == 2 * size(one) and s.get("b", "me") is not None


@pytest.fixture
def small(tmp_path):
    driver = FakeDriver({"llama-8b"})
    limits = dataclasses.replace(Limits(), responses_body_bytes=2000, response_store_entry_bytes=600)
    s = make_settings(tmp_path, catalog=catalog_with_embedder(tmp_path), model_map=MODEL_MAP, limits=limits)
    b = Broker(s, env={}, driver=driver, backends=ToolBackends(driver))
    b.start()
    with TestClient(create_app(b, TOKEN, start=False)) as c:
        yield c
    assert wait_idle(b)
    b.stop()


def test_a_body_over_the_cap_is_413(small):
    big = {"model": "gpt-4o", "input": "x" * 3000}
    r = small.post(EP, json=big, headers=AUTH)
    assert r.status_code == 413 and "responses_body_bytes" in r.json()["error"]["message"]

    def chunks():   # no Content-Length: the cap still holds while reading
        yield json.dumps(big).encode()
    r = small.post(EP, content=chunks(), headers={**AUTH, "content-type": "application/json"})
    assert r.status_code == 413
    assert small.post(EP, json={"model": "gpt-4o", "input": "x" * 100}, headers=AUTH).status_code == 200


def test_the_configured_entry_budget_applies(small):
    r = small.post(EP, json={"model": "gpt-4o", "input": "y" * 1000}, headers=AUTH).json()
    again = small.post(EP, json={"model": "gpt-4o", "input": "z", "previous_response_id": r["id"]}, headers=AUTH)
    assert again.status_code == 400   # over the 600-byte entry budget: never kept


def test_a_body_that_is_not_a_json_object_is_400(small):
    assert small.post(EP, content=b"[1, 2]", headers={**AUTH, "content-type": "application/json"}).status_code == 400
    assert small.post(EP, content=b"{nope", headers={**AUTH, "content-type": "application/json"}).status_code == 400


# ---- 4 -----------------------------------------------------------------------
def one_leading_system(msgs: list[dict[str, Any]]) -> None:
    """What many chat templates enforce ("System message must be at the beginning")."""
    roles = [m["role"] for m in msgs]
    if roles.count("system") > 1 or ("system" in roles and roles[0] != "system"):
        raise ValueError(f"template refuses {roles}")


def dev(text: str) -> dict[str, Any]:
    return {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": text}]}


def test_instructions_and_developer_messages_are_one_leading_system_message():
    chat, turn, _ = to_openai({"instructions": "be brief", "input": [dev("d1"), {"role": "user", "content": "hi"}, dev("d2")]}, [])
    one_leading_system(chat["messages"])
    assert chat["messages"][0] == {"role": "system", "content": "be brief\nd1\nd2"}
    history = [*turn, {"role": "assistant", "content": "ok"}]   # what the store keeps: developer items included
    again, _, _ = to_openai({"instructions": "now terse", "input": [dev("d3"), {"role": "user", "content": "more"}]}, history)
    one_leading_system(again["messages"])
    assert again["messages"][0]["content"] == "now terse\nd1\nd2\nd3"
    assert [m["role"] for m in again["messages"]] == ["system", "user", "assistant", "user"]


def test_replaying_a_stored_conversation_keeps_one_system_message(app_client, dropin_broker):
    first = app_client.post(EP, json={"model": "gpt-4o", "instructions": "i1", "input": [dev("d1"), {"role": "user", "content": "hi"}]},
                            headers=AUTH).json()
    app_client.post(EP, json={"model": "gpt-4o", "instructions": "i2", "previous_response_id": first["id"],
                              "input": [dev("d2"), {"role": "user", "content": "more"}]}, headers=AUTH)
    sent = dropin_broker.backends.sent[-1]["messages"]
    one_leading_system(sent)
    assert sent[0]["content"] == "i2\nd1\nd2"


# ---- 5 -----------------------------------------------------------------------
PATCH = {"type": "custom", "name": "apply_patch", "description": "Apply a patch", "format": {"type": "grammar"}}


def test_a_custom_tool_is_a_function_with_one_string_input():
    chat, _, custom = to_openai({"input": "x", "tools": [PATCH]}, [])
    fn = chat["tools"][0]["function"]
    assert custom == {"apply_patch"} and fn["name"] == "apply_patch"
    assert fn["parameters"] == {"type": "object", "properties": {"input": {"type": "string"}}, "required": ["input"]}
    assert fn["description"].startswith("Apply a patch")


def test_a_custom_call_comes_back_as_custom_tool_call_and_round_trips():
    t = Translator(base("resp_c", {}, "m", {}), frozenset({"apply_patch"}))
    args = json.dumps({"input": "*** Begin Patch\n*** End Patch"})
    evs = stream_of(t, [line({"tool_calls": [{"index": 0, "id": "c1", "function": {"name": "apply_patch", "arguments": args}}]}),
                        line({}, "tool_calls")])
    item = t.response["output"][0]
    assert item == {"id": item["id"], "type": "custom_tool_call", "status": "completed", "call_id": "c1",
                    "name": "apply_patch", "input": "*** Begin Patch\n*** End Patch"}
    kinds = [e["type"] for e in evs]
    assert "response.custom_tool_call_input.delta" in kinds and "response.custom_tool_call_input.done" in kinds
    back = messages([item, {"type": "custom_tool_call_output", "call_id": "c1", "output": "applied"}])
    assert back[0]["tool_calls"][0]["function"] == {"name": "apply_patch", "arguments": args}
    assert back[1] == {"role": "tool", "tool_call_id": "c1", "content": "applied"}


def test_a_function_call_named_like_no_custom_tool_stays_a_function_call():
    t = Translator(base("resp_f", {}, "m", {}), frozenset({"apply_patch"}))
    stream_of(t, [line({"tool_calls": [{"index": 0, "id": "f1", "function": {"name": "exec", "arguments": "{}"}}]})])
    assert t.response["output"][0]["type"] == "function_call"


def test_custom_input_that_is_not_json_is_kept_as_written():
    t = Translator(base("resp_r", {}, "m", {}), frozenset({"apply_patch"}))
    stream_of(t, [line({"tool_calls": [{"index": 0, "id": "c", "function": {"name": "apply_patch", "arguments": "raw text"}}]})])
    assert t.response["output"][0]["input"] == "raw text"


@pytest.mark.parametrize("kind", ["web_search", "web_search_preview_2025_03_11", "file_search", "code_interpreter",
                                  "image_generation", "computer_use_preview", "mcp", "namespace"])
def test_hosted_tools_are_dropped(kind):
    chat, _, _ = to_openai({"input": "x", "tools": [{"type": kind}, {"type": "function", "name": "f"}]}, [])
    assert [t["function"]["name"] for t in chat["tools"]] == ["f"]


@pytest.mark.parametrize("kind", ["local_shell", "something_new"])
def test_other_tool_types_are_refused(app_client, kind):
    r = app_client.post(EP, json={"model": "gpt-4o", "input": "x", "tools": [{"type": kind}]}, headers=AUTH)
    assert r.status_code == 400 and kind in r.json()["error"]["message"]


# ---- 6 -----------------------------------------------------------------------
def private_keys(v: Any) -> list[str]:
    if isinstance(v, dict):
        return [k for k in v if k.startswith("_")] + [k for x in v.values() for k in private_keys(x)]
    if isinstance(v, list):
        return [k for x in v for k in private_keys(x)]
    return []


@pytest.mark.parametrize("failure", [None, {"message": "server went away"}])
def test_no_private_field_is_ever_sent(failure):
    t = Translator(base("resp_p", {}, "m", {}))
    evs = stream_of(t, [line({"reasoning_content": "hmm"}), line({"content": "partial"})], failure)
    assert private_keys(evs) == []


def test_a_broken_stream_closes_its_open_item_as_incomplete():
    t = Translator(base("resp_b", {}, "m", {}))
    evs = stream_of(t, [line({"content": "half an ans"}),
                        line({"tool_calls": [{"index": 0, "id": "x", "function": {"name": "f", "arguments": "{"}}]})],
                    {"message": "boom"})
    done = [e for e in evs if e["type"] == "response.output_item.done"]
    assert done[-1]["item"]["status"] == "incomplete" and done[-1]["item"]["content"][0]["text"] == "half an ans"
    assert evs[-1]["type"] == "response.failed" and evs[-2]["type"] == "response.output_item.done"
    assert [i["type"] for i in evs[-1]["response"]["output"]] == ["message"]   # the half-made call is not offered


# ---- 7 -----------------------------------------------------------------------
LIST = ["x"]
BAD_TAGS = [
    {"input": [{"type": "message", "role": LIST, "content": "x"}]},
    {"input": [{"type": LIST, "role": "user", "content": "x"}]},
    {"input": [{"role": "user", "content": [{"type": LIST, "text": "x"}]}]},
    {"input": "x", "text": {"format": {"type": LIST}}},
    {"input": "x", "tools": [{"type": LIST}]},
    {"input": "x", "tools": [{"type": "function", "name": "f"}], "tool_choice": {"type": LIST}},
    {"input": "x", "tools": [{"type": "function", "name": "f"}], "tool_choice": {"type": "allowed_tools", "mode": LIST, "tools": []}},
]


@pytest.mark.parametrize("body", BAD_TAGS)
def test_a_list_where_a_tag_belongs_is_a_400(app_client, body):
    r = app_client.post(EP, json={"model": "gpt-4o", **body}, headers=AUTH)
    assert r.status_code == 400 and r.json()["error"]["type"] == "invalid_request_error"


# ---- 8 -----------------------------------------------------------------------
def test_allowed_tools_sends_only_the_allowed_subset():
    tools = [{"type": "function", "name": n} for n in ("f", "g", "h")]
    choice = {"type": "allowed_tools", "mode": "auto", "tools": [{"type": "function", "name": "g"}, {"type": "web_search"}]}
    chat, _, _ = to_openai({"input": "x", "tools": tools, "tool_choice": choice}, [])
    assert [t["function"]["name"] for t in chat["tools"]] == ["g"] and chat["tool_choice"] == "auto"
    none, _, _ = to_openai({"input": "x", "tools": tools, "tool_choice": {**choice, "tools": [{"type": "web_search"}]}}, [])
    assert "tools" not in none and "tool_choice" not in none
    with pytest.raises(ValueError, match="requires a tool"):
        to_openai({"input": "x", "tools": tools, "tool_choice": {**choice, "mode": "required", "tools": []}}, [])


# ---- 9 -----------------------------------------------------------------------
def test_interleaved_call_deltas_land_on_the_right_calls():
    t = Translator(base("resp_i", {}, "m", {}))
    stream_of(t, [line({"tool_calls": [{"index": 0, "id": "a", "function": {"name": "f", "arguments": '{"x": '}}]}),
                  line({"tool_calls": [{"index": 1, "id": "b", "function": {"name": "g", "arguments": '{"y": '}}]}),
                  line({"tool_calls": [{"index": 0, "function": {"arguments": "1}"}}]}),
                  line({"tool_calls": [{"index": 1, "function": {"arguments": "2}"}}]}),
                  line({}, "tool_calls")])
    out = [(i["call_id"], i["name"], json.loads(i["arguments"])) for i in t.response["output"]]
    assert out == [("a", "f", {"x": 1}), ("b", "g", {"y": 2})]


def test_a_declared_oversize_body_is_refused_before_it_is_read():
    import asyncio

    from fastapi import HTTPException

    from gpu_broker.web.responses import capped_body

    async def never():
        raise AssertionError("the body was read")
        yield b""
    req = SimpleNamespace(headers={"content-length": "5000"}, stream=never)
    with pytest.raises(HTTPException) as e:
        asyncio.run(capped_body(100)(req))
    assert e.value.status_code == 413
