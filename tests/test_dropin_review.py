"""Review round on the drop-in (PR #14): one test per finding.

1 stop_reason is tool_use whenever a tool_use block went out; 2 stream_options never reaches a
server with stream=false; 3 embedding models never stand in for chat and are not listed;
4 a plain request's `model` is the server's, as before (only mapped names are echoed);
5 the mapping is recorded when the job is created; 6 tool deltas without an index, or with
text between them, stay one call; 7 partial_json is always a string; 8 images in a
tool_result are forwarded; 9 malformed Anthropic input is a 400 and null usage counts are 0;
10 one parse per streamed line, a list `detail` on OpenAI 422s, and admin routes take only
`Authorization: Bearer`."""
from __future__ import annotations

import json

import pytest

from gpu_broker import backends
from gpu_broker.resolve import resolve
from gpu_broker.web import anthropic_sse, completion
from gpu_broker.web.anthropic_req import to_openai
from gpu_broker.web.anthropic_resp import message, usage
from tests.dropin_fakes import EMBED_MODEL, TOOL, app_client, dropin_broker  # noqa: F401
from tests.helpers import TOKEN, wait_idle

MAIN = {"Authorization": f"Bearer {TOKEN}"}
HI = [{"role": "user", "content": "hi"}]


def sse_events(lines):
    out = []
    for frame in anthropic_sse.events(lines, "claude-x", False, {}, {}):
        name, data = frame.strip().split("\n")
        out.append((name.removeprefix("event: "), json.loads(data.removeprefix("data: "))))
    return out


def line(chunk):
    return "data: " + json.dumps(chunk) + "\n\n"


def delta(d, finish=None):
    return line({"choices": [{"index": 0, "delta": d, "finish_reason": finish}]})


# ---- 1 ----------------------------------------------------------------------
def test_tool_use_block_means_tool_use_stop_reason():
    result = {"choices": [{"message": {"content": None, "tool_calls": [
        {"id": "c", "function": {"name": TOOL, "arguments": "{}"}}]}, "finish_reason": "stop"}]}
    assert message(result, "m", False, {})["stop_reason"] == "tool_use"
    evs = sse_events([delta({"tool_calls": [{"index": 0, "id": "c", "function": {"name": TOOL, "arguments": "{}"}}]}),
                      delta({}, "stop")])
    assert next(d for n, d in evs if n == "message_delta")["delta"]["stop_reason"] == "tool_use"
    plain = sse_events([delta({"content": "hi"}), delta({}, "stop")])
    assert next(d for n, d in plain if n == "message_delta")["delta"]["stop_reason"] == "end_turn"


# ---- 2 ----------------------------------------------------------------------
def test_stream_options_is_dropped_unless_streaming():
    hb = backends.HttpBackends.__new__(backends.HttpBackends)
    hb.tokens = {}
    model = {"endpoint": "http://x", "served_name": "s"}
    body = {"messages": HI, "stream": True, "stream_options": {"include_usage": True}}
    for stream in (False, None):
        assert "stream_options" not in json.loads(hb._chat_request(model, body, stream).data)
    assert json.loads(hb._chat_request(model, body, True).data)["stream_options"] == {"include_usage": True}


def test_queued_anthropic_stream_sends_no_stream_options_upstream(app_client, dropin_broker):
    body = {"model": "claude-x", "max_tokens": 8, "messages": HI, "stream": True}
    app_client.post("/v1/messages", headers={**MAIN, "x-priority": "background"}, json=body)
    assert wait_idle(dropin_broker)
    sent = dropin_broker.backends.sent[-1]
    assert "stream_options" in sent   # the payload keeps it; the HTTP layer drops it (test above)


# ---- 3 ----------------------------------------------------------------------
def test_embedders_never_stand_in_for_chat(dropin_broker):
    data = dropin_broker.catalog.data
    for name in ("totally-unknown-chat", "gpt-9"):
        r = resolve(data, name, "llm", [])
        assert r.resolved != EMBED_MODEL
    for m in list(data["models"]):   # with only the embedder left, an unknown chat name has no stand-in
        if m != EMBED_MODEL:
            data["models"][m]["status"] = "missing"
    assert resolve(data, "totally-unknown-chat", "llm", []).resolved is None
    assert resolve(data, "x-embedder", "llm", ["embed"]).resolved == EMBED_MODEL


def test_embedders_are_not_in_the_model_list(app_client):
    ids = [m["id"] for m in app_client.get("/v1/models", headers=MAIN).json()["data"]]
    assert EMBED_MODEL not in ids and ids


# ---- 4 ----------------------------------------------------------------------
@pytest.mark.parametrize("priority", ["", "background"])
def test_plain_request_keeps_the_servers_model(app_client, dropin_broker, priority):
    h = {**MAIN, **({"x-priority": priority} if priority else {})}
    served = dropin_broker.catalog.models["llama-8b"]["served_name"]
    plain = app_client.post("/v1/chat/completions", headers=h, json={"model": "llama", "messages": HI}).json()
    assert plain["model"] == served
    mapped = app_client.post("/v1/chat/completions", headers=h, json={"model": "gpt-4o", "messages": HI}).json()
    assert mapped["model"] == "gpt-4o"
    stream = app_client.post("/v1/chat/completions", headers=h, json={"model": "llama", "messages": HI, "stream": True}).text
    models = {json.loads(p[6:])["model"] for p in stream.split("\n") if p.startswith("data: {")}
    assert models <= {served, "served"} and models
    stream = app_client.post("/v1/chat/completions", headers=h, json={"model": "gpt-4o", "messages": HI, "stream": True}).text
    assert {json.loads(p[6:])["model"] for p in stream.split("\n") if p.startswith("data: {")} == {"gpt-4o"}
    assert wait_idle(dropin_broker)


def test_unmapped_stream_lines_are_relayed_untouched(app_client, monkeypatch):
    calls = []
    monkeypatch.setattr(completion, "echo_model", lambda line, req: calls.append(line) or line)
    app_client.post("/v1/chat/completions", headers=MAIN, json={"model": "llama", "messages": HI, "stream": True})
    assert calls == []


# ---- 5 ----------------------------------------------------------------------
@pytest.mark.parametrize("priority", ["", "background"])
def test_mapping_is_recorded_when_the_job_is_created(app_client, dropin_broker, monkeypatch, priority):
    updates = []
    real = dropin_broker.store.update_job
    monkeypatch.setattr(dropin_broker.store, "update_job", lambda jid, **kw: updates.append(kw) or real(jid, **kw))
    h = {**MAIN, **({"x-priority": priority} if priority else {})}
    r = app_client.post("/v1/chat/completions", headers=h, json={"model": "gpt-4o", "messages": HI}).json()
    assert wait_idle(dropin_broker)
    job = dropin_broker.store.job(r["x_broker"]["job"])
    assert job["requested"] == "gpt-4o" and "model_map" in job["substitution"]
    assert r["x_broker"]["substitution"] == job["substitution"]
    assert [u for u in updates if "substitution" in u] == [updates[0]]   # set once, with resolved, at creation
    if priority:
        ev = [e for e in dropin_broker.store.events(0, 1000) if e["kind"] == "job.substituted"][-1]
        assert ev["data"]["requested"] == "gpt-4o" and ev["data"]["reason"] == job["substitution"]


def test_a_rejected_mapped_call_reports_the_original_name(app_client, dropin_broker):
    dropin_broker.catalog.models["llama-8b"]["status"] = "missing"
    for m in dropin_broker.catalog.models.values():
        m["status"] = "missing"
    r = app_client.post("/v1/chat/completions", headers={**MAIN, "x-priority": "background"},
                        json={"model": "gpt-4o", "messages": HI})
    xb = r.json()["x_broker"]
    assert r.status_code == 503 and xb["requested"] == "gpt-4o"
    assert dropin_broker.store.job(xb["job"])["requested"] == "gpt-4o"


# ---- 6, 7 -------------------------------------------------------------------
def blocks(evs):
    starts = {d["index"]: d["content_block"] for n, d in evs if n == "content_block_start"}
    for n, d in evs:
        if n == "content_block_delta":
            b, dl = starts[d["index"]], d["delta"]
            if dl["type"] == "input_json_delta":
                assert isinstance(dl["partial_json"], str)
                b["args"] = b.get("args", "") + dl["partial_json"]
            elif dl["type"] == "text_delta":
                b["text"] += dl["text"]
    stops = [d["index"] for n, d in evs if n == "content_block_stop"]
    assert sorted(stops) == sorted(starts)   # every block closed exactly once
    return [starts[i] for i in sorted(starts)]


def test_tool_deltas_without_an_index_follow_the_call_id():
    evs = sse_events([
        delta({"tool_calls": [{"id": "a", "function": {"name": "f", "arguments": '{"x"'}}]}),
        delta({"tool_calls": [{"function": {"arguments": ": 1}"}}]}),
        delta({"tool_calls": [{"index": None, "id": "b", "function": {"name": "g", "arguments": "{}"}}]}),
        delta({}, "tool_calls")])
    tools = [b for b in blocks(evs) if b["type"] == "tool_use"]
    assert [(t["id"], t["name"], t["args"]) for t in tools] == [("a", "f", '{"x": 1}'), ("b", "g", "{}")]


def test_text_between_argument_deltas_does_not_split_the_call():
    evs = sse_events([
        delta({"content": "Let me check. "}),
        delta({"tool_calls": [{"index": 0, "id": "a", "function": {"name": "f", "arguments": '{"x"'}}]}),
        delta({"content": "one moment"}),
        delta({"tool_calls": [{"index": 0, "function": {"arguments": ": 1}"}}]}),
        delta({}, "tool_calls")])
    bs = blocks(evs)
    assert [b["type"] for b in bs] == ["text", "tool_use", "text"]
    assert bs[1]["args"] == '{"x": 1}' and bs[0]["text"] == "Let me check. " and bs[2]["text"] == "one moment"


def test_object_arguments_stream_as_json_text():
    evs = sse_events([delta({"tool_calls": [{"index": 0, "id": "a", "function": {"name": "f", "arguments": {"x": 1}}}]})])
    assert json.loads(blocks(evs)[0]["args"]) == {"x": 1}


# ---- 8 ----------------------------------------------------------------------
def test_images_in_a_tool_result_are_forwarded():
    img = {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAA"}}
    body = {"model": "m", "max_tokens": 8, "messages": [
        {"role": "user", "content": "look"},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "shot", "input": {}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": [{"type": "text", "text": "done"}, img]},
                                     {"type": "text", "text": "what is it?"}]}]}
    msgs = to_openai(body)["messages"]
    assert msgs[2] == {"role": "tool", "tool_call_id": "t1", "content": "done"}
    parts = msgs[3]["content"]
    assert msgs[3]["role"] == "user" and "t1" in parts[0]["text"]
    assert parts[1] == {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}}
    assert parts[2] == {"type": "text", "text": "what is it?"}


# ---- 9 ----------------------------------------------------------------------
BASE = {"model": "claude-x", "max_tokens": 8, "messages": HI}
BAD = [
    {"tools": {"name": "f"}},
    {"tools": 5},
    {"tools": [{"name": "f", "input_schema": 5}]},
    {"tools": [{"description": "no name"}]},
    {"tools": [{"name": ""}]},
    {"tools": [{"name": "f", "input_schema": "string"}]},
    {"tools": ["f"]},
    {"tool_choice": {"type": "sometimes"}},
    {"tool_choice": {"type": "tool"}},
    {"tool_choice": "auto"},
    {"max_tokens": "8"},
    {"temperature": "hot"},
    {"stop_sequences": "END"},
    {"messages": [{"role": "user", "content": [{"type": "image", "source": {"type": "base64", "media_type": "image/bmp", "data": "A"}}]}]},
    {"messages": [{"role": "user", "content": [{"type": "image", "source": {"type": "base64", "media_type": "image/png"}}]}]},
    {"messages": [{"role": "user", "content": [{"type": "text", "text": 5}]}]},
    {"messages": [{"role": "assistant", "content": [{"type": "tool_use", "id": "t", "name": "f", "input": "x"}]}]},
    {"messages": [{"role": "user", "content": [{"type": "tool_result", "content": "x"}]}]},
]


@pytest.mark.parametrize("bad", BAD)
def test_malformed_anthropic_input_is_a_400(app_client, bad):
    r = app_client.post("/v1/messages", headers=MAIN, json={**BASE, **bad})
    assert r.status_code == 400 and r.json()["error"]["type"] == "invalid_request_error", bad


def test_null_usage_counts_are_zero():
    assert usage({"prompt_tokens": None, "completion_tokens": None}) == {"input_tokens": 0, "output_tokens": 0}
    assert usage({"prompt_tokens": None}, {"prompt_n": 4, "predicted_n": None}) == {"input_tokens": 4, "output_tokens": 0}
    assert usage({"prompt_tokens": 3, "completion_tokens": 2}) == {"input_tokens": 3, "output_tokens": 2}


# ---- 10 ---------------------------------------------------------------------
def test_each_streamed_line_is_parsed_once_on_the_anthropic_route(app_client, monkeypatch):
    real, count = json.loads, []
    monkeypatch.setattr(anthropic_sse.json, "loads", lambda s, *a, **k: count.append(s) or real(s, *a, **k))
    r = app_client.post("/v1/messages", headers=MAIN, json={**BASE, "stream": True})
    frames = [p for p in r.text.split("\n") if p.startswith("data: ")]
    assert frames and len(count) == len(set(count))


def test_openai_422_keeps_fastapis_detail_list(app_client):
    r = app_client.post("/v1/chat/completions", headers={**MAIN, "Content-Type": "application/json"}, content=b"[1, 2]")
    body = r.json()
    assert r.status_code == 422 and isinstance(body["detail"], list) and body["detail"][0]["loc"]
    assert body["error"]["message"].startswith("invalid request body: ")


def test_admin_routes_take_only_bearer(app_client):
    assert app_client.post("/v1/admin/resume", headers={"x-api-key": TOKEN}).status_code == 401
    assert app_client.post("/v1/admin/resume", headers=MAIN).status_code == 200
    assert app_client.get("/v1/status", headers={"x-api-key": TOKEN}).status_code == 200
