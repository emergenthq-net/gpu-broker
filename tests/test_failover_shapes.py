"""The assemblers below the routes: SSE framing, each API's stream back into its whole answer
(including what the end-to-end fakes do not send: several choices, citations, an empty tool
input), what counts as no whole answer, and how a started stream is gathered."""
from __future__ import annotations

import json

import pytest

from gpu_broker.failover import assemble, shapes, transport
from tests import cloud_shapes
from tests.cloud_shapes import sse


def joined(path, variant="text"):
    return b"".join(cloud_shapes.stream(path, variant))


@pytest.mark.parametrize("path", [cloud_shapes.ANTHROPIC, cloud_shapes.CHAT, cloud_shapes.RESPONSES])
@pytest.mark.parametrize("variant", ["text", "rich"])
def test_every_stream_reassembles_to_its_whole_answer(path, variant):
    assert assemble.assemble(path, joined(path, variant)) == cloud_shapes.whole(path, variant)


def test_as_stream_adds_only_stream_and_chat_usage():
    body = {"model": "m", "messages": [], "stream": False}
    assert assemble.as_stream("/v1/chat/completions", body) == body | {"stream": True, "stream_options": {"include_usage": True}}
    assert assemble.as_stream("/v1/messages", body) == body | {"stream": True}
    assert assemble.as_stream("/v1/responses", body) == body | {"stream": True}
    assert body["stream"] is False   # the caller's body is not changed


def test_events_framing():
    data = b": comment\r\n\r\nevent: a\r\ndata: x\r\ndata:y\r\n\r\nid: 3\n\nevent: b\ndata: {}\n\ndata: \xffz\n\n"
    assert list(shapes.events(data)) == [("a", "x\ny"), ("b", "{}"), ("", "\ufffdz")]   # bad bytes replaced, not fatal


def test_anthropic_citations_empty_tool_input_and_late_usage():
    start = {"id": "m1", "type": "message", "role": "assistant", "model": "c", "content": [], "stop_reason": None,
             "stop_sequence": None, "usage": {"input_tokens": 5, "output_tokens": 1, "cache_read_input_tokens": 2}}
    cite = {"type": "char_location", "cited_text": "x", "document_index": 0, "start_char_index": 0, "end_char_index": 1}
    data = b"".join([
        sse("message_start", {"type": "message_start", "message": start}),
        sse("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
        sse("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "citations_delta", "citation": cite}}),
        sse("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "hi"}}),
        sse("content_block_stop", {"type": "content_block_stop", "index": 0}),
        sse("content_block_start", {"type": "content_block_start", "index": 1,
                                    "content_block": {"type": "tool_use", "id": "t", "name": "now", "input": {}}}),
        sse("content_block_delta", {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": ""}}),
        sse("content_block_stop", {"type": "content_block_stop", "index": 1}),
        sse("message_delta", {"type": "message_delta", "delta": {"stop_reason": "tool_use", "stop_sequence": None},
                              "usage": {"output_tokens": 9, "input_tokens": None, "cache_read_input_tokens": 7}}),
        sse("message_stop", {"type": "message_stop"})])
    m = assemble.assemble("/v1/messages", data)
    assert m["content"] == [{"type": "text", "text": "hi", "citations": [cite]},
                            {"type": "tool_use", "id": "t", "name": "now", "input": {}}]
    assert m["usage"] == {"input_tokens": 5, "output_tokens": 9, "cache_read_input_tokens": 7}   # None never overwrites
    assert m["stop_reason"] == "tool_use"


def test_chat_several_choices_in_index_order_and_no_usage_chunk():
    head = {"id": "c", "object": "chat.completion.chunk", "created": 1, "model": "g"}
    ch = lambda i, d, f=None: sse(None, head | {"choices": [{"index": i, "delta": d, "finish_reason": f}]})  # noqa: E731
    data = b"".join([ch(1, {"role": "assistant", "content": "B"}), ch(0, {"role": "assistant", "content": "A"}),
                     ch(0, {"content": "a"}, "stop"), ch(1, {}, "length"), b"data: [DONE]\n\n"])
    out = assemble.assemble("/v1/chat/completions", data)
    assert [(c["index"], c["message"]["content"], c["finish_reason"]) for c in out["choices"]] == [(0, "Aa", "stop"), (1, "B", "length")]
    assert out["object"] == "chat.completion" and out["usage"] is None and "obfuscation" not in out


def test_anthropic_blocks_that_start_without_their_text_fields():
    start = {"id": "m1", "type": "message", "role": "assistant", "model": "c", "content": []}
    delta = lambda i, d: sse("content_block_delta", {"type": "content_block_delta", "index": i, "delta": d})  # noqa: E731
    data = b"".join([
        sse("message_start", {"type": "message_start", "message": start}),
        sse("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking"}}),
        delta(0, {"type": "thinking_delta", "thinking": "hm"}),
        delta(0, {"type": "signature_delta", "signature": "si"}),
        delta(0, {"type": "signature_delta", "signature": "g"}),
        sse("content_block_start", {"type": "content_block_start", "index": 1, "content_block": {"type": "text"}}),
        delta(1, {"type": "text_delta", "text": "hi"}),
        sse("message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 3}}),
        sse("message_stop", {"type": "message_stop"})])
    m = assemble.assemble("/v1/messages", data)
    assert m["content"] == [{"type": "thinking", "thinking": "hm", "signature": "sig"}, {"type": "text", "text": "hi"}]
    assert m["usage"] == {"output_tokens": 3}   # no usage in message_start: the late usage alone


def test_chat_defaults_and_late_top_level_fields():
    head = {"id": "c", "object": "chat.completion.chunk", "created": 1, "model": "g"}
    calls = [{"index": 0, "id": "x", "function": {"name": "f", "arguments": "{"}},
             {"index": 1, "id": "y", "type": "custom", "function": {"name": "g"}}]
    data = b"".join([sse(None, head | {"choices": [{"index": 0, "delta": {"tool_calls": calls}}]}),
                     sse(None, head | {"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": "}"}}]},
                                                    "finish_reason": "tool_calls"}]}),
                     sse(None, head | {"choices": [], "system_fingerprint": "fp", "service_tier": "default",
                                       "usage": {"total_tokens": 4}}),
                     b"data: [DONE]\n\n"])
    out = assemble.assemble("/v1/chat/completions", data)
    assert (out["system_fingerprint"], out["service_tier"], out["usage"]) == ("fp", "default", {"total_tokens": 4})
    msg = out["choices"][0]["message"]
    assert msg["role"] == "assistant"   # no chunk said so
    assert msg["tool_calls"] == [{"id": "x", "type": "function", "function": {"name": "f", "arguments": "{}"}},
                                 {"id": "y", "type": "custom", "function": {"name": "g", "arguments": ""}}]


def test_chat_tool_call_fragments_without_an_index_follow_on():
    head = {"id": "c", "object": "chat.completion.chunk", "created": 1, "model": "g"}
    d = {"role": "assistant", "content": None, "tool_calls": [{"id": "x", "type": "function", "function": {"name": "f", "arguments": "{}"}}]}
    data = sse(None, head | {"choices": [{"index": 0, "delta": d, "finish_reason": "tool_calls"}]}) + b"data: [DONE]\n\n"
    msg = assemble.assemble("/v1/chat/completions", data)["choices"][0]["message"]
    assert msg["content"] is None and msg["tool_calls"] == [{"id": "x", "type": "function", "function": {"name": "f", "arguments": "{}"}}]


def test_responses_incomplete_is_a_whole_answer_and_failed_is_not():
    resp = cloud_shapes.whole(cloud_shapes.RESPONSES) | {"status": "incomplete"}
    assert assemble.assemble("/v1/responses", sse("response.incomplete", {"type": "response.incomplete", "response": resp})) == resp
    assert assemble.assemble("/v1/responses", sse(None, {"type": "response.completed", "response": resp})) == resp   # type alone
    with pytest.raises(shapes.Broken):
        assemble.assemble("/v1/responses", sse("response.failed", {"type": "response.failed", "response": resp}))


ERROR = "the stream carried an error"


@pytest.mark.parametrize(("path", "data", "why"), [
    ("/v1/messages", lambda: joined("/v1/messages").rsplit(b"event: message_stop", 1)[0], "the stream ended before message_stop"),
    ("/v1/messages", lambda: sse("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {}}),
     "events before message_start"),
    # each error is followed by a whole answer, so only noticing the error can make it Broken
    ("/v1/messages", lambda: sse("error", {"type": "error", "error": {"type": "overloaded_error", "message": "x"}})
     + joined("/v1/messages"), ERROR),
    ("/v1/messages", lambda: b"event: message_start\ndata: {not json\n\n", "an event that is not JSON"),
    ("/v1/messages", lambda: b"data: [1, 2]\n\n", "an event that is not an object"),
    ("/v1/chat/completions", lambda: joined("/v1/chat/completions").replace(b"data: [DONE]\n\n", b""), "the stream ended before [DONE]"),
    ("/v1/chat/completions", lambda: b"data: [DONE]\n\n", "the stream ended with no chunks"),
    ("/v1/chat/completions", lambda: sse(None, {"error": {"message": "x"}}) + joined("/v1/chat/completions"), ERROR),
    ("/v1/chat/completions", lambda: sse(None, {"id": "c", "choices": [{"delta": {}}]}) + b"data: [DONE]\n\n", "a malformed event"),
    ("/v1/responses", lambda: sse("response.created", {"type": "response.created", "response": {}}),
     "the stream ended before response.completed"),
    ("/v1/responses", lambda: sse(None, {"type": "error", "message": "x"}) + joined("/v1/responses"), ERROR),   # type alone
    ("/v1/responses", lambda: sse("error", {"message": "x"}) + joined("/v1/responses"), ERROR),   # event name alone
])
def test_no_whole_answer_is_broken(path, data, why):
    with pytest.raises(shapes.Broken) as e:
        assemble.assemble(path, data())
    assert str(e.value) == why


def test_anthropic_pings_before_message_start_are_skipped():
    assert assemble.assemble("/v1/messages", sse("ping", {"type": "ping"}) + joined("/v1/messages")) == \
        cloud_shapes.whole("/v1/messages")


def answer(body, ctype="text/event-stream", rest=(), **extra):
    closed = []
    a = transport.Answer(200, {"content-type": ctype, "content-length": "9", "transfer-encoding": "chunked",
                               "request-id": "req_1", **extra}, body, iter(rest), stream=True, close=lambda: closed.append(1))
    return a, closed


def test_gather_reassembles_and_fixes_the_headers():
    pieces = cloud_shapes.stream("/v1/messages")
    a, _ = answer(pieces[0], rest=pieces[1:])
    got = assemble.gather("/v1/messages", a)
    assert json.loads(got.body) == cloud_shapes.whole("/v1/messages") and not got.stream
    assert got.headers == {"content-type": "application/json", "request-id": "req_1"}


def test_gather_passes_a_whole_json_answer_through():
    a, _ = answer(b'{"id": 1', ctype="application/json; charset=utf-8", rest=[b"}"])
    got = assemble.gather("/v1/chat/completions", a)
    assert got.body == b'{"id": 1}' and got.headers["content-type"] == "application/json; charset=utf-8"


def test_gather_refuses_an_answer_too_long_and_closes_it(monkeypatch):
    monkeypatch.setattr(assemble, "MAX_BYTES", 10)
    a, closed = answer(b"data: 12345\n\n", rest=[b"data: 6\n\n"])
    with pytest.raises(transport.Unreachable, match="too long"):
        assemble.gather("/v1/messages", a)
    assert closed == [1]


def test_gather_turns_a_broken_stream_into_unreachable_with_fixed_text():
    a, _ = answer(sse("error", {"type": "error", "error": {"type": "x", "message": "SECRET-ECHO"}}))
    with pytest.raises(transport.Unreachable) as e:
        assemble.gather("/v1/messages", a)
    assert "SECRET-ECHO" not in str(e.value)
