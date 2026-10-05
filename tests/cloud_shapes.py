"""What the fake cloud answers: each API's whole (unstreamed) answer, and the stream the real
provider sends for that same answer, built from it the way the provider builds it (Anthropic's
message_start / content blocks / message_delta, OpenAI's chat chunks with a usage chunk and
[DONE], the Responses API's events ending in response.completed). Reassembling the stream must
give back the whole answer exactly. `variant`: "text" (one text block) or "rich" (thinking,
text and a tool call; for chat, two tool calls and logprobs)."""
from __future__ import annotations

import copy
import json
from typing import Any

ANTHROPIC, RESPONSES, CHAT = "/v1/messages", "/v1/responses", "/v1/chat/completions"
WEATHER = {"city": "Rome", "days": 2, "units": ["c", "f"]}


def sse(event: str | None, data: dict[str, Any]) -> bytes:
    return ((f"event: {event}\n" if event else "") + f"data: {json.dumps(data)}\n\n").encode()


def halves(text: str) -> list[str]:
    cut = max(1, len(text) // 2)
    return [text[:cut], text[cut:]] if len(text) > 1 else [text]


def whole(path: str, variant: str = "text") -> dict[str, Any]:
    rich = variant == "rich"
    if path == ANTHROPIC:
        content: list[dict[str, Any]] = [{"type": "text", "text": "from the cloud"}]
        if rich:
            content = [{"type": "thinking", "thinking": "the user wants weather", "signature": "EqQBCkYIBRgCKkCsig=="},
                       {"type": "text", "text": "from the cloud", "citations": None},
                       {"type": "tool_use", "id": "toolu_01A", "name": "get_weather", "input": WEATHER}]
        return {"id": "msg_01Cloud", "type": "message", "role": "assistant", "model": "claude-x", "content": content,
                "stop_reason": "tool_use" if rich else "end_turn", "stop_sequence": None,
                "usage": {"input_tokens": 12, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 4,
                          "output_tokens": 31, "service_tier": "standard"}}
    if path == RESPONSES:
        output: list[dict[str, Any]] = [{"id": "msg_r1", "type": "message", "status": "completed", "role": "assistant",
                                         "content": [{"type": "output_text", "text": "from the cloud", "annotations": [],
                                                      "logprobs": []}]}]
        if rich:
            output.append({"id": "fc_r1", "type": "function_call", "status": "completed", "call_id": "call_r1",
                           "name": "get_weather", "arguments": json.dumps(WEATHER)})
        return {"id": "resp_cloud", "object": "response", "created_at": 1700000000, "model": "gpt-x", "status": "completed",
                "error": None, "incomplete_details": None, "instructions": None, "output": output,
                "parallel_tool_calls": True, "tool_choice": "auto", "tools": [], "temperature": 1.0, "top_p": 1.0,
                "usage": {"input_tokens": 9, "input_tokens_details": {"cached_tokens": 0}, "output_tokens": 5,
                          "output_tokens_details": {"reasoning_tokens": 0}, "total_tokens": 14}}
    message: dict[str, Any] = {"role": "assistant", "content": "from the cloud", "refusal": None}
    logprobs = None
    if rich:
        message = {"role": "assistant", "content": None, "refusal": None, "tool_calls": [
            {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": json.dumps(WEATHER)}},
            {"id": "call_2", "type": "function", "function": {"name": "get_time", "arguments": '{"tz": "CET"}'}}]}
        logprobs = {"content": [{"token": "a", "logprob": -0.1, "bytes": [97], "top_logprobs": []},
                                {"token": "b", "logprob": -0.2, "bytes": [98], "top_logprobs": []}], "refusal": None}
    return {"id": "chatcmpl-cloud", "object": "chat.completion", "created": 1700000000, "model": "gpt-x",
            "choices": [{"index": 0, "message": message, "logprobs": logprobs,
                         "finish_reason": "tool_calls" if rich else "stop"}],
            "usage": {"prompt_tokens": 9, "completion_tokens": 3, "total_tokens": 12,
                      "prompt_tokens_details": {"cached_tokens": 0, "audio_tokens": 0},
                      "completion_tokens_details": {"reasoning_tokens": 0, "audio_tokens": 0}},
            "service_tier": "default", "system_fingerprint": "fp_cloud"}


def stream(path: str, variant: str = "text") -> list[bytes]:
    w = whole(path, variant)
    if path == ANTHROPIC:
        return _anthropic(w)
    if path == RESPONSES:
        return _responses(w)
    return _chat(w)


def _anthropic(w: dict[str, Any]) -> list[bytes]:
    start = copy.deepcopy(w) | {"content": [], "stop_reason": None, "stop_sequence": None}
    start["usage"] = w["usage"] | {"output_tokens": 1}
    out = [sse("message_start", {"type": "message_start", "message": start}), sse("ping", {"type": "ping"})]
    for i, block in enumerate(w["content"]):
        kind = block["type"]
        empty = {"text": {"type": "text", "text": ""} | ({"citations": None} if "citations" in block else {}),
                 "thinking": {"type": "thinking", "thinking": "", "signature": ""},
                 "tool_use": {"type": "tool_use", "id": block.get("id"), "name": block.get("name"), "input": {}}}[kind]
        out.append(sse("content_block_start", {"type": "content_block_start", "index": i, "content_block": empty}))
        if kind == "text":
            deltas = [{"type": "text_delta", "text": t} for t in halves(block["text"])]
        elif kind == "thinking":
            deltas = [{"type": "thinking_delta", "thinking": t} for t in halves(block["thinking"])]
            deltas.append({"type": "signature_delta", "signature": block["signature"]})
        else:
            deltas = [{"type": "input_json_delta", "partial_json": t} for t in ["", *halves(json.dumps(block["input"]))]]
        out += [sse("content_block_delta", {"type": "content_block_delta", "index": i, "delta": d}) for d in deltas]
        out.append(sse("content_block_stop", {"type": "content_block_stop", "index": i}))
    out.append(sse("message_delta", {"type": "message_delta", "delta": {"stop_reason": w["stop_reason"],
               "stop_sequence": w["stop_sequence"]}, "usage": {"output_tokens": w["usage"]["output_tokens"]}}))
    return [*out, sse("message_stop", {"type": "message_stop"})]


def _chat(w: dict[str, Any]) -> list[bytes]:
    head = {k: w[k] for k in ("id", "created", "model", "service_tier", "system_fingerprint")} | {"object": "chat.completion.chunk"}
    choice, msg = w["choices"][0], w["choices"][0]["message"]

    def chunk(delta: dict[str, Any], finish: str | None = None, logprobs: Any = None) -> bytes:
        c = head | {"choices": [{"index": 0, "delta": delta, "logprobs": logprobs, "finish_reason": finish}],
                    "usage": None, "obfuscation": "pad"}
        return sse(None, c)

    out = []
    if msg.get("tool_calls"):
        out.append(chunk({"role": "assistant", "content": None, "refusal": None}))
        for i, call in enumerate(msg["tool_calls"]):
            out.append(chunk({"tool_calls": [{"index": i, "id": call["id"], "type": "function",
                                              "function": {"name": call["function"]["name"], "arguments": ""}}]}))
            out += [chunk({"tool_calls": [{"index": i, "function": {"arguments": a}}]})
                    for a in halves(call["function"]["arguments"])]
    else:
        out.append(chunk({"role": "assistant", "content": "", "refusal": None}))
        out += [chunk({"content": t}) for t in halves(msg["content"])]
    if lp := choice["logprobs"]:
        out += [chunk({}, logprobs={"content": [t], "refusal": None}) for t in lp["content"]]
    out.append(chunk({}, choice["finish_reason"]))
    out.append(sse(None, head | {"choices": [], "usage": w["usage"], "obfuscation": "pad"}))
    return [*out, b"data: [DONE]\n\n"]


def _responses(w: dict[str, Any]) -> list[bytes]:
    seq = iter(range(1000))

    def ev(kind: str, **data: Any) -> bytes:
        return sse(kind, {"type": kind, "sequence_number": next(seq), **data})

    early = w | {"status": "in_progress", "output": [], "usage": None}
    out = [ev("response.created", response=early), ev("response.in_progress", response=early)]
    for i, item in enumerate(w["output"]):
        out.append(ev("response.output_item.added", output_index=i, item=item | {"status": "in_progress"}))
        if item["type"] == "message":
            out += [ev("response.output_text.delta", output_index=i, item_id=item["id"], content_index=0, delta=t, logprobs=[])
                    for t in halves(item["content"][0]["text"])]
        out.append(ev("response.output_item.done", output_index=i, item=item))
    return [*out, ev("response.completed", response=w)]
