"""Each provider's stream put back together into its unstreamed answer: Anthropic Messages
(message_start, content blocks with their deltas, message_delta, message_stop), OpenAI chat
completions (chunks per choice, tool-call fragments by index, the usage chunk, [DONE]) and the
Responses API (the response in response.completed or response.incomplete). Anything else, or a
stream that stops short or carries an error, raises Broken (fixed text, never the provider's).
"""
from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from typing import Any

DONE = "[DONE]"
RESPONSE_END = frozenset({"response.completed", "response.incomplete"})   # both are a finished answer
RESPONSE_FAILED = frozenset({"response.failed", "error"})
STREAMED_INPUT = frozenset({"tool_use", "server_tool_use", "mcp_tool_use"})   # input arrives as JSON text
CHUNK_ONLY = frozenset({"choices", "obfuscation"})   # chat chunk fields with no place in the whole answer


class Broken(ValueError):
    """The stream did not carry a whole answer (cut short, or an error event)."""


def events(data: bytes) -> Iterator[tuple[str, str]]:
    """(event name or "", data) per server-sent event; comments and data-less events skipped."""
    for block in data.decode("utf-8", errors="replace").replace("\r\n", "\n").split("\n\n"):
        name, lines = "", []
        for line in block.split("\n"):
            field, _, value = line.partition(":")
            value = value.removeprefix(" ")
            if field == "event":
                name = value
            elif field == "data":
                lines.append(value)
        if lines:
            yield name, "\n".join(lines)


def _json(text: str) -> dict[str, Any]:
    try:
        got = json.loads(text)
    except ValueError:
        raise Broken("an event that is not JSON") from None
    if not isinstance(got, dict):
        raise Broken("an event that is not an object")
    return got


def anthropic(data: bytes) -> dict[str, Any]:
    message: dict[str, Any] | None = None
    partial: dict[int, str] = {}   # block index -> its input JSON so far
    for _, text in events(data):
        e = _json(text)
        kind = e.get("type")
        if kind == "error":
            raise Broken("the stream carried an error")
        if kind == "message_start":
            message = dict(e["message"])
            message["content"] = []
            continue
        if message is None:
            if kind == "ping":
                continue
            raise Broken("events before message_start")
        if kind == "content_block_start":
            message["content"].append(dict(e["content_block"]))
        elif kind == "content_block_delta":
            _block_delta(message["content"][e["index"]], e["index"], e["delta"], partial)
        elif kind == "content_block_stop":
            block = message["content"][e["index"]]
            if block.get("type") in STREAMED_INPUT and partial.get(e["index"]):
                block["input"] = json.loads(partial[e["index"]])
        elif kind == "message_delta":
            message.update(e.get("delta") or {})
            usage = message.setdefault("usage", {})
            usage.update({k: v for k, v in (e.get("usage") or {}).items() if v is not None})
        elif kind == "message_stop":
            return message
    raise Broken("the stream ended before message_stop")


def _block_delta(block: dict[str, Any], index: int, delta: Mapping[str, Any], partial: dict[int, str]) -> None:
    kind = delta.get("type")
    if kind == "text_delta":
        block["text"] = block.get("text", "") + delta["text"]
    elif kind == "input_json_delta":
        partial[index] = partial.get(index, "") + delta["partial_json"]
    elif kind == "thinking_delta":
        block["thinking"] = block.get("thinking", "") + delta["thinking"]
    elif kind == "signature_delta":
        block["signature"] = block.get("signature", "") + delta["signature"]
    elif kind == "citations_delta":
        block.setdefault("citations", []).append(delta["citation"])


def responses(data: bytes) -> dict[str, Any]:
    for name, text in events(data):
        e = _json(text)
        kind = e.get("type") or name
        if kind in RESPONSE_FAILED:
            raise Broken("the stream carried an error")
        if kind in RESPONSE_END:
            return dict(e["response"])
    raise Broken("the stream ended before response.completed")


def chat(data: bytes) -> dict[str, Any]:
    out: dict[str, Any] | None = None
    choices: dict[int, dict[str, Any]] = {}
    for _, text in events(data):
        if text.strip() == DONE:
            if out is None:
                raise Broken("the stream ended with no chunks")
            out["choices"] = [choices[i] for i in sorted(choices)]
            return out
        c = _json(text)
        if "error" in c:
            raise Broken("the stream carried an error")
        if out is None:
            out = {k: v for k, v in c.items() if k not in CHUNK_ONLY} | {"object": "chat.completion", "usage": None}
        for k in ("usage", "system_fingerprint", "service_tier"):   # the last chunk that says wins
            if c.get(k) is not None:
                out[k] = c[k]
        for ch in c.get("choices") or []:
            _choice(choices.setdefault(ch["index"], {"index": ch["index"], "message": {"role": "assistant",
                    "content": None, "refusal": None}, "logprobs": None, "finish_reason": None}), ch)
    raise Broken("the stream ended before [DONE]")


def _choice(into: dict[str, Any], ch: Mapping[str, Any]) -> None:
    msg = into["message"]
    for k, v in (ch.get("delta") or {}).items():
        if v is None:
            continue
        if k == "tool_calls":
            calls = msg.setdefault("tool_calls", [])
            for d in v:
                _tool_call(calls, d)
        elif k == "role":
            msg["role"] = v
        else:
            msg[k] = _joined(msg.get(k), v)
    if lp := ch.get("logprobs"):
        into["logprobs"] = {k: _joined((into["logprobs"] or {}).get(k), v) for k, v in lp.items()}
    if ch.get("finish_reason") is not None:
        into["finish_reason"] = ch["finish_reason"]


def _tool_call(calls: list[dict[str, Any]], d: Mapping[str, Any]) -> None:
    i = d.get("index", len(calls))
    while len(calls) <= i:
        calls.append({"id": None, "type": "function", "function": {"name": "", "arguments": ""}})
    call = calls[i]
    if d.get("id"):
        call["id"] = d["id"]
    if d.get("type"):
        call["type"] = d["type"]
    fn = d.get("function") or {}
    call["function"]["name"] += fn.get("name") or ""
    call["function"]["arguments"] += fn.get("arguments") or ""


def _joined(old: Any, new: Any) -> Any:
    """Text is appended, lists are extended; anything else is replaced."""
    if isinstance(new, str) and isinstance(old, str):
        return old + new
    if isinstance(new, list):
        return (old or []) + new
    return new
