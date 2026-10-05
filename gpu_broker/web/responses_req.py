"""OpenAI Responses request -> OpenAI chat request, for the broker's chat path (Codex and the
Responses SDK).

Translated: `input` as a string (one user message) or a list of items: `message` items
(input_text / output_text parts; input_image -> an image_url part), `function_call` and
`custom_tool_call` items -> assistant `tool_calls` (consecutive calls join one assistant
message; a custom call's text is the `input` argument), `function_call_output` and
`custom_tool_call_output` items -> `role: tool` messages (images a tool returned follow in a
user message, as on /v1/messages). `instructions` and every system / developer message, from
this request and from a stored conversation alike, become ONE system message at the front:
many chat templates accept a single leading system message and nothing else. `reasoning`
items are dropped: the model regenerates its reasoning, and a local model has no encrypted
reasoning to resume. Function tools, `tool_choice`, `parallel_tool_calls`, `temperature`,
`top_p`, `max_output_tokens` and `text.format` (JSON schema / JSON object) are mapped.

Hosted tools (`web_search`, `file_search`, ...) and `namespace` tool groups cannot run on a
local model. Codex sends both on every request, so refusing them would make Codex unusable:
they are left out, and their names are reported in `x_broker.dropped_tools` (see
`dropped_tools`). Anything else malformed, or a feature a local model cannot provide (input
files, prompt templates, background mode), is a ValueError, answered as a 400.
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from .anthropic_req import BLOCK_JOIN, TOOL_IMAGES
from .anthropic_tools import text_field
from .responses_tools import INPUT, format_, label, scalars, tools

TEXT_PARTS = frozenset({"input_text", "output_text", "text", "summary_text"})
IMAGE_PART, REFUSAL = "input_image", "refusal"
MESSAGE, CALL, OUTPUT, REASONING = "message", "function_call", "function_call_output", "reasoning"
CUSTOM_CALL, CUSTOM_OUTPUT = "custom_tool_call", "custom_tool_call_output"
DROPPED_ITEMS = frozenset({REASONING})
ROLES = {"user": "user", "assistant": "assistant", "system": "system", "developer": "system"}


def _text_parts(content: Any, where: str) -> list[dict[str, Any]]:
    """A message's content (a string or a list of parts) as chat content parts."""
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if not (isinstance(content, list) and all(isinstance(p, Mapping) for p in content)):
        raise ValueError(f"{where}: content must be a string or a list of content parts")
    parts: list[dict[str, Any]] = []
    for p in content:
        kind = label(p.get("type"), f"{where}: a content part's `type`")
        if kind in TEXT_PARTS:
            parts.append({"type": "text", "text": text_field(p.get("text"), f"{where}: a {kind} part's `text`")})
        elif kind == REFUSAL:
            parts.append({"type": "text", "text": text_field(p.get(REFUSAL), f"{where}: a refusal part's `refusal`")})
        elif kind == IMAGE_PART:
            url = p.get("image_url")
            if not isinstance(url, str) or not url:
                raise ValueError(f"{where}: input_image needs `image_url` (a URL or data: URI); file_id is not supported")
            parts.append({"type": "image_url", "image_url": {"url": url}})
        else:
            raise ValueError(f"{where}: content part type {kind!r} is not supported here")
    return parts


def _content(parts: list[dict[str, Any]]) -> str | list[dict[str, Any]]:
    if all(p["type"] == "text" for p in parts):
        return BLOCK_JOIN.join(p["text"] for p in parts)
    return parts


def _arguments(value: Any, where: str) -> str:
    return value if isinstance(value, str) else json.dumps(text_field(value, f"{where}: `arguments`"))


class _Builder:
    """Chat messages in order, joining consecutive function calls into one assistant turn."""

    def __init__(self) -> None:
        self.out: list[dict[str, Any]] = []

    def _assistant(self) -> dict[str, Any]:
        last = self.out[-1] if self.out else None
        if last is not None and last["role"] == "assistant" and "tool_calls" in last:
            return last
        if last is not None and last["role"] == "assistant":   # text then calls: one assistant turn
            last["tool_calls"] = []
            return last
        msg: dict[str, Any] = {"role": "assistant", "content": None, "tool_calls": []}
        self.out.append(msg)
        return msg

    def message(self, item: Mapping[str, Any], where: str) -> None:
        role = ROLES.get(label(item.get("role"), f"{where}: `role`") or "")
        if role is None:
            raise ValueError(f"{where}: role must be one of {sorted(ROLES)}")
        parts = _text_parts(item.get("content"), where)
        if role != "user" and any(p["type"] != "text" for p in parts):
            raise ValueError(f"{where}: images are only accepted in user messages")
        self.out.append({"role": role, "content": _content(parts) or ("" if role != "assistant" else None)})

    def call(self, item: Mapping[str, Any], where: str) -> None:
        call_id = text_field(item.get("call_id"), f"{where}: `call_id`")
        name = text_field(item.get("name"), f"{where}: `name`")
        if item.get("type") == CUSTOM_CALL:
            args = json.dumps({INPUT: text_field(item.get(INPUT, ""), f"{where}: `input`")})
        else:
            args = _arguments(item.get("arguments", "{}"), where)
        self._assistant()["tool_calls"].append({"id": call_id, "type": "function", "function": {"name": name, "arguments": args}})

    def output(self, item: Mapping[str, Any], where: str) -> None:
        call_id = text_field(item.get("call_id"), f"{where}: `call_id`")
        parts = _text_parts(item.get("output", ""), where)
        text = BLOCK_JOIN.join(p["text"] for p in parts if p["type"] == "text")
        self.out.append({"role": "tool", "tool_call_id": call_id, "content": text})
        if images := [p for p in parts if p["type"] != "text"]:
            self.out.append({"role": "user", "content": [{"type": "text", "text": TOOL_IMAGES.format(id=call_id)}, *images]})


def messages(items: Any) -> list[dict[str, Any]]:
    """Responses `input` (a string or a list of items) as chat messages."""
    if isinstance(items, str):
        return [{"role": "user", "content": items}]
    if not isinstance(items, list):
        raise ValueError("`input` must be a string or a list of items")
    b = _Builder()
    for i, item in enumerate(items):
        where = f"input[{i}]"
        if not isinstance(item, Mapping):
            raise ValueError(f"{where} must be an object")
        kind = label(item.get("type", MESSAGE if "role" in item else None), f"{where}: `type`")
        if kind == MESSAGE:
            b.message(item, where)
        elif kind in (CALL, CUSTOM_CALL):
            b.call(item, where)
        elif kind in (OUTPUT, CUSTOM_OUTPUT):
            b.output(item, where)
        elif kind not in DROPPED_ITEMS:
            raise ValueError(f"{where}: item type {kind!r} is not supported here")
    return b.out


def leading_system(instructions: str | None, conversation: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The conversation with `instructions` and every system message joined into one, first."""
    system = [m["content"] for m in conversation if m["role"] == "system" and m["content"]]
    text = BLOCK_JOIN.join([instructions, *system] if instructions else system)
    rest = [m for m in conversation if m["role"] != "system"]
    return [{"role": "system", "content": text}, *rest] if text else rest


def to_openai(body: Mapping[str, Any], history: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]], frozenset[str]]:
    """(the chat request, this turn's input as chat messages, the custom tools' names).
    `history` is the conversation a `previous_response_id` named, kept without its
    instructions (OpenAI does not carry them over) but with its developer messages."""
    if "input" not in body:
        raise ValueError("`input` is required")
    turn = messages(body["input"])
    instructions = body.get("instructions")
    if instructions is not None and not isinstance(instructions, str):
        raise ValueError("`instructions` must be a string")
    out: dict[str, Any] = {"messages": leading_system(instructions, [*history, *turn])}
    if body.get("model") is not None:
        out["model"] = text_field(body["model"], "`model`")
    scalars(body, out)
    custom = tools(body, out)
    format_(body, out)
    if body.get("stream"):
        out.update(stream=True, stream_options={"include_usage": True})
    return out, turn, custom
