"""Anthropic Messages request -> OpenAI chat request, for the broker's OpenAI-compatible path.

Translated: `system` (string or text blocks) -> a system message; text and image blocks
(base64 -> `data:` URI, url -> its URL); `tool_use` blocks -> assistant `tool_calls`;
`tool_result` blocks -> `role: tool` messages (an OpenAI tool message holds text only, so
images a tool returned follow in a user message, labelled with the call they came from);
`tools` / `tool_choice`; `max_tokens`, `temperature`, `top_p`, `top_k`, `stop_sequences` ->
`stop`. `thinking` blocks the client sends back are dropped (the model regenerates its
reasoning). The request's shape is checked as it is translated: a malformed field, or
anything a local model cannot do (hosted server tools, documents, citations), is a
ValueError, which the route answers as a 400 invalid_request_error, never silently ignored.
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from .anthropic_tools import scalars, text_field, tools

PASSTHROUGH = {"model": "model", "max_tokens": "max_tokens", "temperature": "temperature",
               "top_p": "top_p", "top_k": "top_k", "stop_sequences": "stop"}
TEXT, IMAGE, TOOL_USE, TOOL_RESULT = "text", "image", "tool_use", "tool_result"
DROPPED_BLOCKS = frozenset({"thinking", "redacted_thinking"})
THINKING_ON = frozenset({"enabled", "adaptive"})
TOOL_ERROR_PREFIX = "Error: "   # a tool_result with is_error: true, as the model sees it
BLOCK_JOIN = "\n"
IMAGE_TYPES = frozenset({"image/jpeg", "image/png", "image/gif", "image/webp"})
TOOL_IMAGES = "Images returned by tool call {id}:"


def thinking_enabled(body: Mapping[str, Any]) -> bool:
    t = body.get("thinking")
    return isinstance(t, Mapping) and t.get("type") in THINKING_ON


def _blocks(content: Any, where: str) -> list[Mapping[str, Any]]:
    if isinstance(content, str):
        return [{"type": TEXT, "text": content}]
    if not (isinstance(content, list) and all(isinstance(b, Mapping) for b in content)):
        raise ValueError(f"{where}: content must be a string or a list of content blocks")
    return content


def _image(block: Mapping[str, Any]) -> dict[str, Any]:
    src = block.get("source")
    if not isinstance(src, Mapping):
        raise ValueError("image: `source` must be an object")
    if src.get("type") == "base64":
        if src.get("media_type") not in IMAGE_TYPES:
            raise ValueError(f"image: media_type must be one of {sorted(IMAGE_TYPES)}, not {src.get('media_type')!r}")
        url = f"data:{src['media_type']};base64,{text_field(src.get('data'), 'image: source.data')}"
    elif src.get("type") == "url":
        url = text_field(src.get("url"), "image: source.url")
    else:
        raise ValueError(f"image source type {src.get('type')!r} is not supported (use base64 or url)")
    return {"type": "image_url", "image_url": {"url": url}}


def _parts(blocks: list[Mapping[str, Any]], where: str) -> list[dict[str, Any]]:
    """Text and image blocks as OpenAI content parts."""
    parts: list[dict[str, Any]] = []
    for b in blocks:
        kind = b.get("type")
        if kind == TEXT:
            parts.append({"type": TEXT, TEXT: text_field(b.get(TEXT), f"{where}: a text block's `text`")})
        elif kind == IMAGE:
            parts.append(_image(b))
        elif kind not in DROPPED_BLOCKS:
            raise ValueError(f"{where}: content block type {kind!r} is not supported here")
    return parts


def _content(parts: list[dict[str, Any]]) -> str | list[dict[str, Any]]:
    """Text-only content as one string (every chat template reads that); with images, the parts."""
    if all(p["type"] == TEXT for p in parts):
        return BLOCK_JOIN.join(p[TEXT] for p in parts)
    return parts


def _tool_result(b: Mapping[str, Any], where: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """The tool message, and the image parts it carried (an OpenAI tool message is text only)."""
    call = text_field(b.get("tool_use_id"), f"{where}: tool_result `tool_use_id`")
    parts = _parts(_blocks(b.get("content", ""), f"{where}: tool_result"), f"{where}: tool_result")
    text = BLOCK_JOIN.join(p[TEXT] for p in parts if p["type"] == TEXT)
    images = [p for p in parts if p["type"] != TEXT]
    labelled = [{"type": TEXT, TEXT: TOOL_IMAGES.format(id=call)}, *images] if images else []
    return {"role": "tool", "tool_call_id": call, "content": (TOOL_ERROR_PREFIX + text) if b.get("is_error") else text}, labelled


def _user(blocks: list[Mapping[str, Any]], where: str) -> list[dict[str, Any]]:
    """Tool results first (they answer the previous assistant turn), then one user message:
    any images the tools returned, then the user's own content."""
    results = [_tool_result(b, where) for b in blocks if b.get("type") == TOOL_RESULT]
    out = [msg for msg, _ in results]
    rest = [p for _, images in results for p in images] + _parts([b for b in blocks if b.get("type") != TOOL_RESULT], where)
    if rest:
        out.append({"role": "user", "content": _content(rest)})
    return out


def _call(b: Mapping[str, Any], where: str) -> dict[str, Any]:
    args = b.get("input", {})
    if not isinstance(args, Mapping):
        raise ValueError(f"{where}: tool_use `input` must be an object")
    return {"id": text_field(b.get("id"), f"{where}: tool_use `id`"), "type": "function",
            "function": {"name": text_field(b.get("name"), f"{where}: tool_use `name`"), "arguments": json.dumps(args)}}


def _assistant(blocks: list[Mapping[str, Any]], where: str) -> dict[str, Any]:
    calls = [_call(b, where) for b in blocks if b.get("type") == TOOL_USE]
    text = _content(_parts([b for b in blocks if b.get("type") != TOOL_USE], where))
    msg: dict[str, Any] = {"role": "assistant", "content": text or None}
    if calls:
        msg["tool_calls"] = calls
    return msg


def _messages(body: Mapping[str, Any]) -> list[dict[str, Any]]:
    msgs = body.get("messages")
    if not isinstance(msgs, list) or not msgs:
        raise ValueError("`messages` must be a non-empty list")
    out: list[dict[str, Any]] = []
    system = body.get("system")
    if system:
        out.append({"role": "system", "content": _content(_parts(_blocks(system, "system"), "system"))})
    for i, m in enumerate(msgs):
        where = f"messages[{i}]"
        role = m.get("role") if isinstance(m, Mapping) else None
        if role not in ("user", "assistant"):
            raise ValueError(f"{where}: role must be 'user' or 'assistant'")
        blocks = _blocks(m.get("content"), where)
        out.extend(_user(blocks, where) if role == "user" else [_assistant(blocks, where)])
    return out


def to_openai(body: Mapping[str, Any]) -> dict[str, Any]:
    """The OpenAI chat request for an Anthropic Messages request (ValueError on what cannot map)."""
    scalars(body)
    out: dict[str, Any] = {"messages": _messages(body)}
    for src, dst in PASSTHROUGH.items():
        if body.get(src) is not None:
            out[dst] = body[src]
    tools(body, out)
    if body.get("stream"):
        out.update(stream=True, stream_options={"include_usage": True})
    return out
