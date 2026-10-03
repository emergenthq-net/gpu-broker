"""OpenAI chat results -> Anthropic Messages shapes: the message, stop reasons, usage, ids, the
model list and errors.

Stop reasons: stop -> end_turn, length -> max_tokens, tool_calls -> tool_use, and tool_use
whenever the answer carries a tool_use block, whatever the server's finish_reason (some
servers say "stop" after a tool call; the SDKs' tool loops key on tool_use). A server that
reports which stop string matched (llama.cpp's `stopping_word`) gives stop_sequence. A
`reasoning_content` from the server becomes a `thinking` block only when the request
enabled thinking; otherwise it is dropped, as Anthropic does.
"""
from __future__ import annotations

import json
import secrets
from collections.abc import Mapping
from http import HTTPStatus
from typing import Any

from ..catalog import Catalog
from .openai import model_list

TOOL_USE = "tool_use"
STOP_REASONS = {"stop": "end_turn", "length": "max_tokens", "tool_calls": TOOL_USE, "function_call": TOOL_USE}
DEFAULT_STOP = "end_turn"
STOP_SEQUENCE = "stop_sequence"
STOPPING_WORD = "stopping_word"
MSG_PREFIX, TOOL_PREFIX = "msg_", "toolu_"
ID_BYTES = 12
RAW_ARGUMENTS = "_raw_arguments"   # tool input the model produced that is not a JSON object
NO_SIGNATURE = ""                  # local reasoning carries no Anthropic signature
MODEL_CREATED = "1970-01-01T00:00:00Z"
ERROR_TYPES: dict[int, str] = {HTTPStatus.BAD_REQUEST: "invalid_request_error", HTTPStatus.UNAUTHORIZED: "authentication_error",
               HTTPStatus.FORBIDDEN: "permission_error", HTTPStatus.NOT_FOUND: "not_found_error",
               HTTPStatus.REQUEST_ENTITY_TOO_LARGE: "request_too_large", HTTPStatus.TOO_MANY_REQUESTS: "rate_limit_error",
               HTTPStatus.SERVICE_UNAVAILABLE: "overloaded_error", HTTPStatus.UNPROCESSABLE_ENTITY: "invalid_request_error"}
DEFAULT_ERROR = "api_error"


def new_id(prefix: str) -> str:
    return prefix + secrets.token_hex(ID_BYTES)


def stop_reason(finish: str | None, stopping_word: str | None, tool_used: bool = False) -> tuple[str, str | None]:
    """(stop_reason, stop_sequence) for an OpenAI finish_reason; `tool_used` = a tool_use block was emitted."""
    if tool_used:
        return TOOL_USE, None
    if stopping_word:
        return STOP_SEQUENCE, stopping_word
    return STOP_REASONS.get(finish or "", DEFAULT_STOP), None


def usage(raw: Mapping[str, Any] | None, timings: Mapping[str, Any] | None = None) -> dict[str, int]:
    """Token counts from OpenAI `usage`, else llama.cpp `timings` (prompt_n / predicted_n).
    A count that is missing, null or not a number is 0."""
    u = raw if isinstance(raw, Mapping) else {}
    t = timings if isinstance(timings, Mapping) else {}

    def count(*values: Any) -> int:
        return next((int(v) for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)), 0)
    return {"input_tokens": count(u.get("prompt_tokens"), t.get("prompt_n")),
            "output_tokens": count(u.get("completion_tokens"), t.get("predicted_n"))}


def arguments_text(arguments: Any) -> str:
    """Tool-call arguments as the JSON text Anthropic streams in `partial_json` (always a string)."""
    if arguments is None:
        return ""
    return arguments if isinstance(arguments, str) else json.dumps(arguments)


def tool_input(arguments: Any) -> dict[str, Any]:
    if isinstance(arguments, Mapping):
        return dict(arguments)
    try:
        parsed = json.loads(arguments or "{}")
    except ValueError:
        parsed = None
    return parsed if isinstance(parsed, dict) else {RAW_ARGUMENTS: arguments}


def message(result: Mapping[str, Any], requested: str, thinking: bool, meta: Mapping[str, Any]) -> dict[str, Any]:
    """A finished OpenAI chat completion as an Anthropic `message`."""
    choice = (result.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    blocks: list[dict[str, Any]] = []
    if thinking and msg.get("reasoning_content"):
        blocks.append({"type": "thinking", "thinking": msg["reasoning_content"], "signature": NO_SIGNATURE})
    if msg.get("content"):
        blocks.append({"type": "text", "text": msg["content"]})
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function") or {}
        blocks.append({"type": TOOL_USE, "id": tc.get("id") or new_id(TOOL_PREFIX), "name": fn.get("name", ""),
                       "input": tool_input(fn.get("arguments"))})
    tool_used = any(b["type"] == TOOL_USE for b in blocks)
    reason, seq = stop_reason(choice.get("finish_reason"), result.get(STOPPING_WORD) or choice.get(STOPPING_WORD), tool_used)
    return {"id": new_id(MSG_PREFIX), "type": "message", "role": "assistant", "model": requested, "content": blocks,
            "stop_reason": reason, "stop_sequence": seq, "usage": usage(result.get("usage"), result.get("timings")),
            "x_broker": dict(meta)}


def models(catalog: Catalog) -> dict[str, Any]:
    """The ready LLMs in Anthropic's list shape (for clients that send `anthropic-version`)."""
    ids = [m["id"] for m in model_list(catalog)["data"]]
    data = [{"type": "model", "id": i, "display_name": i, "created_at": MODEL_CREATED} for i in ids]
    return {"data": data, "has_more": False, "first_id": ids[0] if ids else None, "last_id": ids[-1] if ids else None}


def error(status: int, text: str) -> dict[str, Any]:
    """Anthropic's error body."""
    kind = ERROR_TYPES.get(status, DEFAULT_ERROR)
    return {"type": "error", "error": {"type": kind, "message": text}}
