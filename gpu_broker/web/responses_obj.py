"""The Responses API `response` object: its settings echoed from the request, and usage."""
from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any

from .anthropic_resp import usage

RESP = "resp_"


def response_usage(raw: Mapping[str, Any] | None, timings: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Token counts in the Responses shape (from OpenAI `usage`, else llama.cpp `timings`)."""
    u = usage(raw, timings)
    details = raw.get("completion_tokens_details") if isinstance(raw, Mapping) else None
    reasoning = details.get("reasoning_tokens") if isinstance(details, Mapping) else None
    cached = (raw.get("prompt_tokens_details") or {}).get("cached_tokens") if isinstance(raw, Mapping) else None
    return {"input_tokens": u["input_tokens"], "input_tokens_details": {"cached_tokens": cached if isinstance(cached, int) else 0},
            "output_tokens": u["output_tokens"],
            "output_tokens_details": {"reasoning_tokens": reasoning if isinstance(reasoning, int) else 0},
            "total_tokens": u["input_tokens"] + u["output_tokens"]}


def base(rid: str, body: Mapping[str, Any], requested: str, meta: Mapping[str, Any]) -> dict[str, Any]:
    """The response object before any output: the request's settings echoed back."""
    fmt = (body.get("text") or {}).get("format") if isinstance(body.get("text"), Mapping) else None
    return {"id": rid, "object": "response", "created_at": int(time.time()), "status": "in_progress", "error": None,
            "incomplete_details": None, "instructions": body.get("instructions"), "model": requested, "output": [],
            "parallel_tool_calls": body.get("parallel_tool_calls", True), "tool_choice": body.get("tool_choice", "auto"),
            "tools": body.get("tools") or [], "temperature": body.get("temperature"), "top_p": body.get("top_p"),
            "max_output_tokens": body.get("max_output_tokens"), "previous_response_id": body.get("previous_response_id"),
            "reasoning": {"effort": None, "summary": None}, "store": body.get("store", True) is not False,
            "text": {"format": fmt or {"type": "text"}}, "truncation": "disabled", "usage": None,
            "metadata": body.get("metadata") or {}, "x_broker": dict(meta)}
