"""OpenAI-client compatibility: the model list and `stream: true` chat responses.

Interactive chat on the resident model streams token by token from the server (see chat.py).
A call that went through the job queue (a background caller, or one that needed a model
switch) is finished before it is returned, so a streamed request then gets the answer as a
single SSE chunk followed by the end marker; streaming clients work unchanged.
"""
from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

from ..catalog import Catalog
from ..constants import OWNER, Kind, ModelStatus

SSE_DONE = "[DONE]"
CHUNK = "chat.completion.chunk"
ROLE = "assistant"
DEFAULT_FINISH = "stop"


def model_list(catalog: Catalog) -> dict[str, Any]:
    """Ready LLMs and their variants."""
    ready = [(k, m) for k, m in catalog.models.items() if m.get("kind") == Kind.LLM and m.get("status") == ModelStatus.READY]
    ids = [k for k, _ in ready] + [v for _, m in ready for v in m.get("variants") or {}]
    data = [{"id": i, "object": "model", "owned_by": OWNER} for i in ids]
    return {"object": "list", "data": data}


def _frame(payload: str) -> str:
    return f"data: {payload}\n\n"


def sse(result: dict[str, Any]) -> Iterator[str]:
    """A finished chat completion as one `chat.completion.chunk` event, then [DONE]."""
    choice = (result.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    delta = {"role": ROLE, "content": msg.get("content") or ""}
    if msg.get("reasoning_content"):
        delta["reasoning_content"] = msg["reasoning_content"]
    chunk: dict[str, Any] = {
        "id": result.get("id", OWNER), "object": CHUNK, "created": result.get("created", 0), "model": result.get("model", ""),
        "choices": [{"index": 0, "delta": delta, "finish_reason": choice.get("finish_reason", DEFAULT_FINISH)}]}
    if result.get("usage"):
        chunk["usage"] = result["usage"]
    yield _frame(json.dumps(chunk))
    yield _frame(SSE_DONE)
