"""What routing a request comes to (Served, Local, Exhausted), and the error event that ends a
stream that broke after it had started, in the request's API shape."""
from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

ANTHROPIC_PATH, RESPONSES_PATH = "/v1/messages", "/v1/responses"


@dataclass
class Served:
    provider: str
    status: int
    headers: dict[str, str]
    body: bytes = b""
    chunks: Iterator[bytes] | None = None   # a successful stream, relayed as it arrives
    skipped: list[str] = field(default_factory=list)   # why earlier providers were passed over


@dataclass
class Local:
    model: str
    reason: str


@dataclass
class Exhausted:
    reason: str
    last: Served | None = None   # the last provider error worth showing (a quota answer), if any
    local: bool = True           # False: the chain was not allowed to reach its local model


def error_event(path: str, message: str) -> bytes:
    """A terminal error event in the request's API shape (Anthropic, Responses, or chat completions)."""
    data: dict[str, Any]
    if path == ANTHROPIC_PATH:
        data = {"type": "error", "error": {"type": "api_error", "message": message}}
        return f"event: error\ndata: {json.dumps(data)}\n\n".encode()
    if path == RESPONSES_PATH:
        data = {"type": "error", "code": "upstream_error", "message": message, "param": None, "sequence_number": 0}
        return f"event: error\ndata: {json.dumps(data)}\n\n".encode()
    return f"data: {json.dumps({'error': {'message': message, 'type': 'upstream_error'}})}\n\n".encode()
