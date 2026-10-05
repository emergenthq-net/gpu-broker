"""An unstreamed request, sent upstream as a stream and put back together into the provider's own
unstreamed answer.

Why: a stream's first byte comes as soon as the provider starts, so the first-byte and idle
timeouts tell a provider that is down from one that is merely writing a long answer, and a
failure is noticed (and failed over) in seconds rather than after a whole-response timeout.
The caller still gets exactly the unstreamed body: the same fields, usage, stop reason, tool
calls and ids. The request changes only by `stream: true` (plus `stream_options.include_usage`
for chat completions, without which a stream carries no usage).

A stream that ends before its finishing event, or that carries an error event, raises Broken:
the caller has seen nothing yet, so the request may still go to another model.
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from . import transport
from .outcome import ANTHROPIC_PATH, RESPONSES_PATH
from .shapes import Broken, anthropic, chat, responses

MAX_BYTES = 64 * 2**20   # a stream longer than this is not reassembled (the caller falls over)
SSE = "text/event-stream"
DROPPED_HEADERS = frozenset({"content-length", "transfer-encoding", "content-type"})


def as_stream(path: str, body: Mapping[str, Any]) -> dict[str, Any]:
    out = {**body, "stream": True}
    if path not in (ANTHROPIC_PATH, RESPONSES_PATH):   # chat completions
        out["stream_options"] = {"include_usage": True}
    return out


def assemble(path: str, data: bytes) -> dict[str, Any]:
    try:
        if path == ANTHROPIC_PATH:
            return anthropic(data)
        if path == RESPONSES_PATH:
            return responses(data)
        return chat(data)
    except (KeyError, IndexError, TypeError, AttributeError, ValueError) as e:   # a malformed stream
        raise e if isinstance(e, Broken) else Broken("a malformed event") from None


def gather(path: str, a: transport.Answer) -> transport.Answer:
    """The started 2xx stream `a` (its opening already read) read to the end and reassembled;
    a 2xx that is not a stream (the provider ignored `stream`) is returned whole as it is.
    Raises transport.Unreachable (fixed text) when there is no whole answer to give."""
    data, chunks = a.body, iter(a.rest)
    for chunk in chunks:   # a drop or idle gap raises Unreachable from transport
        data += chunk
        if len(data) > MAX_BYTES:
            a.close()
            raise transport.Unreachable("the answer was too long to reassemble")
    headers = {k: v for k, v in a.headers.items() if k not in DROPPED_HEADERS}
    if not a.headers.get("content-type", "").startswith(SSE):
        return transport.Answer(a.status, headers | {"content-type": a.headers.get("content-type", "application/json")}, data)
    try:
        whole = assemble(path, data)
    except Broken as e:
        raise transport.Unreachable(f"the stream had no whole answer ({e})") from None
    return transport.Answer(a.status, headers | {"content-type": "application/json"}, json.dumps(whole).encode())
