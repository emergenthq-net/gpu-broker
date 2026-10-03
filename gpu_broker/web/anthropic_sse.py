"""OpenAI chat-completion SSE -> Anthropic Messages SSE.

Anthropic streams a fixed sequence: message_start; for each content block a
content_block_start, its deltas (text_delta, thinking_delta, input_json_delta) and a
content_block_stop; then message_delta (stop_reason, usage) and message_stop. A stream the
model server broke off ends with an `error` event instead, so the client sees the failure.

The input is OpenAI chunk lines, either relayed live from the server or produced by
`openai.sse` from a finished queued job; both look the same here.
"""
from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping
from http import HTTPStatus
from typing import Any

from ..backends import SSE_DATA, SSE_JSON
from .anthropic_resp import (
    MSG_PREFIX,
    NO_SIGNATURE,
    STOPPING_WORD,
    TOOL_PREFIX,
    TOOL_USE,
    arguments_text,
    error,
    new_id,
    stop_reason,
    usage,
)

THINKING, TEXT, TOOL = "thinking", "text", TOOL_USE
DELTA_TYPES = {THINKING: "thinking_delta", TEXT: "text_delta"}
ToolKey = tuple[str, Any]   # ("index", n) or ("id", call id): which OpenAI tool call a delta belongs to


def event(name: str, data: Mapping[str, Any]) -> str:
    return f"event: {name}\ndata: {json.dumps(data)}\n\n"


def chunks(lines: Iterable[str]) -> Iterator[dict[str, Any]]:
    """The JSON chunks in OpenAI SSE lines (comments, blank lines and [DONE] skipped), each parsed once."""
    for line in lines:
        for part in line.splitlines():
            if part.startswith(SSE_JSON):
                try:
                    chunk = json.loads(part.removeprefix(SSE_DATA))
                except ValueError:
                    continue
                if isinstance(chunk, dict):
                    yield chunk


class Translator:
    """Turns a sequence of OpenAI chunks into Anthropic events.

    Blocks are numbered in the order they start. Text and thinking stream into the block of
    their kind that is open; a tool call keeps one block from its first delta to the end of the
    stream or the start of the next tool call, so text the server interleaves between a call's
    argument deltas opens its own block instead of splitting the call. A tool delta belongs to
    its OpenAI `index`; a server that sends no index is followed by call id (the first delta of
    a call carries it), and a delta with neither continues the last call."""

    def __init__(self, thinking: bool) -> None:
        self.thinking = thinking
        self.next_index = 0
        self.streams: dict[str, int] = {}        # open text/thinking block: kind -> block index
        self.tool: tuple[ToolKey, int] | None = None   # the open tool call and its block index
        self.tool_used = False
        self.finish: str | None = None
        self.stopping_word: str | None = None
        self.usage: dict[str, int] = usage(None)

    def _stop(self, index: int) -> str:
        return event("content_block_stop", {"type": "content_block_stop", "index": index})

    def _start(self, block: dict[str, Any]) -> tuple[int, str]:
        i, self.next_index = self.next_index, self.next_index + 1
        return i, event("content_block_start", {"type": "content_block_start", "index": i, "content_block": block})

    def _delta(self, index: int, delta: dict[str, Any]) -> str:
        return event("content_block_delta", {"type": "content_block_delta", "index": index, "delta": delta})

    def _close_streams(self) -> Iterator[str]:
        for i in sorted(self.streams.values()):
            yield self._stop(i)
        self.streams.clear()

    def _streamed(self, kind: str, text: str) -> Iterator[str]:
        if kind not in self.streams:
            yield from self._close_streams()   # text and thinking alternate; an open tool call stays open
            empty = {"type": kind, kind: ""} | ({"signature": NO_SIGNATURE} if kind == THINKING else {})
            self.streams[kind], start = self._start(empty)
            yield start
        yield self._delta(self.streams[kind], {"type": DELTA_TYPES[kind], kind: text})

    def _key(self, tc: Mapping[str, Any]) -> ToolKey:
        if isinstance(tc.get("index"), int):
            return ("index", tc["index"])
        if tc.get("id"):
            return ("id", tc["id"])
        return self.tool[0] if self.tool else ("id", None)

    def _tool_call(self, tc: Mapping[str, Any]) -> Iterator[str]:
        fn = tc.get("function") or {}
        key = self._key(tc)
        if self.tool is None or self.tool[0] != key:
            yield from self._close_streams()
            if self.tool is not None:
                yield self._stop(self.tool[1])
            block: dict[str, Any] = {"type": TOOL, "id": tc.get("id") or new_id(TOOL_PREFIX), "name": fn.get("name") or "", "input": {}}
            index, start = self._start(block)
            self.tool, self.tool_used = (key, index), True
            yield start
        if text := arguments_text(fn.get("arguments")):
            yield self._delta(self.tool[1], {"type": "input_json_delta", "partial_json": text})

    def feed(self, chunk: Mapping[str, Any]) -> Iterator[str]:
        for choice in chunk.get("choices") or []:
            d = choice.get("delta") or {}
            if self.thinking and d.get("reasoning_content"):
                yield from self._streamed(THINKING, d["reasoning_content"])
            if d.get("content"):
                yield from self._streamed(TEXT, d["content"])
            for tc in d.get("tool_calls") or []:
                if isinstance(tc, Mapping):
                    yield from self._tool_call(tc)
            self.finish = choice.get("finish_reason") or self.finish
            self.stopping_word = choice.get(STOPPING_WORD) or self.stopping_word
        if chunk.get("usage") or chunk.get("timings"):
            self.usage = usage(chunk.get("usage"), chunk.get("timings"))

    def end(self) -> Iterator[str]:
        open_blocks = [*self.streams.values(), *([self.tool[1]] if self.tool else [])]
        for i in sorted(open_blocks):
            yield self._stop(i)
        self.streams.clear()
        self.tool = None
        reason, seq = stop_reason(self.finish, self.stopping_word, self.tool_used)
        yield event("message_delta", {"type": "message_delta", "delta": {"stop_reason": reason, "stop_sequence": seq},
                                      "usage": self.usage})
        yield event("message_stop", {"type": "message_stop"})


def events(lines: Iterable[str], requested: str, thinking: bool, meta: Mapping[str, Any],
           failure: Mapping[str, str]) -> Iterator[str]:
    """The whole Anthropic stream. `failure` is filled in by the relay if the server broke off."""
    start: dict[str, Any] = {"id": new_id(MSG_PREFIX), "type": "message", "role": "assistant", "model": requested, "content": [],
             "stop_reason": None, "stop_sequence": None, "usage": usage(None), "x_broker": dict(meta)}
    yield event("message_start", {"type": "message_start", "message": start})
    t = Translator(thinking)
    for chunk in chunks(lines):
        yield from t.feed(chunk)
    if failure:
        yield event("error", error(HTTPStatus.BAD_GATEWAY, failure.get("message", "")))
        return
    yield from t.end()
