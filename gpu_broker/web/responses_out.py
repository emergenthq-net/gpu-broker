"""OpenAI chat chunks -> the Responses API: the streamed events and the finished `response`.

One translator does both: a non-streamed request feeds the same chunks (a finished completion
is one chunk, `openai.sse`) and returns only the final response, so the two never disagree.
Output items come in order, one open at a time, between `response.output_item.added` and
`.done` (Codex ignores a delta with no open item): `reasoning` (one summary part), `message`
(one output_text part), and per tool call a `function_call` (function_call_arguments.delta /
.done) or, for a custom (freeform) tool, a `custom_tool_call` (custom_tool_call_input.*).
Text and reasoning stream as they arrive. Tool calls are collected, keyed by the server's
index (else id; a fragment with neither continues the last call), because a server may
interleave the argument deltas of several calls; each call is emitted whole after the text.
The stream ends with response.completed, response.incomplete (the token limit or a content
filter cut the answer short) or response.failed (the model server broke off; an item still
open is closed as incomplete first). `on_final` runs just before the final event, so a
response is stored before the client can name it. Every event carries a sequence_number;
no event carries the translator's own bookkeeping.
"""
from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Iterator, Mapping
from typing import Any

from .anthropic_resp import arguments_text, new_id
from .anthropic_sse import chunks
from .responses_obj import response_usage
from .responses_tools import INPUT

MSG, FC, CTC, RS, CALL_PREFIX = "msg_", "fc_", "ctc_", "rs_", "call_"
TEXT, REASONING, CALL, CUSTOM_CALL = "message", "reasoning", "function_call", "custom_tool_call"
INCOMPLETE = {"length": "max_output_tokens", "content_filter": "content_filter"}
FAILED_CODE = "server_error"
ToolKey = tuple[str, Any]


def frame(name: str, data: Mapping[str, Any]) -> str:
    return f"event: {name}\ndata: {json.dumps(data)}\n\n"


def custom_input(arguments: str) -> str:
    """A custom call's text: the `input` argument, else the raw arguments as the model wrote them."""
    try:
        args = json.loads(arguments)
    except ValueError:
        return arguments
    return args[INPUT] if isinstance(args, dict) and isinstance(args.get(INPUT), str) else arguments


class Translator:
    def __init__(self, response: dict[str, Any], custom: frozenset[str] = frozenset(),
                 on_final: Callable[[Translator], None] | None = None) -> None:
        self.response, self.custom, self.on_final = response, custom, on_final
        self.items: list[dict[str, Any]] = []   # the output, in order
        self.open: dict[str, Any] | None = None
        self.text = ""                          # the open text/reasoning item's text so far
        self.calls: dict[ToolKey, dict[str, Any]] = {}   # collected tool calls, in first-seen order
        self.last: ToolKey | None = None
        self.seq = 0
        self.finish: str | None = None
        self.usage = response_usage(None)

    def _ev(self, kind: str, /, **data: Any) -> str:
        out = frame(kind, {"type": kind, "sequence_number": self.seq, **data})
        self.seq += 1
        return out

    def start(self) -> Iterator[str]:
        yield self._ev("response.created", response=self.response)
        yield self._ev("response.in_progress", response=self.response)

    def _where(self, item: Mapping[str, Any]) -> dict[str, Any]:
        return {"item_id": item["id"], "output_index": len(self.items) - 1}

    def _open(self, item: dict[str, Any]) -> Iterator[str]:
        yield from self._close()
        self.items.append(item)
        self.open, self.text = item, ""
        yield self._ev("response.output_item.added", output_index=len(self.items) - 1, item=item)
        if item["type"] == TEXT:
            yield self._ev("response.content_part.added", **self._where(item), content_index=0,
                           part={"type": "output_text", "text": "", "annotations": []})
        elif item["type"] == REASONING:
            yield self._ev("response.reasoning_summary_part.added", **self._where(item), summary_index=0,
                           part={"type": "summary_text", "text": ""})

    def _close(self, status: str = "completed") -> Iterator[str]:
        item = self.open
        if item is None:
            return
        where = self._where(item)
        if item["type"] == TEXT:
            part = {"type": "output_text", "text": self.text, "annotations": []}
            item["content"] = [part]
            yield self._ev("response.output_text.done", **where, content_index=0, text=part["text"], logprobs=[])
            yield self._ev("response.content_part.done", **where, content_index=0, part=part)
        elif item["type"] == REASONING:
            part = {"type": "summary_text", "text": self.text}
            item["summary"] = [part]
            yield self._ev("response.reasoning_summary_text.done", **where, summary_index=0, text=part["text"])
            yield self._ev("response.reasoning_summary_part.done", **where, summary_index=0, part=part)
        elif item["type"] == CUSTOM_CALL:
            yield self._ev("response.custom_tool_call_input.done", **where, input=item[INPUT])
        else:
            yield self._ev("response.function_call_arguments.done", **where, name=item["name"], arguments=item["arguments"])
        item["status"] = status
        self.open, self.text = None, ""
        yield self._ev("response.output_item.done", output_index=where["output_index"], item=item)

    def _streamed(self, kind: str, text: str) -> Iterator[str]:
        if self.open is None or self.open["type"] != kind:
            item: dict[str, Any] = ({"id": new_id(MSG), "type": TEXT, "status": "in_progress", "role": "assistant", "content": []}
                                    if kind == TEXT else {"id": new_id(RS), "type": REASONING, "summary": []})
            yield from self._open(item)
        self.text += text
        where = self._where(self.items[-1])
        if kind == TEXT:
            yield self._ev("response.output_text.delta", **where, content_index=0, delta=text, logprobs=[])
        else:
            yield self._ev("response.reasoning_summary_text.delta", **where, summary_index=0, delta=text)

    def _collect(self, tc: Mapping[str, Any]) -> None:
        if isinstance(tc.get("index"), int):
            key: ToolKey = ("index", tc["index"])
        elif tc.get("id"):
            key = ("id", tc["id"])
        else:
            key = self.last or ("id", None)
        fn = tc.get("function") or {}
        call = self.calls.setdefault(key, {"id": None, "name": "", "arguments": ""})
        call["id"] = call["id"] or tc.get("id")
        call["name"] = call["name"] or fn.get("name") or ""
        call["arguments"] += arguments_text(fn.get("arguments"))
        self.last = key

    def _emit_call(self, call: Mapping[str, Any]) -> Iterator[str]:
        common = {"status": "in_progress", "call_id": call["id"] or new_id(CALL_PREFIX), "name": call["name"]}
        if call["name"] in self.custom:
            text = custom_input(call["arguments"])
            item = {"id": new_id(CTC), "type": CUSTOM_CALL, **common, INPUT: text}
            yield from self._open(item)
            yield self._ev("response.custom_tool_call_input.delta", **self._where(item), delta=text)
        else:
            item = {"id": new_id(FC), "type": CALL, **common, "arguments": call["arguments"]}
            yield from self._open(item)
            yield self._ev("response.function_call_arguments.delta", **self._where(item), delta=call["arguments"])

    def feed(self, chunk: Mapping[str, Any]) -> Iterator[str]:
        for choice in chunk.get("choices") or []:
            d = choice.get("delta") or {}
            if d.get("reasoning_content"):
                yield from self._streamed(REASONING, d["reasoning_content"])
            if d.get("content"):
                yield from self._streamed(TEXT, d["content"])
            for tc in d.get("tool_calls") or []:
                if isinstance(tc, Mapping):
                    self._collect(tc)
            self.finish = choice.get("finish_reason") or self.finish
        if chunk.get("usage") or chunk.get("timings"):
            self.usage = response_usage(chunk.get("usage"), chunk.get("timings"))

    def _final(self, kind: str) -> str:
        if self.on_final is not None:
            self.on_final(self)
        return self._ev(kind, response=self.response)

    def end(self) -> Iterator[str]:
        """Close what is open, emit the collected tool calls, and finish the response."""
        yield from self._close()
        for call in self.calls.values():
            yield from self._emit_call(call)
        yield from self._close()
        cut = INCOMPLETE.get(self.finish or "")
        self.response.update(status="incomplete" if cut else "completed", output=self.items, usage=self.usage,
                             incomplete_details={"reason": cut} if cut else None)
        yield self._final("response.incomplete" if cut else "response.completed")

    def fail(self, message: str) -> Iterator[str]:
        """The server broke off: close the open item as incomplete; calls not yet whole are dropped."""
        yield from self._close("incomplete")
        self.response.update(status="failed", output=self.items, error={"code": FAILED_CODE, "message": message})
        yield self._final("response.failed")


def events(t: Translator, lines: Iterable[str], failure: Mapping[str, str]) -> Iterator[str]:
    """The whole stream. `failure` is filled in by the relay if the server broke off."""
    yield from t.start()
    for chunk in chunks(lines):
        yield from t.feed(chunk)
    if failure:
        yield from t.fail(failure.get("message", ""))
        return
    yield from t.end()


def finish(t: Translator, lines: Iterable[str]) -> dict[str, Any]:
    """The finished response object for a non-streamed request."""
    for _ in events(t, lines, {}):
        pass
    return t.response
