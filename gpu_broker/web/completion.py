"""One chat completion through the broker, whichever API it arrived on (OpenAI or Anthropic).

Interactive callers on the resident model are served directly (see `gpu_broker.chat`); with
`stream: true` the server's own SSE lines are relayed as they are generated. Everything else
is a queued job this request waits for (up to `timeouts.chat_wait_s`).

Before either path the hosted name is mapped (see `gpu_broker.modelmap`); the mapping is
recorded with the job when it is created. Only a mapped name is echoed in the response's
`model` field (an app that asked for `gpt-4o` sees `gpt-4o`); a catalog name keeps the
server's own `model`, as before. What really ran is in `x_broker` (`used`, `substitution`).
`chunks` are the server's SSE lines untouched: the route that relays them echoes the model
(`echo_model`) only when mapped, so each line is parsed at most once per route.
"""
from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from http import HTTPStatus
from typing import Any

from fastapi import Request

from ..backends import SSE_DATA, SSE_JSON
from ..broker import Broker, validate_request
from ..chat import Lease, apply_variant
from ..constants import ERR_EVENT, ERR_JOB, PRIORITY_HEADER, REQUESTER_HEADER, TERMINAL, JobState, Kind
from ..modelmap import map_name
from ..resolve import lookup
from .jobs import requester as requester_of

MAX_COMPLETION_TOKENS, MAX_TOKENS = "max_completion_tokens", "max_tokens"
LOCAL = "local"   # x_broker.served_by for an answer from this broker (upstream.py marks hosted ones)


@dataclass
class Outcome:
    requested: str                            # the model name the caller sent
    meta: dict[str, Any]                      # x_broker
    mapped: bool = False                      # requested came through model_map (echo it as `model`)
    result: dict[str, Any] | None = None      # a finished completion, OpenAI shape
    chunks: Iterator[str] | None = None       # the server's SSE lines, relayed live
    failure: dict[str, str] = field(default_factory=dict)   # set when a live stream breaks


class ChatFailed(Exception):
    """The call could not be answered; `status` is the HTTP status to report. `hosted` = the
    caller asked for a hosted model name (so a hosted fallback may answer instead)."""

    def __init__(self, status: HTTPStatus, message: str, meta: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.status, self.message, self.meta = status, message, dict(meta)
        self.hosted = False


class GoHosted(Exception):
    """Raised instead of queueing when the caller allowed a hosted fallback and the local
    model would first need a switch (the GPU is busy with something else)."""


def echo_model(line: str, requested: str) -> str:
    """An SSE `data:` line with its `model` replaced by the name the caller asked for."""
    stripped = line.rstrip("\r\n")
    if not stripped.startswith(SSE_JSON):
        return line
    try:
        chunk = json.loads(stripped.removeprefix(SSE_DATA))
    except ValueError:
        return line
    return SSE_DATA + json.dumps({**chunk, "model": requested}) + line[len(stripped):]


def prepare(broker: Broker, body: Mapping[str, Any]) -> tuple[str, dict[str, Any], str | None]:
    """(name asked for, the body to run, mapping note). `max_completion_tokens` (newer OpenAI
    clients) becomes `max_tokens`, which every OpenAI-compatible server reads."""
    cat = broker.catalog
    requested = body.get("model") or cat.defaults["resident"]
    run = dict(body)
    if MAX_COMPLETION_TOKENS in run:
        run.setdefault(MAX_TOKENS, run.pop(MAX_COMPLETION_TOKENS))
    known = lookup(cat.data, requested) is not None or cat.variant(requested) is not None
    mapped = map_name(broker.settings.model_map, known, requested, cat.defaults["resident"])
    if mapped:
        run["model"] = mapped.target
    run = {**apply_variant(cat, run), "kind": run.get("kind", Kind.LLM)}
    return requested, run, mapped.note if mapped else None


def finished(result: Mapping[str, Any], requested: str, mapped: bool, meta: Mapping[str, Any]) -> dict[str, Any]:
    """A finished completion with x_broker, and `model` echoed only for a mapped name."""
    return {**result, **({"model": requested} if mapped else {}), "x_broker": dict(meta)}


def run(broker: Broker, body: Mapping[str, Any], request: Request, stream: bool, may_forward: bool = False) -> Outcome:
    """`may_forward`: a hosted fallback is available, so an interactive call on a hosted name
    that would wait for a model switch raises GoHosted instead of queueing."""
    validate_request(body)
    requester = requester_of(request, request.headers.get(REQUESTER_HEADER))
    requested, body, note = prepare(broker, body)
    hosted = note is not None
    priority, classes = request.headers.get(PRIORITY_HEADER, ""), broker.scheduler.classes
    interactive = classes.of_request(broker.catalog, body, priority, requester)
    try:
        lease = broker.chat.open(body, requester, requested, note) if interactive else None
        if lease is not None:
            return _direct(broker, lease, body, stream, requested, note)
        target = lookup(broker.catalog.data, body.get("model") or broker.catalog.defaults["resident"])
        if may_forward and hosted and interactive and broker.settings.fallback.when_switching \
                and target != broker.scheduler.pool.resident:
            raise GoHosted(f"'{target}' is not loaded; the GPU would have to switch first")
        return queued(broker, classes.for_queue(body, interactive), requester, requested, note, priority)
    except ChatFailed as e:
        e.hosted = hosted
        raise



def _direct(broker: Broker, lease: Lease, body: dict[str, Any], stream: bool, requested: str, note: str | None) -> Outcome:
    meta = {"job": lease.jid, "requested": requested, "used": lease.key, "direct": True, "substitution": lease.substitution,
            "served_by": LOCAL}
    mapped = note is not None
    if not stream:
        try:
            out = broker.backends.llm_chat(lease.model, body)
        except Exception as e:  # noqa: BLE001 — reported to the caller and recorded on the job
            broker.chat.finish(lease, JobState.FAILED, error=str(e)[:ERR_JOB])
            raise ChatFailed(HTTPStatus.BAD_GATEWAY, str(e)[:ERR_EVENT], meta) from None
        broker.chat.finish(lease, JobState.DONE, result=out)
        return Outcome(requested, meta, mapped, result=finished(out, requested, mapped, meta))
    outcome = Outcome(requested, meta, mapped)

    def relay() -> Iterator[str]:
        summary: dict[str, Any] = {}
        state, error = JobState.DONE, None
        try:
            yield from broker.backends.llm_stream(lease.model, body, summary)
        except Exception as e:  # noqa: BLE001 — the client sees a truncated stream; the job records why
            state, error = JobState.FAILED, str(e)[:ERR_JOB]
            outcome.failure["message"] = error[:ERR_EVENT]
        finally:
            broker.chat.finish(lease, state, result={"streamed": True, **summary}, error=error)
    outcome.chunks = relay()
    return outcome


def queued(broker: Broker, body: dict[str, Any], requester: str, requested: str, note: str | None,
           priority: str = "") -> Outcome:
    """Submit as a job and wait for it (chat that could not go direct, and embeddings)."""
    jid, info = broker.submit(body, requester, requested, note, priority)
    meta = {"job": jid, **info}
    if info.get("error"):
        raise ChatFailed(HTTPStatus.SERVICE_UNAVAILABLE, info["error"], meta)
    j = broker.wait(jid, broker.settings.timeouts.chat_wait_s) or {}
    state = j.get("state")
    if state != JobState.DONE:
        code = HTTPStatus.BAD_GATEWAY if state in TERMINAL else HTTPStatus.GATEWAY_TIMEOUT
        raise ChatFailed(code, j.get("error") or f"still {state}", meta)
    meta = {"job": jid, "requested": requested, "used": j["resolved"], "substitution": j.get("substitution"),
            "served_by": LOCAL}
    mapped = note is not None
    return Outcome(requested, meta, mapped, result=finished(j["result"], requested, mapped, meta))
