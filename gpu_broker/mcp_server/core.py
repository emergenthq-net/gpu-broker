"""What the MCP tools do, on the broker itself; no MCP SDK here (server.py wraps these as tools).

A tool caller is the main token or a client key (web/auth.py rules, checked by http.py). A
client key's name is the requester of its jobs and its id their owner; it sees only the jobs it
owns (names repeat, ids do not): job status, results and gpu_status never show another caller's
job ids or outputs. Generation goes through Broker.submit, so input files get the inputs
layer's checks and limits. A client key sends files as base64: `<slot>_url` inputs make the
broker fetch a caller-chosen URL, so they are for the main token unless the operator sets
`mcp.client_url_inputs`. A model must be runnable now (an unknown or not-yet-downloaded name
would otherwise start a download).
"""
from __future__ import annotations

import posixpath
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import parse_qs, urlsplit

from starlette.requests import Request

from ..backends import OUTPUT_TYPE
from ..broker import Broker
from ..constants import ERR_SHORT, HTTP_SCHEMES, IMAGE_MIME, REQUESTER_HEADER, TERMINAL, URL_SUFFIX, JobState, Kind
from ..drivers import DRIVER_ERRORS
from ..resolve import best_substitute, lookup, runnable
from ..tuning import Mcp
from ..web import completion
from ..web.jobs import CLIENT_ID_STATE, CLIENT_STATE

MAIN, KEY_PREFIX = "main", "key:"   # http.py's caller header: the main token, or `key:<key id>:<key name>`
MAIN_REQUESTER = "mcp"              # requester of a main-token caller's jobs
HIDDEN = "no such job"              # another caller's job reads exactly like a missing one
POLL_HINT = "still running: call job_status (or job_result to wait again) with job_id"
IMAGE_SUFFIXES = {".png": "png", ".jpg": "jpeg", ".jpeg": "jpeg", ".webp": "webp"}
Fetch = Callable[[str, str, str, int], bytes | None]   # (name, subfolder, type, cap) -> body, or None when larger
NO_URL_INPUTS = ("this client key may not send {fields}: the broker would fetch that URL itself. Send the file as base64 "
                 "(a data: URL is fine); the operator can allow URLs with mcp.client_url_inputs")


@dataclass(frozen=True)
class Caller:
    requester: str
    key_id: str | None = None   # a client key's id; None for the main token

    def sees(self, job: Mapping[str, Any]) -> bool:
        return self.key_id is None or job.get("owner") == self.key_id


def caller(header: str | None) -> Caller:
    """The caller http.py vouched for; anything else is refused."""
    if header == MAIN:
        return Caller(MAIN_REQUESTER)
    kid, _, name = (header or "").removeprefix(KEY_PREFIX).partition(":")
    if header and header.startswith(KEY_PREFIX) and kid and name:
        return Caller(name, kid)
    raise PermissionError("not authenticated")


def input_field(slot: str, value: str) -> dict[str, str]:
    """An image or video given as an http(s) URL goes in `<slot>_url` (fetched under inputs.allow_urls
    and the URL guard), anything else (base64, a data: URL) in `<slot>`."""
    return {slot + URL_SUFFIX: value} if urlsplit(value).scheme in HTTP_SCHEMES else {slot: value}


def models(broker: Broker) -> list[dict[str, Any]]:
    data, resident = broker.catalog.data, broker.residency.current
    return [{"name": k, "kind": m.get("kind"), "caps": m.get("caps", []), "image_caps": m.get("image_caps", []),
             "inputs": m.get("inputs", {}), "quality": m.get("quality", 0), "aliases": m.get("aliases", []),
             "loaded": k == resident}
            for k, m in sorted(data["models"].items()) if m.get("kind") != Kind.UI and runnable(data, k)]


def named(broker: Broker, model: str) -> str:
    """The catalog key for a model the caller named, only if it can run now (never a download)."""
    key = lookup(broker.catalog.data, model)
    if key is None or broker.catalog.models[key].get("kind") == Kind.UI or not runnable(broker.catalog.data, key):
        raise ValueError(f"'{model}' is not a model this broker can run now; list_models shows what it can")
    return key


def pick(broker: Broker, kind: Kind, caps: list[str], given: frozenset[str], model: str | None) -> str:
    """The model to use: the one named (a catalog name or alias), else the best ready one for the job."""
    data = broker.catalog.data
    if model:
        return named(broker, model)
    if (best := best_substitute(data, kind, set(caps), images=given)) is None:
        raise ValueError(f"no ready {kind} model can do {caps}; list_models shows what this broker can run")
    return best


def submit(broker: Broker, who: Caller, body: dict[str, Any]) -> str:
    if who.key_id and not broker.settings.mcp.client_url_inputs and (urls := sorted(k for k in body if k.endswith(URL_SUFFIX))):
        raise ValueError(NO_URL_INPUTS.format(fields=", ".join(urls)))
    jid, info = broker.submit(body, who.requester, owner=who.key_id)
    if info.get("error"):
        raise ValueError(info["error"])
    return jid


def report(broker: Broker, who: Caller, jid: str, wait_s: float) -> dict[str, Any]:
    """The job as a tool returns it, after waiting up to `wait_s` for it to finish."""
    job = broker.view(jid)
    if job is None or not who.sees(job):
        raise ValueError(HIDDEN)
    if wait_s > 0 and job["state"] not in TERMINAL:
        job = broker.wait(jid, wait_s) or job
    out: dict[str, Any] = {"job_id": jid, "state": job["state"], "model": job.get("resolved"),
                           "queue_position": broker.scheduler.position(jid)}
    if job.get("substitution"):
        out["substitution"] = job["substitution"]
    if job["state"] == JobState.DONE:
        out["outputs"] = [{k: o[k] for k in ("file", "url") if k in o} for o in (job.get("result") or {}).get("outputs", [])]
    elif job.get("error"):
        out["error"] = job["error"]
    if job["state"] not in TERMINAL:
        out["hint"] = POLL_HINT
    return out


def inline(broker: Broker, job: Mapping[str, Any], cfg: Mcp, fetch: Fetch | None = None) -> list[tuple[str, bytes]]:
    """Small finished images as (MIME type, bytes), read through the broker's ComfyUI client
    (comfy.url and its auth, not the browser URL) with the output's own type (output or temp).
    An output that cannot be read is left to its URL."""
    found: list[tuple[str, bytes]] = []
    for o in job.get("outputs", []):
        kind = IMAGE_SUFFIXES.get(posixpath.splitext(o.get("file", ""))[1].lower())
        if kind is None or "url" not in o or len(found) >= cfg.inline_max_images:
            continue
        sub, name = posixpath.split(o["file"])
        kind_of = (parse_qs(urlsplit(o["url"]).query).get("type") or [OUTPUT_TYPE])[0]
        try:
            data = (fetch or broker.backends.comfy_view)(name, sub, kind_of, cfg.inline_max_bytes)
        except (OSError, ValueError):
            continue
        if data is not None:
            found.append((IMAGE_MIME[kind], data))
    return found


def status(broker: Broker, who: Caller) -> dict[str, Any]:
    """What is loaded, what is running and queued, and VRAM. The main token sees every job; a
    client key sees its own jobs and only a count of everyone else's."""
    queued, running, inflight = broker.scheduler.snapshot()

    def split(jids: list[str]) -> tuple[list[dict[str, Any]], int]:
        jobs = [(jid, broker.view(jid) or {}) for jid in jids]
        mine = [{"job_id": jid, "model": j.get("resolved"), "state": j.get("state")} for jid, j in jobs if who.sees(j)]
        return mine, len(jobs) - len(mine)
    run, run_others = split([j for j in dict.fromkeys([running, *inflight]) if j])
    queue, queue_others = split(queued)
    out: dict[str, Any] = {"loaded_llm": broker.residency.current, "last_comfy_model": broker.residency.last_comfy,
                           "gpu_held": bool(broker.scheduler.hold.get()), "running": run, "queue": queue}
    if who.key_id:
        out |= {"others_running": run_others, "others_queued": queue_others}
    try:
        used, total, util = broker.driver.gpu()
        out["vram"] = {"used_mib": used, "total_mib": total, "util_pct": util}
    except DRIVER_ERRORS as e:
        out["vram"] = {"error": str(e)[:ERR_SHORT]}
    return out


def chat(broker: Broker, who: Caller, body: dict[str, Any]) -> dict[str, Any]:
    """One chat completion down the same path as /v1/chat/completions (a person is waiting, so
    the resident model answers directly when it can), as the caller."""
    if body.get("model"):
        body = {**body, "model": named(broker, str(body["model"]))}   # a chat must not start a download either
    key_name = who.requester if who.key_id else None
    scope = {"type": "http", "method": "POST", "path": "/mcp", "query_string": b"", "client": None,
             "headers": [(REQUESTER_HEADER.encode(), who.requester.encode())],
             "state": {CLIENT_STATE: key_name, CLIENT_ID_STATE: who.key_id}}
    try:
        outcome = completion.run(broker, {**body, "stream": False}, Request(scope), stream=False)
    except completion.ChatFailed as e:
        raise ValueError(e.message) from None
    result = outcome.result or {}
    choices = result.get("choices") or [{}]
    return {"text": str((choices[0].get("message") or {}).get("content") or ""), "model": outcome.meta.get("used") or outcome.meta.get("resolved"),
            "usage": result.get("usage")}
