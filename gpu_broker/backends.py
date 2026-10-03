"""HTTP clients for the two kinds of backend: OpenAI-compatible LLM servers and ComfyUI.

The broker only ever contacts URLs from trusted configuration — `comfy.url` in the config
file and `endpoint` in catalog entries — never a URL taken from a request. Values that come
back from a backend (a ComfyUI prompt id, output file names) are escaped before they are
placed in a URL.
"""
from __future__ import annotations

import contextlib
import json
import secrets
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator, Mapping
from http import HTTPStatus
from typing import Any, Protocol
from urllib.parse import quote, urlencode, urlsplit

from .catalog import Model
from .constants import AUTH_SCHEME, BROKER_FIELDS, ERR_DETAIL, HTTP_SCHEMES, IMAGE_MIME
from .settings import Comfy, Intervals, Timeouts

JSON = {"Content-Type": "application/json"}
HEALTH, CHAT, EMBED = "/health", "/v1/chat/completions", "/v1/embeddings"
SYSTEM_STATS, FREE, PROMPT, HISTORY, VIEW = "/system_stats", "/free", "/prompt", "/history/", "/view"
UPLOAD = "/upload/image"
UPLOAD_FIELDS = {"type": "input", "overwrite": "true"}   # into ComfyUI's input folder, as named
OUTPUT_KINDS = ("images", "videos", "gifs", "audio")
OUTPUT_TYPE = "output"
EXECUTION_ERROR = "execution_error"
STATUS_ERROR = "error"
CLIENT_ID_PREFIX = "broker-"
# OpenAI-style SSE framing, shared by every module that reads or writes it.
SSE_DATA = "data: "
SSE_JSON = SSE_DATA + "{"
SSE_DONE = "[DONE]"
SUMMARY_MARKERS = tuple(f'"{k}"' for k in ("usage", "timings"))   # a line worth parsing for the summary
STREAM_SUMMARY_KEYS = ("usage", "timings")   # kept from a stream for metrics
STREAM_OPTIONS = "stream_options"
ENCODING = "utf-8"
DECIMALS = 1


class Backends(Protocol):
    """What the scheduler and residency need from the backends (tests substitute fakes)."""

    def llm_healthy(self, model: Model) -> bool: ...
    def llm_chat(self, model: Model, payload: Mapping[str, Any]) -> dict[str, Any]: ...
    def llm_stream(self, model: Model, payload: Mapping[str, Any], summary: dict[str, Any]) -> Iterator[str]: ...
    def llm_embed(self, model: Model, payload: Mapping[str, Any]) -> dict[str, Any]: ...
    def comfy_alive(self) -> bool: ...
    def comfy_free(self) -> None: ...
    def comfy_run(self, key: str, graph: dict[str, Any], jid: str) -> dict[str, Any]: ...
    def comfy_queue_len(self) -> int | None: ...
    def comfy_upload(self, name: str, data: bytes, kind: str) -> str: ...


def view_url(browser_url: str, filename: str, subfolder: str, kind: str = OUTPUT_TYPE) -> str:
    """A browser URL for a file in ComfyUI's output (or input) folder, through ComfyUI's /view."""
    return f"{browser_url}{VIEW}?{urlencode({'filename': filename, 'subfolder': subfolder, 'type': kind})}"


def _request(url: str, body: Any = None, headers: Mapping[str, str] | None = None) -> urllib.request.Request:
    if urlsplit(url).scheme not in HTTP_SCHEMES:   # never file:, ftp: or custom handlers
        raise ValueError(f"refusing non-HTTP URL {url!r}")
    data = None if body is None else json.dumps(body).encode()
    return urllib.request.Request(url, data, {**(JSON if data else {}), **(headers or {})})  # noqa: S310 — scheme checked above


class HttpBackends:
    def __init__(self, comfy: Comfy, timeouts: Timeouts, intervals: Intervals, tokens: Mapping[str, str],
                 clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep) -> None:
        self.comfy, self.t, self.i, self.tokens = comfy, timeouts, intervals, tokens
        self.clock, self.sleep = clock, sleep

    def _json(self, req: urllib.request.Request, timeout: float) -> Any:
        with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 — built by _request
            return json.load(r)

    def _ok(self, req: urllib.request.Request) -> bool:
        try:
            with urllib.request.urlopen(req, timeout=self.t.health_s) as r:  # noqa: S310 — built by _request
                return bool(r.status == HTTPStatus.OK)
        except (OSError, ValueError):  # refused, reset, timed out, HTTP error: all mean "not ready"
            return False

    def _auth(self, model: Model) -> dict[str, str]:
        return {"Authorization": f"{AUTH_SCHEME} {self.tokens.get(model.get('auth_env', ''), '')}"}

    # ---- LLM servers ----------------------------------------------------
    def llm_healthy(self, model: Model) -> bool:
        return self._ok(_request(model["endpoint"] + model.get("health_path", HEALTH), headers=self._auth(model)))

    def _chat_request(self, model: Model, payload: Mapping[str, Any], stream: bool | None,
                      path: str = CHAT) -> urllib.request.Request:
        """The server sees its own served name, not the alias asked for, and none of the broker's fields."""
        body = {k: v for k, v in payload.items() if k not in BROKER_FIELDS}
        body.update(model=model["served_name"], **({} if stream is None else {"stream": stream}))
        if not stream:   # stream_options without stream: true is a 400 on OpenAI-compatible servers
            body.pop(STREAM_OPTIONS, None)
        return _request(model["endpoint"] + path, body, self._auth(model))

    def llm_embed(self, model: Model, payload: Mapping[str, Any]) -> dict[str, Any]:
        """One /v1/embeddings call on a server started for embeddings (llama-server --embeddings)."""
        result: dict[str, Any] = self._json(self._chat_request(model, payload, None, EMBED), self.t.llm_call_s)
        return result

    def llm_chat(self, model: Model, payload: Mapping[str, Any]) -> dict[str, Any]:
        """One non-streamed completion."""
        result: dict[str, Any] = self._json(self._chat_request(model, payload, False), self.t.llm_call_s)
        return result

    def llm_stream(self, model: Model, payload: Mapping[str, Any], summary: dict[str, Any]) -> Iterator[str]:
        """Relay the server's SSE lines as they arrive; copy usage/timings from them into `summary`."""
        with urllib.request.urlopen(self._chat_request(model, payload, True), timeout=self.t.llm_call_s) as r:  # noqa: S310 — built by _request
            for raw in r:
                line = raw.decode(ENCODING, "replace")
                if line.startswith(SSE_JSON) and any(m in line for m in SUMMARY_MARKERS):
                    try:
                        chunk = json.loads(line.removeprefix(SSE_DATA))
                    except ValueError:
                        chunk = {}
                    summary.update({k: chunk[k] for k in STREAM_SUMMARY_KEYS if isinstance(chunk, dict) and chunk.get(k)})
                yield line

    # ---- ComfyUI --------------------------------------------------------
    def comfy_alive(self) -> bool:
        return self._ok(_request(self.comfy.url + SYSTEM_STATS))

    def comfy_free(self) -> None:
        """Unload models and free VRAM. Best effort: a ComfyUI that is down holds no VRAM."""
        with contextlib.suppress(OSError, ValueError):
            self._json(_request(self.comfy.url + FREE, {"unload_models": True, "free_memory": True}), self.t.comfy_http_s)

    def comfy_queue_len(self) -> int | None:
        """Prompts queued or running; None when ComfyUI cannot be asked (not the same as idle)."""
        try:
            return int(self._json(_request(self.comfy.url + PROMPT), self.t.health_s)["exec_info"]["queue_remaining"])
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def comfy_upload(self, name: str, data: bytes, kind: str) -> str:
        """Store an input image in ComfyUI (POST /upload/image); returns the name a LoadImage
        node must use (`subfolder/name` when ComfyUI files it in a subfolder)."""
        boundary = secrets.token_hex(16)
        parts = [f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
                 for k, v in UPLOAD_FIELDS.items()]
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="{name}"\r\n'
                     f"Content-Type: {IMAGE_MIME[kind]}\r\n\r\n".encode() + data + b"\r\n")
        body = b"".join(parts) + f"--{boundary}--\r\n".encode()
        req = _request(self.comfy.url + UPLOAD)
        req.data, req.method = body, "POST"
        req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
        try:
            stored = self._json(req, self.t.comfy_http_s)
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"ComfyUI rejected the input image: {e.read().decode()[:ERR_DETAIL]}") from None
        return f"{stored.get('subfolder', '')}/{stored['name']}".lstrip("/")

    def comfy_run(self, key: str, graph: dict[str, Any], jid: str) -> dict[str, Any]:
        """Submit a graph and wait for it; returns output files with browser-reachable URLs."""
        try:
            sent = self._json(_request(self.comfy.url + PROMPT, {"prompt": graph, "client_id": CLIENT_ID_PREFIX + jid}),
                              self.t.comfy_submit_s)
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"ComfyUI rejected the graph: {e.read().decode()[:ERR_DETAIL]}") from None
        pid = str(sent["prompt_id"])
        t0 = self.clock()
        while self.clock() - t0 < self.t.comfy_run_s:
            history = self._json(_request(self.comfy.url + HISTORY + quote(pid, safe="")), self.t.comfy_http_s)
            if pid in history and (done := self._finished(history[pid])) is not None:
                return {"model": key, "outputs": done, "wall_s": round(self.clock() - t0, DECIMALS)}
            self.sleep(self.i.comfy_poll_s)
        raise RuntimeError(f"timed out after {self.t.comfy_run_s}s")

    def _finished(self, entry: Mapping[str, Any]) -> list[dict[str, str]] | None:
        """Output list once the prompt completed, None while it runs; raises on an execution error."""
        status = entry["status"]
        if status.get("status_str") == STATUS_ERROR:
            msgs = [m[1].get("exception_message", "") for m in status.get("messages", []) if m[0] == EXECUTION_ERROR]
            raise RuntimeError(f"ComfyUI execution error: {(msgs or ['unknown'])[0][:ERR_DETAIL]}")
        if not status.get("completed"):
            return None
        outs = []
        for node in entry["outputs"].values():
            for kind in OUTPUT_KINDS:
                for f in node.get(kind, []):
                    sub = f.get("subfolder", "")
                    outs.append({"file": f"{sub}/{f['filename']}".lstrip("/"),
                                 "url": view_url(self.comfy.browser_url, f["filename"], sub, f.get("type", OUTPUT_TYPE))})
        return outs
