"""Hosted fallback: forward a drop-in request to the real provider (off by default).

Used only when `fallback.enabled` is set and the operator's upstream key is in the
environment (tuning.Fallback). The request goes out as the client sent it, minus the
broker's own fields (requester, wait, caps, session, ... — the provider would reject them),
with the operator's key in place of the broker credential; the broker's own token or client
keys are never forwarded. Every answer says which side served it: header `x-broker-served-by`
(`local` | `hosted`) and, on JSON bodies, `x_broker.served_by` with the reason.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator, Mapping
from enum import StrEnum
from http import HTTPStatus
from typing import Any
from urllib.parse import urlsplit

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from ..constants import ANTHROPIC_VERSION_HEADER, AUTH_SCHEME, BROKER_FIELDS, ERR_EVENT, HTTP_SCHEMES, SERVED_BY_HEADER
from ..tuning import Fallback

JSON_TYPE = "application/json"
SSE_TYPE = "text/event-stream"
DEFAULT_ANTHROPIC_VERSION = "2023-06-01"
ANTHROPIC_BETA = "anthropic-beta"
HOSTED = "hosted"
LOCAL = "local"   # same value as completion.LOCAL
FAILED = "hosted fallback failed"
BROKER_ONLY = BROKER_FIELDS - {"model", "stream"}   # both APIs have these; the rest are the broker's


class Api(StrEnum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


def served(side: str) -> dict[str, str]:
    return {SERVED_BY_HEADER: side}


class Upstream:
    def __init__(self, cfg: Fallback, env: Mapping[str, str] | None = None,
                 urlopen: Callable[..., Any] | None = None) -> None:
        self.cfg, self.env, self._urlopen = cfg, os.environ if env is None else env, urlopen

    def urlopen(self, req: urllib.request.Request, timeout: float) -> Any:
        """Looked up per call (not bound at import), so tests' network guard always applies."""
        return (self._urlopen or urllib.request.urlopen)(req, timeout=timeout)

    def _key(self, api: Api) -> str:
        return self.env.get(self.cfg.openai_key_env if api is Api.OPENAI else self.cfg.anthropic_key_env, "")

    def available(self, api: Api) -> bool:
        return self.cfg.enabled and bool(self._key(api))

    def _request(self, api: Api, path: str, body: Mapping[str, Any], request: Request) -> urllib.request.Request:
        base = self.cfg.openai_url if api is Api.OPENAI else self.cfg.anthropic_url
        if urlsplit(base).scheme not in HTTP_SCHEMES:
            raise ValueError(f"fallback URL {base!r} is not http(s)")
        headers = {"Content-Type": JSON_TYPE}
        if api is Api.OPENAI:
            headers["Authorization"] = f"{AUTH_SCHEME} {self._key(api)}"
        else:
            headers.update({"x-api-key": self._key(api),
                            ANTHROPIC_VERSION_HEADER: request.headers.get(ANTHROPIC_VERSION_HEADER, DEFAULT_ANTHROPIC_VERSION)})
            if beta := request.headers.get(ANTHROPIC_BETA):
                headers[ANTHROPIC_BETA] = beta
        sent = {k: v for k, v in body.items() if k not in BROKER_ONLY}
        return urllib.request.Request(base.rstrip("/") + path, json.dumps(sent).encode(), headers, method="POST")  # noqa: S310 — scheme checked

    def forward(self, api: Api, path: str, body: Mapping[str, Any], request: Request, reason: str) -> Response:
        """The provider's answer, passed through (errors included) and marked as hosted."""
        try:
            resp = self.urlopen(self._request(api, path, body, request), timeout=self.cfg.timeout_s)
        except urllib.error.HTTPError as e:   # the provider's own error, in its own shape
            return Response(e.read(), status_code=e.code, media_type=JSON_TYPE, headers=served(HOSTED))
        except OSError as e:
            raise HTTPException(HTTPStatus.BAD_GATEWAY, {"error": f"{FAILED}: {str(e)[:ERR_EVENT]}"}) from None
        if body.get("stream"):
            return StreamingResponse(_relay(resp), media_type=SSE_TYPE, headers=served(HOSTED))
        with resp:
            data = json.load(resp)
        if isinstance(data, dict):
            data["x_broker"] = {"served_by": HOSTED, "reason": reason}
        return JSONResponse(data, headers=served(HOSTED))


def _relay(resp: Any) -> Iterator[bytes]:
    with resp:
        yield from resp
