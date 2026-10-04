"""Which cloud answers count as failed (go local) and which are the caller's to see.

- OK: 2xx.
- FAIL (fail over, and count against the provider's breaker): no connection, DNS, TLS, a
  timeout before the first byte, 408, any 5xx (Anthropic's 529 `overloaded_error` included),
  or a stream whose first event is an error.
- QUOTA (fail over, and stop sending this credential to the provider for a while): 402, or a
  400/429 whose error *type or code* says the quota or credit is used up: OpenAI's
  `insufficient_quota`, Anthropic's `billing_error`. Never the message text: OpenAI's ordinary
  rate-limit message mentions billing, and a plain rate limit is the caller's to see.
- CLIENT (returned as-is, never failed over): every other 4xx — a bad request, a bad key, a
  plain rate limit — because a request that is wrong should not quietly go to another model.
Reasons are built from the status and the provider's error *type* only, never its message,
so nothing the provider echoes (a key prefix, say) reaches the log.
"""
from __future__ import annotations

import json
import re
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from email.utils import parsedate_to_datetime
from enum import StrEnum
from http import HTTPStatus
from typing import Any

QUOTA_CODES = frozenset({"insufficient_quota", "billing_error"})   # error.type or error.code
QUOTA_STATUSES = frozenset({HTTPStatus.BAD_REQUEST, HTTPStatus.PAYMENT_REQUIRED, HTTPStatus.TOO_MANY_REQUESTS})
IDENT = re.compile(r"[a-z][a-z0-9_]{0,47}")
DURATION = re.compile(r"(\d+(?:\.\d+)?)(ms|h|m|s)")
UNIT_S = {"ms": 0.001, "s": 1, "m": 60, "h": 3600}
RESET_HEADERS = ("anthropic-ratelimit-requests-reset", "anthropic-ratelimit-tokens-reset",
                 "anthropic-ratelimit-input-tokens-reset", "anthropic-ratelimit-output-tokens-reset",
                 "x-ratelimit-reset-requests", "x-ratelimit-reset-tokens")


class Kind(StrEnum):
    OK = "ok"
    CLIENT = "client"
    FAIL = "fail"
    QUOTA = "quota"


@dataclass(frozen=True)
class Verdict:
    kind: Kind
    reason: str = ""
    retry_after_s: float = 0.0   # QUOTA: how long the provider says to stay away (0 = not said)


def error_codes(body: bytes) -> list[str]:
    """The provider's error code and type, in that order, those that are identifiers."""
    try:
        data: Any = json.loads(body)
    except ValueError:
        return []
    err = data.get("error") if isinstance(data, dict) else None
    if not isinstance(err, dict):
        return []
    return [v for key in ("code", "type") if isinstance(v := err.get(key), str) and IDENT.fullmatch(v)]


def error_type(body: bytes) -> str:
    """The provider's error code, else its type (`insufficient_quota`, `overloaded_error`)."""
    return next(iter(error_codes(body)), "")


def is_quota(status: int, body: bytes) -> bool:
    if status == HTTPStatus.PAYMENT_REQUIRED:
        return True
    return status in QUOTA_STATUSES and not QUOTA_CODES.isdisjoint(error_codes(body))


def classify(provider: str, status: int, headers: Mapping[str, str], body: bytes,
             wall: Callable[[], float] = time.time) -> Verdict:
    if HTTPStatus.OK <= status < HTTPStatus.MULTIPLE_CHOICES:
        return Verdict(Kind.OK)
    label = f"{provider} {status} {error_type(body)}".rstrip()
    if is_quota(status, body):
        return Verdict(Kind.QUOTA, label, retry_after(headers, wall()))
    if status >= HTTPStatus.INTERNAL_SERVER_ERROR or status == HTTPStatus.REQUEST_TIMEOUT:
        return Verdict(Kind.FAIL, label)
    return Verdict(Kind.CLIENT, label)


def failed(provider: str, what: str) -> Verdict:
    """A transport failure (no answer at all); `what` is a fixed description, not an exception text."""
    return Verdict(Kind.FAIL, f"{provider} {what}")


def first_event(buf: bytes) -> bytes | None:
    """The stream's first real event: the first complete one that is not a comment (`: ping`) or
    a keep-alive (Anthropic's `event: ping`). None if `buf` does not hold one yet."""
    blocks = buf.replace(b"\r\n", b"\n").split(b"\n\n")
    for block in blocks[:-1]:   # the last piece is not known to be complete
        lines = [ln for ln in block.split(b"\n") if ln and not ln.startswith(b":")]
        if lines and b"event: ping" not in lines and b"event:ping" not in lines:
            return b"\n".join(lines)
    return None


def first_event_failed(provider: str, chunk: bytes) -> Verdict | None:
    """A stream that opens with an error event: Anthropic's `event: error` (an `overloaded_error`,
    say) or an OpenAI-style `data: {"error": ...}`. `chunk` is the first real event (first_event)."""
    lines = chunk.decode("utf-8", "replace").splitlines()
    event = next((ln[6:].strip() for ln in lines if ln.startswith("event:")), "")
    payload = next((ln[5:].strip() for ln in lines if ln.startswith("data:")), "")
    try:
        data: Any = json.loads(payload)
    except ValueError:
        data = None
    is_error = isinstance(data, dict) and (data.get("type") == "error" or "error" in data)
    if event != "error" and not is_error:
        return None
    return Verdict(Kind.FAIL, f"{provider} stream error {error_type(payload.encode())}".rstrip())


def retry_after(headers: Mapping[str, str], now: float) -> float:
    """Seconds the provider asks us to wait: Retry-After (seconds or a date) and the rate-limit
    reset headers (an RFC 3339 time for Anthropic, a duration like `6m0s` for OpenAI); the longest."""
    low = {k.lower(): v for k, v in headers.items()}
    waits = [_seconds(low[h], now) for h in ("retry-after", *RESET_HEADERS) if h in low]
    return max([w for w in waits if w is not None], default=0.0)


def _seconds(value: str, now: float) -> float | None:
    value = value.strip()
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    if (parts := DURATION.findall(value)) and "".join(n + u for n, u in parts) == value:
        return sum(float(n) * UNIT_S[u] for n, u in parts)
    try:
        when = parsedate_to_datetime(value).timestamp() if not value[:4].isdigit() else _iso(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, when - now)


def _iso(value: str) -> float:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
