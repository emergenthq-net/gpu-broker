"""Read-only Supercomputer SKU discovery, isolated from broker execution paths."""
from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .provider_quotes import ProviderQuote, parse_supercomputer_skus

CATALOG_URL = "https://api.supercomputer.sh/v1/skus"
MAX_CATALOG_BYTES = 262_144
TIMEOUT_S = 8.0


class CatalogTransport(Protocol):
    def get(self, url: str, *, token: str, timeout_s: float) -> bytes: ...


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req: Request, fp: Any, code: int, msg: str, headers: Any, newurl: str) -> None:
        return None


class ReadOnlyCatalogTransport:
    """Fixed-origin GET only; no session or billing mutation methods."""

    def get(self, url: str, *, token: str, timeout_s: float) -> bytes:
        if url != CATALOG_URL or not token or any(c in token for c in "\r\n"):
            raise ValueError("invalid catalog request")
        # Fixed HTTPS origin is checked above; no caller-selected schemes or redirects.
        request = Request(url, headers={"Authorization": f"Bearer {token}", "Accept": "application/json"}, method="GET")  # noqa: S310 - exact fixed HTTPS URL only
        opener = build_opener(_NoRedirect())
        try:
            with opener.open(request, timeout=timeout_s) as response:
                data: bytes = response.read(MAX_CATALOG_BYTES + 1)
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            raise ValueError("catalog unavailable") from exc
        if len(data) > MAX_CATALOG_BYTES:
            raise ValueError("catalog exceeds size limit")
        return data


def discover_skus(
    *, token: str, observed_at_s: float, transport: CatalogTransport | None = None,
) -> tuple[ProviderQuote, ...]:
    """Fetch and validate catalog only; never infer execution authorization."""
    if not token or any(c in token for c in "\r\n"):
        raise ValueError("missing or invalid API token")
    client = transport if transport is not None else ReadOnlyCatalogTransport()
    data = client.get(CATALOG_URL, token=token, timeout_s=TIMEOUT_S)
    if len(data) > MAX_CATALOG_BYTES:
        raise ValueError("catalog exceeds size limit")
    try:
        payload: Any = json.loads(data)
    except (ValueError, UnicodeDecodeError) as exc:
        raise ValueError("invalid catalog JSON") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("invalid catalog response")
    return parse_supercomputer_skus(payload, observed_at_s=observed_at_s)
