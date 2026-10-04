"""Open WebUI: add the broker as an OpenAI connection through its admin API.

Needs the Open WebUI URL and an admin API key (`--openwebui-url`, `--openwebui-token`).
`GET /openai/config` reads the connections; `POST /openai/config/update` replaces the whole
lists, so both connect and disconnect read the current lists first and change only the
broker's entry: connect adds it (or refreshes the entry an earlier connect added),
disconnect removes it, and every connection and key added in Open WebUI meanwhile stays. A
connection to the broker's URL that the user set up themselves is left exactly as it is
(key and settings) and is not recorded, so disconnect never touches it. The manifest records
only the broker's base URL, how disconnect finds our entry. No Open WebUI keys are stored.
"""
from __future__ import annotations

import json
import urllib.request
from collections.abc import Callable
from typing import Any
from urllib.parse import urlsplit

from . import upstreams
from .core import PROVIDER_ID, ApiChange, Plan, Target, skipped

NAME, LABEL = "openwebui", "Open WebUI"
KEY_MODE = upstreams.SEPARATE
URL_OPT, TOKEN_OPT = "openwebui_url", "openwebui_token"
CONFIG, UPDATE = "/openai/config", "/openai/config/update"
URLS, KEYS, CONFIGS, ENABLE = "OPENAI_API_BASE_URLS", "OPENAI_API_KEYS", "OPENAI_API_CONFIGS", "ENABLE_OPENAI_API"
TIMEOUT_S = 30
SCHEMES = ("http", "https")
Http = Callable[[str, str, str, Any], Any]


def http(method: str, url: str, token: str, body: Any = None) -> Any:
    if urlsplit(url).scheme not in SCHEMES:
        raise ValueError(f"Open WebUI URL {url!r} is not http(s)")
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data, {"Authorization": f"Bearer {token}", "Content-Type": "application/json"},  # noqa: S310 — scheme checked above
                                 method=method)
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:  # noqa: S310 — scheme checked above
        return json.load(r)


def detect(t: Target) -> tuple[bool, str]:
    url = t.options.get(URL_OPT)
    return (True, url) if url else (False, "give --openwebui-url and --openwebui-token (an admin API key) to connect it")


def with_broker(config: dict[str, Any], t: Target) -> dict[str, Any]:
    """The config with the broker's connection added, or its key and settings refreshed."""
    urls, keys = list(config.get(URLS) or []), list(config.get(KEYS) or [])
    keys += [""] * (len(urls) - len(keys))
    configs = dict(config.get(CONFIGS) or {})
    i = urls.index(t.openai_base) if t.openai_base in urls else len(urls)
    if i == len(urls):
        urls.append(t.openai_base)
        keys.append(t.key)
    keys[i] = t.key
    configs[str(i)] = {**configs.get(str(i), {}), "enable": True, "prefix_id": "", "tags": [], "model_ids": [],
                       "connection_type": "external", "name": PROVIDER_ID}
    return {**config, ENABLE: True, URLS: urls, KEYS: keys, CONFIGS: configs}


def plan(t: Target, call: Http = http, ours: bool = False) -> Plan:
    """`ours`: an earlier connect added the broker's connection (the manifest records it)."""
    found, why = detect(t)
    token = t.options.get(TOKEN_OPT, "")
    if not found or not token:
        return skipped(NAME, why)
    base = t.options[URL_OPT].rstrip("/")

    def apply() -> Any:
        current = call("GET", base + CONFIG, token, None)
        if t.openai_base in (current.get(URLS) or []) and not ours:
            return None
        call("POST", base + UPDATE, token, with_broker(current, t))
        return {"base_url": t.openai_base, "existed": False}
    unchanged = f"{t.openai_base} is already a connection in Open WebUI at {base}, configured by you; left untouched"
    return Plan(NAME, api=ApiChange(f"add the broker as an OpenAI connection in Open WebUI at {base}", apply, unchanged))


def without_broker(config: dict[str, Any], base_url: str) -> dict[str, Any] | None:
    """The current config minus the broker's connection (later ones renumbered), or None if absent."""
    urls = list(config.get(URLS) or [])
    if base_url not in urls:
        return None
    i = urls.index(base_url)
    keys = list(config.get(KEYS) or [])
    keys += [""] * (len(urls) - len(keys))
    old = dict(config.get(CONFIGS) or {})
    configs = {str(j if j < i else j - 1): v for k, v in old.items() if k.isdigit() and (j := int(k)) != i}
    configs |= {k: v for k, v in old.items() if not k.isdigit()}   # older URL-keyed entries: not ours to drop
    return {**config, URLS: urls[:i] + urls[i + 1:], KEYS: keys[:i] + keys[i + 1:], CONFIGS: configs}


def restorer(url: str, token: str, call: Http = http) -> Callable[[Any], None]:
    """Disconnect: re-read the current config and remove only the broker's entry."""
    base = url.rstrip("/")

    def restore(record: Any) -> None:
        if not isinstance(record, dict) or "base_url" not in record:
            raise ValueError("the manifest entry predates this version; remove the connection in Open WebUI")
        if record.get("existed"):
            return   # configured before connect: the user's own connection, left in place
        new = without_broker(call("GET", base + CONFIG, token, None), record["base_url"])
        if new is not None:
            call("POST", base + UPDATE, token, new)
    return restore
