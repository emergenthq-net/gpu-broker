"""The `upstreams:` config section: cloud providers, and an ordered fallback chain per model name.

    upstreams:
      providers:
        anthropic: {url: https://api.anthropic.com, api: anthropic}
        openai:    {url: https://api.openai.com, api: openai, key_env: UPSTREAM_OPENAI_API_KEY}
      routes:                       # first matching pattern wins (fnmatch: *, ?, [..])
        "claude-*": [anthropic, my-local-llm]
        "gpt-*":    [openai, my-local-llm]

A chain is one or more providers, optionally ending in one local catalog model (checked
against the catalog when the app starts, `local_names`). A model name no route matches is
served the way it always was. Credentials: `key_env` names an environment variable holding the
operator's key for that provider. The client's own provider key (web/failover.py) goes only to a
provider with `pass_client_key`, which defaults to true just for the official hosts
(https://api.anthropic.com, https://api.openai.com): a key a client meant for OpenAI must not
reach a proxy that also speaks its API. Any other provider uses its `key_env` key, or is skipped.
Keys never come from this file. Off (no providers, no routes) by default.
"""
from __future__ import annotations

import fnmatch
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any
from urllib.parse import urlsplit

from ..constants import HTTP_SCHEMES

OFFICIAL_HOSTS = frozenset({"api.anthropic.com", "api.openai.com"})   # pass_client_key defaults on


class Api(StrEnum):
    ANTHROPIC = "anthropic"   # /v1/messages
    OPENAI = "openai"         # /v1/chat/completions and /v1/responses


@dataclass(frozen=True)
class Provider:
    name: str
    url: str
    api: Api
    key_env: str = ""
    pass_client_key: bool = False   # may the client's own provider key be sent here?

    def configured_key(self, env: Mapping[str, str] | None = None) -> str:
        return (os.environ if env is None else env).get(self.key_env, "") if self.key_env else ""


@dataclass(frozen=True)
class BreakerCfg:
    failures: int = 3            # consecutive failures that open a provider's breaker
    probe_s: float = 15          # first probe after it opens; doubles per failed probe ...
    probe_max_s: float = 300     # ... up to this
    quota_probe_s: float = 900   # after a quota/credit error, unless Retry-After or a reset header says longer
    trial_s: float = 120         # a half-open trial that never reports back is given up after this


@dataclass(frozen=True)
class UpTimeouts:
    connect_s: float = 5         # TCP + TLS to the provider
    probe_s: float = 10          # a breaker's probe (GET /v1/models): the whole answer
    first_byte_s: float = 30     # streamed call: until the first event arrives
    response_s: float = 600      # unstreamed call: the whole answer (it only comes when done)
    idle_s: float = 120          # streamed call: the longest gap between events once it has started


@dataclass(frozen=True)
class Upstreams:
    providers: Mapping[str, Any] = field(default_factory=dict)   # name -> Provider (parsed below)
    routes: Mapping[str, Any] = field(default_factory=dict)      # pattern -> tuple of names (parsed below)
    breaker: BreakerCfg = field(default_factory=BreakerCfg)
    timeouts: UpTimeouts = field(default_factory=UpTimeouts)

    def __post_init__(self) -> None:
        providers = {name: _provider(name, spec) for name, spec in (self.providers or {}).items()}
        routes = {str(pat): _chain(str(pat), chain, providers) for pat, chain in (self.routes or {}).items()}
        object.__setattr__(self, "providers", providers)
        object.__setattr__(self, "routes", routes)

    @property
    def enabled(self) -> bool:
        return bool(self.routes)

    def chain(self, model: str) -> tuple[str, ...] | None:
        """The fallback chain for `model`: the first route whose pattern matches, else None."""
        return next((c for pat, c in self.routes.items() if fnmatch.fnmatchcase(model, pat)), None)

    def local_for(self, model: str) -> str | None:
        """The local model at the end of `model`'s chain, if it has one."""
        chain = self.chain(model)
        return chain[-1] if chain and chain[-1] not in self.providers else None

    def local_names(self) -> set[str]:
        """Every chain entry that is not a provider: each must be a catalog model."""
        return {n for c in self.routes.values() for n in c if n not in self.providers}


def _provider(name: Any, spec: Any) -> Provider:
    if isinstance(spec, Provider):   # already parsed (dataclasses.replace runs __post_init__ again)
        return spec
    where = f"upstreams.providers.{name}"
    if not isinstance(spec, Mapping):
        raise ValueError(f"{where}: expected a mapping with url and api")
    unknown = set(spec) - {"url", "api", "key_env", "pass_client_key"}
    if unknown:
        raise ValueError(f"{where}: unknown keys {sorted(unknown)}")
    url = str(spec.get("url") or "").rstrip("/")
    if urlsplit(url).scheme not in HTTP_SCHEMES or not urlsplit(url).netloc:
        raise ValueError(f"{where}.url: {url!r} is not an http(s) URL")
    try:
        api = Api(spec.get("api", name))
    except ValueError:
        raise ValueError(f"{where}.api: one of {[a.value for a in Api]}") from None
    official = urlsplit(url).scheme == "https" and urlsplit(url).netloc in OFFICIAL_HOSTS
    passing = spec.get("pass_client_key", official)
    if not isinstance(passing, bool):
        raise ValueError(f"{where}.pass_client_key: true or false")
    return Provider(str(name), url, api, str(spec.get("key_env") or ""), passing)


def _chain(pattern: str, chain: Any, providers: Mapping[str, Provider]) -> tuple[str, ...]:
    where = f"upstreams.routes[{pattern!r}]"
    if isinstance(chain, tuple):   # already parsed
        chain = list(chain)
    if not isinstance(chain, list) or not chain or not all(isinstance(n, str) and n for n in chain):
        raise ValueError(f"{where}: expected a list of provider and model names")
    if chain[0] not in providers:
        raise ValueError(f"{where}: must start with a provider ({sorted(providers)})")
    local = [n for n in chain if n not in providers]
    if len(local) > 1 or (local and chain[-1] != local[0]):
        raise ValueError(f"{where}: at most one local model, and it must come last")
    return tuple(chain)
