"""What a connector works with: the target (broker URL, key, model, home) and its plan.

This package is standard-library only: the broker serves it as a zipapp to remote machines
(`connect.sh`), which may have nothing but python3.
"""
from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

OPENAI_PATH = "/v1"            # OpenAI clients want the /v1 prefix; Anthropic clients add it themselves
MCP_PATH = "/mcp"              # the broker's MCP endpoint (Streamable HTTP; gpu_broker.mcp_server)
MCP_OPT = "mcp"                # Target option: set when the broker serves /mcp (its /health says so)
MCP_COMMAND_OPT = "mcp_command"   # Target option: JSON argv of a local `gpu-broker mcp`, when this machine can run one
PROVIDER_ID = "gpu-broker"     # the name every connector gives its entry, so it is easy to find
KEY_PREFIX = "gbk_"             # every client key starts with it (gpu_broker.keys issues them)
KEY_BODY_LEN = 43              # secrets.token_urlsafe(32): 32 bytes as unpadded base64url
KEY_SHAPE = re.compile(rf"{KEY_PREFIX}[A-Za-z0-9_-]{{{KEY_BODY_LEN}}}")
PASSTHROUGH_PATH = "/v1/upstreams/passthrough"   # = constants.PASSTHROUGH_PATH; a client key may read it
BROKER_KEY_HEADER = "x-gpu-broker-key"   # the broker key beside a tool's own provider key (gpu_broker.constants)
OWN_KEY_OPT = {"anthropic": "own_key_anthropic", "openai": "own_key_openai"}   # Target options: see upstreams.py
KEY_ENV = "GPU_BROKER_API_KEY"   # env var the shell block sets for clients that read a key by name
SECRET_MODE = 0o600            # new files that hold the key
Undo = dict[str, Any]          # how to take our change back out of a file someone has edited since
# What a broker URL, key or model id may contain. These values end up in shell rc files,
# YAML, TOML and an installer script; anything outside this set (quotes, $, backticks,
# spaces, newlines) is refused rather than escaped.
SAFE = re.compile(r"[\w.:/@+\[\]-]+")   # [ ] for an IPv6 host
SCHEMES = ("http", "https")


def is_client_key(value: str) -> bool:
    """`value` has the shape of a key gpu_broker.keys issues (the prefix alone is not enough)."""
    return KEY_SHAPE.fullmatch(value) is not None


def checked(url: str, key: str, model: str) -> tuple[str, str, str]:
    """The one check every connector's input goes through (ValueError if any value is unsafe).
    An empty key or model is allowed: `clients` detects without them."""
    for what, value in (("broker URL", url), ("key", key), ("model", model)):
        if value and not SAFE.fullmatch(value):
            raise ValueError(f"{what} {value!r} has characters connect will not write into config files")
    parts = urlsplit(url)
    if parts.scheme not in SCHEMES or not parts.netloc:
        raise ValueError(f"broker URL {url!r} is not an http(s) URL with a host")
    return url, key, model


@dataclass(frozen=True)
class Target:
    url: str                      # the broker, e.g. http://gpu-host:8095 (no /v1)
    key: str                      # the client key this machine uses
    model: str                    # model id clients should ask for by default
    home: Path
    env: Mapping[str, str] = field(default_factory=dict)
    options: Mapping[str, str] = field(default_factory=dict)   # e.g. openwebui_url, claude_code

    def __post_init__(self) -> None:
        checked(self.url, self.key, self.model)

    @property
    def openai_base(self) -> str:
        return self.url.rstrip("/") + OPENAI_PATH

    @property
    def anthropic_base(self) -> str:
        return self.url.rstrip("/")

    @property
    def mcp_url(self) -> str:
        return self.url.rstrip("/") + MCP_PATH

    def mcp_skip(self) -> str | None:
        """Why MCP cannot be registered for this broker, or None."""
        return None if self.options.get(MCP_OPT) else "the broker does not serve MCP (install gpu-broker[mcp] where it runs)"


@dataclass(frozen=True)
class FileChange:
    path: Path
    new: bytes                    # the whole file as it should be
    undo: Undo                    # see edits.undo
    mode: int = SECRET_MODE       # for a file we create; an existing file keeps its mode...
    force_mode: bool = False      # ...unless set (a client that requires `mode`, e.g. Cline's 600)
    # Re-read a moment after writing: why our entry is gone (a running app rewrote the file), or None.
    check: Callable[[bytes | None], str | None] | None = None


@dataclass(frozen=True)
class ApiChange:
    """A change made through a client's own API (Open WebUI): `apply()` returns the record
    disconnect needs, or None when it changed nothing (then `unchanged` is reported)."""
    describe: str
    apply: Callable[[], Any]
    unchanged: str = "already configured; left untouched"


@dataclass
class Plan:
    client: str
    files: list[FileChange] = field(default_factory=list)
    api: ApiChange | None = None
    skip: str | None = None       # why nothing is done for this client
    keys: str = ""                # which credentials the client sends (`clients` shows it), if it matters
    notes: list[str] = field(default_factory=list)


def skipped(client: str, reason: str) -> Plan:
    return Plan(client, skip=reason)
