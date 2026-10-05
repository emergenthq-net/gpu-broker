"""Codex CLI: the broker's MCP tools, as `[mcp_servers.gpu-broker]` in $CODEX_HOME/config.toml.

Unlike the `codex` profile (a separate file, used only with `--profile`), MCP servers belong in
config.toml so plain `codex` gets the tools. The table goes in a marked block (edits.put_block),
so disconnect removes exactly it; the result must parse as TOML, and a `gpu-broker` server
defined outside the block is left alone. Streamable HTTP with the key as a static header
(`http_headers`, the key Codex reads; `headers` would be ignored), and a tool timeout above the
broker's own wait, so a long job comes back as a job id rather than a Codex timeout. config.toml
is kept private once it holds the key.
"""
from __future__ import annotations

import json
import tomllib
from pathlib import Path

from . import edits
from .codex import CONFIG, folder
from .core import PROVIDER_ID, FileChange, Plan, Target, skipped

NAME, LABEL = "codex-mcp", "Codex CLI (MCP tools)"
TOOL_TIMEOUT_S = 600   # above any sensible mcp.wait_s; Codex's default (60 s) is shorter than some


def detect(t: Target) -> tuple[bool, str]:
    d = folder(t)
    return (True, str(d / CONFIG)) if d.is_dir() else (False, f"{d} not found")


def block(t: Target) -> list[str]:
    q = json.dumps   # a JSON string is a valid TOML basic string
    return [f"[mcp_servers.{q(PROVIDER_ID)}]", f"url = {q(t.mcp_url)}",
            f"http_headers = {{ {q('Authorization')} = {q('Bearer ' + t.key)} }}", f"tool_timeout_sec = {TOOL_TIMEOUT_S}"]


def plan(t: Target) -> Plan:
    found, why = detect(t)
    if not found:
        return skipped(NAME, why)
    if why_not := t.mcp_skip():
        return skipped(NAME, why_not)
    path: Path = folder(t) / CONFIG
    text = path.read_text(edits.ENCODING) if path.exists() else ""
    try:
        if PROVIDER_ID in (tomllib.loads(edits.strip_block(text)).get("mcp_servers") or {}):
            return skipped(NAME, f"{path} defines its own '{PROVIDER_ID}' MCP server")
        new = edits.put_block(text, block(t))
        tomllib.loads(new)
    except (tomllib.TOMLDecodeError, edits.Unsupported) as e:
        return skipped(NAME, f"{path} is not TOML connect can extend ({e}); Codex would not start either")
    undo = {"kind": edits.BLOCK, "comment": "#"}
    return Plan(NAME, [FileChange(path, new.encode(edits.ENCODING), undo, force_mode=True)],
                notes=["start a new Codex session; `codex mcp list` shows gpu-broker"])
