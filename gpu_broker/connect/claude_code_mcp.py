"""Claude Code: the broker's MCP tools, as a user-scope server in ~/.claude.json `mcpServers`.

The same entry `claude mcp add --scope user --transport http` writes (Streamable HTTP, the key
as a Bearer header), set as a JSON key so it is backed up and undone like every other edit; the
CLI would take the key on its command line. Unlike `claude-code`, this changes no model: Claude
Code keeps its own and gains the tools. ~/.claude.json is kept private (it now holds the key).
A `gpu-broker` server the user defined is left alone (only connect's own entry is replaced). A
running Claude Code rewrites ~/.claude.json from memory and can drop the entry; connect reads
the file again after writing and says so.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from . import edits
from .core import PROVIDER_ID, FileChange, Plan, Target, skipped
from .state import ours

NAME, LABEL = "claude-code-mcp", "Claude Code (MCP tools)"
CONFIG = Path(".claude.json")


def detect(t: Target) -> tuple[bool, str]:
    path = t.home / CONFIG
    return (True, str(path)) if path.exists() or (t.home / ".claude").is_dir() else (False, f"{path} not found")


def entry(t: Target) -> dict[str, object]:
    return {"type": "http", "url": t.mcp_url, "headers": {"Authorization": f"Bearer {t.key}"}}


def present(raw: bytes | None) -> bool:
    """Whether a ~/.claude.json holds a `gpu-broker` MCP server."""
    try:
        servers: Any = edits.load_json(raw, str(CONFIG)).get("mcpServers")
    except edits.Unsupported:
        return False
    return isinstance(servers, dict) and PROVIDER_ID in servers


def dropped(raw: bytes | None) -> str | None:
    return None if present(raw) else (f"~/{CONFIG} no longer has the '{PROVIDER_ID}' MCP server: a running Claude Code "
                                       "rewrote it. Quit Claude Code, then run `gpu-broker connect --only claude-code-mcp` again")


def plan(t: Target) -> Plan:
    found, why = detect(t)
    if not found:
        return skipped(NAME, why)
    if why_not := t.mcp_skip():
        return skipped(NAME, why_not)
    path = t.home / CONFIG
    raw = path.read_bytes() if path.exists() else None
    if present(raw) and not ours(t.home, path, NAME):
        return skipped(NAME, f"{path} defines its own '{PROVIDER_ID}' MCP server")
    try:
        data = edits.load_json(raw, str(path))
        undo = edits.set_paths(data, {("mcpServers", PROVIDER_ID): entry(t)})
    except edits.Unsupported as e:
        return skipped(NAME, str(e))
    return Plan(NAME, [FileChange(path, edits.dump_json(data, raw), undo, force_mode=True, check=dropped)],
                notes=["restart Claude Code; `/mcp` lists the gpu-broker tools"])
