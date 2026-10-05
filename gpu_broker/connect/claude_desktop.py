"""Claude Desktop: the broker's MCP tools, as `mcpServers.gpu-broker` in claude_desktop_config.json.

Claude Desktop launches local commands only, so the entry runs `gpu-broker mcp` (MCP over
stdio, relayed to the broker's /mcp) with this machine's python, the broker URL as an argument
and the key in the entry's `env` (never on a command line). It is offered only where this
python can run it: gpu-broker with its `mcp` extra installed, which the zipapp from connect.sh
is not. A `gpu-broker` server the user defined is left alone (only connect's own entry is replaced).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from . import edits
from .claude_code_mcp import present
from .core import KEY_ENV, MCP_COMMAND_OPT, PROVIDER_ID, FileChange, Plan, Target, skipped
from .state import ours

NAME, LABEL = "claude-desktop", "Claude Desktop (MCP tools)"
FILE = "claude_desktop_config.json"
APPDATA_ENV, XDG_ENV = "APPDATA", "XDG_CONFIG_HOME"


def folder(t: Target, platform: str = sys.platform) -> Path:
    if platform == "darwin":
        return t.home / "Library" / "Application Support" / "Claude"
    if platform.startswith("win"):
        return Path(t.env.get(APPDATA_ENV) or t.home / "AppData" / "Roaming") / "Claude"
    return Path(t.env.get(XDG_ENV) or t.home / ".config") / "Claude"


def detect(t: Target) -> tuple[bool, str]:
    d = folder(t)
    return (True, str(d / FILE)) if d.is_dir() else (False, f"{d} not found")


def plan(t: Target) -> Plan:
    found, why = detect(t)
    if not found:
        return skipped(NAME, why)
    if why_not := t.mcp_skip():
        return skipped(NAME, why_not)
    if not (command := t.options.get(MCP_COMMAND_OPT)):
        return skipped(NAME, "it launches `gpu-broker mcp`, which needs gpu-broker[mcp] installed on this machine")
    argv = json.loads(command)
    path = folder(t) / FILE
    raw = path.read_bytes() if path.exists() else None
    if present(raw) and not ours(t.home, path, NAME):
        return skipped(NAME, f"{path} defines its own '{PROVIDER_ID}' MCP server")
    server = {"command": argv[0], "args": [*argv[1:], "--url", t.url], "env": {KEY_ENV: t.key}}
    try:
        data = edits.load_json(raw, str(path))
        undo = edits.set_paths(data, {("mcpServers", PROVIDER_ID): server})
    except edits.Unsupported as e:
        return skipped(NAME, str(e))
    return Plan(NAME, [FileChange(path, edits.dump_json(data, raw), undo, force_mode=True)],
                notes=["quit and reopen Claude Desktop; the gpu-broker tools appear under the tools menu"])
