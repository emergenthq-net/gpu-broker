"""What registering the broker's MCP tools needs to know: whether the broker serves /mcp, and
whether this machine can run `gpu-broker mcp` (for apps that launch a command). Standard library."""
from __future__ import annotations

import importlib.util
import json
import sys
from collections.abc import Callable
from typing import Any

from .core import MCP_COMMAND_OPT, MCP_OPT


def serves_mcp(url: str, call: Callable[[str, str, str, Any], Any]) -> bool:
    """The broker's /health says it serves /mcp (no credential needed)."""
    try:
        return bool(call("GET", url + "/health", "", None).get(MCP_OPT))
    except (OSError, ValueError, AttributeError):
        return False


def local_mcp_command() -> str:
    """JSON argv of `gpu-broker mcp` with this python, when it can run it (gpu-broker and its mcp
    extra installed; never from the zipapp, which carries only the connectors), else ""."""
    try:
        ok = importlib.util.find_spec("mcp") is not None and importlib.util.find_spec("gpu_broker.mcp_server") is not None
    except (ImportError, ValueError):
        ok = False
    return json.dumps([sys.executable, "-m", "gpu_broker", "mcp"]) if ok else ""


def options(serves: bool) -> dict[str, str]:
    """The Target options the MCP connectors read (core.MCP_OPT, core.MCP_COMMAND_OPT)."""
    if not serves:
        return {}
    command = local_mcp_command()
    return {MCP_OPT: "1", **({MCP_COMMAND_OPT: command} if command else {})}
