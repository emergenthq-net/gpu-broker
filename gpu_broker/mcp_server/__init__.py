"""gpu-broker as an MCP server, for Claude, ChatGPT and Codex: tools to generate images, video
and 3D, edit images, ask the local LLM and see the GPU.

- core.py: what the tools do (no MCP SDK).
- server.py: the tools, on the MCP Python SDK (`pip install 'gpu-broker[mcp]'`).
- http.py: Streamable HTTP at /mcp on the broker, behind the broker's credentials.
- stdio.py: `gpu-broker mcp`, for apps that launch a local command: relays stdio to /mcp.
"""
from __future__ import annotations

import importlib.util

SDK = "mcp"


def available() -> bool:
    """The MCP SDK is installed (without importing it)."""
    return importlib.util.find_spec(SDK) is not None
