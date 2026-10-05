"""`gpu-broker connect` / `disconnect`: point the AI apps on a machine at the broker, and back.

One module per client; each has NAME, LABEL, detect(target) -> (found, detail) and
plan(target) -> Plan. engine.py applies plans with backups and a manifest and undoes them.
Standard library only (it also runs from the zipapp that `connect.sh` downloads).
"""
from __future__ import annotations

from types import ModuleType

from . import claude_code, claude_code_mcp, claude_desktop, cline, codex, codex_mcp, continue_, openwebui, roo, shell

CLIENTS: dict[str, ModuleType] = {m.NAME: m for m in (shell, continue_, cline, roo, openwebui, codex, claude_code,
                                                              claude_code_mcp, claude_desktop, codex_mcp)}
