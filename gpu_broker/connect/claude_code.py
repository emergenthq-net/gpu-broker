"""Claude Code: ANTHROPIC_BASE_URL and the broker key in ~/.claude/settings.json `env`, plus
ENABLE_TOOL_SEARCH=true (a non-Anthropic base URL turns MCP tool search off otherwise).

Opt-in only (`--claude-code`): this points the user's main coding assistant at the broker.
Which key Claude Code then sends depends on the broker (upstreams.py):
- cloud failover on, with an Anthropic provider that passes the client's key through: Claude
  Code keeps its own login or API key, and the broker key goes in ANTHROPIC_CUSTOM_HEADERS as
  `x-gpu-broker-key` (added to any headers already set there). Claude models are then answered
  by Anthropic on the user's own account, and locally when that fails. A broker key left in
  ANTHROPIC_AUTH_TOKEN by an earlier connect is taken out: Claude Code would send it instead
  of the user's own login.
- otherwise: the broker key in ANTHROPIC_AUTH_TOKEN, which Claude Code sends as a Bearer header
  (no approval prompt, unlike ANTHROPIC_API_KEY), and claude-* names map to a local model.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from . import edits
from .core import BROKER_KEY_HEADER, OWN_KEY_OPT, FileChange, Plan, Target, is_client_key, skipped
from .upstreams import DUAL, SINGLE

NAME, LABEL = "claude-code", "Claude Code (opt-in)"
OPT_IN = "claude_code"
SETTINGS = Path(".claude") / "settings.json"
TOOL_SEARCH, HEADERS, TOKEN = "ENABLE_TOOL_SEARCH", "ANTHROPIC_CUSTOM_HEADERS", "ANTHROPIC_AUTH_TOKEN"
KEY_MODE = f"{DUAL} when the broker passes Anthropic keys through, else {SINGLE}"


def detect(t: Target) -> tuple[bool, str]:
    d = t.home / SETTINGS.parent
    return (True, str(t.home / SETTINGS)) if d.is_dir() else (False, f"{d} not found")


def with_broker_header(current: Any, key: str) -> str:
    """ANTHROPIC_CUSTOM_HEADERS (one `Name: value` per line) with ours replacing any earlier one."""
    lines = current.splitlines() if isinstance(current, str) else []
    kept = [ln for ln in lines if ln.partition(":")[0].strip().lower() != BROKER_KEY_HEADER]
    return "\n".join([*kept, f"{BROKER_KEY_HEADER}: {key}"])


def plan(t: Target) -> Plan:
    found, why = detect(t)
    if not found:
        return skipped(NAME, why)
    if not t.options.get(OPT_IN):
        return skipped(NAME, "it would switch your main Claude Code assistant to the broker; opt in with --claude-code")
    path = t.home / SETTINGS
    raw = path.read_bytes() if path.exists() else None
    own = bool(t.options.get(OWN_KEY_OPT["anthropic"]))
    try:
        data = edits.load_json(raw, str(path))
        found_env = data.get("env")
        env: dict[str, Any] = found_env if isinstance(found_env, dict) else {}
        auth: dict[tuple[str, ...], Any] = ({("env", HEADERS): with_broker_header(env.get(HEADERS), t.key)} if own
                                            else {("env", TOKEN): t.key})
        held = str(env.get(TOKEN, ""))
        if own and (held == t.key or is_client_key(held)):   # ours, from a single-key connect before failover
            auth[("env", TOKEN)] = edits.REMOVE
        undo = edits.set_paths(data, {("env", "ANTHROPIC_BASE_URL"): t.anthropic_base, **auth, ("env", TOOL_SEARCH): "true"})
    except edits.Unsupported as e:
        return skipped(NAME, str(e))
    note = ("Claude Code keeps its own login or API key (the broker sends Claude models to Anthropic with it, "
            "and to a local model when that fails); the broker key travels as x-gpu-broker-key" if own else
            "models it asks for (claude-*) map to the broker's default model")
    return Plan(NAME, [FileChange(path, edits.dump_json(data, raw), undo)], notes=[f"restart Claude Code; {note}"],
                keys=DUAL if own else SINGLE)
