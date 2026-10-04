"""Shells: a marked block in ~/.zshrc, ~/.bashrc and fish's config.fish.

The block sets the variables the OpenAI SDK (and tools built on it) reads, so a new terminal
talks to the broker: OPENAI_BASE_URL, OPENAI_API_KEY, OPENAI_API_BASE (Aider), plus
GPU_BROKER_URL and GPU_BROKER_API_KEY. Two exceptions, so connect never silently takes over
something else:
- an OPENAI_API_KEY already in the environment (a real OpenAI key) is not shadowed: the
  OpenAI variables are left out and only the GPU_BROKER_* ones are set;
- ANTHROPIC_* is set only with --claude-code, since Claude Code and every Anthropic SDK read it.
Values are single-quoted, and were checked against core.SAFE before any plan is made.
A shell is configured when its rc file exists, or when it is the user's login shell ($SHELL)
— then the rc file is created.
"""
from __future__ import annotations

import os
from pathlib import Path

from . import edits, upstreams
from .core import KEY_ENV, FileChange, Plan, Target, is_client_key, skipped

NAME, LABEL = "shell", "Shell (zsh, bash, fish)"
KEY_MODE = upstreams.SHELL
RC = {"zsh": Path(".zshrc"), "bash": Path(".bashrc"), "fish": Path(".config") / "fish" / "config.fish"}
URL_ENV, OPENAI_KEY, ANTHROPIC_KEY = "GPU_BROKER_URL", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"
CLAUDE_CODE = "claude_code"


def _foreign(t: Target, var: str) -> bool:
    """The environment already holds someone else's key in `var` (not one connect issued)."""
    value = t.env.get(var, "")
    return bool(value) and value != t.key and not is_client_key(value)


def variables(t: Target) -> tuple[dict[str, str], list[str]]:
    """(what the block exports, notes on what it left out and why)."""
    out, notes = {URL_ENV: t.url, KEY_ENV: t.key}, []
    if _foreign(t, OPENAI_KEY):
        notes.append(f"{OPENAI_KEY} is already set to another key, so the OpenAI variables are left alone; "
                     f"tools can read {URL_ENV} and {KEY_ENV}")
    else:
        out |= {"OPENAI_BASE_URL": t.openai_base, "OPENAI_API_BASE": t.openai_base, OPENAI_KEY: t.key}
    if not t.options.get(CLAUDE_CODE):
        notes.append("ANTHROPIC_* not exported (it would redirect Claude Code too); use --claude-code to opt in")
    elif _foreign(t, ANTHROPIC_KEY):
        notes.append(f"{ANTHROPIC_KEY} is already set to another key, so ANTHROPIC_* is left alone")
    else:
        out |= {"ANTHROPIC_BASE_URL": t.anthropic_base, ANTHROPIC_KEY: t.key}
    return out, notes


def quote(shell: str, value: str) -> str:
    """A single-quoted literal: nothing inside is expanded by sh, bash, zsh or fish."""
    if shell == "fish":
        return "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"
    return "'" + value.replace("'", "'\\''") + "'"


def _line(shell: str, name: str, value: str) -> str:
    return f"set -gx {name} {quote(shell, value)}" if shell == "fish" else f"export {name}={quote(shell, value)}"


def shells(t: Target) -> list[str]:
    login = os.path.basename(t.env.get("SHELL", ""))
    return [s for s, rc in RC.items() if (t.home / rc).exists() or s == login]


def detect(t: Target) -> tuple[bool, str]:
    found = shells(t)
    return bool(found), ", ".join(found) if found else "no zsh, bash or fish config, and $SHELL is none of them"


def plan(t: Target) -> Plan:
    found = shells(t)
    if not found:
        return skipped(NAME, detect(t)[1])
    exported, notes = variables(t)
    files = []
    for shell in found:
        path = t.home / RC[shell]
        text = path.read_text(edits.ENCODING) if path.exists() else ""
        body = [_line(shell, k, v) for k, v in exported.items()]
        try:
            new = edits.put_block(text, body)
        except edits.Unsupported as e:
            return skipped(NAME, f"{path}: {e}")
        files.append(FileChange(path, new.encode(edits.ENCODING), {"kind": edits.BLOCK, "comment": "#"}))
    return Plan(NAME, files, notes=[f"exports {', '.join(exported)}", *notes,
                                    "open a new terminal (or `source` the file) to pick it up"])
