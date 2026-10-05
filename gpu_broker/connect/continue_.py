"""Continue (continue.dev): one model file in ~/.continue/models/.

Continue loads every `~/.continue/models/*.yaml` block next to config.yaml (source:
core/config/yaml/loadYaml.ts), so the user's config.yaml is never edited: connect writes
`gpu-broker.yaml`, disconnect deletes it. $CONTINUE_GLOBAL_DIR moves the folder.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import edits, upstreams
from .core import PROVIDER_ID, FileChange, Plan, Target, skipped

NAME, LABEL = "continue", "Continue"
KEY_MODE = upstreams.SEPARATE
DIR_ENV = "CONTINUE_GLOBAL_DIR"
DEFAULT_DIR = Path(".continue")
FILE = f"models/{PROVIDER_ID}.yaml"
ROLES = "[chat, edit, apply, summarize]"


def folder(t: Target) -> Path:
    return Path(t.env[DIR_ENV]) if t.env.get(DIR_ENV) else t.home / DEFAULT_DIR


def detect(t: Target) -> tuple[bool, str]:
    d = folder(t)
    return (True, str(d)) if d.is_dir() else (False, f"{d} not found")


def _q(value: str) -> str:
    return json.dumps(value)   # a JSON string is a valid YAML double-quoted scalar


def plan(t: Target) -> Plan:
    found, why = detect(t)
    if not found:
        return skipped(NAME, why)
    text = "\n".join([
        f"# {edits.NOTE}", f"name: {PROVIDER_ID}", "version: 1.0.0", "schema: v1", "models:",
        f"  - name: {_q(PROVIDER_ID + ' (' + t.model + ')')}", "    provider: openai", f"    model: {_q(t.model)}",
        f"    apiBase: {_q(t.openai_base)}", f"    apiKey: {_q(t.key)}", f"    roles: {ROLES}", ""])
    path = folder(t) / FILE
    return Plan(NAME, [FileChange(path, text.encode(edits.ENCODING), {"kind": edits.OWN_FILE})],
                notes=[f"pick '{PROVIDER_ID} ({t.model})' in Continue's model menu"])
