"""Cline (4.x): its provider file, ~/.cline/data/settings/providers.json.

Cline 4 keeps provider settings in that file (sdk/packages/shared/src/storage/paths.ts;
$CLINE_PROVIDER_SETTINGS_PATH, $CLINE_DATA_DIR or $CLINE_DIR move it). Connect adds an
"openai-compatible" provider pointing at the broker and makes it the last-used one; Cline
treats the whole file as empty if any part is invalid, so the values are written exactly in
its schema, the whole file is checked after the change (every provider's `updatedAt` an ISO
time, every `baseUrl` an http(s) URL) and kept at mode 600, as Cline keeps it. If the user's
own entries would fail that check, nothing is written. Older Cline (3.x) kept them in VS Code's internal storage, which nothing else
can write: there the connector skips and says so.
"""
from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from . import edits, upstreams
from .core import FileChange, Plan, Target, skipped

NAME, LABEL = "cline", "Cline"
KEY_MODE = upstreams.SEPARATE
PROVIDER = "openai-compatible"
PATH_ENV, DATA_ENV, DIR_ENV = "CLINE_PROVIDER_SETTINGS_PATH", "CLINE_DATA_DIR", "CLINE_DIR"
DEFAULT_DIR = Path(".cline")
SETTINGS = Path("settings") / "providers.json"
VERSION = 1
ISO = "%Y-%m-%dT%H:%M:%S.000Z"


def settings_path(t: Target) -> Path:
    if t.env.get(PATH_ENV):
        return Path(t.env[PATH_ENV])
    if t.env.get(DATA_ENV):
        return Path(t.env[DATA_ENV]) / SETTINGS
    root = Path(t.env[DIR_ENV]) if t.env.get(DIR_ENV) else t.home / DEFAULT_DIR
    return root / "data" / SETTINGS


def problem(data: dict[str, Any]) -> str | None:
    """Why Cline would reject the file (it then ignores all of it), or None."""
    providers = data.get("providers")
    if not isinstance(providers, dict):
        return "`providers` is not an object"
    for name, p in providers.items():
        if not isinstance(p, dict) or not isinstance(p.get("settings"), dict):
            return f"provider {name!r} has no settings object"
        try:
            datetime.fromisoformat(str(p.get("updatedAt")).replace("Z", "+00:00"))
        except ValueError:
            return f"provider {name!r} has updatedAt {p.get('updatedAt')!r}, not an ISO time"
        url = p["settings"].get("baseUrl")
        if url is not None and (urlsplit(str(url)).scheme not in ("http", "https") or not urlsplit(str(url)).netloc):
            return f"provider {name!r} has baseUrl {url!r}, not an http(s) URL"
    return None


def detect(t: Target) -> tuple[bool, str]:
    root = settings_path(t).parents[2] if not t.env.get(PATH_ENV) else settings_path(t).parent
    if root.is_dir():
        return True, str(settings_path(t))
    return False, f"{root} not found (Cline 4 not installed, or Cline 3, which keeps settings inside VS Code)"


def plan(t: Target) -> Plan:
    found, why = detect(t)
    if not found:
        return skipped(NAME, why)
    path = settings_path(t)
    raw = path.read_bytes() if path.exists() else None
    try:
        data = edits.load_json(raw, str(path))
        settings = {"provider": PROVIDER, "baseUrl": t.openai_base, "apiKey": t.key, "model": t.model}
        ours = (data.get("providers") or {}).get(PROVIDER) or {}
        stamp = ours["updatedAt"] if ours.get("settings") == settings else time.strftime(ISO, time.gmtime())
        values = {("version",): data.get("version", VERSION), ("lastUsedProvider",): PROVIDER,
                  ("providers", PROVIDER): {"settings": settings, "updatedAt": stamp, "tokenSource": "manual"}}
        if "modes" not in data:
            values[("modes",)] = {}
        undo = edits.set_paths(data, values)
    except edits.Unsupported as e:
        return skipped(NAME, str(e))
    if bad := problem(data):
        return skipped(NAME, f"{path}: {bad}; Cline would ignore the whole file, so it is left as it is")
    return Plan(NAME, [FileChange(path, edits.dump_json(data, raw), undo, force_mode=True)],
                notes=["reload the VS Code window if Cline is open"])
