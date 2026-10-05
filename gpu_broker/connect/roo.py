"""Roo Code: a settings file Roo imports at start, named in VS Code's settings.json.

Roo reads `roo-cline.autoImportSettingsPath` when it activates and merges the provider
profiles in that file (src/utils/autoImportSettings.ts, src/core/config/importExport.ts).
Connect writes `~/.gpu-broker/connect/roo-settings.json` with a `gpu-broker` OpenAI-compatible
profile and points the setting at it. VS Code's settings.json is often JSON with comments;
that is not edited (skip, with the reason). Note: the Roo Code repository is archived.
"""
from __future__ import annotations

import json

from . import edits, engine, upstreams, vscode
from .core import PROVIDER_ID, FileChange, Plan, Target, skipped

NAME, LABEL = "roo", "Roo Code (VS Code)"
KEY_MODE = upstreams.SEPARATE
EXTENSION = "rooveterinaryinc.roo-cline"
SETTING = "roo-cline.autoImportSettingsPath"
IMPORT_FILE = "roo-settings.json"


def detect(t: Target) -> tuple[bool, str]:
    ext = vscode.extension(t, EXTENSION)
    return (True, str(ext)) if ext else (False, "the Roo Code extension is not installed in VS Code")


def plan(t: Target) -> Plan:
    found, why = detect(t)
    if not found:
        return skipped(NAME, why)
    profile = {"apiProvider": "openai", "openAiBaseUrl": t.openai_base, "openAiApiKey": t.key, "openAiModelId": t.model}
    imported = {"providerProfiles": {"currentApiConfigName": PROVIDER_ID, "apiConfigs": {PROVIDER_ID: profile},
                                     "modeApiConfigs": {}}, "globalSettings": {}}
    own = engine.state_dir(t.home) / IMPORT_FILE
    settings = vscode.settings_json(t)
    raw = settings.read_bytes() if settings.exists() else None
    try:
        data = edits.load_json(raw, str(settings))
        undo = edits.set_paths(data, {(SETTING,): str(own)})
    except edits.Unsupported as e:
        return skipped(NAME, str(e))
    return Plan(NAME, [FileChange(own, (json.dumps(imported, indent=2) + "\n").encode(), {"kind": edits.OWN_FILE}),
                       FileChange(settings, edits.dump_json(data, raw), undo)],
                notes=["reload VS Code; Roo imports the 'gpu-broker' profile when it starts"])
