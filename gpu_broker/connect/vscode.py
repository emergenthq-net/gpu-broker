"""Where VS Code (and forks that share its layout) keep the user's settings.json and extensions."""
from __future__ import annotations

import sys
from pathlib import Path

from .core import Target

USER_DIRS = {"darwin": Path("Library") / "Application Support" / "Code" / "User",
             "linux": Path(".config") / "Code" / "User"}
EXTENSIONS = Path(".vscode") / "extensions"
SETTINGS = "settings.json"


def settings_json(t: Target) -> Path:
    """The existing settings.json, else the one for this platform."""
    for d in USER_DIRS.values():
        if (t.home / d / SETTINGS).exists():
            return t.home / d / SETTINGS
    return t.home / USER_DIRS.get(sys.platform, USER_DIRS["linux"]) / SETTINGS


def extension(t: Target, publisher_dot_name: str) -> Path | None:
    """The installed extension's folder (`<publisher>.<name>-<version>`), if any."""
    d = t.home / EXTENSIONS
    found = sorted(d.glob(publisher_dot_name.lower() + "-*")) if d.is_dir() else []
    return found[-1] if found else None
