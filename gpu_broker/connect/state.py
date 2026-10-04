"""The connect state on disk: `~/.gpu-broker/connect/` (mode 700) holding the manifest
(mode 600) and timestamped backups of every file before its first change. Writes are atomic
(temp file, then rename)."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .core import SECRET_MODE

STATE_DIR = Path(".gpu-broker") / "connect"
MANIFEST, BACKUPS = "manifest.json", "backups"
STAMP = "%Y%m%dT%H%M%SZ"
PRIVATE_DIR = 0o700
Report = list[str]


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def state_dir(home: Path) -> Path:
    return home / STATE_DIR


def load_manifest(home: Path) -> dict[str, Any]:
    p = state_dir(home) / MANIFEST
    m: dict[str, Any] = json.loads(p.read_text()) if p.exists() else {}
    m.setdefault("files", {})
    m.setdefault("api", {})
    return m


def save_manifest(home: Path, m: Mapping[str, Any]) -> None:
    p = state_dir(home) / MANIFEST
    if not m["files"] and not m["api"] and not m.get("key"):
        p.unlink(missing_ok=True)
        return
    write(p, json.dumps(m, indent=2).encode(), SECRET_MODE)


def read(path: Path) -> bytes | None:
    return path.read_bytes() if path.exists() else None


def private_write(path: Path, data: bytes) -> None:
    """Create `path` (replacing a stale one) at mode 600 from the first byte: the data may be a key."""
    path.unlink(missing_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, SECRET_MODE)
    with os.fdopen(fd, "wb") as f:
        f.write(data)


def write(path: Path, data: bytes, mode: int, force_mode: bool = False) -> None:
    """Atomically, keeping an existing file's mode (a new file, or `force_mode`, gets `mode`).
    The temp file is private while it is written; it takes the final mode just before the rename.
    A symlink is written through (its target changes; the link stays, e.g. a dotfiles repo)."""
    if path.is_symlink():
        path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    keep = path.stat().st_mode & 0o777 if path.exists() and not force_mode else mode
    tmp = path.with_name(f".{path.name}.gpu-broker-tmp")
    private_write(tmp, data)
    os.chmod(tmp, keep)
    os.replace(tmp, path)


def backup(home: Path, path: Path, data: bytes) -> str:
    d = state_dir(home) / BACKUPS
    d.mkdir(parents=True, exist_ok=True)
    for private in (state_dir(home), d):   # backups may hold keys or tokens the user had there
        os.chmod(private, PRIVATE_DIR)
    stamp = time.strftime(STAMP, time.gmtime())
    target = d / f"{stamp}-{sha(str(path).encode())[:8]}-{path.name}"
    private_write(target, data)
    shutil.copymode(path, target)
    return str(target)


def ours(home: Path, path: Path, client: str) -> bool:
    """Whether `client`'s entry in `path` is one connect wrote (the manifest records it)."""
    entry = load_manifest(home)["files"].get(str(path))
    return entry is not None and entry["client"] == client


def remember_key(home: Path, record: Mapping[str, Any]) -> None:
    """Record a key the moment it is issued, so a run that fails afterwards can still revoke it."""
    m = load_manifest(home)
    m["key"] = dict(record)
    save_manifest(home, m)


def forget_key(home: Path) -> None:
    """Drop the key record (after the key has been revoked), so a reconnect issues a new one."""
    m = load_manifest(home)
    m.pop("key", None)
    save_manifest(home, m)
