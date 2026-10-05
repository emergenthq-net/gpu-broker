"""Apply connector plans and take them back out, with backups and a manifest.

Every file is backed up (timestamped) before its first change, and the manifest
(`~/.gpu-broker/connect/manifest.json`, mode 600) records, per file, the backup, whether we
created it, the hash of what we wrote and how to undo it. Disconnect then:
- restores the backup byte for byte (or deletes a file we created) when the file still holds
  exactly what we wrote;
- otherwise removes only our entries (the user has edited the file since), keeps the
  backup, and says so.
Connecting again is idempotent: an unchanged file is not rewritten, and the first backup and
undo record are kept. If the user edited the file between two connects, the backup no longer
reflects what they want back, so the entry becomes strip-only: disconnect removes our
entries and keeps theirs, never restoring the old backup over their edits.

A change whose app may rewrite the file while running (Claude Code and ~/.claude.json) carries a
`check`: a moment after the writes, the file is read again and a dropped entry is reported. A
symlinked config is written through, so the link (a dotfiles repo) stays.

The manifest is saved (write-temp-and-rename) before the first change and after every file or
API change, and an issued key is recorded before any file is written, so a run that fails
part-way leaves every change it made tracked. Disconnect drops an entry only once it is
undone; a failed undo stays recorded for a retry. The key record is never dropped here: only
a successful revoke (cli.py) removes it. A dry run reads only: no backups, no manifest, no
writes.
"""
from __future__ import annotations

import os
import time
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

from . import edits
from .core import SECRET_MODE, FileChange, Plan
from .state import (
    MANIFEST,
    STATE_DIR,
    Report,
    backup,
    forget_key,
    load_manifest,
    read,
    remember_key,
    save_manifest,
    sha,
    state_dir,
    write,
)

SETTLE_S = 2.0   # before re-reading files a running app may rewrite
__all__ = ["MANIFEST", "STATE_DIR", "Report", "connect", "connected", "disconnect", "forget_key", "key_modes", "load_manifest",
           "remember_key",
           "save_manifest", "state_dir", "write"]


def _apply_file(home: Path, m: dict[str, Any], plan: Plan, ch: FileChange, dry: bool) -> tuple[str, bool]:
    """(report line, whether the file was written or its mode changed)."""
    current = read(ch.path)
    if current == ch.new:
        if ch.force_mode and ch.path.stat().st_mode & 0o777 != ch.mode:   # the client needs it (e.g. a key inside)
            if dry:
                return f"{plan.client}: would set {ch.path} to mode {ch.mode:o}", False
            os.chmod(ch.path, ch.mode)
            return f"{plan.client}: {ch.path} already connected; set back to mode {ch.mode:o}", True
        return f"{plan.client}: {ch.path} already connected", False
    if dry:
        return f"{plan.client}: would {'update' if current is not None else 'create'} {ch.path}", False
    entry = m["files"].get(str(ch.path))
    if entry is not None and (current is None or sha(current) != entry["written"]) and not entry["created"]:
        # Deleted by the user since: a new entry (the file is ours now). Edited since: the old
        # backup must never come back over their edits, so only our entries are removed later.
        entry = None if current is None else {**entry, "strip_only": True}
    if entry is not None:   # a later connect may set paths the first did not (Claude Code's key changing header)
        entry = {**entry, "undo": edits.merged(entry["undo"], ch.undo)}
    if entry is None:   # first connect: keep the original and how to undo our change
        entry = {"client": plan.client, "created": current is None, "undo": ch.undo,
                 "backup": backup(home, ch.path, current) if current is not None else None}
    m["files"][str(ch.path)] = {**entry, "written": sha(ch.new), "keys": plan.keys}   # tracked before it is written
    save_manifest(home, m)
    write(ch.path, ch.new, ch.mode, ch.force_mode)
    return f"{plan.client}: {'updated' if current is not None else 'created'} {ch.path}", True


def connect(home: Path, plans: Iterable[Plan], dry: bool = False, key: Mapping[str, Any] | None = None,
            changed: list[Path] | None = None, settle_s: float | None = None, sleep: Callable[[float], None] = time.sleep) -> Report:
    """Apply the plans. `changed`, when given, receives every file written (in order)."""
    m, report = load_manifest(home), []
    checks: list[tuple[str, Path, Callable[[bytes | None], str | None]]] = []
    if key and not dry:   # recorded before anything uses it, so a failed run can still revoke it
        m["key"] = dict(key)
        save_manifest(home, m)
    for plan in plans:
        if plan.skip:
            report.append(f"{plan.client}: skipped: {plan.skip}")
            continue
        for ch in plan.files:
            line, wrote = _apply_file(home, m, plan, ch, dry)
            report.append(line)
            if wrote and changed is not None:
                changed.append(ch.path)
            if ch.check and not dry:
                checks.append((plan.client, ch.path, ch.check))
        if plan.api is not None:
            if dry:
                report.append(f"{plan.client}: would {plan.api.describe}")
            else:
                found = plan.api.apply()   # how disconnect finds our entry; the first connect's record wins
                if found is None:          # nothing changed: not ours, so nothing for disconnect to undo
                    report.append(f"{plan.client}: {plan.api.unchanged}")
                    continue
                m["api"].setdefault(plan.client, found)
                save_manifest(home, m)
                report.append(f"{plan.client}: {plan.api.describe}")
        report += [f"{plan.client}: note: {n}" for n in plan.notes]
    if checks:
        sleep(SETTLE_S if settle_s is None else settle_s)
        report += [f"{client}: warning: {why}" for client, path, check in checks if (why := check(read(path)))]
    return report


def _undo_file(path: Path, entry: Mapping[str, Any], dry: bool) -> tuple[str, bool]:
    """(report line, whether the entry is done with)."""
    try:
        return _undo(path, entry, dry)
    except OSError as e:
        return f"{entry['client']}: {path} could not be undone ({e}); kept in the manifest, run disconnect again", False


def _undo(path: Path, entry: Mapping[str, Any], dry: bool) -> tuple[str, bool]:
    current, who = read(path), entry["client"]
    if current is None:
        return f"{who}: {path} is gone already", True
    if sha(current) == entry["written"] and not entry.get("strip_only"):   # untouched since connect: restore exactly
        if not dry:
            if entry["created"]:
                path.unlink()
            else:
                original = Path(entry["backup"])
                write(path, original.read_bytes(), original.stat().st_mode & 0o777, force_mode=True)
        verb = ("would remove", "removed") if entry["created"] else ("would restore", "restored")   # a file connect made goes
        return f"{who}: {verb[0] if dry else verb[1]} {path}", True
    try:
        stripped = edits.undo(current, entry["undo"])
    except edits.Unsupported as e:
        return (f"{who}: {path} changed since connect and cannot be edited ({e}); original kept at {entry['backup']}; "
                "still recorded"), False
    if stripped is None:
        return f"{who}: {path} changed since connect; left as it is", True
    if not dry:
        write(path, stripped, SECRET_MODE)
    return f"{who}: {path} changed since connect; removed only our entries (original kept at {entry['backup']})", True


def disconnect(home: Path, clients: set[str] | None = None, dry: bool = False,
               restore_api: Mapping[str, Callable[[Any], None]] | None = None) -> Report:
    m, report = load_manifest(home), []
    for path, entry in list(m["files"].items()):
        if clients is None or entry["client"] in clients:
            line, done = _undo_file(Path(path), entry, dry)
            report.append(line)
            if done and not dry:
                del m["files"][path]
                save_manifest(home, m)
    for client, prior in list(m["api"].items()):
        if clients is None or client in clients:
            fn = (restore_api or {}).get(client)
            if fn is None:
                report.append(f"{client}: not restored here (give its URL and admin token)")
                continue
            if not dry:
                try:
                    fn(prior)
                except (OSError, ValueError) as e:
                    report.append(f"{client}: could not be undone ({e}); kept in the manifest, run disconnect again")
                    continue
                del m["api"][client]
                save_manifest(home, m)
            report.append(f"{client}: {'would remove' if dry else 'removed'} the broker's entry")
    if not report:
        report.append("nothing to disconnect")
    return report


def key_modes(home: Path) -> dict[str, str]:
    """Per connected client, the credentials its last connect configured (Plan.keys), where recorded."""
    return {e["client"]: e["keys"] for e in load_manifest(home)["files"].values() if e.get("keys")}


def connected(home: Path) -> set[str]:
    m = load_manifest(home)
    return {e["client"] for e in m["files"].values()} | set(m["api"])
