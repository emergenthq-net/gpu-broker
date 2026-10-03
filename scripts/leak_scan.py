"""Fail if a release tree or a built distribution names anything on a private denylist.

    python scripts/leak_scan.py --denylist FILE PATH...

PATH is a directory (every file under it, and every file name), or a built distribution
(.whl, .zip, .tar.gz: every member). The denylist is kept out of this repository: pass it
with --denylist or GPU_BROKER_LEAK_DENYLIST (a file path). One case-insensitive regular
expression per line; blank lines and lines starting with # are ignored.

A hit is reported by file, line and entry number only, never by the matched text, so the
log of a public CI run does not publish the list. Exit 0: clean; 1: hits; 2: no usable
denylist or an unreadable path (never treated as clean).
"""
from __future__ import annotations

import argparse
import os
import pathlib
import re
import sys
import tarfile
import zipfile
from collections.abc import Iterator

ENV = "GPU_BROKER_LEAK_DENYLIST"
SKIP_DIRS = {".git", ".venv", "build", "dist", "__pycache__", ".mypy_cache", ".ruff_cache", ".pytest_cache", "node_modules"}
CLEAN, HITS, UNUSABLE = 0, 1, 2


def load(path: str | os.PathLike[str]) -> list[re.Pattern[str]]:
    """The denylist's patterns. Raises ValueError when it has none: an empty list scans nothing."""
    lines = pathlib.Path(path).read_text(encoding="utf-8").splitlines()
    pats = [re.compile(s.strip(), re.IGNORECASE) for s in lines if s.strip() and not s.lstrip().startswith("#")]
    if not pats:
        raise ValueError("the denylist has no entries")
    return pats


def from_env() -> list[re.Pattern[str]] | None:
    """The denylist named by GPU_BROKER_LEAK_DENYLIST, or None when it is not set."""
    path = os.environ.get(ENV)
    return load(path) if path else None


def hits(name: str, text: str, pats: list[re.Pattern[str]]) -> Iterator[str]:
    """`name:line: entry N` for every match in the name itself (line 0) and in the text."""
    for n, p in enumerate(pats, 1):
        if p.search(name):
            yield f"{name}:0: entry {n} (in the file name)"
    for i, line in enumerate(text.splitlines(), 1):
        for n, p in enumerate(pats, 1):
            if p.search(line):
                yield f"{name}:{i}: entry {n}"


def members(path: pathlib.Path, skip: set[pathlib.Path]) -> Iterator[tuple[str, bytes]]:
    """(name, contents) of every file in a directory or a built distribution."""
    if path.is_dir():
        for root, dirs, files in os.walk(path):
            dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS and not d.endswith(".egg-info"))
            for f in sorted(files):
                p = pathlib.Path(root, f)
                if p.resolve() not in skip and not p.is_symlink():
                    yield str(p.relative_to(path)), p.read_bytes()
    elif path.suffix in {".whl", ".zip"}:
        with zipfile.ZipFile(path) as z:
            for info in z.infolist():
                if not info.is_dir():
                    yield f"{path.name}!{info.filename}", z.read(info)
    elif path.name.endswith((".tar.gz", ".tgz")):
        with tarfile.open(path) as t:
            for m in t.getmembers():
                fh = t.extractfile(m) if m.isfile() else None
                if fh is not None:
                    yield f"{path.name}!{m.name}", fh.read()
    else:
        yield path.name, path.read_bytes()


def scan(paths: list[pathlib.Path], pats: list[re.Pattern[str]], skip: set[pathlib.Path]) -> list[str]:
    out: list[str] = []
    for path in paths:
        for name, data in members(path, skip):
            out.extend(hits(name, data.decode("utf-8", errors="replace"), pats))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--denylist", default=os.environ.get(ENV), help=f"denylist file (default: ${ENV})")
    ap.add_argument("paths", nargs="+", type=pathlib.Path)
    a = ap.parse_args(argv)
    if not a.denylist:
        print(f"leak-scan: no denylist (--denylist or ${ENV}); refusing to report clean", file=sys.stderr)
        return UNUSABLE
    try:
        pats = load(a.denylist)
        found = scan(a.paths, pats, {pathlib.Path(a.denylist).resolve()})
    except (OSError, ValueError, re.error, zipfile.BadZipFile, tarfile.TarError) as e:
        print(f"leak-scan: {type(e).__name__}: {e}", file=sys.stderr)
        return UNUSABLE
    for line in found:
        print(line)
    print(f"leak-scan: {len(found)} hit(s) for {len(pats)} entries in {', '.join(map(str, a.paths))}", file=sys.stderr)
    return HITS if found else CLEAN


if __name__ == "__main__":
    sys.exit(main())
