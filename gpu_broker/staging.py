"""Accepted input files on disk between submit and run (see media.py).

Each file is stored as <job id>-<name>, where <name> is <slot>[-NN].<ext>. A ComfyUI job
uploads its images to ComfyUI's input folder; an exec job hands the files to its recipe.
"""
from __future__ import annotations

import os
import pathlib
import re
from collections.abc import Callable, Mapping
from typing import Any

from .constants import FRAMES, IMAGE_SLOTS, INPUT_SLOTS
from .media import EXTENSION, InputFile
from .store import JOB_ID_HEX

UPLOAD_PREFIX = "broker-"   # ComfyUI input file: broker-<job id>-<slot><ext>
FILE_MODE = 0o600           # inputs may be private; only the broker's user reads them
KIND = {ext: kind for kind, ext in EXTENSION.items()}
# What put() writes: <job id>-<slot>[-NN]<ext>. clear() deletes only names like this, so a
# staging dir shared with anything else (or set to /var/lib/gpu-broker) loses nothing.
STAGED = re.compile(rf"[0-9a-f]{{{JOB_ID_HEX}}}-({'|'.join(INPUT_SLOTS)})(-[0-9]{{2,}})?"
                    rf"({'|'.join(re.escape(e) for e in EXTENSION.values())})")

Uploader = Callable[[str, bytes, str], str]   # (file name, data, format) -> name ComfyUI stored


class Staging:
    def __init__(self, directory: str) -> None:
        self.dir = pathlib.Path(directory)

    def _files(self, jid: str) -> list[pathlib.Path]:
        return sorted(self.dir.glob(f"{jid}-*")) if self.dir.is_dir() else []

    def put(self, jid: str, files: list[InputFile]) -> None:
        if not files:
            return
        self.dir.mkdir(parents=True, exist_ok=True)
        for f in files:
            fd = os.open(self.dir / f"{jid}-{f.name}", os.O_WRONLY | os.O_CREAT | os.O_TRUNC, FILE_MODE)
            with os.fdopen(fd, "wb") as out:
                out.write(f.data)

    def upload(self, jid: str, uploader: Uploader) -> dict[str, str]:
        """Upload a job's single images; returns {slot: ComfyUI file name} for the template."""
        names = {}
        for p in self._files(jid):
            slot = p.stem.removeprefix(f"{jid}-")
            if slot in IMAGE_SLOTS:
                names[slot] = uploader(UPLOAD_PREFIX + p.name, p.read_bytes(), KIND[p.suffix])
        return names

    def paths(self, jid: str) -> list[tuple[str, pathlib.Path]]:
        """A job's files as (<slot>[-NN].<ext>, path), in name order (frames stay in order)."""
        return [(p.name.removeprefix(f"{jid}-"), p) for p in self._files(jid)]

    def received(self, jid: str) -> dict[str, int]:
        """Staged files per slot ({"frames": 3, "image": 1}), to compare with what submit recorded."""
        out: dict[str, int] = {}
        for p in self._files(jid):
            slot = p.stem.removeprefix(f"{jid}-")
            slot = FRAMES if slot.startswith(FRAMES + "-") else slot
            out[slot] = out.get(slot, 0) + 1
        return out

    def check(self, jid: str, summary: Mapping[str, Any]) -> None:
        """The files on disk must be the ones submit accepted (`summary`: the job's recorded
        inputs); a vanished file fails the job."""
        recorded = {slot: len(v) if isinstance(v, list) else 1 for slot, v in summary.items()}
        if (staged := self.received(jid)) != recorded:
            raise RuntimeError(f"input files missing: expected {recorded}, found {staged}")

    def discard(self, jid: str) -> None:
        for p in self._files(jid):
            p.unlink(missing_ok=True)

    def clear(self, keep: frozenset[str] = frozenset()) -> None:
        """At startup: drop the files of every job from a previous process except `keep` (re-queued)."""
        if self.dir.is_dir():
            for p in self.dir.iterdir():
                if STAGED.fullmatch(p.name) and p.is_file() and p.name[:JOB_ID_HEX] not in keep:
                    p.unlink(missing_ok=True)
