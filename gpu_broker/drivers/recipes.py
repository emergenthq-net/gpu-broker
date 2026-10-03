"""Exec recipes: how a command-line model runs, defined by whoever administers the host.

A recipe is a file `<recipes dir>/<name>.recipe` of `key=value` lines (no spaces around `=`,
`#` starts a comment line). The broker never sends a command: an exec job names a recipe and
carries a job id and input files, and the recipe supplies everything else. The Proxmox host
script (host/gpu-broker-ctl) reads the same format; the local drivers read it here.

    target=101          container id (Proxmox only; local drivers refuse it)
    argv=/opt/tool/bin/predict -i {in_dir} -o {out_dir} -c {checkpoint}
    checkpoint=/models/tool/weights.pt
    in_dir=/var/tmp/gpu-broker/{jid}     input files are written here (removed after the run)
    out_dir=/srv/outputs/broker/{jid}    the program writes its results here
    outputs=*.ply                        which files in out_dir are the job's outputs
    timeout_s=600                        the program is stopped after this long

`outputs` is one or more file name globs separated by spaces (`outputs=world.mp4 run.log`):
the job's outputs are each glob's matching files, sorted by name, in the order the globs are
given, a file matched by an earlier glob not repeated, so the first glob's file is outputs[0].
A file whose name starts with `.` is never an output (find and pathlib disagree on whether `*`
matches one, so both drivers leave them out).

`argv` is split on spaces, with no shell and no quoting; {jid}, {in_dir}, {out_dir} and
{checkpoint} are substituted per word. `in_dir` must be absolute and contain {jid}; `out_dir`
must be absolute and end in /{jid}: only that last directory is created, in a parent that must
already exist (on Proxmox it takes the parent's owner, e.g. ComfyUI's user). A recipe is
stopped at timeout_s with SIGTERM and kill_after_s later with SIGKILL, so the catalog's
exec.timeout_s must exceed that plus the driver's clean time and a margin (execjob.py); the
driver reports its values (`recipe_info`).

host/gpu-broker-ctl parses the same format with the same rules: no carriage returns (CRLF
files are refused, not stripped), every line is a comment, blank, or `key=value`, and
timeout_s is a positive decimal number; the first argv word (the program) has no `=` and does
not start with `-`, because both run it as `env -- GPU_BROKER_JOB=<jid> argv...`.
"""
from __future__ import annotations

import os
import pathlib
import re
import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass

from ..constants import ERR_DETAIL
from . import KILL_AFTER_S, Input, Run, validate

__all__ = ["KILL_AFTER_S", "Recipe", "load", "parse", "run_local"]

SUFFIX = ".recipe"
KEYS = frozenset({"target", "argv", "checkpoint", "in_dir", "out_dir", "outputs", "timeout_s"})
REQUIRED = ("argv", "in_dir", "out_dir", "outputs", "timeout_s")
KEY = re.compile(r"^[a-z_]+$")
WORD = re.compile(r"^[A-Za-z0-9._/@:+=,{}-]+$")
GLOB = re.compile(r"^[A-Za-z0-9._*?-]+$")
TARGET = re.compile(r"^[0-9]+$")
TIMEOUT = re.compile(r"^[0-9]+(\.[0-9]+)?$")
JID = "{jid}"
DIR_MODE = 0o700


@dataclass(frozen=True)
class Recipe:
    name: str
    argv: tuple[str, ...]
    in_dir: str
    out_dir: str
    outputs: tuple[str, ...]
    timeout_s: float
    checkpoint: str = ""
    target: str | None = None

    def dirs(self, jid: str) -> tuple[str, str]:
        return self.in_dir.replace(JID, jid), self.out_dir.replace(JID, jid)

    def command(self, jid: str) -> list[str]:
        in_dir, out_dir = self.dirs(jid)
        values = {JID: jid, "{in_dir}": in_dir, "{out_dir}": out_dir, "{checkpoint}": self.checkpoint}
        out = []
        for word in self.argv:
            filled = word
            for k, v in values.items():
                filled = filled.replace(k, v)
            out.append(filled)
        return out


def parse(name: str, text: str) -> Recipe:
    if "\r" in text:
        raise ValueError(f"recipe {name}: carriage return in the file (save it with LF line ends)")
    fields: dict[str, str] = {}
    for n, line in enumerate(text.splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        key, sep, value = line.partition("=")
        if not sep or not KEY.match(key) or key not in KEYS:
            raise ValueError(f"recipe {name}, line {n}: expected one of {sorted(KEYS)} as key=value")
        fields[key] = value
    if missing := [k for k in REQUIRED if not fields.get(k)]:
        raise ValueError(f"recipe {name}: missing {missing}")
    argv = tuple(fields["argv"].split())
    if "=" in argv[0] or argv[0].startswith("-"):   # `env -- TAG=<jid> argv...` would read it as a variable or option
        raise ValueError(f"recipe {name}: argv must start with a program, not {argv[0]!r}")
    for word in argv:
        if not WORD.match(word) or validate.PARENT in word:
            raise ValueError(f"recipe {name}: argv word {word!r} has characters a recipe may not use")
    for key in ("in_dir", "out_dir"):
        d = fields[key]
        if not d.startswith("/") or JID not in d or validate.PARENT in d or not WORD.match(d):
            raise ValueError(f"recipe {name}: {key} must be an absolute path containing {JID}")
    if not fields["out_dir"].endswith("/" + JID) or fields["out_dir"].count(JID) != 1:
        raise ValueError(f"recipe {name}: out_dir must end in /{JID} (one directory per job, in an existing parent)")
    outputs = tuple(g for g in fields["outputs"].split(" ") if g)   # spaces only, like the host script
    if not outputs or not all(GLOB.match(g) for g in outputs):
        raise ValueError(f"recipe {name}: outputs must be file name globs separated by spaces")
    if not TIMEOUT.match(fields["timeout_s"]) or float(fields["timeout_s"]) <= 0:
        raise ValueError(f"recipe {name}: timeout_s must be a positive number of seconds")
    if "target" in fields and not TARGET.match(fields["target"]):
        raise ValueError(f"recipe {name}: target must be a container id")
    return Recipe(name, argv, fields["in_dir"], fields["out_dir"], outputs, float(fields["timeout_s"]),
                  fields.get("checkpoint", ""), fields.get("target"))


def load(directory: str, name: str) -> Recipe:
    if not validate.RECIPE.match(name):
        raise ValueError(f"bad recipe name {name!r}")
    path = pathlib.Path(directory) / f"{name}{SUFFIX}"
    if not path.is_file():
        raise FileNotFoundError(f"no recipe {name!r} in {directory}")
    return parse(name, path.read_text())


def run_local(recipe: Recipe, jid: str, files: Sequence[tuple[str, Input]], run: Run) -> list[str]:
    """Write the inputs, run the command on this machine (stopped at the recipe's timeout_s,
    with its whole process group), return the output paths."""
    in_dir, out_dir = recipe.dirs(jid)
    try:
        os.makedirs(in_dir, mode=DIR_MODE, exist_ok=True)
        parent = os.path.dirname(out_dir)
        if not os.path.isdir(parent):
            raise RuntimeError(f"recipe {recipe.name}: output parent {parent} does not exist; create it")
        os.mkdir(out_dir)
        for name, data in files:
            with open(pathlib.Path(in_dir) / name, "wb") as out:
                if isinstance(data, bytes):
                    out.write(data)
                else:
                    shutil.copyfileobj(data, out)
        try:
            r = run(recipe.command(jid), timeout=recipe.timeout_s, tag=jid)   # tagged: see reap.py
        except subprocess.TimeoutExpired:   # the process group is already stopped
            raise RuntimeError(f"recipe {recipe.name} timed out after {recipe.timeout_s:g}s and was stopped") from None
        if r.returncode != 0:
            raise RuntimeError(f"recipe {recipe.name} exited {r.returncode}: {r.stderr[-ERR_DETAIL:]}")
        return list(dict.fromkeys(str(p) for glob in recipe.outputs   # in glob order, each file once
                                  for p in sorted(pathlib.Path(out_dir).glob(glob))
                                  if p.is_file() and not p.name.startswith(".")))
    finally:
        shutil.rmtree(in_dir, ignore_errors=True)
