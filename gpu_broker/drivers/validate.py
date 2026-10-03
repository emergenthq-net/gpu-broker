"""Every value a driver passes to a subprocess or the filesystem is checked here first.

The grammars match the Proxmox host script (host/gpu-broker-ctl), which re-checks them on
the host side, so both ends reject the same inputs.
"""
from __future__ import annotations

import os
import re

from ..constants import DownloadKind

SLUG = re.compile(r"^[a-z0-9][a-z0-9.-]{0,63}$")
HF_REPO = re.compile(r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$")
GH_URL = re.compile(r"^https://github\.com/[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$")
INCLUDE = re.compile(r"^[A-Za-z0-9._*?/\[\]-]{1,200}$")   # an `hf download --include` glob
REL_PATH = re.compile(r"^[A-Za-z0-9._/@+-]{1,255}$")
COMFY_SUBDIRS = frozenset({"diffusion_models", "loras", "text_encoders", "vae", "clip_vision", "checkpoints"})
RECIPE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,40}$")
JOB_ID = re.compile(r"^[0-9a-f]{8,32}$")
INPUT_NAME = re.compile(r"^[a-z_]{1,20}(-[0-9]{1,4})?\.[a-z0-9]{1,5}$")   # <slot>[-NN].<ext>
PARENT = ".."
REFS = {DownloadKind.HF: HF_REPO, DownloadKind.GH: GH_URL}


def download(kind: str, ref: str, slug: str, include: list[str]) -> DownloadKind:
    """Validate a download request; returns the parsed kind."""
    try:
        k = DownloadKind(kind)
    except ValueError:
        raise ValueError(f"bad download kind {kind!r}") from None
    if not REFS[k].match(ref) or PARENT in ref:
        raise ValueError(f"bad {k} reference {ref!r}")
    if not SLUG.match(slug):
        raise ValueError(f"bad download slug {slug!r}")
    for pattern in include:
        if not INCLUDE.match(pattern) or PARENT in pattern or pattern.startswith("-"):
            raise ValueError(f"bad include pattern {pattern!r}")
    return k


def link(rel: str, subdir: str) -> None:
    """Validate a request to expose <models root>/<rel> to ComfyUI under models/<subdir>/."""
    if subdir not in COMFY_SUBDIRS:
        raise ValueError(f"bad ComfyUI models subdir {subdir!r}")
    if not REL_PATH.match(rel) or rel.startswith("/") or PARENT in rel.split("/"):
        raise ValueError(f"bad model path {rel!r}")


def recipe_call(recipe: str, jid: str, names: list[str]) -> None:
    """Validate what an exec job sends to a recipe: its name, the job id, input file names.
    These are the only values from a request that reach the host; the command is the recipe's."""
    if not RECIPE.match(recipe):
        raise ValueError(f"bad recipe name {recipe!r}")
    if not JOB_ID.match(jid):
        raise ValueError(f"bad job id {jid!r}")
    for n in names:
        if not INPUT_NAME.match(n):
            raise ValueError(f"bad input file name {n!r}")


def inside(root: str, *parts: str) -> str:
    """Join `parts` under `root` and prove the real path stays inside it (symlinks included)."""
    base = os.path.realpath(root)
    path = os.path.realpath(os.path.join(base, *parts))
    if os.path.commonpath([base, path]) != base:
        raise ValueError(f"{os.path.join(*parts)!r} escapes {root}")
    return path
