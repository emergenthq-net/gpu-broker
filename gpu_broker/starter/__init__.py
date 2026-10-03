"""`gpu-broker init`: write the starter config and catalog, and create the folders they name.

The starter files are examples/config.yaml and examples/catalog.yaml, shipped in the package
(a test keeps the copies identical). Existing files are kept unless `force` is set, so running
init again never undoes an operator's edits.
"""
from __future__ import annotations

import os
import pathlib
from collections.abc import Callable
from importlib.resources import files

from .. import settings

CONFIG, CATALOG = "config.yaml", "catalog.yaml"
DEFAULT_DIR = os.path.dirname(settings.DEFAULT_PATH)
CATALOG_LINE = f"catalog: {DEFAULT_DIR}/{CATALOG}\n"   # the starter config's catalog key, rewritten for another dir


def starter(name: str) -> str:
    return (files(__package__) / name).read_text()


def data_dirs(cfg: settings.Settings) -> list[str]:
    """Folders the config names that must exist before `serve`: the database's and event log's
    parents, the input staging folder and the models root."""
    dirs = [os.path.dirname(cfg.db), cfg.inputs.staging_dir]
    if cfg.events_jsonl:
        dirs.append(os.path.dirname(cfg.events_jsonl))
    if root := cfg.driver.options.get("models_root"):
        dirs.append(str(root))
    return dirs


def write_text(path: pathlib.Path, text: str) -> None:
    path.write_text(text)


def run(dest: str = DEFAULT_DIR, force: bool = False,
        mkdir: Callable[[str], None] = lambda d: os.makedirs(d, exist_ok=True),
        config: str | None = None, catalog: str | None = None,
        write: Callable[[pathlib.Path, str], None] = write_text, keep_hint: str = "--force replaces it") -> list[str]:
    """Write config.yaml and catalog.yaml into `dest` and create the folders the config names.
    `config` and `catalog` replace the starter files (`gpu-broker setup` passes what it
    generated); `mkdir` and `write` let setup make them through sudo. Returns one line per
    action, for the CLI to print."""
    out = pathlib.Path(dest)
    mkdir(str(out))
    if config is None:
        config = starter(CONFIG)
        if out.resolve() != pathlib.Path(DEFAULT_DIR).resolve():
            if CATALOG_LINE not in config:
                raise RuntimeError(f"starter {CONFIG} has no `{CATALOG_LINE.strip()}` line to point at {dest}")
            config = config.replace(CATALOG_LINE, f"catalog: {out.resolve() / CATALOG}\n")
    lines = []
    for name, text in ((CONFIG, config), (CATALOG, catalog if catalog is not None else starter(CATALOG))):
        path = out / name
        if path.exists() and not force:
            lines.append(f"kept     {path} (exists; {keep_hint})")
            continue
        write(path, text)
        lines.append(f"wrote    {path}")
    cfg = settings.load(str(out / CONFIG), env={})
    for d in data_dirs(cfg):
        mkdir(d)
        lines.append(f"folder   {d}")
    return lines
