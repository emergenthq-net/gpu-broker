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


def _mkdir(d: str) -> None:
    os.makedirs(d, exist_ok=True)


def run(dest: str = DEFAULT_DIR, force: bool = False,
        mkdir: Callable[[str], None] = _mkdir,
        config: str | None = None, catalog: str | None = None,
        write: Callable[[pathlib.Path, str], None] = write_text, keep_hint: str = "--force replaces it",
        exists: Callable[[str], bool] = os.path.lexists, catalog_dir: str | None = None,
        data_mkdir: Callable[[str], None] | None = None,
        write_catalog: Callable[[pathlib.Path, str], None] | None = None,
        load: Callable[[str, str], settings.Settings] | None = None) -> list[str]:
    """Write config.yaml into `dest`, catalog.yaml into `catalog_dir` (default `dest`), and
    create the folders the config names. `config` and `catalog` replace the starter files
    (`gpu-broker setup` passes what it generated). Setup also passes its own `mkdir`, `write`
    and `exists` (through sudo), `data_mkdir` and `write_catalog` (owned by the service
    account) and `load` (the config as written, which a dry run never writes). Returns one
    line per action, for the CLI to print."""
    out = pathlib.Path(dest)
    cat_dir = pathlib.Path(catalog_dir or dest)
    mkdir(str(out))
    if cat_dir != out:
        (data_mkdir or mkdir)(str(cat_dir))
    if config is None:
        config = starter(CONFIG)
        if out.resolve() != pathlib.Path(DEFAULT_DIR).resolve():
            if CATALOG_LINE not in config:
                raise RuntimeError(f"starter {CONFIG} has no `{CATALOG_LINE.strip()}` line to point at {dest}")
            config = config.replace(CATALOG_LINE, f"catalog: {out.resolve() / CATALOG}\n")
    lines = []
    for path, text, put in ((out / CONFIG, config, write),
                            (cat_dir / CATALOG, catalog if catalog is not None else starter(CATALOG),
                             write_catalog or write)):
        if exists(str(path)) and not force:
            lines.append(f"kept     {path} (exists; {keep_hint})")
            continue
        put(path, text)
        lines.append(f"wrote    {path}")
    cfg = load(str(out / CONFIG), config) if load else settings.load(str(out / CONFIG), env={})
    for d in data_dirs(cfg):
        (data_mkdir or mkdir)(d)
        lines.append(f"folder   {d}")
    return lines
