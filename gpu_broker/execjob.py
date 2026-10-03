"""Exec jobs: a command-line model (runner `exec`) run by a host recipe on the job's inputs.

The GPU thread has already evicted the resident LLM and freed ComfyUI's weights; this hands
the staged input files to the driver, which runs the recipe the catalog names
(drivers/recipes.py), and turns the output paths into the job result. A recipe that writes
under ComfyUI's output folder (`comfy.output_dir`) gets browser URLs through ComfyUI's /view.
Request keys listed in the catalog's `exec.params` reach the recipe as a params.json input.
Staged files are streamed to the driver as open files, never read into memory.

The catalog's exec.timeout_s bounds the whole job as the broker sees it, so it must outlast
the recipe's own timeout_s, the SIGKILL grace and the driver's reap of leftover processes,
plus EXEC_MARGIN_S (copying inputs, SSH, listing outputs): otherwise the broker would give up
while the program still holds the GPU. Likewise timeouts.exec_clean_s must outlast the
driver's clean (+ CLEAN_MARGIN_S, slack for starting it). The driver reports its own values
(`recipe_info`; the Proxmox driver adds its SSH connect timeout to the host's clean time).
They are read once per recipe (at startup where the host answers) and cached; a failed run
drops the cache entry so the next job reads them afresh. Checked before every run, before
the GPU thread evicts anything.
"""
from __future__ import annotations

import contextlib
import json
import posixpath
import time
from collections.abc import Callable, Mapping
from typing import Any

from .backends import view_url
from .catalog import Catalog, Model
from .constants import ERR_SHORT, Event, Runner
from .drivers import DRIVER_ERRORS, Driver, Input, RecipeInfo
from .settings import Comfy, Timeouts
from .staging import Staging
from .store import Store

PARAMS_FILE = "params.json"
PARAM_STR_MAX = 200
DECIMALS = 1
EXEC_MARGIN_S = 30    # exec.timeout_s - (recipe timeout_s + kill_after_s + reap_s) must be at least this
CLEAN_MARGIN_S = 10   # timeouts.exec_clean_s - the driver's clean_wait_s must be at least this


class TimeoutTooShort(ValueError):
    """A configured timeout would let the broker give up while a recipe may still hold the GPU."""


def params(model: Model, body: Mapping[str, Any]) -> dict[str, Any]:
    """The request values a recipe receives; scalars only, and one of `exec.choices` where the
    catalog lists them (checked at submit: 400 if not)."""
    spec = model.get("exec", {})
    picked = {k: body[k] for k in spec.get("params", []) if k in body}
    for k, v in picked.items():
        ok = isinstance(v, (int, float, bool)) or (isinstance(v, str) and len(v) <= PARAM_STR_MAX)
        if not ok:
            raise ValueError(f"`{k}` must be a number, true/false or a short string")
        allowed = spec.get("choices", {}).get(k)
        if allowed is not None and not any(type(v) is type(c) and v == c for c in allowed):   # True is not 1
            raise ValueError(f"`{k}` must be one of {allowed}, got {v!r}")
    return picked


def check_timeout(key: str, model: Model, i: RecipeInfo, timeouts: Timeouts) -> None:
    spec = model["exec"]
    need = i.timeout_s + i.kill_after_s + i.reap_s + EXEC_MARGIN_S
    if float(spec["timeout_s"]) < need:
        raise TimeoutTooShort(f"{key}: exec.timeout_s {spec['timeout_s']:g} must be at least {need:g} (recipe "
                              f"{spec['recipe']} timeout_s {i.timeout_s:g} + {i.kill_after_s:g} s kill grace + "
                              f"{i.reap_s:g} s reap + {EXEC_MARGIN_S} s)")
    if timeouts.exec_clean_s < i.clean_wait_s + CLEAN_MARGIN_S:
        raise TimeoutTooShort(f"timeouts.exec_clean_s {timeouts.exec_clean_s:g} must be at least "
                              f"{i.clean_wait_s + CLEAN_MARGIN_S:g} (the driver's clean takes up to {i.clean_wait_s:g} s)")


def check_timeouts(catalog: Catalog, jobs: ExecJobs, store: Store) -> None:
    """At startup: a timeout too short for its recipe refuses startup (TimeoutTooShort). Anything
    else — a recipe that does not parse, a host that cannot be reached — is logged and checked
    again before each run."""
    for key, m in catalog.models.items():
        if m.get("runner") != Runner.EXEC:
            continue
        try:
            jobs.check(key, m)
        except TimeoutTooShort:
            raise
        except DRIVER_ERRORS as e:
            store.event(Event.EXEC_UNCHECKED, model=key, error=str(e)[:ERR_SHORT])


class ExecJobs:
    def __init__(self, driver: Driver, staging: Staging, comfy: Comfy, timeouts: Timeouts,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.driver, self.staging, self.comfy, self.clock = driver, staging, comfy, clock
        self.timeouts = timeouts
        self._info: dict[str, RecipeInfo] = {}   # recipe -> the driver's timings, until a run fails

    def check(self, key: str, model: Model) -> None:
        """Raises TimeoutTooShort, or a driver error when the recipe cannot be read."""
        recipe = model["exec"]["recipe"]
        if recipe not in self._info:
            self._info[recipe] = self.driver.recipe_info(recipe)
        try:
            check_timeout(key, model, self._info[recipe], self.timeouts)
        except TimeoutTooShort:
            del self._info[recipe]   # the host may be fixed before the next job: read it afresh
            raise

    def run(self, jid: str, key: str, model: Model, payload: Mapping[str, Any]) -> dict[str, Any]:
        spec = model["exec"]
        self.check(key, model)   # cached: the scheduler checked it before evicting anything
        t0 = self.clock()
        with contextlib.ExitStack() as stack:
            files: list[tuple[str, Input]] = [(name, stack.enter_context(path.open("rb")))
                                              for name, path in self.staging.paths(jid)]
            if spec.get("params"):
                files.append((PARAMS_FILE, json.dumps(params(model, payload)).encode()))
            try:
                paths = self.driver.run_recipe(spec["recipe"], jid, files, float(spec["timeout_s"]))
            except Exception:
                self._info.pop(spec["recipe"], None)   # the host may have changed: read it afresh
                raise
        if not paths:
            raise RuntimeError(f"recipe {spec['recipe']} finished but produced no output file")
        return {"model": key, "outputs": [self.output(p) for p in paths],
                "wall_s": round(self.clock() - t0, DECIMALS)}

    def output(self, path: str) -> dict[str, str]:
        root = self.comfy.output_dir.rstrip("/")
        if root and path.startswith(root + "/"):
            rel = path[len(root) + 1:]
            sub, name = posixpath.split(rel)
            return {"file": rel, "path": path, "url": view_url(self.comfy.browser_url, name, sub)}
        return {"file": posixpath.basename(path), "path": path}
