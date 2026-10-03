"""Drivers for model servers on the machine the broker runs on: systemd units or Docker
containers. They share GPU reading (a gpu_broker.gpu probe: nvidia-smi or amdgpu sysfs) and
model-file handling, and differ only in how a unit is started and stopped.

Per-process VRAM is grouped by what owns the process — the systemd unit or Docker container
found in /proc/<pid>/cgroup, else "host". Inside a container without the host PID namespace
the probe sees no other processes, and the dashboard then shows the total only.
"""
from __future__ import annotations

import contextlib
import os
import pathlib
import re
import shutil
import subprocess
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from typing import Any

from .. import gpu as gpu_probe
from ..constants import DownloadKind, Verb
from ..gpu import amd, auto, nvidia
from ..gpu.auto import ProbeState
from ..settings import Gpu, Timeouts
from . import KILL_AFTER_S, PIPE_DRAIN_S, REAP_WAIT_S, Input, RecipeInfo, Run, check, reap, recipes, run, run_group, validate

HOST_GROUP = "host"
DOCKER_CGROUP = re.compile(r"docker[-/]([0-9a-f]{12})")
SYSTEMD_CGROUP = re.compile(r"/([^/\s]+)\.service")
ACTIVE, RUNNING = "active", "true"   # `systemctl is-active` / `docker inspect .State.Running` output
GIT_DIR = ".git"
NVIDIA_SMI, HF, GIT, DOCKER = nvidia.NVIDIA_SMI, "hf", "git", "docker"   # executables, overridable per driver
SYSTEMD_MODELS_ROOT = "/var/lib/gpu-broker/models"
DOCKER_MODELS_ROOT = "/models"
RECIPES_DIR = "/etc/gpu-broker/recipes"


def group_of(pid: str, proc: str = "/proc") -> str:
    """systemd unit or Docker container (short id) owning `pid`, else "host"."""
    try:
        cgroup = pathlib.Path(proc, pid, "cgroup").read_text()
    except OSError:
        return HOST_GROUP
    if m := DOCKER_CGROUP.search(cgroup):
        return m.group(1)
    if m := SYSTEMD_CGROUP.search(cgroup):
        return m.group(1)
    return HOST_GROUP


NO_EXEC = "the docker driver cannot run exec recipes (use the systemd or proxmox driver)"

class LocalDriver:
    """GPU via a probe (settings `gpu`) and files under `models_root`; subclasses implement
    `unit`. `run`, `group`, `sleep` and the sysfs/proc roots are injectable so tests never
    execute or read anything real."""

    def __init__(self, allowed: frozenset[str] | None, timeouts: Timeouts, sample_s: float,
                 models_root: str, comfy_models_dir: str | None = None, nvidia_smi: str = NVIDIA_SMI,
                 hf: str = HF, git: str = GIT, run: Run = run, group: Callable[[str], str] | None = None,
                 sleep: Callable[[float], None] = time.sleep, recipes_dir: str = RECIPES_DIR,
                 run_recipe_cmd: Run = run_group, gpu: Gpu | None = None,
                 sys_root: str = amd.SYS_ROOT, proc_root: str = amd.PROC_ROOT) -> None:
        self.allowed, self.t, self.sample_s = allowed, timeouts, sample_s
        self.root, self.comfy_dir = models_root, comfy_models_dir
        self.smi, self.hf, self.git = nvidia_smi, hf, git
        self.run, self.sleep = run, sleep
        self.group = group or (lambda pid: group_of(pid, proc_root))
        self.recipes_dir, self.run_recipe_cmd = recipes_dir, run_recipe_cmd
        self._closed = threading.Event()
        self.probe = auto.local(gpu or Gpu(), self.run, self.t.gpu_query_s, self.smi, sys_root, proc_root, sleep)

    def unit(self, spec: Any, verb: Verb) -> bool:
        raise NotImplementedError

    def _local(self, spec: Any, verb: Verb) -> str:
        u = check(spec, verb, self.allowed)
        if u.target is not None:
            raise ValueError(f"{type(self).__name__} runs units on this machine; drop target {u.target!r} from {u.name!r}")
        return u.name

    # ---- GPU ------------------------------------------------------------
    def gpu_probe(self) -> str:
        return self.probe.name

    def gpu_state(self) -> ProbeState:
        return self.probe.state()

    def gpu(self) -> tuple[int, int, int | None]:
        s = self.probe.read()
        return s.used_mib, s.total_mib, s.util_pct

    def sample_line(self) -> str:
        p = self.probe.procs()
        return gpu_probe.line(self.probe.read(), [(self.group(pid), mib) for pid, mib in p.rows], p.unreadable)

    def gpu_stream(self) -> Iterator[str]:
        while not self._closed.is_set():
            yield self.sample_line()
            self.sleep(self.sample_s)

    def close(self) -> None:
        self._closed.set()

    # ---- model files ----------------------------------------------------
    def download(self, kind: str, ref: str, slug: str, include: list[str]) -> subprocess.CompletedProcess[str]:
        k = validate.download(kind, ref, slug, include)
        dest = validate.inside(self.root, slug)
        os.makedirs(dest, exist_ok=True)
        if k is DownloadKind.HF:  # `hf` is the huggingface_hub CLI: pip install 'gpu-broker[download]'
            flags = [a for pattern in include for a in ("--include", pattern)]
            return self.run([self.hf, "download", ref, *flags, "--local-dir", dest], timeout=self.t.download_s)
        if os.path.isdir(os.path.join(dest, GIT_DIR)):
            return self.run([self.git, "-C", dest, "pull", "--ff-only"], timeout=self.t.git_s)
        return self.run([self.git, "clone", "--depth", "1", "--", ref, dest], timeout=self.t.git_s)

    def link(self, rel: str, subdir: str) -> subprocess.CompletedProcess[str]:
        if not self.comfy_dir:
            raise RuntimeError("comfy_models_dir is not configured")
        validate.link(rel, subdir)
        src = validate.inside(self.root, rel)
        if not os.path.isfile(src):
            raise FileNotFoundError(src)
        dst = validate.inside(self.comfy_dir, subdir, os.path.basename(rel))
        if os.path.islink(dst):
            os.unlink(dst)
        os.symlink(src, dst)
        return subprocess.CompletedProcess(["symlink", src, dst], 0, "", "")

    # ---- exec recipes ---------------------------------------------------
    def _recipe(self, recipe: str) -> recipes.Recipe:
        r = recipes.load(self.recipes_dir, recipe)
        if r.target is not None:
            raise ValueError(f"recipe {recipe}: `target` is for the proxmox driver; this one runs on this machine")
        return r

    def recipe_info(self, recipe: str) -> RecipeInfo:
        return RecipeInfo(self._recipe(recipe).timeout_s, KILL_AFTER_S, PIPE_DRAIN_S + REAP_WAIT_S, REAP_WAIT_S)

    def clean_recipe(self, recipe: str, jid: str) -> None:
        """Reap first: that needs only the job id. The recipe is read just to find the inputs
        folder, so a recipe that no longer loads leaves the folder but does not keep the hold."""
        validate.recipe_call(recipe, jid, [])
        if not reap.reap(jid, REAP_WAIT_S):
            raise RuntimeError(f"processes of job {jid} still run (or the scan cannot tell)")
        with contextlib.suppress(OSError, ValueError):
            shutil.rmtree(self._recipe(recipe).dirs(jid)[0], ignore_errors=True)

    def run_recipe(self, recipe: str, jid: str, files: Sequence[tuple[str, Input]], timeout_s: float) -> list[str]:
        """Runs under the recipe's own timeout_s (the catalog's, checked to be longer, bounds the
        job); the process group is gone when this returns, so a timeout is an ordinary failure."""
        validate.recipe_call(recipe, jid, [n for n, _ in files])
        return recipes.run_local(self._recipe(recipe), jid, files, self.run_recipe_cmd)


class SystemdDriver(LocalDriver):
    """Units started with systemctl. Needs root, `sudo: true` with a sudoers rule limited to
    `systemctl start|stop <unit>`, or `user: true` for user units."""

    def __init__(self, sudo: bool = False, user: bool = False, models_root: str = SYSTEMD_MODELS_ROOT,
                 **kw: Any) -> None:
        super().__init__(models_root=models_root, **kw)
        self.base = [*(["sudo", "-n"] if sudo else []), "systemctl", *(["--user"] if user else [])]

    def unit(self, spec: Any, verb: Verb) -> bool:
        r = self.run([*self.base, verb, "--", self._local(spec, verb)], timeout=self.t.unit_s)
        return r.stdout.strip() == ACTIVE if verb == Verb.IS_ACTIVE else r.returncode == 0


class DockerDriver(LocalDriver):
    """Containers started and stopped by name. They must already exist (`docker compose
    create`); the broker needs the Docker socket."""

    def __init__(self, docker: str = DOCKER, models_root: str = DOCKER_MODELS_ROOT, **kw: Any) -> None:
        super().__init__(models_root=models_root, **kw)
        self.docker = docker

    def unit(self, spec: Any, verb: Verb) -> bool:
        name = self._local(spec, verb)
        if verb == Verb.IS_ACTIVE:
            r = self.run([self.docker, "inspect", "-f", "{{.State.Running}}", "--", name], timeout=self.t.unit_s)
            return r.returncode == 0 and r.stdout.strip() == RUNNING
        argv = ([self.docker, "start", "--", name] if verb == Verb.START else
                [self.docker, "stop", "-t", str(self.t.container_stop_s), "--", name])
        return self.run(argv, timeout=self.t.unit_s).returncode == 0

    def recipe_info(self, recipe: str) -> RecipeInfo:
        raise RuntimeError(NO_EXEC)

    def clean_recipe(self, recipe: str, jid: str) -> None:
        raise RuntimeError(NO_EXEC)

    def run_recipe(self, recipe: str, jid: str, files: Sequence[tuple[str, Input]], timeout_s: float) -> list[str]:
        raise RuntimeError(NO_EXEC)

