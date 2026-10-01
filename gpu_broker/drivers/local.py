"""Drivers for model servers on the machine the broker runs on: systemd units or Docker
containers. They share GPU reading (nvidia-smi) and model-file handling, and differ only in
how a unit is started and stopped.

Per-process VRAM is grouped by what owns the process — the systemd unit or Docker container
found in /proc/<pid>/cgroup, else "host". Inside a container without the host PID namespace
nvidia-smi lists no processes, and the dashboard then shows the total only.
"""
from __future__ import annotations

import os
import pathlib
import re
import subprocess
import threading
import time
from collections.abc import Callable, Iterator
from typing import Any

from ..constants import DownloadKind, Verb
from ..settings import Timeouts
from . import Run, check, run, validate

GPU_QUERY = "memory.used,memory.total,utilization.gpu"
STREAM_QUERY = GPU_QUERY + ",power.draw,temperature.gpu,clocks.sm"
APPS_QUERY = "pid,used_memory"
CSV = "--format=csv,noheader,nounits"
HOST_GROUP = "host"
DOCKER_CGROUP = re.compile(r"docker[-/]([0-9a-f]{12})")
SYSTEMD_CGROUP = re.compile(r"/([^/\s]+)\.service")
ACTIVE, RUNNING = "active", "true"   # `systemctl is-active` / `docker inspect .State.Running` output
GIT_DIR = ".git"
NVIDIA_SMI, HF, GIT, DOCKER = "nvidia-smi", "hf", "git", "docker"   # executables, overridable per driver
SYSTEMD_MODELS_ROOT = "/var/lib/gpu-broker/models"
DOCKER_MODELS_ROOT = "/models"


def _fields(csv: str) -> list[str]:
    return [x.strip() for x in csv.strip().splitlines()[0].split(",")]


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


class LocalDriver:
    """GPU via nvidia-smi and files under `models_root`; subclasses implement `unit`.
    `run`, `group` and `sleep` are injectable so tests never execute anything."""

    def __init__(self, allowed: frozenset[str] | None, timeouts: Timeouts, sample_s: float,
                 models_root: str, comfy_models_dir: str | None = None, nvidia_smi: str = NVIDIA_SMI,
                 hf: str = HF, git: str = GIT, run: Run = run, group: Callable[[str], str] = group_of,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self.allowed, self.t, self.sample_s = allowed, timeouts, sample_s
        self.root, self.comfy_dir = models_root, comfy_models_dir
        self.smi, self.hf, self.git = nvidia_smi, hf, git
        self.run, self.group, self.sleep = run, group, sleep
        self._closed = threading.Event()

    def unit(self, spec: Any, verb: Verb) -> bool:
        raise NotImplementedError

    def _local(self, spec: Any, verb: Verb) -> str:
        u = check(spec, verb, self.allowed)
        if u.target is not None:
            raise ValueError(f"{type(self).__name__} runs units on this machine; drop target {u.target!r} from {u.name!r}")
        return u.name

    # ---- GPU ------------------------------------------------------------
    def _smi(self, query: str) -> str:
        return self.run([self.smi, query, CSV], timeout=self.t.gpu_query_s).stdout

    def gpu(self) -> tuple[int, int, int]:
        used, total, util = (int(float(x)) for x in _fields(self._smi(f"--query-gpu={GPU_QUERY}")))
        return used, total, util

    def sample_line(self) -> str:
        head = ",".join(_fields(self._smi(f"--query-gpu={STREAM_QUERY}")))
        procs = []
        for row in self._smi(f"--query-compute-apps={APPS_QUERY}").splitlines():
            pid, _, mib = row.replace(" ", "").partition(",")
            if pid.isdigit() and mib.isdigit():
                procs.append(f"{self.group(pid)}:{mib}")
        return f"{head}|{' '.join(procs)}"

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

