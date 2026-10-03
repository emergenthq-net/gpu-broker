"""Proxmox driver: model servers are systemd units inside LXC containers on a Proxmox host.

The broker holds one SSH key that the host restricts (`command=` in authorized_keys) to
host/gpu-broker-ctl. That script accepts only a few verbs, re-validates every argument and
runs `pct exec`, so the broker has no general access to the hypervisor. Arguments are still
validated here first: the SSH command line is re-split by the remote shell's word rules.
"""
from __future__ import annotations

import subprocess
import threading
from collections.abc import Iterator, Sequence
from typing import Any

from ..constants import ERR_DETAIL, ERR_SHORT, Verb
from ..gpu import opt
from ..gpu.auto import READY, ProbeState
from ..settings import Timeouts
from . import GpuHeld, Input, RecipeInfo, Run, RunInput, check, run, run_input, validate

DEFAULT_KEY = "/etc/gpu-broker/id_ed25519"
HOST_PROBE = "on the Proxmox host (host/gpu-broker-gpu: GPU_VENDOR in /etc/gpu-broker-ctl.conf, default auto)"
ACTIVE = "active"
STREAM_ENDED = "stream ended"
OUTPUT_LINE = "output "   # exec-run prints one `output <path>` line per result file


class Cmd:
    """Verbs understood by host/gpu-broker-ctl."""
    UNIT, GPU, GPU_STREAM, DOWNLOAD, LINK = "unit", "gpu", "gpustream", "download", "comfy-link"
    EXEC_PUT, EXEC_RUN, EXEC_CLEAN, EXEC_INFO = "exec-put", "exec-run", "exec-clean", "exec-info"


INFO_KEYS = ("timeout_s", "kill_after_s", "reap_s", "clean_wait_s")   # exec-info prints <key>=<seconds> lines
SSH_FAILED = 255        # ssh's own failure (or the remote killed by a signal): exec-run's outcome is unknown
SURVIVORS = 7           # exec-run: the program ended but processes of the job still run


class ProxmoxDriver:
    def __init__(self, ssh_target: str, allowed: frozenset[str] | None, timeouts: Timeouts,
                 ssh_key: str = DEFAULT_KEY, run: Run = run, popen: Any = subprocess.Popen,
                 run_input: RunInput = run_input) -> None:
        self.allowed, self.t, self._run, self._popen, self._run_input = allowed, timeouts, run, popen, run_input
        self._streams: set[Any] = set()   # open gpustream SSH sessions, killed by close()
        self._lock = threading.Lock()
        self._closed = threading.Event()
        # IdentitiesOnly: offer only the broker's key, never the user's other keys or agent.
        self.base = ["ssh", "-i", ssh_key, "-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes",
                     "-o", "StrictHostKeyChecking=accept-new", "-o", f"ConnectTimeout={timeouts.ssh_connect_s}",
                     "--", ssh_target]

    def _ctl(self, *args: str, timeout: float) -> subprocess.CompletedProcess[str]:
        return self._run([*self.base, *args], timeout=timeout)

    def unit(self, spec: Any, verb: Verb) -> bool:
        u = check(spec, verb, self.allowed)
        if u.target is None:
            raise ValueError(f"proxmox units need a container id: {{name: {u.name}, target: <id>}}")
        r = self._ctl(Cmd.UNIT, u.target, u.name, verb, timeout=self.t.unit_s)
        return r.stdout.strip() == ACTIVE if verb == Verb.IS_ACTIVE else r.returncode == 0

    def gpu_probe(self) -> str:
        return HOST_PROBE

    def gpu_state(self) -> ProbeState:
        return ProbeState(READY, HOST_PROBE)   # the host script decides, within its own budget

    def gpu(self) -> tuple[int, int, int | None]:
        used, total, util = (opt(x) for x in self._ctl(Cmd.GPU, timeout=self.t.gpu_query_s).stdout.split(","))
        if used is None or total is None:
            raise RuntimeError("the host reported no GPU memory used or total")
        return int(used), int(total), None if util is None else int(util)

    def gpu_stream(self) -> Iterator[str]:
        """One long-lived SSH session; the host script prints a sample line periodically."""
        with self._lock:
            if self._closed.is_set():
                return
            p = self._popen([*self.base, Cmd.GPU_STREAM], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            self._streams.add(p)
        try:
            yield from p.stdout
            if self._closed.is_set():
                return
            raise RuntimeError((p.stderr.read() or STREAM_ENDED)[-ERR_SHORT:])
        finally:
            p.kill()
            with self._lock:
                self._streams.discard(p)

    def close(self) -> None:
        """Kill the SSH session(s): the reader sees EOF and the sampler thread ends."""
        with self._lock:
            self._closed.set()
            streams = list(self._streams)
        for p in streams:
            p.kill()

    def download(self, kind: str, ref: str, slug: str, include: list[str]) -> subprocess.CompletedProcess[str]:
        validate.download(kind, ref, slug, include)
        return self._ctl(Cmd.DOWNLOAD, kind, ref, slug, *include, timeout=self.t.download_s)

    def link(self, rel: str, subdir: str) -> subprocess.CompletedProcess[str]:
        validate.link(rel, subdir)
        return self._ctl(Cmd.LINK, rel, subdir, timeout=self.t.comfy_http_s)

    def recipe_info(self, recipe: str) -> RecipeInfo:
        """The host's own values: its recipe's timeout_s, and its KILL_AFTER_S and reap/clean bounds.
        clean_wait_s also counts this side's SSH connect (ConnectTimeout); execjob.CLEAN_MARGIN_S
        covers the rest (key exchange, reading the recipe, opening the job's lock file)."""
        if not validate.RECIPE.match(recipe):
            raise ValueError(f"bad recipe name {recipe!r}")
        r = self._ctl(Cmd.EXEC_INFO, recipe, timeout=self.t.unit_s)
        found = dict(x.split("=", 1) for x in r.stdout.splitlines() if x.split("=", 1)[0] in INFO_KEYS)
        if r.returncode != 0 or set(found) != set(INFO_KEYS):
            raise RuntimeError(f"reading recipe {recipe} failed: {r.stderr[-ERR_DETAIL:]}")
        timeout_s, kill_after_s, reap_s, clean_wait_s = (float(found[k]) for k in INFO_KEYS)
        return RecipeInfo(timeout_s, kill_after_s, reap_s, clean_wait_s + self.t.ssh_connect_s)

    def clean_recipe(self, recipe: str, jid: str) -> None:
        validate.recipe_call(recipe, jid, [])
        c = self._ctl(Cmd.EXEC_CLEAN, recipe, jid, timeout=self.t.exec_clean_s)
        if c.returncode != 0:
            raise RuntimeError(f"exec-clean of job {jid} failed ({c.returncode}): {c.stderr[-ERR_DETAIL:]}")

    def run_recipe(self, recipe: str, jid: str, files: Sequence[tuple[str, Input]], timeout_s: float) -> list[str]:
        """Each input goes over its own SSH call on stdin (`exec-put`); `exec-run` then runs the
        host's recipe in its container. Only the recipe name, job id and file names cross SSH.

        exec-run's exit status says how it ended (it reaps the job's processes itself), except
        when its state is unknown: no answer within timeout_s, SSH failure (255), or job
        processes that survived its reap (7). Then, and whenever it was not reached,
        `exec-clean` stops whatever runs for the job and removes its inputs. If the program may
        have started and the clean cannot confirm it is gone, GpuHeld: the broker must not hand
        the GPU to the next job."""
        validate.recipe_call(recipe, jid, [n for n, _ in files])
        started = settled = False
        try:
            for name, data in files:
                put = self._run_input([*self.base, Cmd.EXEC_PUT, recipe, jid, name], data, timeout=self.t.exec_put_s)
                if put.returncode != 0:
                    raise RuntimeError(f"copying {name} for recipe {recipe} failed: "
                                       f"{put.stderr.decode(errors='replace')[-ERR_DETAIL:]}")
            started = True
            try:
                r = self._ctl(Cmd.EXEC_RUN, recipe, jid, timeout=timeout_s)
            except subprocess.TimeoutExpired:
                raise RuntimeError(f"recipe {recipe} did not finish within {timeout_s:g}s") from None
            if r.returncode in (SSH_FAILED, SURVIVORS):
                raise RuntimeError(f"recipe {recipe}: exec-run ended {r.returncode} with the job's state unknown: "
                                   f"{r.stderr[-ERR_DETAIL:]}")
            settled = True
        finally:
            if not settled:
                self._clean(recipe, jid, started)
        if r.returncode != 0:
            raise RuntimeError(f"recipe {recipe} exited {r.returncode}: {r.stderr[-ERR_DETAIL:]}")
        return [line.removeprefix(OUTPUT_LINE) for line in r.stdout.splitlines() if line.startswith(OUTPUT_LINE)]

    def _clean(self, recipe: str, jid: str, started: bool) -> None:
        try:
            self.clean_recipe(recipe, jid)
        except (OSError, subprocess.SubprocessError, RuntimeError) as e:
            if started:
                raise GpuHeld(f"recipe {recipe} for job {jid} may still be running on the host: "
                              f"{str(e)[-ERR_DETAIL:]}") from e
