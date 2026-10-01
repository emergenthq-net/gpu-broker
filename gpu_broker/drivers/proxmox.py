"""Proxmox driver: model servers are systemd units inside LXC containers on a Proxmox host.

The broker holds one SSH key that the host restricts (`command=` in authorized_keys) to
host/gpu-broker-ctl. That script accepts only a few verbs, re-validates every argument and
runs `pct exec`, so the broker has no general access to the hypervisor. Arguments are still
validated here first: the SSH command line is re-split by the remote shell's word rules.
"""
from __future__ import annotations

import subprocess
import threading
from collections.abc import Iterator
from typing import Any

from ..constants import ERR_SHORT, Verb
from ..settings import Timeouts
from . import Run, check, run, validate

DEFAULT_KEY = "/etc/gpu-broker/id_ed25519"
ACTIVE = "active"
STREAM_ENDED = "stream ended"


class Cmd:
    """Verbs understood by host/gpu-broker-ctl."""
    UNIT, GPU, GPU_STREAM, DOWNLOAD, LINK = "unit", "gpu", "gpustream", "download", "comfy-link"


class ProxmoxDriver:
    def __init__(self, ssh_target: str, allowed: frozenset[str] | None, timeouts: Timeouts,
                 ssh_key: str = DEFAULT_KEY, run: Run = run, popen: Any = subprocess.Popen) -> None:
        self.allowed, self.t, self._run, self._popen = allowed, timeouts, run, popen
        self._streams: set[Any] = set()   # open gpustream SSH sessions, killed by close()
        self._lock = threading.Lock()
        self._closed = threading.Event()
        self.base = ["ssh", "-i", ssh_key, "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new",
                     "-o", f"ConnectTimeout={timeouts.ssh_connect_s}", "--", ssh_target]

    def _ctl(self, *args: str, timeout: float) -> subprocess.CompletedProcess[str]:
        return self._run([*self.base, *args], timeout=timeout)

    def unit(self, spec: Any, verb: Verb) -> bool:
        u = check(spec, verb, self.allowed)
        if u.target is None:
            raise ValueError(f"proxmox units need a container id: {{name: {u.name}, target: <id>}}")
        r = self._ctl(Cmd.UNIT, u.target, u.name, verb, timeout=self.t.unit_s)
        return r.stdout.strip() == ACTIVE if verb == Verb.IS_ACTIVE else r.returncode == 0

    def gpu(self) -> tuple[int, int, int]:
        used, total, util = (int(x) for x in self._ctl(Cmd.GPU, timeout=self.t.gpu_query_s).stdout.split(","))
        return used, total, util

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
