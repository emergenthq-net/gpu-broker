"""Host drivers: how the broker starts and stops model servers, reads the GPU and fetches
model files.

Every driver implements `Driver`, so the scheduler, residency and the dashboard never know
whether a model server is a systemd unit on this machine, a Docker container, or a unit
inside a Proxmox container reached through a restricted SSH key. Drivers only ever touch
allowlisted units; by default that is exactly the units the catalog and config name.
"""
from __future__ import annotations

import subprocess
from collections.abc import Callable, Iterator
from typing import Any, Protocol

from ..constants import Verb
from ..settings import Settings
from ..units import UnitRef, unit_ref

Run = Callable[..., subprocess.CompletedProcess[str]]
# What a driver call may raise: exec/SSH failures, timeouts, rejected arguments, refused units.
DRIVER_ERRORS = (OSError, ValueError, subprocess.SubprocessError, RuntimeError)


def run(cmd: list[str], timeout: float) -> subprocess.CompletedProcess[str]:
    """The one place drivers execute anything: an argv list, never a shell string."""
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)  # noqa: S603 — argv validated by callers


class Driver(Protocol):
    allowed: frozenset[str] | None

    def unit(self, spec: Any, verb: Verb) -> bool:
        """start/stop: True on success. is-active: True if running."""

    def gpu(self) -> tuple[int, int, int]:
        """(used MiB, total MiB, utilisation %)."""

    def gpu_stream(self) -> Iterator[str]:
        """Sample lines `used,total,util,power,temp,sm|group:mib ...`; returns when the source ends."""

    def close(self) -> None:
        """Shutdown: end any open `gpu_stream` now (a reader blocked on it must wake up)."""

    def download(self, kind: str, ref: str, slug: str, include: list[str]) -> subprocess.CompletedProcess[str]:
        """Fetch `ref` (a Hugging Face repo id or GitHub URL) into <models root>/<slug>."""

    def link(self, rel: str, subdir: str) -> subprocess.CompletedProcess[str]:
        """Expose <models root>/<rel> to ComfyUI under models/<subdir>/."""


def check(spec: Any, verb: str, allowed: frozenset[str] | None) -> UnitRef:
    """Validate a unit request: a clean name, a known verb, and (when given) allowlisted."""
    u = unit_ref(spec)
    if verb not in set(Verb):
        raise ValueError(f"bad verb {verb!r}")
    if allowed is not None and u.key not in allowed:
        raise PermissionError(f"unit {u.key!r} is not allowlisted")
    return u


def build(settings: Settings, catalog_units: list[UnitRef]) -> Driver:
    """Construct the configured driver. Construction runs nothing on the host."""
    d = settings.driver
    if d.allowed_units is not None:
        allowed = frozenset(u.key for u in d.allowed_units)
    else:
        extra = [settings.comfy.unit] if settings.comfy.unit else []
        allowed = frozenset(u.key for u in [*catalog_units, *extra])
    common: dict[str, Any] = {"allowed": allowed, "timeouts": settings.timeouts, **d.options}
    if d.kind == "systemd":
        from .local import SystemdDriver
        return SystemdDriver(sample_s=settings.intervals.gpu_sample_s, **common)
    if d.kind == "docker":
        from .local import DockerDriver
        return DockerDriver(sample_s=settings.intervals.gpu_sample_s, **common)
    if d.kind == "proxmox":
        from .proxmox import ProxmoxDriver
        return ProxmoxDriver(**common)
    raise ValueError(f"unknown driver kind {d.kind!r} (systemd|docker|proxmox)")
