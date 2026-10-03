"""Host drivers: how the broker starts and stops model servers, reads the GPU and fetches
model files.

Every driver implements `Driver`, so the scheduler, residency and the dashboard never know
whether a model server is a systemd unit on this machine, a Docker container, or a unit
inside a Proxmox container reached through a restricted SSH key. Drivers only ever touch
allowlisted units; by default that is exactly the units the catalog and config name.
"""
from __future__ import annotations

import contextlib
import os
import signal
import subprocess
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import IO, Any, Protocol

from ..constants import Verb
from ..gpu.auto import ProbeState
from ..settings import Settings
from ..units import UnitRef, unit_ref
from . import reap

Run = Callable[..., subprocess.CompletedProcess[str]]
RunInput = Callable[..., subprocess.CompletedProcess[bytes]]
Input = bytes | IO[bytes]   # an exec input file: small generated data, or an open file streamed as-is
KILL_AFTER_S = 10   # local drivers: an exec recipe gets SIGTERM at its timeout_s and SIGKILL this much later
REAP_WAIT_S = 10    # local drivers: after a recipe, how long its leftover tagged processes may take to die
PIPE_DRAIN_S = 5    # after SIGKILL, how long the recipe's pipes may take to close
# What a driver call may raise: exec/SSH failures, timeouts, rejected arguments, refused units.
DRIVER_ERRORS = (OSError, ValueError, subprocess.SubprocessError, RuntimeError)


class GpuHeld(RuntimeError):
    """An exec recipe may still be running (and holding the GPU) after its job ended: the
    scheduler stops taking jobs until a clean confirms it is gone or an operator clears it."""


@dataclass(frozen=True)
class RecipeInfo:
    """What the broker needs to bound an exec job, as the driver that runs it applies it."""
    timeout_s: float      # the recipe's own: SIGTERM then
    kill_after_s: float   # SIGKILL this much later
    reap_s: float         # after that, at most this long making sure no process of the job is left
    clean_wait_s: float   # longest a separate clean (clean_recipe) takes to confirm the job is gone


def run(cmd: list[str], timeout: float) -> subprocess.CompletedProcess[str]:
    """The one place drivers execute anything: an argv list, never a shell string."""
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)  # noqa: S603 — argv validated by callers


def run_input(cmd: list[str], data: Input, timeout: float) -> subprocess.CompletedProcess[bytes]:
    """As `run`, with `data` on the command's stdin (how a file reaches a remote host). An open
    file is streamed from its descriptor, never read into memory."""
    if isinstance(data, bytes):
        return subprocess.run(cmd, input=data, capture_output=True, timeout=timeout, check=False)  # noqa: S603 — argv validated by callers
    return subprocess.run(cmd, stdin=data, capture_output=True, timeout=timeout, check=False)  # noqa: S603 — argv validated by callers


def run_group(cmd: list[str], timeout: float, kill_after: float = KILL_AFTER_S, tag: str | None = None,
              drain_s: float = PIPE_DRAIN_S, reap_s: float = REAP_WAIT_S,
              proc: str = reap.PROC) -> subprocess.CompletedProcess[str]:
    """As `run`, in a session of its own: at `timeout` the whole process group gets SIGTERM, and
    SIGKILL `kill_after` seconds later. Raises subprocess.TimeoutExpired once the group is gone.

    A worker that left the group (setsid, double fork) survives that. With `tag` (a job id) the
    command runs with GPU_BROKER_JOB=<tag> in its environment, and afterwards every process
    still carrying it is killed (reap.py). GpuHeld if either says one may survive: the pipes
    stay open after SIGKILL, or a tagged process outlives `reap_s` (or the scan cannot tell)."""
    env = {**os.environ, reap.MARK: tag} if tag else None
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,  # noqa: S603 — argv validated by callers
                         start_new_session=True, env=env)
    held: GpuHeld | None = None
    try:
        try:
            out, err = p.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                _stop_group(p, kill_after, drain_s)
            except GpuHeld as e:
                held = e
            raise
    finally:
        # The scan runs even when the pipes already said "held": it kills what it can find. Its
        # visibility check uses our child's pid while that is still ours (not yet waited for).
        alive = p.pid if p.returncode is None else None
        if tag and not reap.reap(tag, reap_s, proc, alive=alive):
            raise GpuHeld(f"processes of job {tag} survive their recipe (or the scan cannot tell)")
        if held is not None:
            raise held
    return subprocess.CompletedProcess(cmd, p.returncode, out, err)


def _stop_group(p: subprocess.Popen[str], kill_after: float, drain_s: float) -> None:
    """SIGTERM the group, SIGKILL it `kill_after` s later; then the pipes must close within
    `drain_s`, else something outside the group holds them (and maybe the GPU)."""
    for sig, wait in ((signal.SIGTERM, kill_after), (signal.SIGKILL, drain_s)):
        with contextlib.suppress(ProcessLookupError, PermissionError):   # EPERM: macOS, a group of zombies
            os.killpg(p.pid, sig)
        try:
            p.communicate(timeout=wait)
            return
        except subprocess.TimeoutExpired:
            continue
    for pipe in (p.stdout, p.stderr):
        if pipe is not None:
            pipe.close()
    raise GpuHeld(f"recipe process group {p.pid} was killed but its output pipes stay open: "
                  "a worker that left the group may still hold the GPU")


class Driver(Protocol):
    allowed: frozenset[str] | None

    def unit(self, spec: Any, verb: Verb) -> bool:
        """start/stop: True on success. is-active: True if running."""

    def gpu_probe(self) -> str:
        """Which GPU reader is used (resolving `gpu.vendor: auto`); raises when there is none."""

    def gpu_state(self) -> ProbeState:
        """Whether the GPU reader is ready, failed or still being chosen; never blocks."""

    def gpu(self) -> tuple[int, int, int | None]:
        """(used MiB, total MiB, utilisation % or None when the card does not report it)."""

    def gpu_stream(self) -> Iterator[str]:
        """Sample lines (gpu_broker.gpu.line); returns when the source ends."""

    def close(self) -> None:
        """Shutdown: end any open `gpu_stream` now (a reader blocked on it must wake up)."""

    def download(self, kind: str, ref: str, slug: str, include: list[str]) -> subprocess.CompletedProcess[str]:
        """Fetch `ref` (a Hugging Face repo id or GitHub URL) into <models root>/<slug>."""

    def link(self, rel: str, subdir: str) -> subprocess.CompletedProcess[str]:
        """Expose <models root>/<rel> to ComfyUI under models/<subdir>/."""

    def recipe_info(self, recipe: str) -> RecipeInfo:
        """How long the recipe may run and be stopped, for checking the catalog's exec.timeout_s."""

    def clean_recipe(self, recipe: str, jid: str) -> None:
        """Stop whatever still runs for job `jid` and remove its inputs; raises unless that is
        confirmed (how a GPU hold clears itself)."""

    def run_recipe(self, recipe: str, jid: str, files: Sequence[tuple[str, Input]], timeout_s: float) -> list[str]:
        """Run exec recipe `recipe` (drivers/recipes.py) for job `jid` on its input `files`
        ((name, data or open file) pairs); returns the output file paths as the recipe's host
        sees them. Raises RuntimeError when the recipe fails, and GpuHeld when it may still be
        running after this returns."""


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
        return SystemdDriver(sample_s=settings.intervals.gpu_sample_s, gpu=settings.gpu, **common)
    if d.kind == "docker":
        from .local import DockerDriver
        return DockerDriver(sample_s=settings.intervals.gpu_sample_s, gpu=settings.gpu, **common)
    if d.kind == "proxmox":
        from .proxmox import ProxmoxDriver
        return ProxmoxDriver(**common)
    raise ValueError(f"unknown driver kind {d.kind!r} (systemd|docker|proxmox)")
