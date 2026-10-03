"""`gpu.vendor: auto`: which probe to use, decided once, on first use, by one caller at a time.

nvidia when nvidia-smi is installed and answers, else amd when an amdgpu card exists. An
installed nvidia-smi that fails or hangs is retried, all within `budget_s` of wall-clock time
(each check is given only what is left), then an error: never a reason to read another
vendor's card. A failed choice is remembered for `budget_s`, so callers in that window get the
error at once instead of waiting through the retries again.

`state()` never blocks: it reports ready, failed or probing, and starts the choice in a
background thread when nobody is making it. Request handlers use it, so a hung nvidia-smi
can never hold up a web request; the sampler thread, which may wait, reads through `probe()`.
"""
from __future__ import annotations

import contextlib
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final, Literal

from ..settings import Gpu
from . import AMD, AUTO, NVIDIA, GpuProbe, GpuSample, NvidiaState, Procs, amd, check_vendor, nvidia
from .nvidia import Run

RETRY_PAUSE_S = 1
NO_GPU = ("no GPU found: nvidia-smi is not installed and no amdgpu card is listed under "
          "{sys}/class/drm (set gpu.vendor to force one)")
NVIDIA_FAILING = ("nvidia-smi is installed but did not answer within {s:g} s ({why}); not falling back "
                  "to another vendor's card (set gpu.vendor: amd to read an AMD card instead)")
log = logging.getLogger(__package__)


State = Literal["ready", "probing", "failed"]
READY: Final = "ready"
PROBING: Final = "probing"
FAILED: Final = "failed"


@dataclass(frozen=True)
class ProbeState:
    state: State
    detail: str = ""   # ready: the probe's name; failed: why


class Auto:
    """Delegates to the chosen probe; `factories` build each one. `nvidia_state(timeout_s)` checks
    nvidia-smi once, giving up after `timeout_s`."""

    def __init__(self, vendor: str, factories: dict[str, Callable[[], GpuProbe]],
                 nvidia_state: Callable[[float], tuple[NvidiaState, str]], amd_present: Callable[[], bool],
                 sys_root: str, budget_s: float, clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self.vendor, self.factories = check_vendor(vendor), factories
        self.nvidia_state, self.amd_present, self.sys_root = nvidia_state, amd_present, sys_root
        self.budget_s, self.clock, self.sleep = budget_s, clock, sleep
        self._probe: GpuProbe | None = None
        self._failed: tuple[str, float] | None = None      # (why, until when it stands)
        self._lock = threading.Lock()                       # one choice at a time
        self._bg: threading.Thread | None = None
        self._bg_lock = threading.Lock()

    def probe(self) -> GpuProbe:
        if (p := self._probe) is not None:
            return p
        with self._lock:
            if self._probe is not None:                      # chosen while we waited
                return self._probe
            if (f := self._failed) and self.clock() < f[1]:
                raise RuntimeError(f[0])
            try:
                self._probe = self.factories[self._choose()]()
            except RuntimeError as e:
                self._failed = (str(e), self.clock() + self.budget_s)
                raise
            self._failed = None
            return self._probe

    def state(self) -> ProbeState:
        """Never blocks; starts the choice in the background when it is due."""
        if (p := self._probe) is not None:
            return ProbeState(READY, p.name)
        if (f := self._failed) and self.clock() < f[1]:
            return ProbeState(FAILED, f[0])
        with self._bg_lock:
            if self._bg is None or not self._bg.is_alive():
                self._bg = threading.Thread(target=self._try, daemon=True, name="gpu-auto")
                self._bg.start()
        return ProbeState(PROBING)

    def _try(self) -> None:
        with contextlib.suppress(RuntimeError):   # remembered in _failed; state() reports it
            self.probe()

    def _choose(self) -> str:
        if self.vendor != AUTO:
            return self.vendor
        end = self.clock() + self.budget_s
        state, why = self.nvidia_state(self.budget_s)
        if state == "absent":
            if self.amd_present():
                log.info("gpu.vendor auto: amd (nvidia-smi is not installed; an amdgpu card is)")
                return AMD
            raise RuntimeError(NO_GPU.format(sys=self.sys_root))
        while state != "ok":   # once a second, within the budget (as host/gpu-broker-gpu)
            if (left := end - self.clock()) <= RETRY_PAUSE_S:
                raise RuntimeError(NVIDIA_FAILING.format(s=self.budget_s, why=why))
            self.sleep(RETRY_PAUSE_S)
            state, why = self.nvidia_state(left - RETRY_PAUSE_S)
        if self.amd_present():
            log.warning("gpu.vendor auto: nvidia (nvidia-smi answers); an amdgpu card is also present "
                        "and is not read: set gpu.vendor: amd to read it instead")
        else:
            log.info("gpu.vendor auto: nvidia (nvidia-smi answers)")
        return NVIDIA

    @property
    def name(self) -> str:
        return self.probe().name

    def read(self) -> GpuSample:
        return self.probe().read()

    def procs(self) -> Procs:
        return self.probe().procs()


def local(g: Gpu, run: Run, query_s: float, smi: str, sys_root: str, proc_root: str,
          sleep: Callable[[float], None] = time.sleep) -> Auto:
    """The probe for a GPU on this machine, as the systemd and docker drivers read it."""
    return Auto(g.vendor,
                {NVIDIA: lambda: nvidia.NvidiaProbe(run, query_s, g.index, smi),
                 AMD: lambda: amd.AmdProbe(g.index, sys_root, proc_root)},
                lambda t: nvidia.state(run, t, smi), lambda: amd.present(sys_root), sys_root,
                budget_s=query_s, sleep=sleep)
