"""GPU readings, per vendor: a `GpuProbe` reads the card's memory, utilisation, power,
temperature and clock, and lists the processes using its memory.

Probes: `nvidia` (nvidia-smi) and `amd` (the amdgpu driver's sysfs files and the DRM fdinfo of
/proc; no ROCm tools needed). `vendor: auto` is decided on first use, not at construction
(gpu/auto.py): nvidia when nvidia-smi is installed (it must answer, retried for a while: a
slow or failing nvidia-smi is an error, never a reason to read another vendor's card), else
amd when an amdgpu card exists. Only used and total memory are required; utilisation, power, temperature
and clock may be unknown (None), and the sample line then leaves the field empty.

The sample line is the wire format between a driver and metrics.parse_sample, and also what
host/gpu-broker-gpu prints: `used,total,util,power,temp,clock|group:mib ...` (MiB, %, W, °C,
MHz; one `group:mib` per process, grouped by whoever owns it). `PROCS_UNREADABLE` in the
group list means some processes' memory could not be read (it has no `:`, so parsers that
predate it skip it).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal, Protocol

VENDORS = ("auto", "nvidia", "amd")
AUTO, NVIDIA, AMD = VENDORS
INDEX = re.compile(r"^(0|[1-9][0-9]*)$")   # a card index; host/gpu-broker-gpu checks GPU_INDEX the same way
POWER_DECIMALS = 1
PROCS_UNREADABLE = "!procs-unreadable"
UNREADABLE_HINT = "per-process memory needs root or CAP_SYS_PTRACE"
UNKNOWN = re.compile(r"^\[.*\]$")          # nvidia-smi's "[N/A]", "[Not Supported]", "[Unknown Error]"...
NvidiaState = Literal["absent", "ok", "failing"]


def check_vendor(vendor: str) -> str:
    if vendor not in VENDORS:
        raise ValueError(f"gpu.vendor must be one of {', '.join(VENDORS)}, not {vendor!r}")
    return vendor


def check_index(raw: object) -> int:
    if not INDEX.fullmatch(str(raw)):   # str(True) is "True", str(1.0) is "1.0": refused too
        raise ValueError(f"gpu.index must be a card number (0, 1, ...), not {raw!r}")
    return int(str(raw))


def opt(text: str) -> float | None:
    """One optional field: empty, or any bracketed vendor marker, is unknown (None)."""
    text = text.strip()
    return None if not text or UNKNOWN.match(text) else float(text)


def fmt(v: float | None) -> str:
    return "" if v is None else str(v)


@dataclass(frozen=True)
class GpuSample:
    used_mib: int
    total_mib: int
    util_pct: int | None = None
    power_w: float | None = None
    temp_c: int | None = None
    clock_mhz: int | None = None


@dataclass(frozen=True)
class Procs:
    rows: list[tuple[str, int]] = field(default_factory=list)   # (pid, MiB)
    unreadable: bool = False   # some processes could not be inspected: `rows` may be missing some


class GpuProbe(Protocol):
    name: str   # what was chosen, for `gpu-broker check` and the dashboard

    def read(self) -> GpuSample:
        """The card's current reading; raises if its memory cannot be read."""

    def procs(self) -> Procs:
        """(pid, MiB) per process using the card's memory, as far as this namespace can see."""


def line(s: GpuSample, groups: list[tuple[str, int]], unreadable: bool = False) -> str:
    """The wire format; `groups` are (owner group, MiB) pairs."""
    power = None if s.power_w is None else round(s.power_w, POWER_DECIMALS)
    head = ",".join([str(s.used_mib), str(s.total_mib), fmt(s.util_pct), fmt(power), fmt(s.temp_c), fmt(s.clock_mhz)])
    tail = [f"{g}:{mib}" for g, mib in groups] + ([PROCS_UNREADABLE] if unreadable else [])
    return f"{head}|{' '.join(tail)}"


from .auto import Auto as Auto  # noqa: E402 — auto.py builds on the names above
