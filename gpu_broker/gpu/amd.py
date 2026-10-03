"""AMD: the amdgpu driver's sysfs files, readable without ROCm tools and inside any container
that sees /sys (Documentation/gpu/amdgpu/thermal.rst and driver-misc.rst in the kernel tree).

Card: the `index`-th of /sys/class/drm/card<N> (by N) whose device has vendor 0x1002 and
`mem_info_vram_total` (an amdgpu card). Memory: mem_info_vram_used / _total (bytes), the only
required values. Utilisation: gpu_busy_percent. From device/hwmon/hwmon*/: power1_average,
else power1_input (µW; RDNA3 has only the latter), temp1_input (m°C, the edge sensor),
freq1_input (Hz, sclk). Utilisation and hwmon files may be unreadable while the card is
runtime-suspended (BACO): an unknown value, never an error.

Processes: the fds in /proc/<pid>/fd that link into /dev/dri/ (no other fd is opened), and for
each the fdinfo of an amdgpu DRM file on this card (drm-driver, drm-pdev;
Documentation/gpu/drm-usage-stats.rst), with its VRAM from drm-memory-vram (else
drm-resident-vram). A client has one drm-client-id however many fds (or processes) share it,
and the id need only be unique per device, so each (drm-pdev, client) is counted once.
Another user's fds are readable only with root or CAP_SYS_PTRACE: when any read is refused
the result says so (`Procs.unreadable`) rather than looking empty. Without the host PID
namespace only this container's own processes are visible; the dashboard then shows the
total only.
"""
from __future__ import annotations

import os
import pathlib
import re

from . import GpuSample, Procs

SYS_ROOT, PROC_ROOT = "/sys", "/proc"
DRM_DIR = "class/drm"
CARD = re.compile(r"^card(\d+)$")
AMD_VENDOR = "0x1002"
VRAM_USED, VRAM_TOTAL, BUSY = "mem_info_vram_used", "mem_info_vram_total", "gpu_busy_percent"
POWER_FILES = ("power1_average", "power1_input")   # µW, in order of preference
TEMP_FILE, CLOCK_FILE = "temp1_input", "freq1_input"
SLOT_KEY = "PCI_SLOT_NAME="                         # in device/uevent: the card's PCI address
MIB, KIB = 1024 * 1024, 1024
MICRO, MILLI, MEGA = 1_000_000, 1000, 1_000_000
DRIVER, PDEV, CLIENT = "drm-driver", "drm-pdev", "drm-client-id"
VRAM_KEYS = ("drm-memory-vram", "drm-resident-vram")   # older and newer amdgpu name it differently
UNITS = {"": 1, "KiB": KIB, "MiB": MIB}               # drm-usage-stats: bytes, or KiB / MiB
AMDGPU = "amdgpu"
DRI = "/dev/dri/"                                  # DRM device nodes: fds linking here


def _int(p: pathlib.Path) -> int | None:
    try:
        return int(p.read_text().strip())
    except (OSError, ValueError):
        return None


def cards(sys_root: str = SYS_ROOT) -> list[pathlib.Path]:
    """amdgpu card device directories, by card number."""
    drm = pathlib.Path(sys_root, DRM_DIR)
    try:
        names = sorted((int(m.group(1)), e.name) for e in drm.iterdir() if (m := CARD.match(e.name)))
    except OSError:
        return []
    out = []
    for _, name in names:
        dev = drm / name / "device"
        try:
            vendor = (dev / "vendor").read_text().strip()
        except OSError:
            continue
        if vendor == AMD_VENDOR and (dev / VRAM_TOTAL).is_file():
            out.append(dev)
    return out


def present(sys_root: str = SYS_ROOT) -> bool:
    return bool(cards(sys_root))


class AmdProbe:
    def __init__(self, index: int = 0, sys_root: str = SYS_ROOT, proc_root: str = PROC_ROOT) -> None:
        found = cards(sys_root)
        if index >= len(found):
            raise RuntimeError(f"gpu.index {index}: {len(found)} amdgpu card(s) under {sys_root}/{DRM_DIR}")
        self.dev, self.proc_root = found[index], proc_root
        self.pdev = self._slot()
        self.name = f"amd (amdgpu sysfs, {self.dev.parent.name}, {self.pdev or 'no PCI address'})"

    def _slot(self) -> str:
        try:
            lines = (self.dev / "uevent").read_text().splitlines()
        except OSError:
            return ""
        return next((ln[len(SLOT_KEY):] for ln in lines if ln.startswith(SLOT_KEY)), "")

    def _hwmon(self, *names: str) -> int | None:
        for d in sorted((self.dev / "hwmon").glob("hwmon*")):
            for n in names:
                if (v := _int(d / n)) is not None:
                    return v
        return None

    def read(self) -> GpuSample:
        used, total, busy = (_int(self.dev / f) for f in (VRAM_USED, VRAM_TOTAL, BUSY))
        if used is None or total is None:
            raise RuntimeError(f"cannot read {VRAM_USED} or {VRAM_TOTAL} under {self.dev}")
        power, temp, clock = self._hwmon(*POWER_FILES), self._hwmon(TEMP_FILE), self._hwmon(CLOCK_FILE)
        return GpuSample(used // MIB, total // MIB, busy, None if power is None else power / MICRO,
                         None if temp is None else temp // MILLI, None if clock is None else clock // MEGA)

    def procs(self) -> Procs:
        per_pid: dict[str, int] = {}
        seen: set[str] = set()
        unreadable = False
        for pid in sorted((p for p in os.listdir(self.proc_root) if p.isdigit()), key=int):
            try:
                fds = _drm_fds(pathlib.Path(self.proc_root, pid))
                infos = [_fields(pathlib.Path(self.proc_root, pid, "fdinfo", fd)) for fd in fds]
            except PermissionError:
                unreadable = True
                continue
            for f in infos:
                if f.get(DRIVER) != AMDGPU or (self.pdev and f.get(PDEV) != self.pdev):
                    continue
                client, size = f.get(CLIENT, ""), _bytes(f)
                key = f"{f.get(PDEV, '')}/{client}"
                if size is None or key in seen:
                    continue
                if client:
                    seen.add(key)
                per_pid[pid] = per_pid.get(pid, 0) + size
        return Procs([(pid, b // MIB) for pid, b in per_pid.items()], unreadable)


def _drm_fds(proc: pathlib.Path) -> list[str]:
    """The process's fds that link into /dev/dri/, in numeric order; [] if it is gone.
    PermissionError when its fds may not be read."""
    out = []
    try:
        names = os.listdir(proc / "fd")
    except PermissionError:
        raise
    except OSError:
        return []
    for fd in sorted((n for n in names if n.isdigit()), key=int):
        try:
            target = os.readlink(proc / "fd" / fd)
        except PermissionError:
            raise
        except OSError:
            continue
        if target.startswith(DRI):
            out.append(fd)
    return out


def _fields(p: pathlib.Path) -> dict[str, str]:
    try:
        text = p.read_text()
    except PermissionError:
        raise
    except OSError:
        return {}
    return {k.strip(): v.strip() for k, sep, v in (ln.partition(":") for ln in text.splitlines()) if sep}


def _bytes(f: dict[str, str]) -> int | None:
    raw = next((f[k] for k in VRAM_KEYS if k in f), None)
    if raw is None:
        return None
    num, _, unit = raw.partition(" ")
    if not num.isdigit() or unit.strip() not in UNITS:
        return None
    return int(num) * UNITS[unit.strip()]
