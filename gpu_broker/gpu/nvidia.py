"""NVIDIA: nvidia-smi, one card selected by index (`-i`). Any bracketed value nvidia-smi prints
("[N/A]", "[Not Supported]", "[Unknown Error]", "[Insufficient Permissions]") is unknown, not
an error; only used and total memory are required."""
from __future__ import annotations

from collections.abc import Callable
from subprocess import CompletedProcess, SubprocessError

from . import GpuSample, NvidiaState, Procs, opt

NVIDIA_SMI = "nvidia-smi"
READ_QUERY = "memory.used,memory.total,utilization.gpu,power.draw,temperature.gpu,clocks.sm"
APPS_QUERY = "pid,used_memory"
CSV = "--format=csv,noheader,nounits"
LIST = "-L"                     # lists the cards; exit 0 means the driver answers

Run = Callable[..., CompletedProcess[str]]


def _int(v: float | None) -> int | None:
    return None if v is None else int(v)


class NvidiaProbe:
    def __init__(self, run: Run, timeout_s: float, index: int = 0, smi: str = NVIDIA_SMI) -> None:
        self.run, self.timeout_s, self.index, self.smi = run, timeout_s, index, smi
        self.name = f"nvidia ({smi}, card {index})"

    def _query(self, query: str) -> str:
        r = self.run([self.smi, "-i", str(self.index), query, CSV], timeout=self.timeout_s)
        if r.returncode != 0:
            raise RuntimeError(f"{self.smi} failed ({r.returncode}): {r.stderr.strip() or r.stdout.strip()}")
        return str(r.stdout)

    def read(self) -> GpuSample:
        used, total, util, power, temp, clock = (opt(x) for x in self._query(f"--query-gpu={READ_QUERY}").strip()
                                                 .splitlines()[0].split(","))
        if used is None or total is None:
            raise RuntimeError(f"{self.smi} reported no memory used or total")
        return GpuSample(int(used), int(total), _int(util), power, _int(temp), _int(clock))

    def procs(self) -> Procs:
        out = []
        for row in self._query(f"--query-compute-apps={APPS_QUERY}").splitlines():
            pid, _, mib = row.replace(" ", "").partition(",")
            if pid.isdigit() and mib.isdigit():
                out.append((pid, int(mib)))
        return Procs(out)


def state(run: Run, timeout_s: float, smi: str = NVIDIA_SMI) -> tuple[NvidiaState, str]:
    """absent (not installed), ok (lists cards), or failing (installed; errors or hangs), and why."""
    try:
        r = run([smi, LIST], timeout=timeout_s)
    except FileNotFoundError:
        return "absent", f"{smi} not found"
    except (OSError, SubprocessError) as e:
        return "failing", f"{smi} {LIST}: {e}"
    if r.returncode == 0:
        return "ok", ""
    return "failing", f"{smi} {LIST} exited {r.returncode}: {(r.stderr or r.stdout).strip()[:200]}"
