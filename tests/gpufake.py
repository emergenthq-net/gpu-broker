"""Runs host/gpu-broker-gpu (and gpu-broker-ctl's GPU verbs) for real against the amdgpu fixture
trees, with fakes on PATH: `nvidia-smi` (answers, fails, hangs or is absent, and can start failing
after N calls) and `timeout` (logs its arguments, then runs the command)."""
import os
import subprocess
import time

import pytest

from tests.helpers import FIX, ROOT

HELPER = ROOT / "host/gpu-broker-gpu"
CTL = ROOT / "host/gpu-broker-ctl"
AMD = FIX / "amdgpu"
FAKE_SMI = r"""#!/bin/bash
echo "$*" >> "$SMI_LOG"
n=$(wc -l < "$SMI_LOG")
[[ -n "${SMI_DOWN:-}" ]] && exit 9
[[ -n "${SMI_L_TAKES:-}" && "$*" == -L ]] && sleep "$SMI_L_TAKES"
[[ -n "${SMI_HANG:-}" && "$*" != -L ]] && exec sleep 60
[[ -n "${SMI_FAIL_AFTER:-}" ]] && (( n > SMI_FAIL_AFTER )) && { echo "GPU is lost" >&2; exit 15; }
case "$*" in
  -L) echo "GPU 0: NVIDIA GeForce RTX 4090" ;;
  *compute-apps*) echo "4242, 7000" ;;
  *power.draw*) echo "${SMI_ROW:-8000, 24564, 37, [N/A], 50, 2500}" ;;
  *) echo "${SMI_ROW:-8000, 24564, 37}" ;;
esac
"""
FAKE_TIMEOUT = '#!/bin/bash\necho "$*" >> "$TIMEOUT_LOG"\n[[ $1 == -k ]] && shift 2\nshift\nexec "$@"\n'


class Gpu:
    def __init__(self, tmp_path):
        self.tmp, self.bin = tmp_path, tmp_path / "bin"
        self.bin.mkdir()
        self.smi_log, self.timeout_log = tmp_path / "smi.log", tmp_path / "timeout.log"
        for name, body in (("nvidia-smi", FAKE_SMI), ("timeout", FAKE_TIMEOUT)):
            (self.bin / name).write_text(body)
            (self.bin / name).chmod(0o755)

    def env(self, card="vega10", nv="absent", **env):
        smi, off = self.bin / "nvidia-smi", self.tmp / "nvidia-smi.absent"
        if nv == "absent" and smi.exists():
            smi.rename(off)
        elif nv != "absent" and off.exists():
            off.rename(smi)
        self.smi_log.touch()
        return {"PATH": f"{self.bin}:/usr/bin:/bin", "SYSFS_ROOT": str(AMD / card / "sys"),
                "PROC_ROOT": str(AMD / card / "proc"), "SAMPLE_S": "0", "NV_TIMEOUT_S": "0",
                "SMI_LOG": str(self.smi_log), "TIMEOUT_LOG": str(self.timeout_log),
                **({"SMI_DOWN": "1"} if nv == "down" else {}), **{k: str(v) for k, v in env.items()}}

    def __call__(self, verb, card="vega10", nv="absent", script=HELPER, **env):
        args = ["bash", str(script), *([verb] if script == HELPER else [])]
        return subprocess.run(args, env=self.env(card, nv, **env), capture_output=True, text=True, timeout=30)

    def stream(self, card="vega10", nv="absent", lines=1, between=None, script=HELPER, **env):
        """The first `lines` lines (calling `between()` after the first), then the exit code if
        it ended by itself within a few seconds, else None (killed)."""
        args = ["bash", str(script), *(["stream"] if script == HELPER else [])]
        with subprocess.Popen(args, env=self.env(card, nv, **env), stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, text=True) as p:
            assert p.stdout is not None and p.stderr is not None
            out = [p.stdout.readline()]
            if between:
                between()
            out += [p.stdout.readline() for _ in range(lines - 1)]
            end = time.monotonic() + 5
            while p.poll() is None and time.monotonic() < end and between:
                time.sleep(0.05)
            rc = p.poll()
            if rc is None:
                p.kill()
            return [ln for ln in out if ln], rc, p.stderr.read()

    def smi_calls(self):
        return self.smi_log.read_text().splitlines()

    def timeout_calls(self):
        return self.timeout_log.read_text().splitlines() if self.timeout_log.exists() else []


@pytest.fixture
def gpu(tmp_path):
    return Gpu(tmp_path)


def no_root():
    return pytest.mark.skipif(os.geteuid() == 0, reason="root reads every /proc/<pid>/fd")
