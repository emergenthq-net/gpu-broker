"""Local drivers reading the GPU: a failing nvidia-smi is an error (never a fall-back to AMD),
and processes whose fds could not be read are flagged in the sample line."""
import os

import pytest

from tests.helpers import amdgpu_fixture
from tests.test_drivers import DockerDriver, Rec, T, local


def test_local_drivers_flag_processes_they_could_not_read(tmp_path, monkeypatch):
    from gpu_broker.gpu import PROCS_UNREADABLE
    from gpu_broker.gpu import amd as amd_mod
    amd, real = amdgpu_fixture() / "rdna3", os.listdir

    def listdir(p):
        if str(p).endswith("2002/fd"):
            raise PermissionError(13, "Permission denied", str(p))
        return real(p)
    monkeypatch.setattr(amd_mod.os, "listdir", listdir)
    d = local(DockerDriver, tmp_path, Rec(missing={"nvidia-smi"}), sys_root=str(amd / "sys"), proc_root=str(amd / "proc"))
    assert d.sample_line() == f"8192,24576,99,287.0,64,|llama-server:6144 {PROCS_UNREADABLE}"


def test_local_drivers_never_read_amd_when_nvidia_smi_is_installed_but_failing(tmp_path):
    amd, slept = amdgpu_fixture() / "rdna3", []
    rec = Rec(rc=9)
    d = local(DockerDriver, tmp_path, rec, sys_root=str(amd / "sys"), proc_root=str(amd / "proc"))
    now = [0.0]
    d.probe.clock, d.probe.sleep = (lambda: now[0]), (lambda s: slept.append(s) or now.__setitem__(0, now[0] + s))
    with pytest.raises(RuntimeError, match=f"did not answer within {T.gpu_query_s:g} s"):
        d.gpu()
    assert len(slept) == T.gpu_query_s - 1 and all(c == ["nvidia-smi", "-L"] for c in rec.calls)
