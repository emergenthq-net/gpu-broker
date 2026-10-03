"""host/gpu-broker-gpu on NVIDIA and vendor auto: bracketed values, every nvidia-smi call under
`timeout`, a failing nvidia-smi never falls back to AMD, the choice cached until the conf
changes; and gpu-broker-ctl's GPU verbs, with the old NVIDIA-only path when the helper is not
installed yet."""
import os
import re

import pytest

from gpu_broker import gpu as gpu_mod
from tests.gpufake import CTL, HELPER, gpu  # noqa: F401 — the fixture

OLD_PARSE = re.compile(r"^(\d+(\.\d+)?,){5}\d+(\.\d+)?\|")   # what metrics.parse_sample accepted before AMD support


def test_auto_takes_a_working_nvidia_smi_and_reads_brackets_as_unknown(gpu):
    assert gpu("read", nv="ok").stdout == "8000,24564,37\n"
    assert gpu.stream(nv="ok")[0] == ["8000,24564,37,,50,2500| host:7000\n"]
    row = "8000, 24564, [Unknown Error], [Not Supported], [Insufficient Permissions], [N/A]"
    assert gpu.stream(nv="ok", SMI_ROW=row)[0] == ["8000,24564,,,,| host:7000\n"]
    assert gpu("read", GPU_VENDOR="amd", nv="ok").stdout == "4096,16384,37\n"


def test_nvidia_without_memory_is_an_error(gpu):
    r = gpu("read", nv="ok", SMI_ROW="[N/A], 24564, 37")
    assert r.returncode == 6 and r.stdout == "" and "no GPU memory used or total" in r.stderr
    lines, rc, err = gpu.stream(nv="ok", SMI_ROW="8000, [N/A], 37, 1, 1, 1", between=lambda: None)
    assert lines == [] and rc == 6 and "no GPU memory used or total" in err


def test_every_nvidia_smi_call_is_bounded(gpu):
    gpu.stream(nv="ok", NV_TIMEOUT_S="7")
    calls = gpu.timeout_calls()
    assert len(calls) >= 3 and all(c.startswith("-k 1 7 nvidia-smi ") for c in calls)
    assert len(gpu.smi_calls()) == len(calls)


def test_a_failing_nvidia_smi_is_retried_then_an_error_never_amd(gpu):
    r = gpu("read", nv="down", GPU_BUDGET_S="4")
    assert r.returncode == 6 and r.stdout == ""
    assert "nvidia-smi is installed but did not answer within 4 s" in r.stderr and "GPU_VENDOR=amd" in r.stderr
    assert len(gpu.smi_calls()) >= 2          # retried within GPU_BUDGET_S
    assert gpu("read", nv="down", GPU_VENDOR="amd").stdout == "4096,16384,37\n"


def test_no_gpu_at_all(gpu, tmp_path):
    r = gpu("read", SYSFS_ROOT=tmp_path / "none")
    assert r.returncode == 6 and "no GPU found: nvidia-smi is not installed" in r.stderr and "amdgpu card" in r.stderr


def test_both_vendors_present_auto_says_which_it_chose(gpu):
    r = gpu("read", nv="ok")
    assert r.stdout == "8000,24564,37\n" and "auto chose nvidia" in r.stderr and "GPU_VENDOR=amd" in r.stderr


def test_auto_is_decided_once_until_the_conf_changes(gpu, tmp_path):
    conf, cache = tmp_path / "ctl.conf", tmp_path / "cache"
    conf.write_text("")
    os.utime(conf, (1, 1))
    env = {"nv": "ok", "GPU_CACHE_DIR": cache, "GPU_CONF": conf}
    gpu("read", **env)
    assert (cache / "vendor").read_text() == "auto 0 nvidia\n"
    first = gpu.smi_calls().count("-L")
    gpu("read", **env)
    assert gpu.smi_calls().count("-L") == first              # cached: no new probe
    conf.write_text("GPU_INDEX=0\n")                          # the conf changed: decide again
    assert gpu("read", **{**env, "nv": "absent"}).stdout == "4096,16384,37\n"
    assert (cache / "vendor").read_text() == "auto 0 amd\n"
    (cache / "vendor").write_text("auto 1 nvidia\n")          # another card's choice is not reused
    os.utime(conf, (1, 1))
    assert gpu("read", **{**env, "nv": "absent"}).stdout == "4096,16384,37\n"


def test_a_failure_mid_stream_ends_it_with_an_error(gpu):
    lines, rc, err = gpu.stream(nv="ok", SAMPLE_S="0.3", SMI_FAIL_AFTER="4", between=lambda: None)
    assert lines == ["8000,24564,37,,50,2500| host:7000\n"] and rc == 6 and "failed or did not answer" in err


@pytest.fixture
def ctl(gpu, tmp_path):
    conf = tmp_path / "ctl.conf"

    def run(verb, helper=HELPER, stream=False, **conf_vars):
        conf.write_text(f"LOG={tmp_path}/ctl.log\nLOCKS={tmp_path}/locks\nGPU_HELPER={helper}\n"
                        + "".join(f"{k}={v}\n" for k, v in conf_vars.items()))
        env = {"SSH_ORIGINAL_COMMAND": verb, "GPU_BROKER_CTL_CONF": conf, "nv": conf_vars.pop("nv", "ok")}
        return gpu.stream(script=CTL, **env)[0] if stream else gpu("", script=CTL, **env)
    return run


def test_ctl_verbs_hand_their_conf_to_the_helper(ctl, tmp_path):
    assert ctl("gpu", GPU_VENDOR="amd").stdout == "4096,16384,37\n"
    assert ctl("gpustream", stream=True, GPU_VENDOR="amd")[0].startswith("4096,16384,37,42.0,51,1500|")
    assert ctl("gpu").stdout == "8000,24564,37\n"
    assert (tmp_path / "locks/vendor").read_text() == "auto 0 nvidia\n"   # the ctl's lock dir holds the cache


def test_ctl_without_the_helper_uses_the_old_nvidia_path(ctl, tmp_path):
    missing = tmp_path / "not-installed"
    assert ctl("gpu", helper=missing).stdout == "8000, 24564, 37\n"
    (line,) = ctl("gpustream", helper=missing, stream=True)
    assert line == "8000,24564,37,[N/A],50,2500| host:7000\n"


def test_old_and_new_lines_cross_parse_on_nvidia():
    """Old host script -> new broker: "[N/A]" is unknown, nothing dropped. New host script ->
    old broker: on a card that reports all six values, the line is what the old parser took."""
    from gpu_broker.metrics import parse_sample
    old = parse_sample("8000,24564,37,[N/A],50,2500| host:7000", 0)
    assert old and (old["power_w"], old["clock_mhz"], old["sm_mhz"], old["by_group"]) == (None, 2500, 2500, {"host": 7000})
    new = gpu_mod.line(gpu_mod.GpuSample(8000, 24564, 37, 287.04, 64, 2520), [("101", 7000)])
    assert OLD_PARSE.match(new) and new == "8000,24564,37,287.0,64,2520|101:7000"
