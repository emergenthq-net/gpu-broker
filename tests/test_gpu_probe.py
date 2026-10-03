"""GPU settings, the nvidia-smi probe, and the sample line with unknown values (the `auto` choice
is in test_gpu_auto.py)."""
import dataclasses
import subprocess
import time

import pytest

from gpu_broker import gpu, settings
from gpu_broker.gpu import GpuSample, nvidia
from gpu_broker.metrics import parse_sample


def test_vendor_and_index_are_checked_in_one_place():
    with pytest.raises(ValueError, match="auto, nvidia, amd"):
        gpu.Auto("intel", {}, lambda t: ("ok", ""), lambda: True, "/sys", budget_s=1)
    with pytest.raises(ValueError, match=r"gpu\.vendor"):
        settings.Gpu(vendor="intel")
    for bad in (-1, True, 1.0, "01", "+1", " 1"):
        with pytest.raises(ValueError, match=r"gpu\.index"):
            settings.Gpu(index=bad)
    assert gpu.check_index("10") == 10 and gpu.check_index(0) == 0


def test_env_overrides_the_gpu_section(tmp_path):
    (tmp_path / "c.yaml").write_text("gpu: {vendor: nvidia, index: 3}\n")
    assert settings.load(str(tmp_path / "c.yaml"), env={}).gpu == settings.Gpu("nvidia", 3)
    s = settings.load(str(tmp_path / "c.yaml"), env={"BROKER_GPU_VENDOR": "amd", "BROKER_GPU_INDEX": "1"})
    assert s.gpu == settings.Gpu("amd", 1)
    for bad in ("01", "1.5", "-1", "x"):
        with pytest.raises(ValueError, match=r"gpu\.index"):
            settings.load(str(tmp_path / "c.yaml"), env={"BROKER_GPU_INDEX": bad})


def test_the_line_leaves_unknown_values_empty_and_parses_back_to_none():
    line = gpu.line(GpuSample(10, 20, None, None, 40, None), [("ct", 5)])
    assert line == "10,20,,,40,|ct:5"
    s = parse_sample(line, 1.0)
    assert (s["util_pct"], s["power_w"], s["temp_c"], s["clock_mhz"], s["by_group"]) == (None, None, 40, None, {"ct": 5})
    assert gpu.line(GpuSample(1, 2, 3, 120.456, 50, 2500), []) == "1,2,3,120.5,50,2500|"
    assert parse_sample("1,2,3,120.456,50,2500|", 0)["power_w"] == 120.5    # one rounding rule, gpu.POWER_DECIMALS
    assert gpu.line(GpuSample(1, 2), [("ct", 5)], unreadable=True) == f"1,2,,,,|ct:5 {gpu.PROCS_UNREADABLE}"


def wait_for(cond, timeout=5):
    end = time.monotonic() + timeout
    while not cond():
        if time.monotonic() > end:
            return False
        time.sleep(0.02)
    return True


def run_answering(out, rc=0):
    calls = []
    def run(cmd, timeout):
        calls.append(cmd)
        key = next(k for k in out if k in " ".join(cmd))
        return subprocess.CompletedProcess(cmd, rc, out[key], "boom")
    return run, calls


@pytest.mark.parametrize("marker", ["[N/A]", "[Not Supported]", "[Unknown Error]", "[Insufficient Permissions]"])
def test_nvidia_probe_reads_any_bracketed_value_as_unknown(marker):
    run, calls = run_answering({"power.draw": f"8000, 24564, {marker}, {marker}, {marker}, {marker}\n",
                                "compute-apps": "123, 7000\nbad\n"})
    p = nvidia.NvidiaProbe(run, 5, index=1)
    assert p.read() == GpuSample(8000, 24564) and p.procs() == gpu.Procs([("123", 7000)])
    assert calls[0][:3] == ["nvidia-smi", "-i", "1"] and "card 1" in p.name


@pytest.mark.parametrize("row", ["[N/A], 24564, 37, 1, 1, 1\n", "8000, [Unknown Error], 37, 1, 1, 1\n"])
def test_nvidia_without_used_or_total_memory_raises(row):
    with pytest.raises(RuntimeError, match="no memory used or total"):
        nvidia.NvidiaProbe(run_answering({"": row})[0], 5).read()


def test_nvidia_failures_raise_and_state_tells_absent_from_failing():
    run, _ = run_answering({"": ""}, rc=9)
    with pytest.raises(RuntimeError, match=r"nvidia-smi failed \(9\): boom"):
        nvidia.NvidiaProbe(run, 5).read()
    assert nvidia.state(run, 5) == ("failing", "nvidia-smi -L exited 9: boom")
    assert nvidia.state(run_answering({"-L": "GPU 0: x"})[0], 5) == ("ok", "")
    def missing(cmd, timeout):
        raise FileNotFoundError(cmd[0])
    def hangs(cmd, timeout):
        raise subprocess.TimeoutExpired(cmd, timeout)
    def denied(cmd, timeout):
        raise PermissionError(cmd[0])
    assert nvidia.state(missing, 5)[0] == "absent"
    assert nvidia.state(hangs, 5)[0] == nvidia.state(denied, 5)[0] == "failing"


def test_the_broker_serves_without_a_gpu_and_shows_why(tmp_path):
    from fastapi.testclient import TestClient

    from gpu_broker.broker import Broker
    from gpu_broker.web.app import create_app
    from tests.helpers import TOKEN, FakeBackends, FakeDriver, make_settings
    d = FakeDriver({"llama-8b"})
    d.no_gpu = True
    s = dataclasses.replace(make_settings(tmp_path), gpu_stream=True)
    b = Broker(s, env={}, driver=d, backends=FakeBackends(d))
    with TestClient(create_app(b, TOKEN)) as c:
        h = {"Authorization": f"Bearer {TOKEN}"}
        assert len(b._threads) == 3                       # started anyway: scheduler, downloads, sampler
        assert c.get("/v1/gpu", headers=h).json() == {"error": "no GPU found"}
        assert c.get("/v1/status", headers=h).status_code == 200
        assert wait_for(lambda: c.get("/v1/metrics", headers=h).json()["gpu_error"] == "no GPU found")
        d.no_gpu, d.probe_name = False, "nvidia (test)"    # the card comes back: the sampler recovers
        assert wait_for(lambda: c.get("/v1/metrics", headers=h).json()["gpu"])
        assert c.get("/v1/gpu", headers=h).json()["probe"] == "nvidia (test)"
