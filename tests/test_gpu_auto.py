"""`gpu.vendor: auto` (gpu/auto.py): nvidia-smi, then amdgpu, else a clear error; the whole choice
bounded by wall-clock time, made once by one caller at a time, a failure remembered for a while,
and never waited on by a web request."""
import stat
import threading
import time

import pytest

from gpu_broker import gpu
from gpu_broker.gpu import GpuSample


class Fake:
    def __init__(self, name):
        self.name = name

    def read(self):
        return GpuSample(1, 2, 3)

    def procs(self):
        return gpu.Procs([("7", 1)])

def auto(vendor, nv, amd, built=None, budget_s=3, cost_s=0.0):
    """`nv`: "ok" / "absent" / "failing", or a list of those, one per check. Time is fake: each
    check costs `cost_s` (at most the timeout it is given), each sleep its argument."""
    built = [] if built is None else built
    states = list(nv) if isinstance(nv, list) else None
    checks, slept, given, now = [], [], [], [0.0]

    def nv_state(timeout_s):
        given.append(timeout_s)
        now[0] += min(cost_s, timeout_s)
        st = states.pop(0) if states else nv
        checks.append(st)
        return st, "why" if st != "ok" else ""

    def sleep(s):
        slept.append(s)
        now[0] += s

    def make(n):
        return lambda: built.append(n) or Fake(n)
    a = gpu.Auto(vendor, {gpu.NVIDIA: make(gpu.NVIDIA), gpu.AMD: make(gpu.AMD)}, nv_state, lambda: amd, "/sys",
                 budget_s=budget_s, clock=lambda: now[0], sleep=sleep)
    a.checks, a.slept, a.given, a.now = checks, slept, given, now
    return a


@pytest.mark.parametrize(("vendor", "nv", "amd", "chosen"), [
    ("auto", "ok", True, "nvidia"), ("auto", "absent", True, "amd"), ("nvidia", "absent", True, "nvidia"),
    ("amd", "ok", False, "amd"), ("auto", ["failing", "failing", "ok"], True, "nvidia")])
def test_choice(vendor, nv, amd, chosen):
    assert auto(vendor, nv, amd).name == chosen


def test_a_failing_nvidia_smi_is_retried_within_the_budget_then_raises_and_never_picks_amd():
    a = auto("auto", "failing", True, budget_s=3)
    with pytest.raises(RuntimeError, match=r"nvidia-smi is installed but did not answer within 3 s \(why\).*gpu.vendor: amd"):
        a.read()
    assert a.checks == ["failing"] * 3 and a.slept == [1, 1]   # the first, then once a second within 3 s
    assert a.now[0] <= 3


def test_a_hung_nvidia_smi_is_bounded_by_wall_clock_not_by_tries():
    """Each check hangs until its timeout: the whole choice still ends within the budget, because
    every check is given only what is left of it (not the full budget each time)."""
    a = auto("auto", "failing", True, budget_s=20, cost_s=20)
    with pytest.raises(RuntimeError, match="within 20 s"):
        a.read()
    assert a.now[0] <= 20 and a.given[0] == 20 and len(a.checks) == 1
    b = auto("auto", "failing", True, budget_s=20, cost_s=8)
    with pytest.raises(RuntimeError):
        b.read()
    assert b.now[0] <= 20 and b.given == [20, 11, 2]   # 20, then 20-8-1, then 20-8-1-8-1


def test_a_failed_choice_is_remembered_for_the_budget_then_retried():
    a = auto("auto", "failing", True, budget_s=3)
    with pytest.raises(RuntimeError):
        a.read()
    checks = len(a.checks)
    a.nvidia_state = lambda t: ("absent", "")
    with pytest.raises(RuntimeError, match="did not answer"):
        a.read()                                       # at once: no new check
    assert len(a.checks) == checks and a.state().state == "failed"
    a.now[0] += 3
    assert a.name == "amd"                             # the retry time passed: chosen afresh
    assert a.state() == gpu.auto.ProbeState("ready", "amd")


def test_both_vendors_present_logs_the_choice(caplog):
    with caplog.at_level("INFO", logger="gpu_broker.gpu"):
        auto("auto", "ok", True).read()
        auto("auto", "ok", False).read()
        auto("auto", "absent", True).read()
    msgs = [r.getMessage() for r in caplog.records]
    assert "amdgpu card is also present" in msgs[0] and caplog.records[0].levelname == "WARNING"
    assert msgs[1:] == ["gpu.vendor auto: nvidia (nvidia-smi answers)",
                        "gpu.vendor auto: amd (nvidia-smi is not installed; an amdgpu card is)"]


def test_no_gpu_names_both_and_the_override():
    with pytest.raises(RuntimeError, match=r"nvidia-smi is not installed.*amdgpu card.*/sys/class/drm.*gpu.vendor"):
        auto("auto", "absent", False).read()


def test_concurrent_callers_make_one_choice():
    gate, built = threading.Event(), []

    def nv_state(t):
        gate.wait(5)
        return "ok", ""
    a = gpu.Auto("auto", {gpu.NVIDIA: lambda: built.append(1) or Fake("nvidia")}, nv_state, lambda: False, "/sys",
                 budget_s=3)
    ts = [threading.Thread(target=a.read) for _ in range(8)]
    for t in ts:
        t.start()
    time.sleep(0.05)
    gate.set()
    for t in ts:
        t.join(5)
    assert built == [1]


def test_state_never_blocks_and_chooses_in_the_background():
    gate = threading.Event()

    def nv_state(t):
        gate.wait(5)
        return "ok", ""
    a = gpu.Auto("auto", {gpu.NVIDIA: lambda: Fake("nvidia")}, nv_state, lambda: False, "/sys", budget_s=3)
    t0 = time.monotonic()
    assert a.state().state == "probing" and a.state().state == "probing"
    assert time.monotonic() - t0 < 0.5
    gate.set()
    end = time.monotonic() + 5
    while a.state().state == "probing" and time.monotonic() < end:
        time.sleep(0.01)
    assert a.state() == gpu.auto.ProbeState("ready", "nvidia")


def test_the_choice_is_made_once_and_lazily():
    built = []
    a = auto("auto", "ok", True, built)
    assert built == []                               # constructing reads nothing
    a.read(), a.procs(), a.name
    assert built == ["nvidia"]


def test_a_hung_nvidia_smi_never_stalls_the_gpu_endpoint(tmp_path):
    """The real systemd driver with an nvidia-smi that hangs: /v1/gpu answers "probing" at once
    (the choice runs in the background, never under the endpoint's cache lock), and then the
    error, also at once, for as long as the failed choice is remembered."""
    import dataclasses

    from fastapi.testclient import TestClient

    from gpu_broker import settings
    from gpu_broker.broker import Broker
    from gpu_broker.drivers.systemd import SystemdDriver
    from gpu_broker.web.app import create_app
    from tests.helpers import TOKEN, FakeBackends, FakeDriver, make_settings
    smi = tmp_path / "hung-smi"
    smi.write_text("#!/bin/sh\nexec sleep 30\n")
    smi.chmod(smi.stat().st_mode | stat.S_IEXEC)
    s = make_settings(tmp_path)
    t = dataclasses.replace(s.timeouts, gpu_query_s=2.0)
    driver = SystemdDriver(allowed=frozenset(), timeouts=t, sample_s=1, models_root=str(tmp_path),
                           nvidia_smi=str(smi), gpu=settings.Gpu("auto"), sys_root=str(tmp_path / "sys"))
    fake = FakeDriver()
    b = Broker(dataclasses.replace(s, timeouts=t, gpu_stream=False), env={}, driver=driver,
               backends=FakeBackends(fake))
    h = {"Authorization": f"Bearer {TOKEN}"}
    with TestClient(create_app(b, TOKEN)) as c:
        for _ in range(3):
            t0 = time.monotonic()
            r = c.get("/v1/gpu", headers=h).json()
            assert time.monotonic() - t0 < 1 and r == {"state": "probing"}
        end = time.monotonic() + 10
        while "error" not in r and time.monotonic() < end:
            time.sleep(0.1)
            r = c.get("/v1/gpu", headers=h).json()
        t0 = time.monotonic()
        r = c.get("/v1/gpu", headers=h).json()
        assert time.monotonic() - t0 < 1 and "did not answer within 2 s" in r["error"]
