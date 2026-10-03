"""Shutdown must not wait on the GPU sample stream: on Proxmox it is a long-lived SSH child that
otherwise outlives the app and holds `systemctl stop` until the unit's stop timeout."""
import dataclasses
import itertools
import subprocess
import sys
import time

import pytest
from fastapi.testclient import TestClient

from gpu_broker import settings
from gpu_broker.broker import Broker
from gpu_broker.drivers.proxmox import ProxmoxDriver
from gpu_broker.web.app import create_app
from tests.helpers import WAIT_S, FakeBackends, FakeDriver, make_settings

# Prints one sample, then blocks like `gpu-broker-ctl gpustream` between samples.
STREAM = [sys.executable, "-c", "import time; print('1,2,3,4,5,6|', flush=True); time.sleep(600)"]


def wait_for(cond, timeout=WAIT_S):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.01)
    return False


@pytest.fixture
def procs():
    started = []
    yield started
    for p in started:  # never leak a sleeping child, even when an assertion failed
        p.kill()
        p.wait()


def proxmox_with_real_stream(procs):
    def popen(argv, **kw):
        assert argv[-1] == "gpustream"
        p = subprocess.Popen(STREAM, **kw)
        procs.append(p)
        return p
    unreachable = lambda cmd, timeout: subprocess.CompletedProcess(cmd, 255, "", "ssh: no route")  # noqa: E731
    return ProxmoxDriver("root@pve", None, settings.Timeouts(), popen=popen, run=unreachable)


def test_app_shutdown_kills_the_stream_process_and_ends_the_sampler(tmp_path, procs):
    driver = proxmox_with_real_stream(procs)
    s = dataclasses.replace(make_settings(tmp_path), gpu_stream=True)
    fake = FakeDriver()
    b = Broker(s, env={}, driver=driver, backends=FakeBackends(fake))
    b.residency.detect = lambda: None
    with TestClient(create_app(b, "t")):
        assert wait_for(lambda: b.sampler.since(0)), "sampler never read the stream"
        assert procs and procs[0].poll() is None
    assert procs[0].wait(timeout=WAIT_S) is not None
    sampler = [t for t in b._threads if t.name == b.sampler.loop.__qualname__]
    assert sampler and not sampler[0].is_alive()
    assert len(procs) == 1, "the sampler reconnected after shutdown"


def test_close_before_the_stream_opens_starts_no_process(procs):
    driver = proxmox_with_real_stream(procs)
    driver.close()
    assert list(driver.gpu_stream()) == []
    assert procs == []


def test_a_stream_that_drops_on_its_own_is_still_an_error():
    class Gone:
        stdout = iter([])

        class stderr:
            @staticmethod
            def read():
                return "connection reset"

        def kill(self):
            pass
    d = ProxmoxDriver("root@pve", None, settings.Timeouts(), popen=lambda *a, **k: Gone())
    try:
        list(d.gpu_stream())
    except RuntimeError as e:
        assert "connection reset" in str(e)
    else:
        raise AssertionError("a dropped stream must raise so the dashboard shows why")


def test_local_stream_ends_on_close(tmp_path):
    from gpu_broker.drivers.systemd import SystemdDriver
    d = SystemdDriver(allowed=None, timeouts=settings.Timeouts(), sample_s=0, models_root=str(tmp_path),
                      run=lambda *a, **k: None, sleep=lambda _: None)
    d.sample_line = lambda: "x"
    it = d.gpu_stream()
    assert next(it) == "x"
    d.close()
    assert list(itertools.islice(it, 3)) == []


def test_serve_caps_uvicorns_graceful_shutdown(monkeypatch, tmp_path):
    import uvicorn

    from gpu_broker import cli
    seen = {}
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: seen.update(kw))
    monkeypatch.setattr("gpu_broker.broker.Broker", lambda *a, **k: FakeDriver())
    monkeypatch.setattr("gpu_broker.web.app.create_app", lambda *a, **k: None)
    cfg = tmp_path / "c.yaml"
    cfg.write_text("server: {graceful_shutdown_s: 7}\n")
    assert cli.main(["--config", str(cfg), "serve"], {"BROKER_TOKEN": "t"}) == 0
    assert seen["timeout_graceful_shutdown"] == 7
    assert settings.Server().graceful_shutdown_s == 10


def test_shipped_unit_stops_a_little_after_the_graceful_cap():
    import pathlib
    import re
    unit = (pathlib.Path(__file__).parents[1] / "examples/systemd/gpu-broker.service").read_text()
    stop = int(re.search(r"^TimeoutStopSec=(\d+)$", unit, re.M).group(1))
    assert settings.Server().graceful_shutdown_s < stop <= settings.Server().graceful_shutdown_s + 10
