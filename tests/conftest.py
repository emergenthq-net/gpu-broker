"""Shared fixtures. Tests never reach real services: network and host binaries are blocked
for the whole session, and the broker runs on in-memory fakes of its driver and backends.
(A test that once left a job in flight went on to render real videos on a live GPU.)"""
from __future__ import annotations

import os
import subprocess
import urllib.request

import pytest

from gpu_broker.broker import Broker
from tests.helpers import TOKEN, FakeBackends, FakeDriver, make_settings, wait_idle

HOST_BINARIES = {"ssh", "docker", "systemctl", "sudo", "nvidia-smi", "pct", "hf", "git"}


@pytest.fixture(autouse=True, scope="session")
def _no_network():
    def boom(req, *a, **k):
        raise AssertionError(f"network call in test: {getattr(req, 'full_url', req)}")
    real, urllib.request.urlopen = urllib.request.urlopen, boom
    real_open = urllib.request.OpenerDirector.open
    urllib.request.OpenerDirector.open = lambda self, req, *a, **k: boom(req)  # custom openers too
    yield
    urllib.request.urlopen, urllib.request.OpenerDirector.open = real, real_open


@pytest.fixture(autouse=True, scope="session")
def _no_host_commands():
    """Drivers take an injectable `run`; a host binary reaching the real subprocess is a test
    that escaped its fakes."""
    real_run, real_popen = subprocess.run, subprocess.Popen

    def refuse(cmd):
        argv0 = os.path.basename(str(cmd[0] if isinstance(cmd, (list, tuple)) else cmd).split()[0])
        if argv0 in HOST_BINARIES:
            raise AssertionError(f"host command in test: {cmd}")

    def guarded_run(cmd, *a, **k):
        refuse(cmd)
        return real_run(cmd, *a, **k)

    class GuardedPopen(real_popen):   # a class, not a function: code that writes Popen[bytes]
        def __init__(self, cmd, *a, **k):   # (the mcp SDK, at import) or isinstance() still works
            refuse(cmd)
            super().__init__(cmd, *a, **k)
    subprocess.run, subprocess.Popen = guarded_run, GuardedPopen
    yield
    subprocess.run, subprocess.Popen = real_run, real_popen


@pytest.fixture
def broker(tmp_path):
    driver = FakeDriver({"llama-8b"})
    b = Broker(make_settings(tmp_path), env={}, driver=driver, backends=FakeBackends(driver))
    b.start()
    yield b
    assert wait_idle(b), "test left a broker job in flight"
    b.stop()


@pytest.fixture
def client(broker):
    from fastapi.testclient import TestClient

    from gpu_broker.web.app import create_app
    with TestClient(create_app(broker, TOKEN, start=False)) as c:
        c.headers["Authorization"] = f"Bearer {TOKEN}"
        yield c




@pytest.fixture(autouse=True)
def _no_settle(monkeypatch):
    """connect re-reads files a running app may rewrite after a pause; tests need no pause."""
    from gpu_broker.connect import engine
    monkeypatch.setattr(engine, "SETTLE_S", 0)
