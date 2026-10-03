"""drivers.run_group: a recipe that outlives its timeout is stopped with every process it
started (its whole group), first with SIGTERM, then SIGKILL. Runs real child processes."""
import os
import subprocess
import sys
import time

import pytest

from gpu_broker import drivers

# A parent that starts a grandchild (same group) and prints its pid; both ignore SIGTERM.
PROG = """
import os, signal, subprocess, sys, time
signal.signal(signal.SIGTERM, signal.SIG_IGN)
child = subprocess.Popen([sys.executable, "-c", "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)"])
print(child.pid, flush=True)
time.sleep(60)
"""


def group_alive(pgid):
    try:
        os.killpg(pgid, 0)
    except (ProcessLookupError, PermissionError):   # macOS answers EPERM for a group of zombies
        return False
    return True


def test_the_whole_group_is_killed_at_the_timeout(tmp_path, monkeypatch):
    pids = []
    real_popen = subprocess.Popen

    def popen(cmd, **kw):
        p = real_popen(cmd, **kw)
        pids.append(p.pid)
        return p
    monkeypatch.setattr(subprocess, "Popen", popen)
    t0 = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired):
        drivers.run_group([sys.executable, "-c", PROG], timeout=1.0, kill_after=0.5)
    assert time.monotonic() - t0 < 5
    deadline = time.monotonic() + 5   # the grandchild is reparented to init and reaped there
    while time.monotonic() < deadline and group_alive(pids[0]):
        time.sleep(0.05)
    assert not group_alive(pids[0])

def test_a_prompt_program_returns_its_output():
    r = drivers.run_group([sys.executable, "-c", "print('ok')"], timeout=10)
    assert (r.returncode, r.stdout) == (0, "ok\n")


def test_run_input_streams_an_open_file_from_its_descriptor(tmp_path, monkeypatch):
    """The file reaches the command as its stdin; Python never holds its bytes."""
    big = tmp_path / "in.bin"
    big.write_bytes(os.urandom(1 << 20))
    seen = []
    real = subprocess.run
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: seen.append(k) or real(*a, **k))
    with big.open("rb") as f:
        r = drivers.run_input([sys.executable, "-c", "import sys; print(len(sys.stdin.buffer.read()))"], f, timeout=20)
    assert r.stdout.strip() == b"1048576"
    assert "input" not in seen[0] and seen[0]["stdin"].name == str(big)
    assert drivers.run_input([sys.executable, "-c", "import sys; print(sys.stdin.read())"], b"hi", 20).stdout == b"hi\n"


# A program whose worker leaves the group (a new session) but keeps stdout open, and writes its pid.
ESCAPE = """
import subprocess, sys, time
w = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
open(sys.argv[1], "w").write(str(w.pid))
time.sleep(60)
"""


def test_a_worker_that_left_the_group_and_holds_the_pipes_is_gpu_held(tmp_path):
    pidfile = tmp_path / "worker.pid"
    t0 = time.monotonic()
    try:
        with pytest.raises(drivers.GpuHeld, match="output pipes stay open"):
            drivers.run_group([sys.executable, "-c", ESCAPE, str(pidfile)], timeout=1.0, kill_after=0.2, drain_s=0.5)
        assert time.monotonic() - t0 < 5   # bounded: never waits for the worker's EOF
    finally:
        if pidfile.exists():
            os.kill(int(pidfile.read_text()), 9)


def test_a_tagged_run_carries_the_job_id_and_is_reaped(tmp_path):
    fake_proc = tmp_path / "proc"
    fake_proc.mkdir()
    r = drivers.run_group([sys.executable, "-c", "import os; print(os.environ['GPU_BROKER_JOB'])"], timeout=10,
                          tag="ab12cd34ef56", proc=str(fake_proc))
    assert r.stdout == "ab12cd34ef56\n"
    with pytest.raises(drivers.GpuHeld, match="survive their recipe"):   # /proc unreadable: never "gone"
        drivers.run_group([sys.executable, "-c", "pass"], timeout=10, tag="ab12cd34ef56", proc=str(tmp_path / "none"),
                          reap_s=0)


def test_open_pipes_hold_the_gpu_even_when_the_tag_scan_finds_nothing(tmp_path, monkeypatch):
    """Either signal is enough: a worker that dropped the tag (cleared its environment) but
    holds the pipes is still a worker."""
    pidfile = tmp_path / "worker.pid"
    monkeypatch.setattr(drivers.reap, "reap", lambda *a, **k: True)
    try:
        with pytest.raises(drivers.GpuHeld, match="output pipes stay open"):
            drivers.run_group([sys.executable, "-c", ESCAPE, str(pidfile)], timeout=1.0, kill_after=0.2,
                              drain_s=0.5, tag="ab12cd34ef56")
    finally:
        if pidfile.exists():
            os.kill(int(pidfile.read_text()), 9)


def test_the_scan_checks_the_unreaped_childs_pid_is_visible(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(drivers.reap, "reap", lambda jid, wait_s, proc, alive=None: seen.append(alive) or True)
    drivers.run_group([sys.executable, "-c", "pass"], timeout=10, tag="ab12cd34ef56")
    assert seen == [None]   # waited for: its pid may be reused, so it proves nothing
