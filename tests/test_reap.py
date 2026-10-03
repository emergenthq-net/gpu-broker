"""drivers/reap.py: a job's processes are found by the tag in their environment (never by
command line or group), killed by pid and by group (never group 1 or the broker's own), and
"gone" is only ever the answer of a scan that could read /proc."""
import os
import signal

from gpu_broker.drivers import reap
from tests.ctlfake import proc

JID, OTHER = "ab12cd34ef56", "0123456789ab"


class Kernel:
    """kill/killpg over a fake /proc: a kill removes the entry unless the pid is unkillable."""

    def __init__(self, root, unkillable=()):
        self.root, self.unkillable, self.calls = root, set(unkillable), []

    def _die(self, pid):
        d = self.root / str(pid)
        if pid not in self.unkillable and d.exists():
            for f in d.iterdir():
                f.unlink()
            d.rmdir()

    def kill(self, pid, sig):
        self.calls.append(("kill", pid, sig))
        self._die(pid)

    def killpg(self, pgid, sig):
        self.calls.append(("killpg", pgid, sig))
        for d in list(self.root.iterdir()):
            if reap._group(int(d.name), str(self.root)) == pgid:
                self._die(int(d.name))


def setup(tmp_path, procs, unkillable=()):
    root = tmp_path / "proc"
    proc(root, 1, 1)
    for p in procs:
        proc(root, *p)
    return root, Kernel(root, unkillable)


def test_tagged_finds_the_job_by_environment_only(tmp_path):
    root, _ = setup(tmp_path, [(10, 10, JID), (11, 10, None), (12, 12, OTHER), (13, 13, JID),
                               (15, 15, JID + "ff")])   # a longer id that starts with this one is another job
    (root / "14").mkdir()   # exited meanwhile: no environ
    assert sorted(reap.tagged(JID, str(root))) == [10, 13]
    assert reap.tagged(JID, str(tmp_path / "missing")) is None


def test_reap_kills_pid_and_group_until_none_is_left(tmp_path):
    root, k = setup(tmp_path, [(10, 10, JID), (11, 10, None), (13, 1, JID), (20, 20, OTHER)])
    assert reap.reap(JID, 5, str(root), k.kill, k.killpg, sleep=lambda s: None)
    assert sorted(int(d.name) for d in root.iterdir()) == [1, 20]   # 11 went with its group
    assert ("killpg", 1, signal.SIGKILL) not in k.calls and ("kill", 13, signal.SIGKILL) in k.calls


def test_reap_never_signals_the_brokers_own_group(tmp_path):
    root, k = setup(tmp_path, [(10, os.getpgrp(), JID)])
    assert reap.reap(JID, 5, str(root), k.kill, k.killpg, sleep=lambda s: None)
    assert [c[0] for c in k.calls] == ["kill"]


def test_survivors_or_an_unreadable_proc_are_never_gone(tmp_path):
    root, k = setup(tmp_path, [(10, 10, JID)], unkillable=[10])
    ticks = iter(range(100))
    assert not reap.reap(JID, 3, str(root), k.kill, k.killpg, clock=lambda: next(ticks), sleep=lambda s: None)
    assert len([c for c in k.calls if c[0] == "kill"]) == 2   # bounded by the clock, not a try count
    assert not reap.reap(JID, 3, str(tmp_path / "missing"), k.kill, k.killpg, sleep=lambda s: None)


def test_a_process_that_exits_mid_scan_is_gone_but_an_unreadable_one_of_ours_is_unknown(tmp_path):
    root = tmp_path / "proc"
    proc(root, 1, 1)
    (root / "702").mkdir()                       # listed, then gone before its environ is read
    assert reap.tagged(JID, str(root)) == []
    proc(root, 703, 703, JID)
    (root / "703" / "environ").chmod(0)          # ours, but not readable (setuid, non-dumpable)
    try:
        assert reap.tagged(JID, str(root)) is None
    finally:
        (root / "703" / "environ").chmod(0o600)


def test_another_users_unreadable_process_is_not_ours(tmp_path, monkeypatch):
    root = tmp_path / "proc"
    proc(root, 1, 1)
    proc(root, 704, 704, OTHER)
    (root / "704" / "environ").chmod(0)
    monkeypatch.setattr(reap, "_owner", lambda entry: os.getuid() + 1)
    try:
        assert reap.tagged(JID, str(root)) == []
    finally:
        (root / "704" / "environ").chmod(0o600)


def test_a_proc_that_hides_a_pid_known_alive_cannot_say_gone(tmp_path):
    """hidepid: other processes vanish from the listing, so an empty scan would mean nothing."""
    root = tmp_path / "proc"
    proc(root, 1, 1)
    assert reap.tagged(JID, str(root), alive=4242) is None
    assert not reap.reap(JID, 0, str(root), alive=4242)
    proc(root, 4242, 4242)
    assert reap.tagged(JID, str(root), alive=4242) == []
