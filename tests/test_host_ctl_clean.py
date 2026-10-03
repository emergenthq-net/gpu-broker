"""host/gpu-broker-ctl: exec-run's output folder (only <job id>, in an existing parent, owned
like it) and exit codes, the job reap (by environment tag, pid and process group, bounded in
time, never trusting a failed scan), exec-clean against an exec-run in flight, and exec-info.
Run for real with the fakes of tests/ctlfake.py."""
import time

import pytest

from tests import ctlfake
from tests.test_host_ctl_exec import JID, RECIPE

IN = f"/var/tmp/gpu-broker/{JID}"
OTHER = "0123456789ab"
SCAN = f"scan 101 GPU_BROKER_JOB={JID}"


@pytest.fixture
def ctl(tmp_path):
    return ctlfake.make(tmp_path, RECIPE)


def test_a_missing_output_parent_fails_clearly_and_creates_nothing(ctl):
    r, log, *_ = ctl(f"exec-run t {JID}", no_parent=True)
    assert r.returncode == 8 and b"output folder /srv/outputs/broker is missing in CT 101" in r.stderr
    assert not any("mkdir" in c or "timeout" in c for c in log)
    assert log[-1] == f"exec 101 -- rm -rf -- {IN}"


def test_only_the_jobs_folder_is_created_and_it_takes_the_parents_owner(ctl):
    r, log, *_ = ctl(f"exec-run t {JID}")
    assert r.returncode == 0, r.stderr
    made = [c for c in log if "mkdir" in c or "chown" in c]
    out = f"/srv/outputs/broker/{JID}"
    assert made == [f"exec 101 -- mkdir -m 755 -- {out}", f"exec 101 -- chown -- {ctlfake.OWNER} {out}",
                    f"exec 101 -- chown -R -- {ctlfake.OWNER} {out}"]   # no -p: the parent is never created


def test_the_failed_programs_outputs_are_still_handed_to_the_owner(ctl):
    r, log, *_ = ctl(f"exec-run t {JID}", recipe=RECIPE.replace("--job {jid}", "--job FAIL"))
    assert r.returncode == 9 and any(c.startswith("exec 101 -- chown -R") for c in log)


def test_exec_run_reaps_workers_that_left_the_programs_group(ctl):
    """A double-forked worker keeps the tag, not the group: exec-run kills it (and its group)."""
    r, _, _, killed, left = ctl(f"exec-run t {JID}", leave=[(500, 500, JID), (501, 500, JID), (600, 600, None)])
    assert r.returncode == 0, r.stderr
    assert left == [600] and "-500" in killed and "600" not in killed   # untagged processes are never touched


def test_exec_run_reports_survivors_as_7_never_as_success(ctl):
    r, *_ = ctl(f"exec-run t {JID}", leave=[(500, 500, JID, True)], conf="CLEAN_WAIT_S=1")
    assert r.returncode == 7 and b"still run" in r.stderr


def test_a_step_that_fails_exits_10_not_its_own_code(tmp_path):
    """pct itself may exit 255; exec-run must never pass that on (255 means "ssh failed")."""
    ctl = ctlfake.make(tmp_path, RECIPE)
    (tmp_path / "bin/pct").write_text((tmp_path / "bin/pct").read_text().replace(
        'case "$4" in', '[[ "$4" == mkdir && "$5" == -m ]] && exit 255\ncase "$4" in'))
    r, *_ = ctl(f"exec-run t {JID}")
    assert r.returncode == 10


def test_clean_with_nothing_running_removes_the_inputs(ctl):
    r, log, _, killed, _ = ctl(f"exec-clean t {JID}")
    assert r.returncode == 0, r.stderr
    assert (log, killed) == ([SCAN, f"exec 101 -- rm -rf -- {IN}"], [])


def test_clean_kills_by_tag_pid_and_group_and_leaves_other_jobs_alone(ctl):
    procs = [(700, 700, JID), (701, 700, None), (702, 702, JID), (800, 800, OTHER), (801, 1, JID)]
    r, log, _, killed, left = ctl(f"exec-clean t {JID}", procs=procs)
    assert r.returncode == 0, r.stderr
    assert left == [800] and log.count(SCAN) == 2   # 701 died with its group; 801 by pid
    assert "-1" not in killed and "801" in killed   # init's group (1) is never signalled


def test_a_failed_scan_never_counts_as_gone(ctl):
    """By count, not by time: a 1 s wait that starts just before SECONDS ticks may allow one scan."""
    r, log, *_ = ctl(f"exec-clean t {JID}", scan_fails=1, conf="CLEAN_WAIT_S=30")
    assert r.returncode == 0 and log.count(SCAN) == 2   # the failed scan was retried, not taken as "none"
    r, log, *_ = ctl(f"exec-clean t {JID}", scan_fails="always", conf="CLEAN_WAIT_S=1")
    assert r.returncode == 7 and log.count(SCAN) >= 1 and log[-1] == f"exec 101 -- rm -rf -- {IN}"


def test_a_container_without_a_readable_proc_never_counts_as_gone(tmp_path):
    ctl = ctlfake.make(tmp_path, RECIPE)
    (tmp_path / "bin/pct").write_text((tmp_path / "bin/pct").read_text().replace(
        'exec sh -c', 'rm -rf "$FAKE_PROC/1"; exec sh -c'))
    r, *_ = ctl(f"exec-clean t {JID}", conf="CLEAN_WAIT_S=1")
    assert r.returncode == 7


def test_clean_reports_a_job_that_will_not_die_after_its_wait_in_seconds(ctl):
    t0 = time.monotonic()
    r, log, *_ = ctl(f"exec-clean t {JID}", procs=[(700, 700, JID, True)], conf="CLEAN_WAIT_S=2")
    assert r.returncode == 7 and b"still running" in r.stderr
    assert 1 <= time.monotonic() - t0 < 6 and log.count(SCAN) > 2   # bounded by time, not by a try count
    assert log[-1] == f"exec 101 -- rm -rf -- {IN}"


def test_clean_waits_for_an_exec_run_in_flight_and_cancels_a_later_one(ctl, tmp_path):
    r, log, *_ = ctl(f"exec-clean t {JID}", busy=3)   # exec-run holds the job lock for 3 more scans
    assert r.returncode == 0 and log.count(SCAN) == 4
    assert (tmp_path / f"locks/{JID}.cancel").exists()
    r, log, *_ = ctl(f"exec-run t {JID}")   # its SSH session was cut before it started the program
    assert r.returncode == 11 and not any("timeout" in c for c in log)
    r, *_ = ctl(f"exec-clean t {JID}", busy="always", conf="CLEAN_WAIT_S=1")
    assert r.returncode == 7   # a run that never lets go is not "gone"


def test_clean_validates_like_the_other_verbs(ctl):
    for cmd in (f"exec-clean ../t {JID}", "exec-clean t nothex"):
        r, log, *_ = ctl(cmd)
        assert r.returncode == 2 and log == [], cmd


def test_info_prints_the_hosts_own_timings_and_runs_nothing(ctl):
    conf = "CLEAN_WAIT_S=60\nSCAN_S=15\nSCAN_KILL_S=4\nCLEAN_POLL_S=1\nKILL_AFTER_S=12"
    r, log, *_ = ctl("exec-info t", conf=conf)
    assert (r.returncode, log) == (0, [])
    # A bounded step is 15 + 4 (timeout -k): reap 60 + 1 + 1 + 19, clean the reap + one more step.
    assert r.stdout.decode().split() == ["timeout_s=600", "kill_after_s=12", "reap_s=81", "clean_wait_s=100"]


def test_every_bounded_container_step_also_gets_its_kill_time(ctl, tmp_path):
    r, *_ = ctl(f"exec-clean t {JID}", conf="SCAN_S=15\nSCAN_KILL_S=4", procs=[(700, 700, JID)])
    calls = (tmp_path / "timeout.log").read_text().splitlines()
    assert r.returncode == 0 and len(calls) == 3   # two scans and the input removal
    assert all(c == "-k 4 15 pct exec 101 --" for c in calls), calls


def test_timings_in_the_conf_must_be_whole_seconds(ctl):
    r, *_ = ctl("exec-info t", conf="CLEAN_WAIT_S=1e9")
    assert r.returncode == 2
