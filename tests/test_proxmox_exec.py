"""Proxmox driver, exec recipes: each input crosses SSH on stdin (streamed from its open file),
then one exec-run call. Only the recipe name, job id and file names are ever on the SSH command
line. Unless exec-run answers with a definite outcome, exec-clean runs: no answer, ssh failure
(255) and surviving job processes (7) are not definite. If the program may have started and the
clean cannot confirm it is gone, GpuHeld."""
import io
import subprocess

import pytest

from gpu_broker import settings
from gpu_broker.drivers import GpuHeld, RecipeInfo
from gpu_broker.drivers.proxmox import ProxmoxDriver

T = settings.Timeouts(exec_put_s=7, exec_clean_s=9, unit_s=11)
JID = "ab12cd34ef56"
INFO = "timeout_s=600\nkill_after_s=12\nreap_s=77\nclean_wait_s=92\n"
OUTPUTS = "output /o/broker/ab12cd34ef56/a.ply\nlog line\noutput /o/broker/ab12cd34ef56/b.ply\n"


class Ssh:
    """Answers each verb from `rc` (exit codes) or raises `raises[verb]`; records every call."""

    def __init__(self, rc=None, raises=None, stdout=OUTPUTS):
        self.puts, self.runs, self.rc, self.raises, self.stdout = [], [], rc or {}, raises or {}, stdout

    def run(self, cmd, timeout):
        verb = cmd[cmd.index("root@pve") + 1]
        self.runs.append((cmd[cmd.index("root@pve") + 1:], timeout))
        if verb in self.raises:
            raise self.raises[verb]
        out = {"exec-run": self.stdout, "exec-info": INFO}.get(verb, "")
        return subprocess.CompletedProcess(cmd, self.rc.get(verb, 0), out, "Traceback: boom")

    def run_input(self, cmd, data, timeout):
        self.puts.append((cmd, data if isinstance(data, bytes) else ("file", data.read()), timeout))
        return subprocess.CompletedProcess(cmd, self.rc.get("exec-put", 0), b"", b"input larger than MAX_PUT_BYTES")


def driver(ssh):
    return ProxmoxDriver("root@pve", None, T, run=ssh.run, run_input=ssh.run_input)


def test_puts_each_file_then_runs_and_returns_the_output_lines():
    ssh = Ssh()
    d = driver(ssh)
    outs = d.run_recipe("sharp", JID, [("image.png", io.BytesIO(b"PNG")), ("params.json", b"{}")], 660)
    assert outs == ["/o/broker/ab12cd34ef56/a.ply", "/o/broker/ab12cd34ef56/b.ply"]
    assert ssh.puts == [([*d.base, "exec-put", "sharp", JID, "image.png"], ("file", b"PNG"), 7),
                        ([*d.base, "exec-put", "sharp", JID, "params.json"], b"{}", 7)]
    assert ssh.runs == [(["exec-run", "sharp", JID], 660)]   # exec-run cleaned up itself: no exec-clean


def test_a_failed_put_cleans_up_and_never_runs():
    ssh = Ssh(rc={"exec-put": 6})
    with pytest.raises(RuntimeError, match=r"copying image\.png for recipe sharp failed: input larger"):
        driver(ssh).run_recipe("sharp", JID, [("image.png", b"PNG")], 660)
    assert ssh.runs == [(["exec-clean", "sharp", JID], 9)]


def test_a_failed_clean_before_the_program_started_is_not_gpu_held():
    ssh = Ssh(rc={"exec-put": 6, "exec-clean": 7})
    with pytest.raises(RuntimeError, match="copying") as e:
        driver(ssh).run_recipe("sharp", JID, [("image.png", b"PNG")], 660)
    assert not isinstance(e.value, GpuHeld)


def test_a_failed_run_reports_the_exit_code_and_stderr():
    ssh = Ssh(rc={"exec-run": 2})
    with pytest.raises(RuntimeError, match="recipe sharp exited 2: Traceback: boom"):
        driver(ssh).run_recipe("sharp", JID, [], 660)
    assert [verb for (verb, *_), _ in ssh.runs] == ["exec-run"]   # the host script cleaned up


def test_a_broker_side_timeout_kills_the_job_on_the_host():
    ssh = Ssh(raises={"exec-run": subprocess.TimeoutExpired("ssh", 660)})
    with pytest.raises(RuntimeError, match="did not finish within 660s") as e:
        driver(ssh).run_recipe("sharp", JID, [], 660)
    assert not isinstance(e.value, GpuHeld)
    assert ssh.runs == [(["exec-run", "sharp", JID], 660), (["exec-clean", "sharp", JID], 9)]


@pytest.mark.parametrize("clean", [{"rc": {"exec-clean": 7}}, {"raises": {"exec-clean": OSError("ssh: no route")}}])
@pytest.mark.parametrize("run", [subprocess.TimeoutExpired("ssh", 660), OSError("connection reset")])
def test_a_program_that_may_still_run_is_gpu_held(run, clean):
    ssh = Ssh(rc=clean.get("rc"), raises={"exec-run": run, **clean.get("raises", {})})
    with pytest.raises(GpuHeld, match=f"recipe sharp for job {JID} may still be running"):
        driver(ssh).run_recipe("sharp", JID, [], 660)


@pytest.mark.parametrize("rc", [255, 7])
@pytest.mark.parametrize(("clean", "held"), [({}, False), ({"exec-clean": 7}, True)])
def test_an_exec_run_with_an_unknown_outcome_is_cleaned_and_held_if_that_fails(rc, clean, held):
    """255: ssh failed (or the host script was killed); 7: job processes survived its reap."""
    ssh = Ssh(rc={"exec-run": rc, **clean})
    with pytest.raises(RuntimeError, match=r"state unknown|may still be running") as e:
        driver(ssh).run_recipe("sharp", JID, [], 660)
    assert isinstance(e.value, GpuHeld) is held
    assert ssh.runs == [(["exec-run", "sharp", JID], 660), (["exec-clean", "sharp", JID], 9)]


@pytest.mark.parametrize("rc", [8, 9, 10, 11])
def test_a_definite_exec_run_failure_needs_no_clean(rc):
    ssh = Ssh(rc={"exec-run": rc})
    with pytest.raises(RuntimeError, match=f"recipe sharp exited {rc}"):
        driver(ssh).run_recipe("sharp", JID, [], 660)
    assert len(ssh.runs) == 1


def test_recipe_info_is_the_hosts_own_values():
    ssh = Ssh()
    # The host's clean_wait_s plus this side's SSH ConnectTimeout.
    assert driver(ssh).recipe_info("sharp") == RecipeInfo(600, 12, 77, 92 + T.ssh_connect_s)
    assert ssh.runs == [(["exec-info", "sharp"], 11)]
    with pytest.raises(ValueError):
        driver(ssh).recipe_info("sharp;id")
    with pytest.raises(RuntimeError, match="reading recipe sharp failed"):
        driver(Ssh(rc={"exec-info": 5})).recipe_info("sharp")
    old_host = Ssh()
    old_host.run = lambda cmd, timeout: subprocess.CompletedProcess(cmd, 0, "timeout_s=600\n", "")
    with pytest.raises(RuntimeError, match="reading recipe sharp failed"):   # every value, or none
        driver(old_host).recipe_info("sharp")


def test_clean_recipe_raises_unless_the_host_confirms():
    driver(ssh := Ssh()).clean_recipe("sharp", JID)
    assert ssh.runs == [(["exec-clean", "sharp", JID], 9)]
    with pytest.raises(RuntimeError, match=r"exec-clean of job .* failed \(7\)"):
        driver(Ssh(rc={"exec-clean": 7})).clean_recipe("sharp", JID)
    with pytest.raises(ValueError):
        driver(Ssh()).clean_recipe("sharp", "x y")


@pytest.mark.parametrize(("recipe", "jid", "name"), [("sharp;id", JID, "image.png"), ("sharp", "x y", "image.png"),
                                                      ("sharp", JID, "../../etc/passwd"), ("sharp", JID, "-rf.png")])
def test_nothing_crosses_ssh_unless_every_value_is_clean(recipe, jid, name):
    ssh = Ssh()
    with pytest.raises(ValueError):
        driver(ssh).run_recipe(recipe, jid, [(name, b"x")], 660)
    assert ssh.puts == [] and ssh.runs == []


def test_ssh_uses_only_the_broker_key():
    """No fallback to other keys or an agent: only the configured identity is offered."""
    base = ProxmoxDriver("root@pve", None, T, ssh_key="/etc/gpu-broker/id_ed25519").base
    assert base[:5] == ["ssh", "-i", "/etc/gpu-broker/id_ed25519", "-o", "IdentitiesOnly=yes"]
    assert base.index("IdentitiesOnly=yes") < base.index("--")
