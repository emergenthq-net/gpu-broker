"""host/gpu-broker-ctl exec-put / exec-run, run for real with a fake `pct` on PATH. The host
builds the command from its own recipe file; the parity test proves it builds the same argv
as the broker's parser (drivers/recipes.py) from the same file."""
import pytest

from gpu_broker.drivers import recipes
from tests import ctlfake

JID = "ab12cd34ef56"
RECIPE = """# test recipe
target=101
argv=/opt/t/bin/predict -i {in_dir} -o {out_dir} -c {checkpoint} --job {jid}
checkpoint=/mnt/models/t.pt
in_dir=/var/tmp/gpu-broker/{jid}
out_dir=/srv/outputs/broker/{jid}
outputs=*.ply
timeout_s=600
"""


@pytest.fixture
def ctl(tmp_path):
    return ctlfake.make(tmp_path, RECIPE)


def test_exec_put_pushes_stdin_into_the_jobs_input_dir(ctl):
    r, log, pushed, _, _ = ctl(f"exec-put t {JID} image.png", b"\x89PNG data")
    assert r.returncode == 0, r.stderr
    assert log == [f"exec 101 -- mkdir -p -m 700 -- /var/tmp/gpu-broker/{JID}",
                   f"push 101 {log[1].split()[2]} /var/tmp/gpu-broker/{JID}/image.png"]
    assert (pushed / "image.png").read_bytes() == b"\x89PNG data"


def test_exec_put_refuses_oversize_input_and_bad_names(ctl):
    r, log, *_ = ctl(f"exec-put t {JID} image.png", b"x" * 11, conf="MAX_PUT_BYTES=10")
    assert r.returncode == 6 and log == []
    for name in ("../x.png", "Image.png", "image", "a.b.png"):
        r, log, *_ = ctl(f"exec-put t {JID} {name}", b"x")
        assert r.returncode == 2 and log == [], name


def test_exec_run_builds_the_same_argv_as_the_broker_and_prints_only_outputs(ctl):
    r, log, *_ = ctl(f"exec-run t {JID}")
    assert r.returncode == 0, r.stderr
    argv = recipes.parse("t", RECIPE).command(JID)
    out = f"/srv/outputs/broker/{JID}"
    assert log == ["exec 101 -- stat -c %u:%g -- /srv/outputs/broker",
                   f"exec 101 -- mkdir -m 755 -- {out}", f"exec 101 -- chown -- {ctlfake.OWNER} {out}",
                   f"exec 101 -- timeout --kill-after=10 600 env -- GPU_BROKER_JOB={JID} " + " ".join(argv),
                   f"scan 101 GPU_BROKER_JOB={JID}",   # the job's processes are reaped before anything else
                   f"exec 101 -- chown -R -- {ctlfake.OWNER} {out}",
                   f"exec 101 -- find {out} -maxdepth 1 -type f -name *.ply ! -name .*",
                   f"exec 101 -- rm -rf -- /var/tmp/gpu-broker/{JID}"]
    assert r.stdout.decode() == f"output /srv/outputs/broker/{JID}/scene.ply\n"   # program chatter went to stderr


def test_exec_run_lists_outputs_in_glob_order_each_sorted_and_once(ctl):
    recipe = RECIPE.replace("outputs=*.ply", "outputs=w.mp4 *.log *")
    r, log, *_ = ctl(f"exec-run t {JID}", recipe=recipe, files=("z.log", "b.ply", "w.mp4", ".hidden", "a.log"))
    assert r.returncode == 0, r.stderr
    out = f"/srv/outputs/broker/{JID}"
    assert [c for c in log if " find " in c] == [f"exec 101 -- find {out} -maxdepth 1 -type f -name {g} ! -name .*"
                                                 for g in ("w.mp4", "*.log", "*")]
    listed = [line.removeprefix(f"output {out}/") for line in r.stdout.decode().splitlines()]
    assert listed == ["w.mp4", "a.log", "z.log", "b.ply"]   # the same order run_local gives; no dotfile


@pytest.mark.parametrize("outputs", ["FAIL*.ply *.log", "*.log FAIL*.ply"])
def test_exec_run_fails_when_listing_any_glob_fails(ctl, outputs):
    r, log, *_ = ctl(f"exec-run t {JID}", recipe=RECIPE.replace("outputs=*.ply", f"outputs={outputs}"),
                     files=("a.log", "b.ply"))
    assert r.returncode == 12 and b"could not list the outputs" in r.stderr
    assert b"output " not in r.stdout   # nothing listed, not even the globs that worked
    assert log[-1] == f"exec 101 -- rm -rf -- /var/tmp/gpu-broker/{JID}"


def test_exec_run_with_no_outputs_lists_nothing(ctl):
    r, *_ = ctl(f"exec-run t {JID}", files=("a.log",))
    assert r.returncode == 0 and b"output " not in r.stdout


def test_exec_run_failure_exits_nonzero_and_still_removes_the_inputs(ctl):
    r, log, *_ = ctl(f"exec-run t {JID}", recipe=RECIPE.replace("--job {jid}", "--job FAIL"))
    assert r.returncode == 9 and b"recipe t exited 3" in r.stderr   # the program's own code never passes through
    assert log[-1] == f"exec 101 -- rm -rf -- /var/tmp/gpu-broker/{JID}" and not any("find" in c for c in log)


@pytest.mark.parametrize("recipe", [
    RECIPE.replace("target=101", "target="), RECIPE + "shell=sh\n", RECIPE.replace("{jid}\nout", "shared\nout"),
    RECIPE.replace("broker/{jid}", "{jid}/broker"), RECIPE.replace("timeout_s=600", "timeout_s=0"),
    RECIPE.replace("\n", "\r\n"), RECIPE + "argv\n",
    RECIPE.replace("outputs=*.ply", "outputs=../*.ply"), RECIPE.replace("timeout_s=600", "timeout_s=10;id"),
    RECIPE.replace("--job {jid}", "--job $(id)"), RECIPE.replace("-c {checkpoint}", "-c /a/../b"),
    RECIPE.replace("--job {jid}", "--job /bin/bas?"), RECIPE.replace("outputs=*.ply", "outputs=*.ply ../x"),
    RECIPE.replace("outputs=*.ply", "outputs= ")])   # a glob is refused, never expanded (to /bin/bash)
def test_a_bad_recipe_runs_nothing_and_matches_the_broker_parser(ctl, recipe):
    r, log, *_ = ctl(f"exec-run t {JID}", recipe=recipe)
    assert r.returncode == 5 and log == [], r.stderr
    with pytest.raises(ValueError):
        recipes.parse("t", recipe)


def test_exec_verbs_validate_names_and_need_the_recipe(ctl):
    for cmd in (f"exec-run ../t {JID}", "exec-run t nothex", f"exec-run T {JID}"):
        r, log, *_ = ctl(cmd)
        assert r.returncode == 2 and log == [], cmd
    r, log, *_ = ctl(f"exec-run missing {JID}")
    assert r.returncode == 4 and log == []
