"""Exec recipes: the file format, argv substitution, and running one on this machine."""
import io
import subprocess

import pytest

from gpu_broker.drivers import KILL_AFTER_S, PIPE_DRAIN_S, REAP_WAIT_S, RecipeInfo, recipes
from gpu_broker.drivers.local import DockerDriver, SystemdDriver
from gpu_broker.settings import Timeouts
from tests.helpers import ROOT

JID = "ab12cd34ef56"
GOOD = """# a comment
argv=/opt/t/bin/predict -i {in_dir} -o {out_dir} -c {checkpoint} --tag {jid}
checkpoint=/models/t.pt
in_dir=/tmp/x/in/{jid}
out_dir=/tmp/x/out/{jid}

outputs=*.ply
timeout_s=600
"""


def test_parse_and_substitute():
    r = recipes.parse("t", GOOD)
    assert (r.target, r.timeout_s, r.outputs) == (None, 600.0, ("*.ply",))
    assert r.dirs(JID) == (f"/tmp/x/in/{JID}", f"/tmp/x/out/{JID}")
    assert r.command(JID) == ["/opt/t/bin/predict", "-i", f"/tmp/x/in/{JID}", "-o", f"/tmp/x/out/{JID}",
                              "-c", "/models/t.pt", "--tag", JID]
    assert recipes.parse("t", GOOD + "target=101\n").target == "101"
    assert recipes.parse("t", GOOD + "outputs=world.mp4  *.log\n").outputs == ("world.mp4", "*.log")
    assert recipes.parse("t", GOOD.replace("timeout_s=600", "timeout_s=0.5")).timeout_s == 0.5


@pytest.mark.parametrize(("change", "msg"), [
    ("shell=bash\n", "expected one of"), ("argv /bin/x\n", "expected one of"), (" argv=/bin/x\n", "expected one of"),
    ("argv=/bin/sh -c rm;ls\n", "may not use"), ("argv=/bin/x ../etc/passwd\n", "may not use"),
    ("argv=/bin/x $(id)\n", "may not use"),
    ("in_dir=/tmp/shared\n", "in_dir must be an absolute path"), ("out_dir=relative/{jid}\n", "out_dir must be"),
    ("in_dir=/tmp/{jid}/../etc\n", "in_dir must be"), ("outputs=../*.ply\n", "outputs must be"),
    ("outputs=/abs/*.ply\n", "outputs must be"),
    ("outputs=a.mp4 ../b.log\n", "outputs must be"), ("outputs=a.mp4\tb.log\n", "outputs must be"),
    ("outputs= \n", "outputs must be"), ("timeout_s=1e9\n", "timeout_s must be"), ("timeout_s=\n", "missing"),
    ("timeout_s=0\n", "positive"), ("timeout_s=0.00\n", "positive"),
    ("out_dir=/tmp/{jid}/out\n", "end in /{jid}"), ("out_dir=/tmp/{jid}/{jid}\n", "end in /{jid}"),
    ("target=101;id\n", "target must be"), ("argv=\n", "missing"),
    ("timeout_s=600\r\n", "carriage return"), ("# note\r\n", "carriage return")])
def test_parse_rejects(change, msg):
    with pytest.raises(ValueError, match=msg):
        recipes.parse("t", GOOD + change)


def test_crlf_files_are_refused_not_stripped():
    with pytest.raises(ValueError, match="carriage return"):
        recipes.parse("t", GOOD.replace("\n", "\r\n"))


def test_load_by_name_only(tmp_path):
    (tmp_path / "t.recipe").write_text(GOOD)
    assert recipes.load(str(tmp_path), "t").name == "t"
    with pytest.raises(FileNotFoundError):
        recipes.load(str(tmp_path), "nope")
    with pytest.raises(ValueError, match="bad recipe name"):
        recipes.load(str(tmp_path), "../t")


def test_the_example_recipe_parses():
    r = recipes.load(str(ROOT / "examples/recipes"), "sharp")
    assert r.target is None and r.command(JID)[:2] == ["/opt/ml-sharp/.venv/bin/sharp", "predict"]


def local_recipe(tmp_path, timeout_s=600, parent=True):
    if parent:
        (tmp_path / "out").mkdir(exist_ok=True)
    return recipes.parse("t", GOOD.replace("/tmp/x", str(tmp_path)).replace("timeout_s=600", f"timeout_s={timeout_s}"))


def test_run_local_streams_inputs_runs_argv_and_returns_outputs(tmp_path):
    r, seen = local_recipe(tmp_path), []

    def run(cmd, timeout, tag):
        assert tag == JID   # the job's processes carry its id (reap.py)
        in_dir = tmp_path / "in" / JID
        seen.append((cmd, timeout, sorted(p.name for p in in_dir.iterdir()), (in_dir / "image.png").read_bytes()))
        (tmp_path / "out" / JID / "image.ply").write_text("ply")
        (tmp_path / "out" / JID / "log.txt").write_text("not an output")
        (tmp_path / "out" / JID / "cache.ply").mkdir()                 # matches the glob, but is no file
        return subprocess.CompletedProcess(cmd, 0, "", "")
    outs = recipes.run_local(r, JID, [("image.png", io.BytesIO(b"png")), ("params.json", b"{}")], run)
    assert outs == [str(tmp_path / "out" / JID / "image.ply")]
    assert seen == [(r.command(JID), 600, ["image.png", "params.json"], b"png")]
    assert not (tmp_path / "in" / JID).exists()          # inputs removed after the run


def test_run_local_lists_outputs_in_glob_order_each_sorted_and_once(tmp_path):
    (tmp_path / "out").mkdir()
    r = recipes.parse("t", GOOD.replace("/tmp/x", str(tmp_path)).replace("outputs=*.ply", "outputs=w.mp4 *.log *"))

    def run(cmd, timeout, tag):
        for name in ("z.log", "w.mp4", "a.log", "b.ply", ".hidden"):
            (tmp_path / "out" / JID / name).write_text("x")
        return subprocess.CompletedProcess(cmd, 0, "", "")
    outs = recipes.run_local(r, JID, [], run)
    assert [o.rsplit("/", 1)[1] for o in outs] == ["w.mp4", "a.log", "z.log", "b.ply"]


def test_run_local_failure_raises_with_stderr_and_still_cleans_up(tmp_path):
    r = local_recipe(tmp_path, timeout_s=5)
    timeouts = []

    def fail(cmd, timeout, tag):
        timeouts.append(timeout)
        return subprocess.CompletedProcess(cmd, 3, "", "CUDA out of memory")
    with pytest.raises(RuntimeError, match="exited 3: CUDA out of memory"):
        recipes.run_local(r, JID, [("image.png", b"png")], fail)
    assert timeouts == [5] and not (tmp_path / "in" / JID).exists()   # the recipe's own limit


def test_a_timeout_is_an_ordinary_failure_once_the_group_is_stopped(tmp_path):
    def slow(cmd, timeout, tag):
        raise subprocess.TimeoutExpired(cmd, timeout)
    with pytest.raises(RuntimeError, match="timed out after 5s and was stopped"):
        recipes.run_local(local_recipe(tmp_path, timeout_s=5), JID, [], slow)
    assert not (tmp_path / "in" / JID).exists()


def test_the_output_folder_is_only_the_jobs_and_its_parent_must_exist(tmp_path):
    r = local_recipe(tmp_path, parent=False)
    with pytest.raises(RuntimeError, match=r"output parent .*/out does not exist"):
        recipes.run_local(r, JID, [("image.png", b"png")], lambda *a, **k: pytest.fail("must not run"))
    assert not (tmp_path / "in" / JID).exists() and not (tmp_path / "out").exists()
    (tmp_path / "out" / JID).mkdir(parents=True)   # already there: created inside the try, so cleaned up
    with pytest.raises(FileExistsError):
        recipes.run_local(r, JID, [("image.png", b"png")], lambda *a, **k: pytest.fail("must not run"))
    assert not (tmp_path / "in" / JID).exists()


def systemd(tmp_path, run):
    (tmp_path / "t.recipe").write_text(GOOD.replace("/tmp/x", str(tmp_path)))
    (tmp_path / "out").mkdir(exist_ok=True)
    return SystemdDriver(allowed=None, timeouts=Timeouts(), sample_s=1, run_recipe_cmd=run, recipes_dir=str(tmp_path),
                         run=lambda *a, **k: pytest.fail("recipes run through run_recipe_cmd"))


def test_systemd_driver_runs_a_recipe_and_validates_what_the_job_sends(tmp_path):
    def run(cmd, timeout, tag):
        (tmp_path / "out/abcdef12/s.ply").write_text("ply")
        return subprocess.CompletedProcess(cmd, 0, "", "")
    d = systemd(tmp_path, run)
    assert d.recipe_info("t") == RecipeInfo(600, KILL_AFTER_S, PIPE_DRAIN_S + REAP_WAIT_S, REAP_WAIT_S)
    assert d.run_recipe("t", "abcdef12", [("image.png", b"x")], 60) == [str(tmp_path / "out/abcdef12/s.ply")]
    for recipe, jid, name in (("../t", "abcdef12", "image.png"), ("t", "../abcdef12", "image.png"),
                              ("t", "abcdef12", "../image.png"), ("t", "abcdef12", "a b.png")):
        with pytest.raises(ValueError):
            d.run_recipe(recipe, jid, [(name, b"x")], 60)


def test_local_drivers_refuse_proxmox_recipes_and_docker_refuses_all(tmp_path):
    d = systemd(tmp_path, lambda *a, **k: pytest.fail("must not run"))
    (tmp_path / "ct.recipe").write_text(GOOD + "target=101\n")
    for call in (lambda: d.run_recipe("ct", "abcdef12", [], 60), lambda: d.recipe_info("ct")):
        with pytest.raises(ValueError, match="for the proxmox driver"):
            call()
    docker = DockerDriver(allowed=None, timeouts=Timeouts(), sample_s=1)
    for call in (lambda: docker.run_recipe("t", "abcdef12", [], 60), lambda: docker.recipe_info("t"), lambda: docker.clean_recipe("t", "abcdef12")):
        with pytest.raises(RuntimeError, match="docker driver cannot"):
            call()


def test_a_local_clean_removes_the_inputs_only_once_the_job_is_gone(tmp_path, monkeypatch):
    from gpu_broker.drivers import local
    d = systemd(tmp_path, lambda *a, **k: pytest.fail("a clean runs nothing"))
    (tmp_path / "in/abcdef12").mkdir(parents=True)
    monkeypatch.setattr(local.reap, "reap", lambda jid, wait_s: False)
    with pytest.raises(RuntimeError, match="processes of job abcdef12 still run"):
        d.clean_recipe("t", "abcdef12")
    assert (tmp_path / "in/abcdef12").exists()
    monkeypatch.setattr(local.reap, "reap", lambda jid, wait_s: jid == "abcdef12")
    d.clean_recipe("t", "abcdef12")
    assert not (tmp_path / "in/abcdef12").exists()


def test_a_local_clean_needs_only_the_reap_not_a_loadable_recipe(tmp_path, monkeypatch):
    """The reap goes by job id; the recipe only names the inputs folder, so a recipe that no
    longer loads (deleted, broken, or for another driver) must not keep the GPU held."""
    from gpu_broker.drivers import local
    d = systemd(tmp_path, lambda *a, **k: pytest.fail("a clean runs nothing"))
    reaped = []
    monkeypatch.setattr(local.reap, "reap", lambda jid, wait_s: reaped.append(jid) or True)
    (tmp_path / "ct.recipe").write_text(GOOD + "target=101\n")
    (tmp_path / "bad.recipe").write_text("not a recipe\n")
    for name in ("gone", "bad", "ct"):
        d.clean_recipe(name, "abcdef12")
    assert reaped == ["abcdef12"] * 3
