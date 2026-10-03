"""examples/systemd/gpu-broker-exec-prune@.{path,service}: exec job folders (<root>/<job id>/)
are removed once old, triggered by changes to the root, never stopped by the start limit, and
ComfyUI's own output files beside them are left alone. The find command runs on a temp root."""
import os
import pathlib
import shlex
import shutil
import subprocess
import time

import pytest

from gpu_broker.drivers import recipes
from gpu_broker.tuning import Timeouts
from tests.helpers import ROOT
from tests.test_comfy_prune import sections

UNIT = ROOT / "examples/systemd/gpu-broker-exec-prune@.service"
PATH_UNIT = UNIT.with_suffix(".path")
JID = "0123456789ab"


def argv(root):
    (line,) = [x for x in sections(UNIT)["Service"] if x.startswith("ExecStart=")]
    return [root if a == "%f" else a for a in shlex.split(line.removeprefix("ExecStart="))]


@pytest.mark.skipif(shutil.which("find") is None, reason="needs find")
def test_prunes_only_old_job_folders(tmp_path):
    old = time.time() - 3 * 3600
    for name, mtime in {JID: old, "ba9876543210": time.time(), "notajobfolder": old, "0123456789abc": old}.items():
        (tmp_path / name).mkdir()
        (tmp_path / name / "scene.ply").write_text("ply")
        os.utime(tmp_path / name, (mtime, mtime))
    (tmp_path / f"{JID}_00001_.png").write_text("png")   # a ComfyUI output in the same root
    (tmp_path / "aaaaaaaaaaaa").write_text("a file named like a job")   # only folders go
    os.utime(tmp_path / "aaaaaaaaaaaa", (old, old))
    os.utime(tmp_path / f"{JID}_00001_.png", (old, old))
    (tmp_path / "deep" / JID).mkdir(parents=True)        # -maxdepth 1
    os.utime(tmp_path / "deep" / JID, (old, old))
    subprocess.run(argv(str(tmp_path)), check=True, timeout=20)
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(
        ["ba9876543210", "notajobfolder", "0123456789abc", f"{JID}_00001_.png", "deep", "aaaaaaaaaaaa"])
    assert (tmp_path / "deep" / JID).exists()


def test_the_age_outlasts_the_longest_graph_and_bursts_cannot_stop_it():
    args = argv("/x")
    assert int(args[args.index("-mmin") + 1].lstrip("+")) * 60 > Timeouts().comfy_run_s
    assert "StartLimitIntervalSec=0" in sections(UNIT)["Unit"]
    assert "ReadWritePaths=%f" in sections(UNIT)["Service"]


def test_the_path_unit_watches_the_root_and_starts_the_same_instance():
    lines = sections(PATH_UNIT)["Path"]
    assert "PathChanged=%f" in lines and "Unit=gpu-broker-exec-prune@%i.service" in lines


@pytest.mark.parametrize("recipe_dir", [ROOT / "examples/recipes"])
def test_every_shipped_recipe_puts_job_folders_where_the_prune_looks(recipe_dir):
    """The documented roots are the parents of the recipes' out_dir."""
    doc = UNIT.read_text()
    for f in recipe_dir.glob("*.recipe"):
        root = pathlib.PurePosixPath(recipes.load(str(recipe_dir), f.stem).dirs(JID)[1]).parent
        assert f"systemd-escape --path {root})" in doc, f
