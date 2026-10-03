"""examples/systemd/comfyui-input-prune.{path,service}: the path unit starts the prune when
ComfyUI's input folder changes (no timer), and its find command, run on a temp folder, deletes
only old files the broker uploaded (broker-<job id>-<slot>.<ext>)."""
import os
import pathlib
import shlex
import shutil
import subprocess
import time

import pytest

from gpu_broker.staging import UPLOAD_PREFIX
from gpu_broker.tuning import Timeouts

UNIT = pathlib.Path(__file__).parents[1] / "examples/systemd/comfyui-input-prune.service"
PATH_UNIT = UNIT.with_suffix(".path")
COMFY_INPUT = "/opt/ComfyUI/input"
JID = "0123456789ab"


def argv(folder):
    line = next(x for x in UNIT.read_text().splitlines() if x.startswith("ExecStart="))
    return [folder if a == COMFY_INPUT else a for a in shlex.split(line.removeprefix("ExecStart="))]


def test_the_age_outlasts_the_longest_graph():
    args = argv(COMFY_INPUT)
    assert int(args[args.index("-mmin") + 1].lstrip("+")) * 60 > Timeouts().comfy_run_s
    assert f"ReadWritePaths={COMFY_INPUT}" in UNIT.read_text()


@pytest.mark.skipif(shutil.which("find") is None, reason="needs find")
def test_prunes_only_old_broker_uploads(tmp_path):
    old = time.time() - 3 * 3600
    names = {f"{UPLOAD_PREFIX}{JID}-image.png": (old, False), f"{UPLOAD_PREFIX}{JID}-end_image.jpg": (old, False),
             f"{UPLOAD_PREFIX}{JID}-image.webp": (time.time(), True),   # its job may still be running
             f"{UPLOAD_PREFIX}abc-image.png": (old, True), "portrait.png": (old, True),
             f"x{UPLOAD_PREFIX}{JID}-image.png": (old, True)}
    for n, (mtime, _) in names.items():
        (tmp_path / n).write_bytes(b"x")
        os.utime(tmp_path / n, (mtime, mtime))
    (tmp_path / f"{UPLOAD_PREFIX}{JID}-image.jpg").mkdir()   # a directory, not an upload
    os.utime(tmp_path / f"{UPLOAD_PREFIX}{JID}-image.jpg", (old, old))
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / f"{UPLOAD_PREFIX}{JID}-image.png").write_bytes(b"x")
    os.utime(tmp_path / "sub" / f"{UPLOAD_PREFIX}{JID}-image.png", (old, old))
    subprocess.run(argv(str(tmp_path)), check=True, timeout=20)
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted([*(n for n, (_, keep) in names.items() if keep), "sub",
                                                                 f"{UPLOAD_PREFIX}{JID}-image.jpg"])
    assert (tmp_path / "sub" / f"{UPLOAD_PREFIX}{JID}-image.png").exists()   # -maxdepth 1


def test_pruning_is_triggered_by_uploads_not_a_clock():
    lines = PATH_UNIT.read_text().splitlines()
    assert f"PathChanged={COMFY_INPUT}" in lines and f"Unit={UNIT.name}" in lines
    assert not UNIT.with_suffix(".timer").exists()
    assert not any(x.startswith(("OnCalendar", "OnUnitActiveSec", "OnBootSec")) for x in lines)


def sections(path):
    out, current = {}, None
    for line in path.read_text().splitlines():
        if line.startswith("[") and line.endswith("]"):
            current = out.setdefault(line[1:-1], [])
        elif current is not None and line and not line.startswith("#"):
            current.append(line)
    return out


def test_uploads_in_bursts_cannot_trip_the_start_limit():
    """Without it, more than 5 starts in 10 s (a burst of uploads) fails the service, and the
    .path unit gives up with start-limit-hit: pruning would stop silently. It belongs in [Unit]."""
    assert "StartLimitIntervalSec=0" in sections(UNIT)["Unit"]
    assert not any(x.startswith("StartLimit") for x in sections(UNIT)["Service"])
