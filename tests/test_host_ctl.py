"""host/gpu-broker-ctl (the Proxmox host script) run for real with a fake `pct` on PATH."""
import os
import pathlib
import subprocess

import pytest

SCRIPT = pathlib.Path(__file__).parents[1] / "host/gpu-broker-ctl"


@pytest.fixture
def ctl(tmp_path):
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    (bin_ / "pct").write_text('#!/bin/sh\necho "pct $*"\n')
    (bin_ / "pct").chmod(0o755)
    models = tmp_path / "models"
    (models / "m").mkdir(parents=True)
    (models / "m/l.safetensors").write_text("x")

    def run(cmd, conf=""):
        cf = tmp_path / "ctl.conf"
        cf.write_text(f"LOG={tmp_path}/ctl.log\nMODELS={models}\n{conf}")
        env = {"PATH": f"{bin_}:/usr/bin:/bin", "SSH_ORIGINAL_COMMAND": cmd, "GPU_BROKER_CTL_CONF": str(cf)}
        return subprocess.run(["bash", str(SCRIPT)], env=env, capture_output=True, text=True, timeout=20)
    return run


def test_no_conf_allows_no_unit(ctl):
    r = ctl("unit 101 llm start")
    assert r.returncode == 3 and "not allowlisted" in r.stderr


def test_allowlisted_unit_runs_pct(ctl):
    r = ctl("unit 101 llm stop", 'ALLOW_UNITS="101:llm 102:comfyui"')
    assert r.returncode == 0 and r.stdout.strip() == "pct exec 101 -- systemctl stop llm"
    assert ctl("unit 101 llm restart", 'ALLOW_UNITS="101:llm"').returncode == 2


def test_bad_arguments_are_refused(ctl):
    assert "bad arg" in ctl("unit 101 ../etc start", 'ALLOW_UNITS="101:llm"').stderr
    assert ctl("rm -rf /").returncode == 2
    assert ctl("download hf ../x slug").returncode == 2


def test_comfy_link_needs_comfy_ct(ctl):
    assert "disabled" in ctl("comfy-link m/l.safetensors loras").stderr
    r = ctl("comfy-link m/l.safetensors loras", "COMFY_CT=102\nCOMFY_MODELS_MOUNT=/mnt/m\nCOMFY_MODELS_DIR=/c/models")
    assert r.stdout.strip() == "pct exec 102 -- ln -sfn /mnt/m/m/l.safetensors /c/models/loras/l.safetensors"


@pytest.mark.skipif(not os.path.exists("/bin/bash"), reason="needs bash")
def test_script_parses():
    assert subprocess.run(["bash", "-n", str(SCRIPT)]).returncode == 0
