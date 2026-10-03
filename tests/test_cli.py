"""`gpu-broker check` validates the shipped examples without running anything."""
from gpu_broker import cli, settings
from tests.helpers import ROOT, amdgpu_fixture

EX = ROOT / "examples"
VEGA = amdgpu_fixture() / "vega10"


def conf(tmp_path, catalog, sys_root=VEGA / "sys"):
    """The local drivers read the GPU from the fixture amdgpu sysfs (driver options sys_root/proc_root)."""
    c = tmp_path / "c.yaml"
    c.write_text(f"catalog: {catalog}\ncomfy: {{unit: comfyui}}\ngpu: {{vendor: amd}}\n"
                 f"driver: {{kind: systemd, sys_root: {sys_root}, proc_root: {VEGA / 'proc'}}}\n")
    return str(c)


def test_check_passes_on_the_examples(tmp_path, capsys):
    assert cli.main(["-c", conf(tmp_path, EX / "catalog.yaml"), "check"], env={}) == cli.EXIT_OK
    out = capsys.readouterr().out
    assert "SystemdDriver" in out and "['comfyui', 'llama-server']" in out and "127.0.0.1:8095" in out
    assert "gpu:     amd (amdgpu sysfs, card0, 0000:03:00.0)" in out
    assert "sample:  4096,16384,37,42.0,51,1500|comfyui:512 llama-server:3072" in out


def test_check_without_a_gpu_says_so_and_still_passes(tmp_path, capsys):
    assert cli.main(["-c", conf(tmp_path, EX / "catalog.yaml", sys_root=tmp_path / "nosys"), "check"], env={}) == cli.EXIT_OK
    assert "gpu:     none usable here (gpu.index 0: 0 amdgpu card(s)" in capsys.readouterr().out


def test_check_reports_unknown_template(tmp_path, capsys):
    cat = tmp_path / "cat.yaml"
    cat.write_text("defaults: {resident: a}\nmodels: {a: {kind: image, runner: comfy, status: ready, template: nope}}\n")
    assert cli.main(["-c", conf(tmp_path, cat), "check"], env={}) == cli.EXIT_PROBLEMS
    assert "unknown template 'nope'" in capsys.readouterr().out


def test_serve_refuses_without_token(tmp_path):
    assert cli.main(["-c", conf(tmp_path, EX / "catalog.yaml"), "serve"], env={}) == cli.EXIT_NO_TOKEN


def test_check_reports_an_exec_timeout_shorter_than_the_recipes(tmp_path, capsys, monkeypatch):
    from gpu_broker import drivers
    from gpu_broker.drivers.local import SystemdDriver
    from gpu_broker.settings import Timeouts
    rdir = tmp_path / "recipes"
    rdir.mkdir()
    (rdir / "sharp.recipe").write_text((EX / "recipes/sharp.recipe").read_text().replace("timeout_s=600", "timeout_s=630"))
    monkeypatch.setattr(drivers, "build", lambda cfg, units: SystemdDriver(
        allowed=None, timeouts=Timeouts(), sample_s=1, recipes_dir=str(rdir), sys_root=str(VEGA / "sys"),
        gpu=settings.Gpu("amd")))
    assert cli.main(["-c", conf(tmp_path, EX / "catalog.yaml"), "check"], env={}) == cli.EXIT_PROBLEMS
    assert "apple-sharp: exec.timeout_s 660 must be at least 685" in capsys.readouterr().out   # 630 + 10 + 15 + 30
    (rdir / "sharp.recipe").unlink()   # unreadable here: reported, not a problem
    assert cli.main(["-c", conf(tmp_path, EX / "catalog.yaml"), "check"], env={}) == cli.EXIT_OK
    assert "apple-sharp: recipe not checked here" in capsys.readouterr().out
