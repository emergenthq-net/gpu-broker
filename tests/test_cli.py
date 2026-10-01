"""`gpu-broker check` validates the shipped examples without running anything."""
from gpu_broker import cli
from tests.helpers import ROOT

EX = ROOT / "examples"


def conf(tmp_path, catalog):
    c = tmp_path / "c.yaml"
    c.write_text(f"catalog: {catalog}\ncomfy: {{unit: comfyui}}\n")
    return str(c)


def test_check_passes_on_the_examples(tmp_path, capsys):
    assert cli.main(["-c", conf(tmp_path, EX / "catalog.yaml"), "check"], env={}) == cli.EXIT_OK
    out = capsys.readouterr().out
    assert "SystemdDriver" in out and "['comfyui', 'llama-server']" in out and "127.0.0.1:8095" in out


def test_check_reports_unknown_template(tmp_path, capsys):
    cat = tmp_path / "cat.yaml"
    cat.write_text("defaults: {resident: a}\nmodels: {a: {kind: image, runner: comfy, status: ready, template: nope}}\n")
    assert cli.main(["-c", conf(tmp_path, cat), "check"], env={}) == cli.EXIT_PROBLEMS
    assert "unknown template 'nope'" in capsys.readouterr().out


def test_serve_refuses_without_token(tmp_path):
    assert cli.main(["-c", conf(tmp_path, EX / "catalog.yaml"), "serve"], env={}) == cli.EXIT_NO_TOKEN
