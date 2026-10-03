"""`gpu-broker init`: the starter files match examples/, land where asked, and are never clobbered."""
from gpu_broker import cli, settings, starter
from tests.helpers import ROOT

EX = ROOT / "examples"


def test_the_shipped_starters_are_the_examples():
    for name in (starter.CONFIG, starter.CATALOG):
        assert starter.starter(name) == (EX / name).read_text(), f"gpu_broker/starter/{name} differs from examples/{name}"


def test_init_writes_a_config_whose_catalog_is_beside_it_and_creates_its_folders(tmp_path):
    made = []
    lines = starter.run(str(tmp_path), mkdir=made.append)
    cfg = settings.load(str(tmp_path / starter.CONFIG), env={})
    assert cfg.catalog == str((tmp_path / starter.CATALOG).resolve())
    assert (tmp_path / starter.CATALOG).read_text() == (EX / starter.CATALOG).read_text()
    assert made == [str(tmp_path), "/var/lib/gpu-broker", "/var/lib/gpu-broker/inputs", "/var/log/gpu-broker",
                    "/var/lib/gpu-broker/models"]
    assert [ln.split()[0] for ln in lines] == ["wrote", "wrote", "folder", "folder", "folder", "folder"]


def test_init_keeps_existing_files_unless_forced(tmp_path):
    (tmp_path / starter.CATALOG).write_text("mine")
    lines = starter.run(str(tmp_path), mkdir=lambda d: None)
    assert (tmp_path / starter.CATALOG).read_text() == "mine" and lines[1].startswith("kept")
    starter.run(str(tmp_path), force=True, mkdir=lambda d: None)
    assert (tmp_path / starter.CATALOG).read_text() == (EX / starter.CATALOG).read_text()


def test_the_cli_runs_init_without_loading_a_config(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(starter, "data_dirs", lambda cfg: [])
    assert cli.main(["-c", "/nonexistent.yaml", "init", "--dir", str(tmp_path)], env={}) == cli.EXIT_OK
    out = capsys.readouterr().out
    assert "wrote" in out and "check" in out and (tmp_path / starter.CONFIG).exists()


def test_init_without_permission_says_to_use_sudo(tmp_path, capsys):
    locked = tmp_path / "locked"
    locked.mkdir(mode=0o500)
    try:
        assert cli.main(["init", "--dir", str(locked)], env={}) == cli.EXIT_PROBLEMS
    finally:
        locked.chmod(0o700)
    assert "sudo" in capsys.readouterr().err
