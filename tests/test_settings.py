"""Settings: defaults, file over defaults, environment over file, strict keys, coercion."""
import pytest

from gpu_broker import settings
from gpu_broker.units import UnitRef


def load(tmp_path, text="", env=None):
    p = tmp_path / "c.yaml"
    p.write_text(text)
    return settings.load(str(p), env or {})


def test_defaults_bind_to_localhost_and_use_systemd(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "DEFAULT_PATH", str(tmp_path / "absent.yaml"))
    s = settings.load(None, {})   # no explicit file and nothing at the default path
    assert s.server.host == "127.0.0.1" and s.driver.kind == "systemd" and s.comfy.unit is None
    assert s.source is None and s.comfy.browser_url == s.comfy.url


def test_explicit_missing_file_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        settings.load(None, {settings.CONFIG_ENV: str(tmp_path / "absent.yaml")})


def test_file_then_environment(tmp_path):
    s = load(tmp_path, "server: {port: 9000}\ntimeouts: {chat_wait_s: 30}\ncomfy: {url: http://c:1/, unit: {ct: 7, name: ui}}\n",
             {"BROKER_PORT": "9100", "BROKER_GPU_STREAM": "off", "BROKER_CHAT_WAIT_S": "12.5"})
    assert s.server.port == 9100 and s.gpu_stream is False and s.timeouts.chat_wait_s == 12.5
    assert s.comfy.url == "http://c:1" and s.comfy.unit == UnitRef("ui", "7")


def test_unknown_keys_are_rejected_at_any_depth(tmp_path):
    with pytest.raises(ValueError, match="top level"):
        load(tmp_path, "chat_wait: 5\n")
    with pytest.raises(ValueError, match="timeouts"):
        load(tmp_path, "timeouts: {llm_strat_s: 5}\n")


def test_driver_options_pass_through_and_units_are_validated(tmp_path):
    s = load(tmp_path, "driver: {kind: proxmox, ssh_target: root@pve, allowed_units: [{name: a, target: 1}]}\n")
    assert s.driver.options == {"ssh_target": "root@pve"} and s.driver.allowed_units == (UnitRef("a", "1"),)
    with pytest.raises(ValueError):
        load(tmp_path, "comfy: {unit: '../x'}\n")


def test_input_settings_from_file_and_environment(tmp_path):
    s = load(tmp_path, "inputs: {types: [png], video_types: [mp4], max_bytes: 1000, max_frames: 8}\n"
                       "comfy: {output_dir: /srv/comfy/output/}\n",
             {"BROKER_INPUT_URLS": "yes", "BROKER_INPUT_MAX_BYTES": "2048", "BROKER_INPUT_DIR": "/tmp/in"})
    i = s.inputs
    assert i.types == ("png",) and i.video_types == ("mp4",) and i.max_bytes == 2048 and i.max_frames == 8
    assert i.allow_urls is True and i.staging_dir == "/tmp/in" and s.comfy.output_dir == "/srv/comfy/output/"
    d = load(tmp_path).inputs
    assert d.allow_urls is False and d.types == ("png", "jpeg", "webp") and d.video_types == ("mp4", "mov", "webm")


def test_the_scheduler_policy_defaults_to_fair_and_fifo_is_the_rollback(tmp_path):
    assert settings.Settings().scheduler.policy == "fair"
    assert settings.load(env={"BROKER_SCHEDULER_POLICY": "fifo", "BROKER_CONFIG": ""}).scheduler.policy == "fifo"
    (tmp_path / "c.yaml").write_text("scheduler: {policy: lifo}\n")
    with pytest.raises(ValueError, match=r"scheduler\.policy must be one of"):
        settings.load(str(tmp_path / "c.yaml"), env={})
