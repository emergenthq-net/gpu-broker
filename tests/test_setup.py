"""`gpu-broker setup`: detection on fakes, the files it generates, and the whole run with every
machine action faked (nothing here touches systemd, sudo, the network or the real GPU)."""
from __future__ import annotations

import os
import stat
import subprocess

import pytest
import yaml

from gpu_broker import settings
from gpu_broker.catalog import Catalog, validate
from gpu_broker.constants import TOKEN_ENV
from gpu_broker.setup import Machine, Options, detect, generate, host, main
from gpu_broker.setup.detect import COMFYUI, LLAMA, OLLAMA, VLLM, Container, FoundGpu, Probes

GB = 1024 ** 3
GPU = FoundGpu("nvidia", "nvidia (nvidia-smi, card 0)", 24564, "NVIDIA GeForce RTX 4090, 24 GB")
LLAMA_MODELS = {"object": "list", "data": [{"id": "qwen3-8b-q4.gguf", "owned_by": "llamacpp",
                                            "meta": {"size": 5 * GB}}]}
OLLAMA_TAGS = {"models": [{"name": "llama3.1:8b", "size": 4 * GB}, {"name": "qwen2.5-coder:7b", "size": 4 * GB}]}
VLLM_MODELS = {"data": [{"id": "Qwen/Qwen3-8B", "owned_by": "vllm"}]}
COMFY_STATS = {"system": {"os": "posix"}}
CHECKPOINTS = ["sd_xl_base_1.0.safetensors", "v1-5-pruned.safetensors", "juggernautXL_v9.safetensors"]
UNIT_FILES = """llama-server.service enabled enabled
ollama.service disabled enabled
vllm.service disabled enabled
comfyui.service enabled enabled
llama@.service static -
sshd.service enabled enabled
"""
LIST_UNITS = ("systemctl", "list-unit-files", "--type=service", "--no-legend", "--plain")


def probes(http=None, units="", user_units="", containers=None, gpu=GPU):
    http = http or {}

    def get_gpu():
        if gpu is None:
            raise RuntimeError("no GPU found")
        return gpu

    def run(argv):
        if tuple(argv) == LIST_UNITS:
            return units
        if tuple(argv) == (LIST_UNITS[0], "--user", *LIST_UNITS[1:]):
            return user_units
        return None
    return Probes(get_gpu, http.get, run, lambda: containers)


def everything():
    return probes({"http://127.0.0.1:8080/v1/models": LLAMA_MODELS, "http://127.0.0.1:11434/api/tags": OLLAMA_TAGS,
                   "http://127.0.0.1:8188/system_stats": COMFY_STATS,
                   "http://127.0.0.1:8188/models/checkpoints": CHECKPOINTS}, units=UNIT_FILES)


# Detection.

def test_finds_servers_on_their_ports_and_pairs_them_with_their_units():
    f = detect.find(everything())
    by = {s.kind: s for s in f.servers}
    assert set(by) == {LLAMA, OLLAMA, COMFYUI}
    assert by[LLAMA].unit == "llama-server" and by[OLLAMA].unit == "ollama"   # "ollama" contains "llama"
    assert by[COMFYUI].unit == "comfyui"
    assert [m.name for m in by[OLLAMA].models] == ["llama3.1:8b", "qwen2.5-coder:7b"]
    assert by[LLAMA].models[0].size_mib == 5 * 1024
    assert f.idle() == ["vllm"]                 # installed, not answering: a hint, not an entry
    assert f.checkpoints == CHECKPOINTS
    assert "llama@" not in [u.name for u in f.units] and "sshd" not in [u.name for u in f.units]


def test_owned_by_says_which_openai_server_answered():
    f = detect.find(probes({"http://127.0.0.1:8000/v1/models": VLLM_MODELS,
                            "http://127.0.0.1:8080/v1/models": {"data": [{"id": "m", "owned_by": "vllm"}]}}))
    assert [s.kind for s in f.servers] == [VLLM, VLLM]


def test_odd_answers_are_not_servers():
    f = detect.find(probes({"http://127.0.0.1:8080/v1/models": {"data": "nope"},
                            "http://127.0.0.1:11434/api/tags": ["x"], "http://127.0.0.1:8188/system_stats": {}}))
    assert f.servers == []


def test_no_gpu_is_reported_not_raised():
    f = detect.find(probes(gpu=None))
    assert f.gpu is None and "no GPU" in f.gpu_error


def test_containers_stand_in_for_units():
    f = detect.find(probes({"http://127.0.0.1:11434/api/tags": OLLAMA_TAGS},
                           containers=[Container("llm", "ollama/ollama:latest"), Container("db", "postgres")]))
    assert f.servers[0].container == "llm" and [c.name for c in f.containers] == ["llm"]
    assert generate.driver_kind(f) == "docker"


# What it writes.

LAY = generate.Layout("/etc/gb", "/var/lib/gb", "/var/log/gb", True)


def test_generated_catalog_validates_and_estimates_safely():
    f = detect.find(everything())
    cat = generate.catalog(f)
    validate(cat)
    models = cat["models"]
    assert cat["defaults"]["resident"] == "llama3.1-8b"        # first server found: Ollama
    assert models["llama3.1-8b"]["health_path"] == "/api/version" and models["llama3.1-8b"]["unit"] == "ollama"
    assert models["qwen3-8b-q4.gguf"]["vram_mib"] == 7000           # 5 GiB * 1.15 + 1 GiB, rounded up to 100
    assert sorted(k for k, m in models.items() if m["kind"] == "image") == ["juggernautxl_v9", "sd_xl_base_1.0"]
    assert cat["defaults"]["vram_total_mib"] == GPU.total_mib


def test_unknown_sizes_assume_the_whole_card_and_vllm_its_share():
    assert generate.vram_estimate(LLAMA, None, 24564) == 24564 - generate.RESERVE_MIB
    assert generate.vram_estimate(VLLM, None, 24564) == 22200
    assert generate.vram_estimate(LLAMA, 40000, 24564) == 24564 - generate.RESERVE_MIB   # never over the card


def test_generated_config_loads_and_names_what_was_found(tmp_path):
    f = detect.find(everything())
    p = tmp_path / "c.yaml"
    p.write_text(generate.text("config", generate.config(f, LAY), LAY))
    cfg = settings.load(str(p), env={})
    assert cfg.comfy.unit and cfg.comfy.unit.key == "comfyui"
    assert cfg.driver.kind == "systemd" and cfg.driver.options["user"] is False
    assert cfg.catalog == "/etc/gb/catalog.yaml" and cfg.db == "/var/lib/gb/broker.db"
    assert cfg.ui.gpu_label == GPU.label and set(cfg.ui.groups) == {"llama-server", "ollama", "comfyui"}


def test_user_units_set_driver_user():
    f = detect.find(probes({"http://127.0.0.1:8080/v1/models": LLAMA_MODELS}, user_units="llama-server.service x\n"))
    assert generate.config(f, LAY)["driver"]["user"] is True


def test_nothing_found_means_the_starter_catalog():
    assert generate.catalog(detect.find(probes())) is None


# The whole run, faked.

class Fake:
    """A Machine whose every action is recorded instead of done."""

    def __init__(self, tmp_path, monkeypatch, *, euid=1000, sudo=False, systemd=False, healthy_after=0,
                 interactive=False, prefix="/opt/venv", found=None):
        self.cmds, self.opened, self.started = [], [], []
        self.t = 0.0
        self.polls = 0
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        for k in ("SSH_CONNECTION", "SSH_TTY"):
            monkeypatch.delenv(k, raising=False)
        self.env = {"XDG_CONFIG_HOME": str(tmp_path / "cfg"), "XDG_DATA_HOME": str(tmp_path / "data"),
                    "XDG_STATE_HOME": str(tmp_path / "state"), "DISPLAY": ":0"}

        def healthy(base):
            self.polls += 1
            return self.polls > healthy_after

        def run(argv, **kw):
            if argv[:3] == ["sudo", "-n", "test"]:      # reads through sudo: answered from the disk
                return subprocess.CompletedProcess(argv, 0 if os.path.exists(argv[-1]) else 1, "", "")
            self.cmds.append(list(argv))
            out = str(tmp_path / "uvbin") + "\n" if argv[:3] == ["uv", "tool", "dir"] else ""
            return subprocess.CompletedProcess(argv, 0, out, "")

        def sleep(s):
            self.t += s

        self.m = Machine(probes=lambda: found or everything(), euid=euid, can_sudo=lambda: sudo,
                         systemd=lambda: systemd, run=run, healthy=healthy, sleep=sleep, clock=lambda: self.t,
                         opener=self.opened.append, ask=lambda q: "n", interactive=interactive,
                         platform="linux", prefix=prefix, exe=["/opt/venv/bin/gpu-broker"],
                         which=lambda name: "/usr/bin/" + name,
                         start=lambda argv, env: self.started.append((argv, env)) or Proc(),
                         system_dirs=(str(tmp_path / "etc"), str(tmp_path / "lib"), str(tmp_path / "log")),
                         unit_path=str(tmp_path / "gpu-broker.service"))
        self.out: list[str] = []

    def __call__(self, **opts):
        self.out = []
        return main(Options(**opts), self.env, self.m, self.out.append)

    def text(self):
        return "\n".join(self.out)


class Proc:
    def wait(self):
        return 0


@pytest.fixture
def check_ok(monkeypatch):
    from gpu_broker import cli
    monkeypatch.setattr(cli, "check", lambda cfg: 0)


def test_without_root_it_writes_user_files_and_prints_the_serve_command(tmp_path, monkeypatch, check_ok):
    fake = Fake(tmp_path, monkeypatch)
    assert fake(yes=True) == 0
    cfg_dir = tmp_path / "cfg" / "gpu-broker"
    env_file = cfg_dir / "broker.env"
    token = env_file.read_text().strip().partition("=")[2]
    assert stat.S_IMODE(os.stat(env_file).st_mode) == 0o600 and len(token) >= 32
    assert token not in fake.text() and token[:4] + "…" in fake.text()
    Catalog(str(cfg_dir / "catalog.yaml"))
    assert settings.load(str(cfg_dir / "config.yaml"), env={}).db.startswith(str(tmp_path / "data"))
    assert f"serve    set -a; . {env_file}; set +a; /opt/venv/bin/gpu-broker -c {cfg_dir}/config.yaml serve" in fake.out
    assert fake.cmds == [] and fake.started == [] and fake.opened == []   # --yes never runs it in the foreground
    assert "`sudo gpu-broker setup` installs it as a service" in fake.text()   # the units found are system ones


def test_running_it_twice_keeps_every_file_and_the_token(tmp_path, monkeypatch, check_ok):
    fake = Fake(tmp_path, monkeypatch)
    fake(yes=True)
    cfg_dir = tmp_path / "cfg" / "gpu-broker"
    (cfg_dir / "catalog.yaml").write_text((cfg_dir / "catalog.yaml").read_text() + "# mine\n")
    before = {p.name: p.read_text() for p in cfg_dir.iterdir()}
    assert fake(yes=True) == 0
    assert {p.name: p.read_text() for p in cfg_dir.iterdir()} == before
    assert fake.text().count("setup never replaces it") == 2 and "(kept, in" in fake.text()


def test_dry_run_changes_nothing(tmp_path, monkeypatch):
    fake = Fake(tmp_path, monkeypatch, euid=0, systemd=True)
    assert fake(dry_run=True) == 0
    assert not (tmp_path / "etc").exists() and not (tmp_path / "gpu-broker.service").exists()
    assert fake.cmds == []
    text = fake.text()
    assert f"would    write    {tmp_path}/etc/config.yaml (mode 644)" in text
    assert f"would    write    {tmp_path}/etc/broker.env (mode 600)" in text
    assert "would    run      systemctl enable --now gpu-broker" in text


def test_as_root_with_systemd_it_installs_starts_and_opens_the_dashboard(tmp_path, monkeypatch, check_ok):
    fake = Fake(tmp_path, monkeypatch, euid=0, systemd=True, healthy_after=2)
    assert fake() == 0
    unit = (tmp_path / "gpu-broker.service").read_text()
    assert f"ExecStart=/opt/venv/bin/gpu-broker -c {tmp_path}/etc/config.yaml serve" in unit
    assert f"EnvironmentFile={tmp_path}/etc/broker.env" in unit
    assert fake.cmds == [["systemctl", "daemon-reload"], ["systemctl", "enable", "gpu-broker"],
                         ["systemctl", "restart", "gpu-broker"]]
    token = (tmp_path / "etc" / "broker.env").read_text().strip().partition("=")[2]
    assert fake.opened == [f"http://127.0.0.1:8095/dash#token={token}"]
    assert "open     http://127.0.0.1:8095/dash" in fake.text() and token not in fake.text()
    fake.cmds.clear()
    assert fake() == 0                           # again: nothing changed, so start (not restart)
    assert fake.cmds == [["systemctl", "enable", "gpu-broker"], ["systemctl", "start", "gpu-broker"]]


def test_over_ssh_the_dashboard_is_only_printed(tmp_path, monkeypatch, check_ok):
    fake = Fake(tmp_path, monkeypatch, euid=0, systemd=True)
    fake.env["SSH_CONNECTION"] = "192.0.2.1 22 192.0.2.2 22"
    assert fake() == 0
    assert fake.opened == [] and "open     http://127.0.0.1:8095/dash" in fake.text()


def test_a_broker_that_never_answers_is_a_problem(tmp_path, monkeypatch, check_ok):
    fake = Fake(tmp_path, monkeypatch, euid=0, systemd=True, healthy_after=10 ** 6)
    assert fake() == 1
    assert "did not answer within 60 s" in fake.text() and fake.opened == []


def test_check_problems_stop_before_the_service(tmp_path, monkeypatch):
    from gpu_broker import cli
    monkeypatch.setattr(cli, "check", lambda cfg: 1)
    fake = Fake(tmp_path, monkeypatch, euid=0, systemd=True)
    assert fake() == 1
    assert fake.cmds == [] and "problem  `check` found problems" in fake.text()


def test_with_passwordless_sudo_every_change_goes_through_sudo(tmp_path, monkeypatch):
    fake = Fake(tmp_path, monkeypatch, sudo=True, systemd=True)
    assert fake(dry_run=True) == 0
    assert "layout   " + str(tmp_path / "etc") in fake.text()          # the system layout, not ~/.config
    h = host.Host(dry=False, sudo=True, run=fake.m.run)
    h.write("/etc/gb/broker.env", "X=1\n", host.PRIVATE)
    assert fake.cmds[-1] == ["sudo", "-n", "install", "-m", "600", "--", "/dev/stdin", "/etc/gb/broker.env"]


def test_interactive_without_root_it_runs_serve_in_the_foreground(tmp_path, monkeypatch, check_ok):
    fake = Fake(tmp_path, monkeypatch, interactive=True)
    assert fake() == 0
    ((argv, env),) = fake.started
    assert argv[-1] == "serve" and env[TOKEN_ENV] and len(fake.opened) == 1


def test_a_uvx_run_installs_itself_for_the_service(tmp_path, monkeypatch, check_ok):
    fake = Fake(tmp_path, monkeypatch, euid=0, systemd=True, prefix="/root/.cache/uv/archive-v0/abc")
    assert fake() == 0
    assert fake.cmds[0][:3] == ["uv", "tool", "install"]
    assert f"ExecStart={tmp_path}/uvbin/gpu-broker -c" in (tmp_path / "gpu-broker.service").read_text()


def test_nothing_found_writes_the_starter_catalog(tmp_path, monkeypatch, check_ok):
    fake = Fake(tmp_path, monkeypatch, found=probes(gpu=None))
    assert fake(yes=True) == 0
    cat = yaml.safe_load((tmp_path / "cfg" / "gpu-broker" / "catalog.yaml").read_text())
    assert "llama-3.1-8b" in cat["models"] and "writing the starter catalog" in fake.text()


def test_the_token_file_keeps_other_lines(tmp_path):
    env = tmp_path / "broker.env"
    env.write_text("UPSTREAM_TOKEN_X=abc\n")
    tok, new = host.token(host.Host(dry=False, sudo=False), str(env))
    assert new and env.read_text() == f"UPSTREAM_TOKEN_X=abc\n{TOKEN_ENV}={tok}\n"
    assert host.token(host.Host(dry=False, sudo=False), str(env)) == (tok, False)
