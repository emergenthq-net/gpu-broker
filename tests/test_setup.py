"""`gpu-broker setup`: detection on fakes, the files it generates, and the whole run with every
machine action faked (nothing here touches systemd, sudo, the network or the real GPU)."""
from __future__ import annotations

import os
import pathlib
import signal
import stat
import subprocess
import sys

import pytest
import yaml

from gpu_broker import settings
from gpu_broker.catalog import Catalog, validate
from gpu_broker.constants import TOKEN_ENV
from gpu_broker.setup import Machine, Options, detect, generate, host, main, service
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


def probes(http=None, units="", user_units="", containers=None, gpu=GPU, active=(), serves=None):
    http, serves = http or {}, serves or {}

    def get_gpu():
        if gpu is None:
            raise RuntimeError("no GPU found")
        return gpu

    def run(argv):
        if tuple(argv) == LIST_UNITS:
            return units
        if tuple(argv) == (LIST_UNITS[0], "--user", *LIST_UNITS[1:]):
            return user_units
        if "is-active" in argv:
            return "active\n" if argv[-1] in active else None
        return None
    return Probes(get_gpu, http.get, run, lambda: containers, lambda unit, port: serves.get(unit.name))


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
    assert cfg.catalog == "/var/lib/gb/catalog.yaml" and cfg.db == "/var/lib/gb/broker.db"   # the service rewrites it
    assert cfg.ui.gpu_label == GPU.label and set(cfg.ui.groups) == {"llama-server", "ollama", "comfyui"}


def test_user_units_set_driver_user():
    f = detect.find(probes({"http://127.0.0.1:8080/v1/models": LLAMA_MODELS}, user_units="llama-server.service x\n"))
    assert generate.config(f, LAY)["driver"]["user"] is True


def test_nothing_found_means_the_starter_catalog():
    assert generate.catalog(detect.find(probes())) is None


def test_a_unit_shown_to_hold_the_port_wins_and_one_shown_not_to_is_never_picked():
    two = "llama-server.service enabled enabled\nllama-big.service enabled enabled\n"
    http = {"http://127.0.0.1:8080/v1/models": LLAMA_MODELS}
    f = detect.find(probes(http, units=two, serves={"llama-big": True}))
    assert f.servers[0].unit == "llama-big" and f.servers[0].rivals == ()
    f = detect.find(probes(http, units=two, serves={"llama-big": False}))
    assert f.servers[0].unit == "llama-server"
    f = detect.find(probes(http, units="llama-big.service x\n", serves={"llama-big": False}))
    assert f.servers[0].unit is None                     # the only candidate is shown not to serve it


def test_a_running_unit_beats_a_stopped_one_and_a_tie_is_reported(tmp_path, monkeypatch):
    two = "llama-server.service enabled enabled\nllama-big.service enabled enabled\n"
    http = {"http://127.0.0.1:8080/v1/models": LLAMA_MODELS}
    f = detect.find(probes(http, units=two, active={"llama-server"}))
    assert (f.servers[0].unit, f.servers[0].rivals, f.servers[0].unit_inactive) == ("llama-server", (), False)
    f = detect.find(probes(http, units=two))
    assert (f.servers[0].unit, f.servers[0].rivals) == ("llama-big", ("llama-server",))
    out: list[str] = []
    from gpu_broker.setup import report
    report(f, out.append)
    assert any("llama-big, llama-server could each run llama.cpp" in line for line in out)


def fake_proc(tmp_path, port, held_by_pid):
    """A /proc and cgroup tree: pid 41 in the unit's cgroup; the listening socket's inode 777."""
    proc, cg = tmp_path / "proc", tmp_path / "cg"
    (proc / "net").mkdir(parents=True)
    head = "  sl  local_address rem_address   st tx_queue rx_queue tr tm->when retrnsmt   uid  timeout inode\n"
    (proc / "net" / "tcp").write_text(head + f"   0: 00000000:{port:04X} 00000000:0000 0A 00000000:00000000 00:00000000 "
                                              "00000000  1000        0 777 1\n")
    (proc / "net" / "tcp6").write_text(head)
    (proc / "41" / "fd").mkdir(parents=True)
    os.symlink(f"socket:[{777 if held_by_pid else 555}]", proc / "41" / "fd" / "3")
    (cg / "system.slice" / "llama-server.service").mkdir(parents=True)
    (cg / "system.slice" / "llama-server.service" / "cgroup.procs").write_text("41\n")
    return str(proc), str(cg)


@pytest.mark.parametrize("held", [True, False])
def test_unit_serves_looks_at_the_sockets_of_the_units_processes(tmp_path, held):
    proc, cg = fake_proc(tmp_path, 8080, held)

    def run(argv):
        return "/system.slice/llama-server.service\n" if "ControlGroup" in argv else "/usr/bin/llama-server\n"
    assert detect.unit_serves(detect.Unit("llama-server"), 8080, run, proc, cg) is held


def test_unit_serves_falls_back_to_the_command_line(tmp_path):
    def run(argv):
        return "" if "ControlGroup" in argv else "{ path=/usr/bin/llama-server ; argv[]=llama-server --port 8080 }\n"
    unit = detect.Unit("llama-server")
    assert detect.unit_serves(unit, 8080, run, str(tmp_path), str(tmp_path)) is True
    assert detect.unit_serves(unit, 18080, run, str(tmp_path), str(tmp_path)) is None   # 8080 is not 18080


# Host: writes, commands, the token, unit quoting.

def test_writes_go_through_a_new_file_renamed_into_place(tmp_path):
    target = tmp_path / "broker.env"
    target.write_text("old\n")
    before = os.stat(target).st_ino
    host.Host(dry=False, sudo=False).write(target, "new\n", host.PRIVATE)
    assert target.read_text() == "new\n" and os.stat(target).st_ino != before
    assert stat.S_IMODE(os.stat(target).st_mode) == 0o600 and os.listdir(tmp_path) == ["broker.env"]


@pytest.mark.parametrize("dangling", [True, False])
def test_a_symlink_is_never_written_through(tmp_path, dangling):
    outside = tmp_path / "outside"
    if not dangling:
        outside.write_text("theirs\n")
    (tmp_path / "broker.env").symlink_to(outside)
    h = host.Host(dry=False, sudo=False)
    with pytest.raises(host.UnsafePath):
        h.write(tmp_path / "broker.env", "BROKER_TOKEN=x\n", host.PRIVATE)
    assert outside.exists() is not dangling and (dangling or outside.read_text() == "theirs\n")
    assert h.exists(str(tmp_path / "broker.env")) and h.read(str(tmp_path / "broker.env")) is None


def test_through_sudo_a_write_is_a_new_root_file_moved_into_place():
    cmds = []

    def run(argv, **kw):
        cmds.append(argv)
        if argv[3:4] == ["-L"] or argv[3:4] == ["-d"]:
            return subprocess.CompletedProcess(argv, 1, "", "")
        return subprocess.CompletedProcess(argv, 0, "/etc/gb/.broker.env.Ab12Cd34\n" if "mktemp" in argv else "", "")
    host.Host(dry=False, sudo=True, run=run).write("/etc/gb/broker.env", "X=1\n", host.PRIVATE, owner="svc")
    tmp = "/etc/gb/.broker.env.Ab12Cd34"
    assert cmds[2:] == [["sudo", "-n", "mktemp", "--", "/etc/gb/.broker.env.XXXXXXXX"], ["sudo", "-n", "tee", "--", tmp],
                        ["sudo", "-n", "chmod", "600", "--", tmp], ["sudo", "-n", "chown", "--", "svc:", tmp],
                        ["sudo", "-n", "mv", "-fT", "--", tmp, "/etc/gb/broker.env"]]


@pytest.mark.parametrize("err", [subprocess.TimeoutExpired(["x"], 60), FileNotFoundError(2, "No such file")])
def test_a_command_that_cannot_run_is_a_failed_result(err):
    def run(argv, **kw):
        raise err
    r = host.Host(dry=False, sudo=False, run=run).cmd(["systemctl", "restart", "gpu-broker"])
    assert r.returncode == host.NO_RESULT and r.stderr.startswith("systemctl: ")


def test_the_env_file_is_read_as_systemd_reads_it():
    assert host.parse_env('# c\n; c\nexport BROKER_TOKEN="a b"\nX=1\nBROKER_TOKEN=\'last\'\n\n') == \
        {TOKEN_ENV: "last", "X": "1"}


def test_the_token_file_keeps_other_lines(tmp_path):
    env = tmp_path / "broker.env"
    env.write_text("UPSTREAM_TOKEN_X=abc\n")
    tok, new = host.token(host.Host(dry=False, sudo=False), str(env))
    assert new and env.read_text() == f"UPSTREAM_TOKEN_X=abc\n{TOKEN_ENV}={tok}\n"
    assert host.token(host.Host(dry=False, sudo=False), str(env)) == (tok, False)


def test_a_quoted_exported_token_is_kept_and_a_blank_one_replaced_not_doubled(tmp_path):
    env = tmp_path / "broker.env"
    env.write_text(f'export {TOKEN_ENV}="kept-token"\n')
    assert host.token(host.Host(dry=False, sudo=False), str(env)) == ("kept-token", False)
    env.write_text(f'{TOKEN_ENV}=""\nX=1\nexport {TOKEN_ENV}=\n')
    tok, new = host.token(host.Host(dry=False, sudo=False), str(env))
    assert new and env.read_text() == f"X=1\n{TOKEN_ENV}={tok}\n"


def test_unit_values_are_escaped_the_systemd_way():
    assert host.exec_word("/opt/gb/bin/gpu-broker") == "/opt/gb/bin/gpu-broker"
    assert host.exec_word("/opt/my gb%1/$x") == '"/opt/my gb%%1/$$x"'
    assert host.exec_word('/a"b\\c') == '"/a\\"b\\\\c"'
    assert host.path_value("/etc/my gb%1/broker.env") == "/etc/my gb%%1/broker.env"
    for bad in ("/a\nb", "relative/broker.env", " /lead"):
        with pytest.raises(host.UnsafePath):
            host.path_value(bad)
    with pytest.raises(host.UnsafePath):
        host.exec_word("/a\nExecStartPre=/bin/evil")


def test_a_unit_for_a_path_with_spaces_and_a_percent_sign():
    unit = service.system_unit(["/opt/my gb%1/bin/gpu-broker"], "/etc/my gb%1/config.yaml",
                               "/etc/my gb%1/broker.env", "alice", ["video", "render"])
    assert 'ExecStart="/opt/my gb%%1/bin/gpu-broker" -c "/etc/my gb%%1/config.yaml" serve' in unit
    assert "EnvironmentFile=/etc/my gb%%1/broker.env\n" in unit
    assert "User=alice\nSupplementaryGroups=video render\n" in unit


# Who may change what the service runs.

def install(tmp_path):
    venv = tmp_path / "venv"
    (venv / "lib" / "site-packages" / "gpu_broker").mkdir(parents=True)
    (venv / "lib" / "site-packages" / "gpu_broker" / "__init__.py").write_text("")
    (venv / "bin").mkdir()
    py = venv / "bin" / "python"
    py.write_text("")
    exe = venv / "bin" / "gpu-broker"
    exe.write_text(f"#!{py}\nimport gpu_broker\n")
    for f in (py, exe):
        f.chmod(0o755)
    return venv, service.Install(str(exe), str(py), str(venv / "lib" / "site-packages"))


def me():
    acct = service.account(os.getuid())
    assert acct is not None
    return acct


def test_an_installation_only_its_owner_can_change_is_safe(tmp_path):
    _, inst = install(tmp_path)
    assert service.Checker(me()).problems(inst, service.walk) == []


def test_a_file_others_can_write_anywhere_in_the_installation_is_not(tmp_path):
    venv, inst = install(tmp_path)
    deep = venv / "lib" / "site-packages" / "gpu_broker" / "__init__.py"
    deep.chmod(0o666)
    assert service.Checker(me()).problems(inst, service.walk) == [f"{os.path.realpath(deep)} is writable by everyone"]


def as_root(paths_under):
    """lstat that reports everything under `paths_under` as root's, world-readable, not group-writable."""
    def lstat(p):
        st = os.lstat(p)
        if not str(p).startswith(str(paths_under)) and not str(paths_under).startswith(str(p)):
            return st
        fields = list(st)
        executable = stat.S_ISDIR(st.st_mode) or st.st_mode & stat.S_IXUSR
        fields[0] = (st.st_mode & ~0o022) | 0o044 | (0o011 if executable else 0)
        fields[4] = fields[5] = 0
        return os.stat_result(fields)
    return lstat


def test_another_account_must_be_able_to_read_it(tmp_path):
    _, inst = install(tmp_path)
    svc = service.Account("gpu-broker", 990, 990, "/var/lib/gpu-broker")
    root_lstat = as_root(tmp_path)
    assert service.Checker(svc, root_lstat).problems(inst, service.walk) == []

    def private_exe(p):
        st = root_lstat(p)
        if str(p).endswith("gpu-broker"):
            return os.stat_result([st.st_mode & ~0o077, *list(st)[1:]])
        return st
    assert service.Checker(svc, private_exe).problems(inst, service.walk) == \
        [f"{os.path.realpath(inst.exe)} is not readable by gpu-broker"]


def test_the_interpreter_comes_from_the_shebang_or_pips_trampoline(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.write_text("#!/opt/v/bin/python3 -I\n")
    b.write_text("#!/bin/sh\n'''exec' \"/opt/my v/bin/python\" \"$0\" \"$@\"\n' '''\n")
    read = host.Host(dry=False, sudo=False).read
    assert service.interpreter([str(a)], read) == "/opt/v/bin/python3"
    assert service.interpreter([str(b)], read) == "/opt/my v/bin/python"
    assert service.interpreter(["/usr/bin/python3", "-m", "gpu_broker"], read) == "/usr/bin/python3"


def test_the_sudoers_rule_names_exactly_the_units():
    text, skipped = service.sudoers("svc", "/usr/bin/systemctl", ["ollama", "comfyui", "ollama", "a:b"])
    assert "svc ALL=(root) NOPASSWD: /usr/bin/systemctl start -- comfyui, /usr/bin/systemctl stop -- comfyui, " \
           "/usr/bin/systemctl is-active -- comfyui, /usr/bin/systemctl start -- ollama," in text
    assert skipped == ["a:b"] and "a:b" not in text


# The whole run, faked.

class Fake:
    """A Machine whose every command is recorded instead of run. The installation is a real
    venv under tmp_path, owned by whoever runs the tests (or reported as root's)."""

    def __init__(self, tmp_path, monkeypatch, *, euid=None, sudo=False, systemd=False, healthy_after=0,
                 interactive=False, prefix=None, found=None, root_owned=False, linger="yes", fail=(),
                 child_exits=None):
        self.cmds, self.kw, self.opened, self.started, self.stopped = [], [], [], [], []
        self.t = 0.0
        self.polls = 0
        self.dedicated = None
        self.tmp = tmp_path
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        for k in ("SSH_CONNECTION", "SSH_TTY"):
            monkeypatch.delenv(k, raising=False)
        self.env = {"XDG_CONFIG_HOME": str(tmp_path / "cfg"), "XDG_DATA_HOME": str(tmp_path / "data"),
                    "XDG_STATE_HOME": str(tmp_path / "state"), "DISPLAY": ":0"}
        venv, self.inst = install(tmp_path)

        def healthy(base):
            self.polls += 1
            return self.polls > healthy_after

        def run(argv, **kw):
            if argv[:3] == ["sudo", "-n", "test"]:      # reads through sudo: answered from the disk
                p = argv[-1]
                ok = {"-e": os.path.exists, "-L": os.path.islink, "-d": os.path.isdir}[argv[3]](p)
                return subprocess.CompletedProcess(argv, 0 if ok else 1, "", "")
            if argv[:3] == ["sudo", "-n", "cat"]:
                p = argv[-1]
                return subprocess.CompletedProcess(argv, 0, open(p).read(), "") if os.path.isfile(p) else \
                    subprocess.CompletedProcess(argv, 1, "", "no such file")
            if argv[1:3] == ["-I", "-c"]:
                return subprocess.CompletedProcess(argv, 0, self.inst.site + "\n", "")
            if argv[:2] == ["loginctl", "show-user"]:
                return subprocess.CompletedProcess(argv, 0, linger + "\n", "")
            self.cmds.append(list(argv))
            self.kw.append(kw)
            if any(argv[:len(f)] == list(f) or argv[2:2 + len(f)] == list(f) for f in fail):
                return subprocess.CompletedProcess(argv, 1, "", "boom")
            if argv[0] == "useradd":
                self.dedicated = service.Account("gpu-broker", 990, 990, argv[argv.index("--home-dir") + 1])
            if argv[:3] == ["mv", "-fT", "--"]:          # the staged sudoers rule moving into place
                os.replace(argv[3], argv[4])
            if "mktemp" in argv:
                return subprocess.CompletedProcess(argv, 0, argv[-1].replace("XXXXXXXX", "Ab12Cd34") + "\n", "")
            out = str(tmp_path / "uvbin") + "\n" if argv[:3] == ["uv", "tool", "dir"] else ""
            return subprocess.CompletedProcess(argv, 0, out, "")

        def sleep(s):
            self.t += s

        def account(key):
            if key == "gpu-broker":
                return self.dedicated
            return service.account(os.getuid() if key in (0, 1000) else key)

        def start(argv, env):
            self.started.append((argv, env))
            return Child(self, child_exits)

        self.m = Machine(probes=lambda: found or everything(), euid=os.getuid() if euid is None else euid,
                         can_sudo=lambda: sudo, systemd=lambda: systemd, run=run, healthy=healthy, sleep=sleep,
                         clock=lambda: self.t, opener=self.opened.append, ask=lambda q: "n", interactive=interactive,
                         platform="linux", prefix=prefix or str(venv), exe=[self.inst.exe],
                         which=lambda name: "/usr/bin/" + name, start=start,
                         system_dirs=(str(tmp_path / "etc"), str(tmp_path / "lib"), str(tmp_path / "log")),
                         unit_path=str(tmp_path / "gpu-broker.service"), sudoers_path=str(tmp_path / "sudoers-gb"),
                         account=account, group_exists=lambda g: g in ("video", "render", "docker"),
                         lstat=as_root(venv) if root_owned else os.lstat)
        self.out: list[str] = []

    def __call__(self, **opts):
        self.out = []
        return main(Options(**opts), self.env, self.m, self.out.append)

    def text(self):
        return "\n".join(self.out)


class Child:
    def __init__(self, fake, exits_with):
        self.fake, self.code = fake, exits_with

    def poll(self):
        return self.code

    def wait(self):
        return 0

    def stop(self):
        self.fake.stopped.append(True)

    def tail(self):
        return ["Traceback (most recent call last):", "OSError: [Errno 98] Address already in use"]


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
    assert f"serve    set -a; . {env_file}; set +a; {fake.inst.exe} -c {cfg_dir}/config.yaml serve" in fake.out
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


def test_as_root_the_service_runs_as_the_installations_owner_never_root(tmp_path, monkeypatch, check_ok):
    fake = Fake(tmp_path, monkeypatch, euid=0, systemd=True, healthy_after=2)
    assert fake() == 0
    unit = (tmp_path / "gpu-broker.service").read_text()
    assert f"User={me().name}\nSupplementaryGroups=video render\n" in unit
    assert f"ExecStart={fake.inst.exe} -c {tmp_path}/etc/config.yaml serve" in unit
    assert f"EnvironmentFile={tmp_path}/etc/broker.env" in unit
    assert fake.cmds[-3:] == [["systemctl", "daemon-reload"], ["systemctl", "enable", "gpu-broker"],
                              ["systemctl", "restart", "gpu-broker"]]
    token = (tmp_path / "etc" / "broker.env").read_text().strip().partition("=")[2]
    assert fake.opened == [f"http://127.0.0.1:8095/dash#token={token}"]
    assert "open     http://127.0.0.1:8095/dash" in fake.text() and token not in fake.text()
    assert (tmp_path / "lib" / "catalog.yaml").is_file() and not (tmp_path / "etc" / "catalog.yaml").exists()
    fake.cmds.clear()
    assert fake() == 0                           # again: nothing changed, so start (not restart)
    assert fake.cmds == [["systemctl", "enable", "gpu-broker"], ["systemctl", "start", "gpu-broker"]]


def test_it_starts_and_stops_system_units_through_a_sudoers_rule_visudo_checked(tmp_path, monkeypatch, check_ok):
    fake = Fake(tmp_path, monkeypatch, euid=0, systemd=True)
    assert fake() == 0
    rule = (tmp_path / "sudoers-gb").read_text()
    assert f"{me().name} ALL=(root) NOPASSWD: /usr/bin/systemctl start -- comfyui," in rule
    assert "/usr/bin/systemctl stop -- llama-server" in rule and "/usr/bin/systemctl is-active -- ollama" in rule
    staged = f"{tmp_path}/sudoers-gb.new"
    assert fake.cmds[:2] == [["visudo", "-c", "-q", "-f", staged], ["mv", "-fT", "--", staged, f"{tmp_path}/sudoers-gb"]]
    assert stat.S_IMODE(os.stat(tmp_path / "sudoers-gb").st_mode) == 0o440
    assert settings.load(str(tmp_path / "etc" / "config.yaml"), env={}).driver.options["sudo"] is True


def test_a_rule_visudo_rejects_is_removed_and_nothing_starts(tmp_path, monkeypatch, check_ok):
    fake = Fake(tmp_path, monkeypatch, euid=0, systemd=True, fail=[("visudo",)])
    assert fake() == 1
    assert ["rm", "-f", "--", f"{tmp_path}/sudoers-gb.new"] in fake.cmds
    assert not any(c[0] == "systemctl" for c in fake.cmds) and "visudo rejected" in fake.text()


def test_a_root_owned_installation_runs_as_a_dedicated_account(tmp_path, monkeypatch, check_ok):
    fake = Fake(tmp_path, monkeypatch, euid=0, systemd=True, root_owned=True)
    monkeypatch.setattr(host, "_ids", lambda owner: (os.getuid(), os.getgid()))   # chown to an account that is not real
    assert fake() == 0
    assert fake.cmds[0][0] == "useradd" and fake.cmds[0][-1] == "gpu-broker"
    assert "User=gpu-broker\n" in (tmp_path / "gpu-broker.service").read_text()


def test_code_others_could_change_is_refused_before_anything_is_written(tmp_path, monkeypatch, check_ok):
    fake = Fake(tmp_path, monkeypatch, euid=0, systemd=True)
    (pathlib.Path(fake.inst.site) / "gpu_broker" / "__init__.py").chmod(0o666)
    assert fake() == 1
    assert "is writable by everyone" in fake.text() and fake.cmds == []
    assert not (tmp_path / "etc").exists() and not (tmp_path / "gpu-broker.service").exists()


def test_docker_containers_need_the_docker_group(tmp_path, monkeypatch, check_ok):
    found = probes({"http://127.0.0.1:11434/api/tags": OLLAMA_TAGS}, containers=[Container("llm", "ollama/ollama")])
    fake = Fake(tmp_path, monkeypatch, euid=0, systemd=True, found=found)
    assert fake() == 0
    assert "SupplementaryGroups=video render docker\n" in (tmp_path / "gpu-broker.service").read_text()
    assert not (tmp_path / "sudoers-gb").exists()         # no system units to start: no sudoers rule


def test_with_passwordless_sudo_every_change_goes_through_sudo(tmp_path, monkeypatch, check_ok):
    fake = Fake(tmp_path, monkeypatch, sudo=True, systemd=True)
    assert fake() == 0
    assert "layout   " + str(tmp_path / "etc") in fake.text()          # the system layout, not ~/.config
    root_cmds = [c for c in fake.cmds if c[:2] != ["sudo", "-n"]]
    assert root_cmds == []                                            # every change was made through sudo
    catalog = f"{tmp_path}/lib/catalog.yaml"
    assert ["sudo", "-n", "chown", "--", f"{me().name}:", f"{tmp_path}/lib/.catalog.yaml.Ab12Cd34"] in fake.cmds
    assert ["sudo", "-n", "mv", "-fT", "--", f"{tmp_path}/lib/.catalog.yaml.Ab12Cd34", catalog] in fake.cmds
    assert ["sudo", "-n", "chown", "--", f"{me().name}:", f"{tmp_path}/lib/inputs"] in fake.cmds
    assert not any("/etc/" in c[-1] for c in fake.cmds if "chown" in c)


def test_through_sudo_existing_files_are_seen_with_sudo_test(tmp_path, monkeypatch, check_ok):
    fake = Fake(tmp_path, monkeypatch, sudo=True, systemd=True)
    seen = []
    run = fake.m.run

    def root_only(argv, **kw):                   # /etc/gpu-broker/config.yaml exists, but only root can see it
        if argv[:4] == ["sudo", "-n", "test", "-e"] and argv[-1].endswith("etc/config.yaml"):
            seen.append(argv[-1])
            return subprocess.CompletedProcess(argv, 0, "", "")
        return run(argv, **kw)
    fake.m.run = root_only
    assert fake() == 0
    assert seen and f"kept     {tmp_path}/etc/config.yaml" in fake.text()
    assert not any(c[-1] == f"{tmp_path}/etc/config.yaml" for c in fake.cmds)


def test_user_units_get_a_user_service_that_lingers(tmp_path, monkeypatch, check_ok):
    found = probes({"http://127.0.0.1:8080/v1/models": LLAMA_MODELS}, user_units="llama-server.service x\n")
    fake = Fake(tmp_path, monkeypatch, systemd=True, sudo=True, found=found, linger="no")
    assert fake() == 0
    unit = (tmp_path / "cfg" / "systemd" / "user" / "gpu-broker.service").read_text()
    assert "User=" not in unit and "WantedBy=default.target" in unit
    assert fake.cmds == [["systemctl", "--user", "daemon-reload"], ["systemctl", "--user", "enable", "gpu-broker"],
                         ["systemctl", "--user", "restart", "gpu-broker"], ["loginctl", "enable-linger", me().name]]
    cfg = settings.load(str(tmp_path / "cfg" / "gpu-broker" / "config.yaml"), env={})
    assert cfg.driver.options["user"] is True and cfg.driver.options["sudo"] is False


def test_a_user_service_says_when_it_cannot_linger(tmp_path, monkeypatch, check_ok):
    found = probes({"http://127.0.0.1:8080/v1/models": LLAMA_MODELS}, user_units="llama-server.service x\n")
    fake = Fake(tmp_path, monkeypatch, systemd=True, found=found, linger="no", fail=[("loginctl",)])
    assert fake() == 0
    assert f"`sudo loginctl enable-linger {me().name}`" in fake.text()


@pytest.mark.parametrize(("units", "user_units", "euid"), [
    ("ollama.service x\n", "llama-server.service x\n", None),     # a mix: one broker cannot control both
    ("", "llama-server.service x\n", 0),                          # root cannot run a user's units
])
def test_setups_that_cannot_work_are_refused_before_writing(tmp_path, monkeypatch, units, user_units, euid):
    found = probes({"http://127.0.0.1:8080/v1/models": LLAMA_MODELS, "http://127.0.0.1:11434/api/tags": OLLAMA_TAGS},
                   units=units, user_units=user_units)
    fake = Fake(tmp_path, monkeypatch, systemd=True, found=found, euid=euid)
    assert fake() == 1
    assert "problem  " in fake.text() and fake.cmds == []
    assert not (tmp_path / "cfg").exists() and not (tmp_path / "etc").exists()


def test_a_planted_symlink_is_refused(tmp_path, monkeypatch, check_ok):
    fake = Fake(tmp_path, monkeypatch)
    cfg_dir = tmp_path / "cfg" / "gpu-broker"
    cfg_dir.mkdir(parents=True)
    (cfg_dir / "broker.env").symlink_to(tmp_path / "elsewhere")
    assert fake(yes=True) == 1
    assert "is a symlink" in fake.text() and not (tmp_path / "elsewhere").exists()


def test_a_command_that_cannot_run_is_reported_not_raised(tmp_path, monkeypatch, check_ok):
    fake = Fake(tmp_path, monkeypatch, euid=0, systemd=True)

    def missing(argv, **kw):
        if argv[:2] == ["systemctl", "restart"]:
            raise FileNotFoundError(2, "No such file or directory")
        return run(argv, **kw)
    run, fake.m.run = fake.m.run, missing
    assert fake() == 1
    assert "problem  could not start gpu-broker: systemctl: [Errno 2] No such file or directory" in fake.text()


def test_a_uvx_run_installs_itself_with_a_long_timeout(tmp_path, monkeypatch, check_ok):
    fake = Fake(tmp_path, monkeypatch, euid=0, systemd=True, prefix="/root/.cache/uv/archive-v0/abc")
    (tmp_path / "uvbin").mkdir()
    os.symlink(fake.inst.exe, tmp_path / "uvbin" / "gpu-broker")
    assert fake() == 0
    assert fake.cmds[0][:3] == ["uv", "tool", "install"] and fake.kw[0]["timeout"] == host.INSTALL_TIMEOUT_S
    assert f"ExecStart={tmp_path}/uvbin/gpu-broker -c" in (tmp_path / "gpu-broker.service").read_text()


def test_dry_run_changes_nothing(tmp_path, monkeypatch, check_ok):
    fake = Fake(tmp_path, monkeypatch, euid=0, systemd=True)
    assert fake(dry_run=True) == 0
    assert not (tmp_path / "etc").exists() and not (tmp_path / "gpu-broker.service").exists()
    assert fake.cmds == []
    text = fake.text()
    assert f"would    write    {tmp_path}/etc/config.yaml (mode 644)" in text
    assert f"would    write    {tmp_path}/lib/catalog.yaml (mode 644, owner {me().name})" in text
    assert f"would    write    {tmp_path}/etc/broker.env (mode 600)" in text
    assert "would    run      visudo -c -q -f" in text
    assert "would    run      systemctl restart gpu-broker" in text


def test_dry_run_takes_the_same_branches_as_a_real_run(tmp_path, monkeypatch, check_ok):
    fake = Fake(tmp_path, monkeypatch, euid=0, systemd=True)
    assert fake() == 0
    (tmp_path / "gpu-broker.service").write_text("# edited\n")
    fake.cmds.clear()
    assert fake(dry_run=True) == 0
    text = fake.text()
    assert fake.cmds == []
    assert f"kept     {tmp_path}/gpu-broker.service (exists and differs" in text
    assert "would    run      systemctl start gpu-broker" in text and "systemctl restart" not in text
    assert "would    write" not in text and "daemon-reload" not in text


def test_dry_run_of_a_uvx_run_resolves_where_it_would_install(tmp_path, monkeypatch, check_ok):
    fake = Fake(tmp_path, monkeypatch, euid=0, systemd=True, prefix="/root/.cache/uv/archive-v0/abc")
    assert fake(dry_run=True) == 0
    text = fake.text()
    assert "would    run      uv tool install gpu-broker" in text
    assert "would    run      useradd --system" in text      # root's `uv tool install` would be root's
    assert f"would    check that only root or gpu-broker can change {tmp_path}/uvbin/gpu-broker" in text


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


def test_interactive_without_root_it_runs_serve_in_the_foreground(tmp_path, monkeypatch, check_ok):
    fake = Fake(tmp_path, monkeypatch, interactive=True)
    assert fake() == 0
    ((argv, env),) = fake.started
    assert argv[-1] == "serve" and env[TOKEN_ENV] and len(fake.opened) == 1 and fake.stopped == [True]


def test_a_foreground_serve_that_exits_is_reported_at_once(tmp_path, monkeypatch, check_ok):
    fake = Fake(tmp_path, monkeypatch, interactive=True, healthy_after=10 ** 6, child_exits=3)
    assert fake() == 3
    assert "serve exited with code 3 before" in fake.text() and "Address already in use" in fake.text()
    assert fake.polls == 0 and fake.t == 0 and fake.stopped == [True]


def test_a_foreground_serve_that_never_answers_is_stopped(tmp_path, monkeypatch, check_ok):
    fake = Fake(tmp_path, monkeypatch, interactive=True, healthy_after=10 ** 6)
    assert fake() == 1
    assert "did not answer within 60 s" in fake.text() and fake.stopped == [True]


def test_foreground_keeps_the_tail_and_stop_leaves_nothing_running():
    from gpu_broker.setup import Foreground
    code = "import sys; [print(f'line {i}', file=sys.stderr) for i in range(30)]; sys.exit(3)"
    child = Foreground([sys.executable, "-c", code], os.environ)
    assert child.wait() == 3 and child.tail()[-1] == "line 29" and len(child.tail()) == 20
    sleeper = Foreground([sys.executable, "-c", "import time; time.sleep(60)"], os.environ)
    sleeper.stop()
    assert sleeper.poll() == -signal.SIGTERM               # asked to stop (serve shuts down cleanly), not killed


def test_nothing_found_writes_the_starter_catalog(tmp_path, monkeypatch, check_ok):
    fake = Fake(tmp_path, monkeypatch, found=probes(gpu=None))
    assert fake(yes=True) == 0
    cat = yaml.safe_load((tmp_path / "cfg" / "gpu-broker" / "catalog.yaml").read_text())
    assert "llama-3.1-8b" in cat["models"] and "writing the starter catalog" in fake.text()
