"""Host drivers with subprocess replaced by a recorder: argv, allowlists, parsing, files."""
import dataclasses
import subprocess

import pytest

from gpu_broker import drivers, settings
from gpu_broker.constants import Verb
from gpu_broker.drivers.local import DockerDriver, SystemdDriver, group_of
from gpu_broker.drivers.proxmox import ProxmoxDriver
from gpu_broker.units import unit_ref
from tests.helpers import amdgpu_fixture

T = settings.Timeouts()
CAT_UNITS = [unit_ref("llm-a"), unit_ref({"name": "llm-b"})]


class Rec:
    """Stands in for drivers.run: records argv, answers from {substring of argv: stdout}."""

    def __init__(self, out=None, rc=0, missing=()):
        self.calls, self.out, self.rc, self.missing = [], out or {}, rc, missing

    def __call__(self, cmd, timeout):
        self.calls.append(cmd)
        if cmd[0] in self.missing:
            raise FileNotFoundError(cmd[0])
        key = next((k for k in self.out if k in " ".join(cmd)), None)
        return subprocess.CompletedProcess(cmd, self.rc, self.out.get(key, ""), "")


def local(cls, tmp_path, rec=None, **kw):
    return cls(allowed=frozenset({"llm-a", "comfyui"}), timeouts=T, sample_s=0, models_root=str(tmp_path / "models"),
               run=rec or Rec(), sleep=lambda _: None, **kw)


def cfg(kind, allowed=None, **options):
    return dataclasses.replace(settings.Settings(), driver=settings.Driver(kind, allowed, options),
                               comfy=settings.Comfy(unit=unit_ref("comfyui")))


@pytest.mark.parametrize("kind,cls,opts", [("systemd", SystemdDriver, {}), ("docker", DockerDriver, {}),
                                           ("proxmox", ProxmoxDriver, {"ssh_target": "root@pve"})])
def test_build_dispatches_on_kind(kind, cls, opts):
    assert type(drivers.build(cfg(kind, **opts), CAT_UNITS)) is cls


def test_build_rejects_unknown_kinds_and_options():
    with pytest.raises(ValueError, match="unknown driver"):
        drivers.build(cfg("kubernetes"), CAT_UNITS)
    with pytest.raises(TypeError):
        drivers.build(cfg("systemd", typo_option=1), CAT_UNITS)


def test_default_allowlist_is_catalog_units_plus_comfy():
    assert drivers.build(cfg("systemd"), CAT_UNITS).allowed == {"llm-a", "llm-b", "comfyui"}
    assert drivers.build(cfg("docker", allowed=(unit_ref("x"),)), CAT_UNITS).allowed == {"x"}


def test_systemd_argv(tmp_path):
    rec = Rec({"is-active": "active\n"})
    d = local(SystemdDriver, tmp_path, rec, sudo=True)
    assert d.unit("llm-a", Verb.IS_ACTIVE) is True and d.unit("llm-a", Verb.START) is True
    assert rec.calls == [["sudo", "-n", "systemctl", "is-active", "--", "llm-a"], ["sudo", "-n", "systemctl", "start", "--", "llm-a"]]
    with pytest.raises(PermissionError):
        d.unit("other", Verb.START)
    d.allowed = None   # without an allowlist the target check is what refuses
    with pytest.raises(ValueError, match="drop target"):
        d.unit({"name": "llm-a", "target": 1}, Verb.START)
    assert rec.calls[2:] == []


def test_docker_argv(tmp_path):
    rec = Rec({"inspect": "true\n"})
    d = local(DockerDriver, tmp_path, rec)
    assert d.unit("llm-a", Verb.IS_ACTIVE) and d.unit("llm-a", Verb.STOP)
    assert rec.calls[1] == ["docker", "stop", "-t", str(T.container_stop_s), "--", "llm-a"]


def test_local_gpu_parsing_and_sample_line(tmp_path):
    rec = Rec({"compute-apps": "123, 7000\nbad\n", "power.draw": "8000, 24564, 37, 120.5, 50, 2500\n",
               "query-gpu=memory": "8000, 24564, 37\n"})
    d = local(SystemdDriver, tmp_path, rec, group=lambda pid: "unit-" + pid)
    assert d.gpu() == (8000, 24564, 37)
    assert next(d.gpu_stream()) == "8000,24564,37,120.5,50,2500|unit-123:7000"
    assert d.gpu_probe() == "nvidia (nvidia-smi, card 0)" and rec.calls[0] == ["nvidia-smi", "-L"]


def test_local_drivers_read_an_amd_card_when_nvidia_smi_is_not_installed(tmp_path):
    amd = amdgpu_fixture() / "rdna3"
    d = local(DockerDriver, tmp_path, Rec(missing={"nvidia-smi"}), sys_root=str(amd / "sys"), proc_root=str(amd / "proc"))
    assert d.gpu() == (8192, 24576, 99) and d.gpu_probe().startswith("amd (amdgpu sysfs, card0")
    assert d.sample_line() == "8192,24576,99,287.0,64,|llama-server:6144 comfyui:1024"   # groups from the fixture's cgroups


def test_build_passes_the_gpu_section_to_local_drivers():
    c = dataclasses.replace(cfg("systemd"), gpu=settings.Gpu("amd", 2))
    with pytest.raises(RuntimeError, match=r"gpu\.index 2: 0 amdgpu card"):
        drivers.build(dataclasses.replace(c, driver=settings.Driver("systemd", None, {"sys_root": "/nonexistent"})),
                      CAT_UNITS).gpu_probe()


def test_group_of_reads_the_cgroup(tmp_path):
    (tmp_path / "1").mkdir()
    (tmp_path / "1" / "cgroup").write_text("0::/system.slice/docker-0123456789abcdef.scope\n")
    (tmp_path / "2").mkdir()
    (tmp_path / "2" / "cgroup").write_text("0::/system.slice/llama.service\n")
    assert group_of("1", str(tmp_path)) == "0123456789ab"
    assert group_of("2", str(tmp_path)) == "llama"
    assert group_of("3", str(tmp_path)) == "host"


def test_downloads_stay_under_the_models_root(tmp_path):
    rec = Rec()
    d = local(SystemdDriver, tmp_path, rec)
    d.download("hf", "org/repo", "m", ["*.gguf"])
    d.download("gh", "https://github.com/o/r", "g", [])
    root = (tmp_path / "models").resolve()
    assert rec.calls[0] == ["hf", "download", "org/repo", "--include", "*.gguf", "--local-dir", str(root / "m")]
    assert rec.calls[1] == ["git", "clone", "--depth", "1", "--", "https://github.com/o/r", str(root / "g")]
    with pytest.raises(ValueError):
        d.download("hf", "org/repo", "../m", [])
    assert len(rec.calls) == 2


def test_link_confines_both_ends(tmp_path):
    comfy = tmp_path / "comfy"
    (comfy / "loras").mkdir(parents=True)
    d = local(SystemdDriver, tmp_path, comfy_models_dir=str(comfy))
    (tmp_path / "models" / "m").mkdir(parents=True)
    (tmp_path / "models" / "m" / "x.safetensors").write_text("w")
    d.link("m/x.safetensors", "loras")
    assert (comfy / "loras" / "x.safetensors").resolve() == (tmp_path / "models" / "m" / "x.safetensors").resolve()
    for rel, sub in [("../secret", "loras"), ("m/x.safetensors", "../../etc")]:
        with pytest.raises(ValueError):
            d.link(rel, sub)
    with pytest.raises(FileNotFoundError):
        d.link("m/absent.safetensors", "loras")


def test_proxmox_argv_and_stream():
    rec = Rec({" unit ": "active\n", " gpu": "1,2,3\n"})

    class Proc:
        stdout = iter(["a\n", "b\n"])

        class stderr:
            @staticmethod
            def read():
                return "gone"

        def kill(self):
            pass

    d = ProxmoxDriver("root@pve", frozenset({"7:llm"}), T, run=rec, popen=lambda *a, **k: Proc())
    assert d.unit({"name": "llm", "target": 7}, Verb.IS_ACTIVE) is True
    assert rec.calls[0][-6:] == ["--", "root@pve", "unit", "7", "llm", "is-active"]
    assert d.gpu() == (1, 2, 3) and "host/gpu-broker-gpu" in d.gpu_probe()
    stream = d.gpu_stream()
    assert [next(stream), next(stream)] == ["a\n", "b\n"]
    with pytest.raises(RuntimeError, match="gone"):
        next(stream)
    with pytest.raises(PermissionError):
        d.unit("llm", Verb.START)
    d.allowed = None
    with pytest.raises(ValueError, match="container id"):
        d.unit("llm", Verb.START)
    with pytest.raises(ValueError):
        d.download("hf", "org/repo", "m", ["$(reboot)"])


def test_download_refuses_a_slug_that_is_a_symlink_out_of_the_root(tmp_path):
    root = tmp_path / "models"
    root.mkdir()
    (tmp_path / "outside").mkdir()
    (root / "escape").symlink_to(tmp_path / "outside")
    rec = Rec()
    with pytest.raises(ValueError, match="escapes"):
        local(SystemdDriver, tmp_path, rec).download("hf", "org/repo", "escape", [])
    assert rec.calls == []
