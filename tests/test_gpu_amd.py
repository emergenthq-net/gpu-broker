"""The amdgpu probe against fixture sysfs/proc trees written to the kernel's documented formats
(tests/fixtures/amdgpu/README.md): card choice, units, missing hwmon files, fdinfo accounting."""
import os
import pathlib
import shutil

import pytest

from gpu_broker.gpu import GpuSample, Procs, amd
from tests.helpers import FIX

AMD = FIX / "amdgpu"


def probe(name, index=0):
    return amd.AmdProbe(index, str(AMD / name / "sys"), str(AMD / name / "proc"))


def test_vega10_reads_every_value_in_its_unit():
    p = probe("vega10")
    assert p.read() == GpuSample(4096, 16384, 37, 42.0, 51, 1500)
    assert p.pdev == "0000:03:00.0" and "card0" in p.name


def test_vega10_counts_only_amdgpu_files_of_this_card():
    """pid 1201: 3 GiB in KiB; pid 99: drm-resident-vram only; i915, another card, a socket and
    pid 1600's log file (whose fdinfo claims amdgpu VRAM) are skipped."""
    assert probe("vega10").procs() == Procs([("99", 512), ("1201", 3072)])


def test_power1_average_wins_over_power1_input(tmp_path):
    root = tmp_path / "sys"
    shutil.copytree(AMD / "vega10/sys", root)
    (root / "class/drm/card0/device/hwmon/hwmon2/power1_input").write_text("99000000\n")
    assert probe_at(root).read().power_w == 42.0


def test_rdna3_power1_input_and_no_clock():
    assert probe("rdna3").read() == GpuSample(8192, 24576, 99, 287.0, 64, None)


def test_rdna3_one_client_on_two_fds_and_two_processes_counts_once():
    assert probe("rdna3").procs() == Procs([("2001", 6144), ("2002", 1024)])


def test_only_amdgpu_cards_count_and_the_index_selects_among_them(tmp_path):
    root = tmp_path / "sys"
    shutil.copytree(AMD / "vega10/sys", root)
    assert len(amd.cards(str(root))) == 1 and amd.present(str(root))
    for n, used_mib in ((2, 1), (10, 2)):
        shutil.copytree(root / "class/drm/card0", root / f"class/drm/card{n}")
        (root / f"class/drm/card{n}/device/mem_info_vram_used").write_text(f"{used_mib * amd.MIB}\n")
    shutil.copytree(root / "class/drm/card0", root / "class/drm/card3")   # VRAM files, but an Intel vendor id
    (root / "class/drm/card3/device/vendor").write_text("0x8086\n")
    assert [d.parent.name for d in amd.cards(str(root))] == ["card0", "card2", "card10"]   # by number, not by name
    assert [amd.AmdProbe(i, str(root), str(AMD / "vega10/proc")).read().used_mib for i in (1, 2)] == [1, 2]
    with pytest.raises(RuntimeError, match=r"gpu\.index 3: 3 amdgpu card"):
        amd.AmdProbe(3, str(root))


def test_an_amd_vendor_without_vram_files_is_not_an_amdgpu_card(tmp_path):
    root = tmp_path / "sys"
    shutil.copytree(AMD / "vega10/sys", root)
    (root / "class/drm/card0/device/mem_info_vram_total").unlink()
    assert amd.cards(str(root)) == [] and not amd.present(str(root)) and not amd.present(str(tmp_path / "none"))


def test_only_memory_is_required(tmp_path):
    """Runtime PM / BACO: utilisation and hwmon files unreadable are unknown values; memory
    unreadable is an error."""
    root = tmp_path / "sys"
    shutil.copytree(AMD / "vega10/sys", root)
    dev = root / "class/drm/card0/device"
    shutil.rmtree(dev / "hwmon")
    assert probe_at(root).read() == GpuSample(4096, 16384, 37, None, None, None)
    (dev / "gpu_busy_percent").write_text("busy\n")
    assert probe_at(root).read() == GpuSample(4096, 16384, None, None, None, None)
    for f in ("mem_info_vram_used", "mem_info_vram_total"):
        shutil.copy(AMD / "vega10/sys/class/drm/card0/device" / f, dev / f)
        (dev / f).write_text("?\n")
        with pytest.raises(RuntimeError, match="mem_info_vram_used or mem_info_vram_total"):
            probe_at(root).read()
        shutil.copy(AMD / "vega10/sys/class/drm/card0/device" / f, dev / f)


def test_without_a_pci_address_every_amdgpu_file_counts(tmp_path):
    root = tmp_path / "sys"
    shutil.copytree(AMD / "vega10/sys", root)
    (root / "class/drm/card0/device/uevent").unlink()
    assert dict(probe_at(root).procs().rows) == {"99": 512, "1201": 3072, "1500": 1024}


@pytest.mark.parametrize(("raw", "mib"), [("2097152", 2), ("2048 KiB", 2), ("2 MiB", 2), ("2 GiB", None), ("x KiB", None)])
def test_vram_units(raw, mib):
    b = amd._bytes({"drm-memory-vram": raw})
    assert (None if b is None else b // amd.MIB) == mib


def probe_at(root):
    return amd.AmdProbe(0, str(root), str(AMD / "vega10/proc"))


def test_only_fds_linking_into_dev_dri_are_opened(monkeypatch):
    """listdir + readlink pick the DRM fds; only their fdinfo is read (never a socket's or a log
    file's, and never another fd's)."""
    opened = []
    real = pathlib.Path.read_text
    monkeypatch.setattr(pathlib.Path, "read_text", lambda self, *a, **k: opened.append(self) or real(self, *a, **k))
    probe("vega10").procs()
    fdinfo = sorted(str(p.relative_to(AMD / "vega10/proc")) for p in opened if "fdinfo" in p.parts)
    assert fdinfo == ["1201/fdinfo/7", "1400/fdinfo/5", "1500/fdinfo/6", "99/fdinfo/4"]


@pytest.mark.parametrize("where", ["fd", "readlink", "fdinfo"])
def test_a_refused_read_is_reported_not_an_empty_list(monkeypatch, where):
    """Another user's /proc/<pid>/fd needs root or CAP_SYS_PTRACE: what can be read is counted,
    and `unreadable` says the list may be incomplete."""
    def refuse(real, part):
        def f(path, *a, **k):
            if "/99/" in str(path) and part in str(path):
                raise PermissionError(13, "Permission denied", str(path))
            return real(path, *a, **k)
        return f
    if where == "fd":
        monkeypatch.setattr(amd.os, "listdir", refuse(os.listdir, "/fd"))
    elif where == "readlink":
        monkeypatch.setattr(amd.os, "readlink", refuse(os.readlink, "/fd/"))
    else:
        monkeypatch.setattr(pathlib.Path, "read_text", refuse(pathlib.Path.read_text, "fdinfo"))
    assert probe("vega10").procs() == Procs([("1201", 3072)], unreadable=True)


def test_a_process_that_exits_is_skipped_not_unreadable(tmp_path):
    root = tmp_path / "proc"
    shutil.copytree(AMD / "vega10/proc", root, symlinks=True)
    shutil.rmtree(root / "99/fd")
    assert amd.AmdProbe(0, str(AMD / "vega10/sys"), str(root)).procs() == Procs([("1201", 3072)])
