"""host/gpu-broker-gpu on AMD (behind gpu-broker-ctl's `gpu` and `gpustream`): the amdgpu sysfs
and fdinfo reading against the fixture trees, the same numbers as the broker's own probe
(gpu_broker/gpu/amd.py), unknown values, unreadable processes, and failures that end the stream."""
import shutil

from gpu_broker.gpu import PROCS_UNREADABLE, amd
from gpu_broker.metrics import parse_sample
from tests.gpufake import AMD, gpu, no_root  # noqa: F401 — the fixture

VEGA10_LINE = "4096,16384,37,42.0,51,1500| host:512 101:3072\n"   # pid 1201 is in LXC 101; 1600's log fd never counts


def copy(tmp_path, card="vega10"):
    root = tmp_path / card
    shutil.copytree(AMD / card, root, symlinks=True)
    return root


def test_amd_read_and_stream_vega10(gpu):
    assert gpu("read").stdout == "4096,16384,37\n"
    assert gpu.stream()[0] == [VEGA10_LINE]


def test_the_host_script_prefers_power1_average_and_orders_cards_by_number(gpu, tmp_path):
    root = copy(tmp_path) / "sys"
    (root / "class/drm/card0/device/hwmon/hwmon2/power1_input").write_text("99000000\n")
    for n, used in ((2, 1048576), (10, 2097152)):
        shutil.copytree(root / "class/drm/card0", root / f"class/drm/card{n}")
        (root / f"class/drm/card{n}/device/mem_info_vram_used").write_text(f"{used}\n")
    shutil.copytree(root / "class/drm/card0", root / "class/drm/card3")
    (root / "class/drm/card3/device/vendor").write_text("0x8086\n")
    assert gpu.stream(SYSFS_ROOT=root)[0][0].startswith("4096,16384,37,42.0,")
    assert [gpu("read", SYSFS_ROOT=root, GPU_INDEX=i).stdout for i in "12"] == ["1,16384,37\n", "2,16384,37\n"]
    assert gpu("read", SYSFS_ROOT=root, GPU_INDEX="3").returncode == 6


def test_amd_stream_rdna3_leaves_the_clock_empty_and_counts_a_shared_client_once(gpu):
    assert gpu.stream("rdna3")[0] == ["8192,24576,99,287.0,64,| 104:6144 host:1024\n"]


def test_the_host_script_reads_what_the_brokers_probe_reads(gpu):
    for card in ("vega10", "rdna3"):
        p = amd.AmdProbe(0, str(AMD / card / "sys"), str(AMD / card / "proc"))
        s, procs = p.read(), p.procs()
        mine = parse_sample(gpu.stream(card)[0][0], 0)
        theirs = parse_sample(f"{s.used_mib},{s.total_mib},{s.util_pct},{s.power_w},{s.temp_c},{s.clock_mhz or ''}|", 0)
        assert mine and theirs
        assert {k: v for k, v in mine.items() if k != "by_group"} == {k: v for k, v in theirs.items() if k != "by_group"}
        assert sorted(mine["by_group"].values()) == sorted(mib for _, mib in procs.rows)


def test_only_memory_is_required_utilisation_and_hwmon_may_be_unreadable(gpu, tmp_path):
    """Runtime PM / BACO: gpu_busy_percent and the hwmon files fail to read; the line has them
    empty. Memory unreadable is an error."""
    root = copy(tmp_path)
    dev = root / "sys/class/drm/card0/device"
    (dev / "gpu_busy_percent").write_text("busy\n")
    for f in (dev / "hwmon/hwmon2").iterdir():
        f.write_text("x\n")
    assert gpu("read", SYSFS_ROOT=root / "sys").stdout == "4096,16384,\n"
    assert gpu.stream(SYSFS_ROOT=root / "sys")[0] == ["4096,16384,,,,| host:512 101:3072\n"]
    (dev / "mem_info_vram_used").unlink()
    r = gpu("read", SYSFS_ROOT=root / "sys")
    assert r.returncode == 6 and r.stdout == "" and "mem_info_vram_used" in r.stderr


def test_a_failure_mid_stream_ends_it_with_an_error(gpu, tmp_path):
    root = copy(tmp_path)
    used = root / "sys/class/drm/card0/device/mem_info_vram_used"
    lines, rc, err = gpu.stream(SYSFS_ROOT=root / "sys", SAMPLE_S="0.3", between=used.unlink)
    assert lines == [VEGA10_LINE] and rc == 6 and "cannot read mem_info_vram_used" in err


@no_root()
def test_processes_whose_fds_cannot_be_read_are_reported_not_silently_dropped(gpu, tmp_path):
    root = copy(tmp_path)
    (root / "proc/99/fd").chmod(0)   # another user's process, without root or CAP_SYS_PTRACE
    try:
        line = gpu.stream(PROC_ROOT=root / "proc")[0][0]
    finally:
        (root / "proc/99/fd").chmod(0o755)
    assert line == f"4096,16384,37,42.0,51,1500| 101:3072 {PROCS_UNREADABLE}\n"
    assert parse_sample(line, 0)["procs_unreadable"] is True


def test_only_fds_linking_into_dev_dri_are_read(gpu, tmp_path):
    """pid 1600's fd 8 is a log file whose fdinfo claims amdgpu VRAM: counted only if it were
    read. Linking it into /dev/dri makes it count, which shows the link is what decides."""
    root = copy(tmp_path)
    assert gpu.stream(PROC_ROOT=root / "proc")[0] == [VEGA10_LINE]
    (root / "proc/1600/fd/8").unlink()
    (root / "proc/1600/fd/8").symlink_to("/dev/dri/renderD128")
    assert gpu.stream(PROC_ROOT=root / "proc")[0] == ["4096,16384,37,42.0,51,1500| host:512 101:3072 101:9216\n"]


def test_bad_settings_are_refused(gpu):
    for env in ({"GPU_VENDOR": "intel"}, {"GPU_INDEX": "x"}, {"GPU_INDEX": "01"}, {"GPU_INDEX": "+1"},
                {"GPU_INDEX": "1.0"}, {"NV_TIMEOUT_S": "soon"}):
        assert gpu("read", **env).returncode == 2, env
    assert gpu("bogus").returncode == 2
    r = gpu("read", GPU_INDEX="1")
    assert r.returncode == 6 and "GPU_INDEX 1: no such amdgpu card" in r.stderr


def test_the_host_script_shares_the_brokers_rules():
    from gpu_broker import gpu as g
    from tests.gpufake import HELPER
    script = HELPER.read_text()
    assert f'"$GPU_INDEX" =~ {g.INDEX.pattern}' in script
    assert f"POWER_FMT=%.{g.POWER_DECIMALS}f" in script and f"UNREADABLE='{g.PROCS_UNREADABLE}'" in script
