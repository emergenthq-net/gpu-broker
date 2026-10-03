# amdgpu fixtures

Fake `/sys` and `/proc` trees for `gpu_broker/gpu/amd.py` and `host/gpu-broker-gpu`. No AMD card
was available, so every file follows the kernel's documented formats and none was copied from
a live card:

- `class/drm/card<N>/device/`: `vendor` (`0x1002`), `uevent` (`PCI_SLOT_NAME=`),
  `mem_info_vram_total` / `mem_info_vram_used` (bytes), `gpu_busy_percent` (%):
  Documentation/gpu/amdgpu/driver-misc.rst and thermal.rst.
- `hwmon/hwmon*/`: `power1_average` or `power1_input` (microwatts), `temp1_input`
  (millidegrees C, label `edge`), `freq1_input` (Hz, label `sclk`): amdgpu thermal.rst.
- `/proc/<pid>/fdinfo/<fd>`: `drm-driver`, `drm-pdev`, `drm-client-id`, `drm-memory-vram` (the
  amdgpu-only alias of `drm-resident-vram`), units bytes / KiB / MiB:
  Documentation/gpu/drm-usage-stats.rst. The `pos/flags/mnt_id/ino` lines are the generic
  fdinfo header.
- `/proc/<pid>/fd/<fd>`: symlinks, as the kernel shows them; only those into `/dev/dri/` are DRM
  files whose fdinfo is read. They are listed in `fd-links.txt` instead of committed (they
  dangle here, and an sdist drops dangling symlinks); `tests.helpers.amdgpu_fixture()` copies
  this tree to a temp dir and recreates them.

`vega10/`: a Radeon Pro WX 9100-style card (16 GiB, `power1_average`), an NVIDIA card and a
second amdgpu device's process that must be skipped, an i915 file, a non-DRM fd, and pid 1600
whose fd 8 is a log file with an (impossible) amdgpu fdinfo: it is never read, so never counted.
`rdna3/`: an RX 7900 XTX-style card (24 GiB) with only `power1_input` and no `freq1_input`;
client 17 is open on two fds of pid 2001 and inherited by pid 2003 (counted once).
