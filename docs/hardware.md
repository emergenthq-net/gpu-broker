# Hosts and GPUs

## Host drivers

| `driver.kind` | model servers are | start/stop via | GPU readings |
|---|---|---|---|
| `systemd` (default) | units on this machine | `systemctl [--user]`, optionally `sudo -n` | this machine's GPU |
| `docker` | existing containers | `docker start/stop` over the socket | from the broker container (`nvidia-smi`, or `/sys` for AMD) |
| `proxmox` | systemd units inside LXCs | SSH to a forced-command script on the host | the host's GPU, read by `host/gpu-broker-gpu` |

- Catalog units are driver-neutral: `unit: llama-server`, or
  `unit: {name: llama-server, target: 101}` where `target` is the Proxmox container id.
- systemd: the broker runs `systemctl start/stop` on the units named in the catalog.
  - Run it as root, or set `driver.sudo: true` with a sudoers rule limited to those units.
- Docker: containers must already exist (`docker compose create`). The Docker socket is
  root-equivalent on the host.
- Proxmox: see the [Proxmox setup](../README.md#proxmox) and [SECURITY.md](../SECURITY.md).

## Supported GPUs

| GPU | read through | model servers |
|---|---|---|
| NVIDIA | `nvidia-smi` | CUDA builds (llama.cpp `server-cuda`, ComfyUI on CUDA PyTorch) |
| AMD | amdgpu sysfs files and `/proc/*/fdinfo`; no ROCm tools needed | ROCm or Vulkan builds ([`compose.rocm.yaml`](../examples/docker/compose.rocm.yaml)) |
| Intel, Apple | not supported yet | |

## Choosing the card

`gpu: {vendor: auto, index: 0}` picks the card. Env: `BROKER_GPU_VENDOR`, `BROKER_GPU_INDEX`.

How `auto` decides, on first use:

| found | chosen |
|---|---|
| `nvidia-smi` installed | NVIDIA |
| no `nvidia-smi` | the first amdgpu card |
| both an NVIDIA and an amdgpu card | NVIDIA; the log says the AMD card is not read |

- If `nvidia-smi` fails or hangs, it is retried once a second within `timeouts.gpu_query_s`.
  - Then it is reported as an error, remembered for that long so callers get it at once.
  - The broker never falls back to AMD because `nvidia-smi` misbehaved. Set `vendor: amd` for that.
- `serve` starts even when no GPU can be read yet.
  - The dashboard and `/v1/gpu` say why, and the sampler keeps retrying.
  - While the choice is being made, `/v1/gpu` answers `{"state": "probing"}`.
- `gpu-broker check` prints the choice and one reading.

### On Proxmox

The host script decides instead. Settings in `/etc/gpu-broker-ctl.conf`:

| setting | what it does |
|---|---|
| `GPU_VENDOR`, `GPU_INDEX` | as `gpu.vendor` and `gpu.index` |
| `NV_TIMEOUT_S` | limit on each `nvidia-smi` call |
| `GPU_BUDGET_S` | limit on a whole reading, including `auto`'s choice and retries |

- Keep `GPU_BUDGET_S` + `timeouts.ssh_connect_s` under `timeouts.gpu_query_s` (default 8 + 10 < 20).
- Install `host/gpu-broker-gpu` next to `gpu-broker-ctl`.
- It decides `auto` once and caches it in `/run/gpu-broker-ctl` until the conf file changes.

## What the dashboard shows

- Required: VRAM used and total.
- On AMD, also:
  - utilisation (`gpu_busy_percent`)
  - board power (`power1_average`, or `power1_input` on RDNA3)
  - edge temperature
  - graphics clock (`freq1_input`)
- A value the card does not report is shown as "—".
  - So is any value it cannot report while runtime-suspended.
  - So is any bracketed `nvidia-smi` value (`[N/A]`, `[Not Supported]`, ...).

## Per-process VRAM on AMD

- Comes from each DRM client's `drm-memory-vram`, counted once per client.
- Read only from fds that link into `/dev/dri/`.
- Reading another user's `/proc/<pid>/fd` needs root or `CAP_SYS_PTRACE`.
  - Without it, the processes the broker can read are still shown.
  - The dashboard says "per-process memory needs root or CAP_SYS_PTRACE" (`procs_unreadable`
    in `/v1/metrics`), so the list never looks complete when it is not.
