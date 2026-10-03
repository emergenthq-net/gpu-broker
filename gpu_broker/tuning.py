"""Timeouts, poll intervals, size limits and the GPU choice: the tuning sections of the settings."""
from __future__ import annotations

from dataclasses import dataclass

from .gpu import check_index, check_vendor


@dataclass(frozen=True)
class Timeouts:
    llm_start_s: float = 240      # an LLM unit must answer its health check within this
    comfy_start_s: float = 180    # ComfyUI must answer /system_stats within this after a start
    llm_call_s: float = 1800      # one chat completion
    comfy_run_s: float = 3600     # one ComfyUI graph, queued to finished
    comfy_submit_s: float = 600   # POST /prompt
    comfy_http_s: float = 15      # /free, /history
    health_s: float = 5           # one health probe
    unit_s: float = 180           # one start/stop/is-active
    gpu_query_s: float = 20       # one GPU reading (one nvidia-smi call)
    download_s: float = 21600     # one model download
    git_s: float = 3600           # one git clone/pull
    exec_put_s: float = 120       # copying one input file to an exec recipe's host
    exec_clean_s: float = 150     # one clean of an exec job: at least the driver's clean_wait_s + 10 (checked)
    exec_vram_s: float = 120      # before an exec recipe: freed VRAM must show up in the GPU reading
    ssh_connect_s: int = 10
    container_stop_s: int = 30    # docker stop grace period
    chat_wait_s: float = 570      # /v1/chat/completions: queue + switch + run
    job_wait_s: float = 3600      # POST /v1/jobs with wait=true
    quiesce_wait_s: float = 900   # POST /v1/admin/quiesce default


@dataclass(frozen=True)
class Intervals:
    worker_poll_s: float = 5      # GPU worker wakes this often when idle (idle restore check)
    paused_s: float = 1           # GPU worker re-checks a quiesce this often
    health_poll_s: float = 2      # while waiting for an LLM or ComfyUI to come up
    comfy_poll_s: float = 2       # while waiting for a ComfyUI graph to finish
    session_poll_s: float = 5     # interactive session activity check
    gpu_sample_s: float = 2       # local GPU sampler period (the Proxmox host script has its own)
    sampler_retry_s: float = 5    # reconnect delay after the GPU stream drops
    gpu_cache_s: float = 5        # /v1/gpu (and the exec VRAM wait) use a cached reading at most this old
    held_retry_s: float = 60      # while the GPU is held: retry the held job's clean this often


@dataclass(frozen=True)
class Limits:
    gpu_samples: int = 1800       # dashboard history (~1 h at 2 s)
    metrics_window_s: float = 3600
    metrics_jobs: int = 200
    stats_window_s: float = 86400
    event_lookback_s: float = 7200  # a job's phase events may predate its window by this much
    status_downloads: int = 30
    status_recent: int = 20
    events_page: int = 500        # most events one /v1/events call returns


@dataclass(frozen=True)
class Gpu:
    """Which GPU the broker reads (local drivers; the proxmox driver's host script decides there)."""
    vendor: str = "auto"   # auto | nvidia | amd (gpu_broker.gpu: auto tries nvidia-smi, then amdgpu sysfs)
    index: int = 0         # which card of that vendor: nvidia-smi's index, or the n-th amdgpu card

    def __post_init__(self) -> None:
        check_vendor(self.vendor)
        check_index(self.index)
