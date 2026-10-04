"""Timeouts, poll intervals, size limits and the GPU choice: the tuning sections of the settings."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from .gpu import check_index, check_vendor
from .policy import DEFAULT_EVICT_WAIT_S, DEFAULT_MAX_WAIT_S, DEFAULT_POLICY, POLICIES


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
    quiesced_retry_s: int = 5     # Retry-After on a 503 while quiesced (a restart takes a few seconds)


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
    responses_body_bytes: int = 32 * 2**20          # largest /v1/responses request (else 413)
    response_store_entry_bytes: int = 8 * 2**20     # largest conversation kept for previous_response_id
    response_store_bytes: int = 128 * 2**20         # all kept conversations together (oldest go first)


@dataclass(frozen=True)
class Gpu:
    """Which GPU the broker reads (local drivers; the proxmox driver's host script decides there)."""
    vendor: str = "auto"   # auto | nvidia | amd (gpu_broker.gpu: auto tries nvidia-smi, then amdgpu sysfs)
    index: int = 0         # which card of that vendor: nvidia-smi's index, or the n-th amdgpu card

    def __post_init__(self) -> None:
        check_vendor(self.vendor)
        check_index(self.index)


@dataclass(frozen=True)
class Fallback:
    """Optional hosted fallback for the drop-in routes (docs/drop-in.md). Off by default.

    When on, a request the local side cannot serve goes to the real provider with the
    operator's own upstream key: the requested model is not a catalog name and the local call
    failed, or the resident model would first need a switch (the GPU is busy elsewhere), or the
    request uses a feature only the hosted API has. Keys come from the environment variables
    named here, never from the config file, and are never logged."""
    enabled: bool = False
    openai_url: str = "https://api.openai.com"
    anthropic_url: str = "https://api.anthropic.com"
    openai_key_env: str = "UPSTREAM_OPENAI_API_KEY"         # not OPENAI_API_KEY: connect points that at the broker
    anthropic_key_env: str = "UPSTREAM_ANTHROPIC_API_KEY"   # nor ANTHROPIC_API_KEY
    when_switching: bool = True   # also forward an interactive call that would wait for a model switch
    timeout_s: float = 600        # one forwarded call


@dataclass(frozen=True)
class Scheduling:
    """The GPU thread's queue. `fair` (interactive first, then the requester with the least
    expected GPU time used) or `fifo` (arrival order, head-of-line blocking: the behaviour before
    `fair`, kept as a rollback; it also leaves a /v1/jobs job's priority as the caller sent it)."""
    policy: str = DEFAULT_POLICY
    cost_lookback_s: float = 7 * 86400   # expected run time per model: median over this window
    cost_refresh_s: float = 600          # recomputed this often (on its own thread; never under fifo)
    max_wait_s: float = DEFAULT_MAX_WAIT_S   # a background job waiting this long counts as interactive
    evict_wait_s: float = DEFAULT_EVICT_WAIT_S   # a call waiting for a slot holds off a switch this long at most
    # Who may claim `interactive` (x-priority or the request's flag); None: every requester not in
    # the catalog's background_requesters. Anyone may claim `background`.
    may_claim_interactive: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if self.policy not in POLICIES:
            raise ValueError(f"scheduler.policy must be one of {sorted(POLICIES)}, not {self.policy!r}")


@dataclass(frozen=True)
class Ui:
    gpu_label: str = "GPU"
    resident_label: str = "the default model"
    power_max_w: float = 450      # top of the dashboard's power chart
    temp_max_c: float = 90        # top of the temperature chart
    groups: Mapping[str, Mapping[str, str]] = field(default_factory=dict)  # {group: {label, color}}


@dataclass(frozen=True)
class Mcp:
    """The MCP server (gpu_broker/mcp_server): served at /mcp when the `mcp` SDK is installed."""
    enabled: bool = True
    wait_s: float = 45            # a generate tool waits this long, then returns the job id to poll
    inline_max_bytes: int = 1024 * 1024   # a finished image this small is also returned inline
    inline_max_images: int = 4
    max_body_bytes: int = 64 * 1024 * 1024   # one /mcp request (base64 inputs); larger files go by URL
    client_url_inputs: bool = False   # may a client key send `<slot>_url` inputs (the broker fetches them)? main token: always

    def __post_init__(self) -> None:
        if self.wait_s < 0 or self.inline_max_bytes < 0 or self.max_body_bytes <= 0:
            raise ValueError("mcp: wait_s and inline_max_bytes must be >= 0, max_body_bytes > 0")
