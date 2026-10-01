"""Deployment settings: one YAML file plus environment overrides, as typed, frozen dataclasses.

Precedence is environment > file > the defaults below. The file is $BROKER_CONFIG, else
DEFAULT_PATH; a missing default file means "all defaults" (systemd driver on this machine,
ComfyUI and the API on localhost). An unknown key is an error, so a typo never silently
falls back to a default. Secrets (the API token, model-server keys) are environment-only.
"""
from __future__ import annotations

import dataclasses
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import yaml

from .units import UnitRef, unit_ref

DEFAULT_PATH = "/etc/gpu-broker/config.yaml"
CONFIG_ENV = "BROKER_CONFIG"
TRUE_WORDS = frozenset({"1", "true", "yes", "on"})


@dataclass(frozen=True)
class Server:
    host: str = "127.0.0.1"   # bind address; expose deliberately (and behind TLS)
    port: int = 8095
    # Shutdown waits at most this long for open HTTP connections (a streaming chat, the
    # dashboard's live feed) before closing them. Keep the service manager's stop timeout above it.
    graceful_shutdown_s: int = 10


@dataclass(frozen=True)
class Comfy:
    url: str = "http://127.0.0.1:8188"   # as the broker reaches ComfyUI
    public_url: str = ""                 # as browsers reach it (dashboard, output URLs); "" = url
    unit: UnitRef | None = None          # started if ComfyUI is found down; None = never started

    @property
    def browser_url(self) -> str:
        return self.public_url or self.url


@dataclass(frozen=True)
class Driver:
    kind: str = "systemd"                         # systemd | docker | proxmox
    allowed_units: tuple[UnitRef, ...] | None = None  # None = catalog units + comfy.unit
    options: Mapping[str, Any] = field(default_factory=dict)  # driver-specific keyword arguments


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
    gpu_query_s: float = 20       # one nvidia-smi call
    download_s: float = 21600     # one model download
    git_s: float = 3600           # one git clone/pull
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
    job_wait_poll_s: float = 1    # blocking API calls re-read the job this often
    gpu_sample_s: float = 2       # local GPU sampler period (the Proxmox host script has its own)
    sampler_retry_s: float = 5    # reconnect delay after the GPU stream drops
    gpu_cache_s: float = 5        # /v1/gpu serves a cached reading at most this old


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
class Scheduling:
    policy: str = "balanced"      # balanced | fifo
    aging_s: float = 300          # waiting this long promotes a job one priority band


@dataclass(frozen=True)
class Ui:
    gpu_label: str = "GPU"
    resident_label: str = "the default model"
    power_max_w: float = 450      # top of the dashboard's power chart
    temp_max_c: float = 90        # top of the temperature chart
    groups: Mapping[str, Mapping[str, str]] = field(default_factory=dict)  # {group: {label, color}}


@dataclass(frozen=True)
class Settings:
    catalog: str = "/etc/gpu-broker/catalog.yaml"
    db: str = "/var/lib/gpu-broker/broker.db"
    events_jsonl: str = "/var/log/gpu-broker/events.jsonl"  # "" = SQLite only
    gpu_stream: bool = True
    server: Server = field(default_factory=Server)
    comfy: Comfy = field(default_factory=Comfy)
    driver: Driver = field(default_factory=Driver)
    timeouts: Timeouts = field(default_factory=Timeouts)
    intervals: Intervals = field(default_factory=Intervals)
    limits: Limits = field(default_factory=Limits)
    scheduling: Scheduling = field(default_factory=Scheduling)
    ui: Ui = field(default_factory=Ui)
    source: str | None = None     # the file these came from, for the startup event


ENV: Mapping[str, tuple[str, ...]] = {   # env var -> settings path
    "BROKER_CATALOG": ("catalog",),
    "BROKER_DB": ("db",),
    "BROKER_JSONL": ("events_jsonl",),
    "BROKER_GPU_STREAM": ("gpu_stream",),
    "BROKER_HOST": ("server", "host"),
    "BROKER_PORT": ("server", "port"),
    "BROKER_GRACEFUL_SHUTDOWN_S": ("server", "graceful_shutdown_s"),
    "BROKER_COMFY_URL": ("comfy", "url"),
    "BROKER_DRIVER": ("driver", "kind"),
    "BROKER_CHAT_WAIT_S": ("timeouts", "chat_wait_s"),
    "BROKER_SCHEDULER_POLICY": ("scheduling", "policy"),
    "BROKER_SCHEDULER_AGING_S": ("scheduling", "aging_s"),
}


def load(path: str | None = None, env: Mapping[str, str] | None = None) -> Settings:
    env = os.environ if env is None else env
    explicit = path or env.get(CONFIG_ENV)
    path = explicit or DEFAULT_PATH
    raw: dict[str, Any] = {}
    if os.path.exists(path):
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
    elif explicit:
        raise FileNotFoundError(f"config file {path} does not exist")
    for var, keys in ENV.items():
        if var in env:
            node = raw
            for k in keys[:-1]:
                node = node.setdefault(k, {})
            node[keys[-1]] = env[var]
    settings: Settings = _build(Settings, raw, "")
    return dataclasses.replace(settings, source=path if os.path.exists(path) else None)


def _build(cls: type[Any], data: Mapping[str, Any], where: str) -> Any:
    """Construct dataclass `cls` from a mapping, coercing values (env strings) to the field types."""
    if cls is Driver:  # driver options are free-form; the driver's constructor validates them
        data = {"kind": data.get("kind", Driver.kind), "allowed_units": data.get("allowed_units"),
                "options": {k: v for k, v in data.items() if k not in ("kind", "allowed_units")}}
    fields = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(data) - set(fields) - {"source"}
    if unknown:
        raise ValueError(f"unknown config keys at {where or 'top level'}: {sorted(unknown)}")
    defaults = cls()
    kwargs: dict[str, Any] = {}
    for name, value in data.items():
        if name == "source":
            continue
        current = getattr(defaults, name)
        if dataclasses.is_dataclass(current):
            kwargs[name] = _build(type(current), value or {}, f"{where}{name}.")
        else:
            kwargs[name] = _coerce(name, value, str(fields[name].type))
    return cls(**kwargs)


SCALARS: Mapping[str, type] = {"int": int, "float": float, "str": str}


def _coerce(name: str, value: Any, annotation: str) -> Any:
    if name == "unit":
        return None if value in (None, "") else unit_ref(value)
    if name == "allowed_units":
        return None if value is None else tuple(unit_ref(u) for u in value)
    if annotation == "bool":
        return value.strip().lower() in TRUE_WORDS if isinstance(value, str) else bool(value)
    if annotation in SCALARS and value is not None:
        value = SCALARS[annotation](float(value) if annotation == "int" else value)
    if isinstance(value, str) and name.endswith("url"):
        return value.rstrip("/")
    return value
