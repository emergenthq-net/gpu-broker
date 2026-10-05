"""The deployment settings: typed, frozen dataclasses with documented defaults.

One dataclass per config section; the tuning sections (timeouts, intervals, limits, the GPU
choice, scheduling, fallback, the dashboard) are in tuning.py. settings.py builds a `Settings`
from the config file and the environment, and re-exports every section.
"""
from __future__ import annotations

import ipaddress
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .failover.config import Upstreams
from .tuning import Fallback, Gpu, Intervals, Limits, Mcp, Scheduling, Timeouts, Ui
from .units import UnitRef

Network = ipaddress.IPv4Network | ipaddress.IPv6Network


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
    # ComfyUI's output folder as exec recipes see it: an exec output under it gets a /view URL.
    output_dir: str = ""
    public_url: str = ""                 # as browsers reach it (dashboard, output URLs); "" = url
    unit: UnitRef | None = None          # started if ComfyUI is found down; None = never started
    auth_env: str = ""                   # ComfyUI behind auth: env var (UPSTREAM_TOKEN_*) with its Bearer token

    @property
    def browser_url(self) -> str:
        return self.public_url or self.url


@dataclass(frozen=True)
class Driver:
    kind: str = "systemd"                         # systemd | docker | proxmox
    allowed_units: tuple[UnitRef, ...] | None = None  # None = catalog units + comfy.unit
    options: Mapping[str, Any] = field(default_factory=dict)  # driver-specific keyword arguments


@dataclass(frozen=True)
class Inputs:
    """Input files: images (`image`, `end_image`, `frames`) and a `video`, checked at submit."""
    max_bytes: int = 20 * 1024 * 1024          # per image, after base64 decoding
    video_max_bytes: int = 100 * 1024 * 1024   # the video, after decoding
    max_frames: int = 64                       # images in one `frames` list
    types: tuple[str, ...] = ("png", "jpeg", "webp")        # image formats, checked by magic bytes
    video_types: tuple[str, ...] = ("mp4", "mov", "webm")   # video containers, checked the same way
    allow_urls: bool = False   # accept `<slot>_url`: the broker then fetches caller-chosen URLs
    fetch_s: float = 30        # one `<slot>_url` download, start to last byte
    staging_dir: str = "/var/lib/gpu-broker/inputs"   # held here until the job hands them on
    # `<slot>_url` may only reach public addresses; CIDRs listed here are allowed as well
    # (e.g. a LAN file host). Loopback, private, link-local and unspecified are refused otherwise.
    # The broker's own ComfyUI (comfy.url / comfy.public_url host and port) is always allowed.
    # Parsed once, here: CIDR strings from YAML become network objects.
    url_allow_networks: tuple[Network, ...] = ()

    def __post_init__(self) -> None:
        nets = []
        for cidr in self.url_allow_networks:
            try:
                nets.append(ipaddress.ip_network(cidr, strict=False))
            except ValueError:
                raise ValueError(f"inputs.url_allow_networks: {cidr!r} is not a network (e.g. 192.168.1.0/24)") from None
        object.__setattr__(self, "url_allow_networks", tuple(nets))


@dataclass(frozen=True)
class Settings:
    catalog: str = "/etc/gpu-broker/catalog.yaml"
    db: str = "/var/lib/gpu-broker/broker.db"
    events_jsonl: str = "/var/log/gpu-broker/events.jsonl"  # "" = SQLite only
    gpu_stream: bool = True
    server: Server = field(default_factory=Server)
    comfy: Comfy = field(default_factory=Comfy)
    driver: Driver = field(default_factory=Driver)
    gpu: Gpu = field(default_factory=Gpu)
    timeouts: Timeouts = field(default_factory=Timeouts)
    intervals: Intervals = field(default_factory=Intervals)
    limits: Limits = field(default_factory=Limits)
    inputs: Inputs = field(default_factory=Inputs)
    ui: Ui = field(default_factory=Ui)
    # Hosted names -> catalog models (modelmap.py). None = the built-in default map, {} = off.
    # Env: BROKER_MODEL_MAP as a JSON object.
    model_map: Mapping[str, str] | None = None
    fallback: Fallback = field(default_factory=Fallback)
    scheduler: Scheduling = field(default_factory=Scheduling)
    upstreams: Upstreams = field(default_factory=Upstreams)   # cloud first, local on failure (failover/)
    mcp: Mcp = field(default_factory=Mcp)
    source: str | None = None     # the file these came from, for the startup event
