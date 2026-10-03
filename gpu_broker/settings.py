"""Deployment settings: one YAML file plus environment overrides, as typed, frozen dataclasses.

Precedence is environment > file > the defaults below. The file is $BROKER_CONFIG, else
DEFAULT_PATH; a missing default file means "all defaults" (systemd driver on this machine,
ComfyUI and the API on localhost). An unknown key is an error, so a typo never silently
falls back to a default. Secrets (the API token, model-server keys) are environment-only.
"""
from __future__ import annotations

import dataclasses
import ipaddress
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import yaml

from .gpu import check_index
from .modelmap import parse_map
from .tuning import Gpu as Gpu
from .tuning import Intervals as Intervals
from .tuning import Limits as Limits
from .tuning import Timeouts as Timeouts
from .units import UnitRef, unit_ref

Network = ipaddress.IPv4Network | ipaddress.IPv6Network

DEFAULT_PATH, CONFIG_ENV = "/etc/gpu-broker/config.yaml", "BROKER_CONFIG"
TRUE_WORDS = frozenset({"1", "true", "yes", "on"})
SCALARS: Mapping[str, type] = {"int": int, "float": float, "str": str}


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
    gpu: Gpu = field(default_factory=Gpu)
    timeouts: Timeouts = field(default_factory=Timeouts)
    intervals: Intervals = field(default_factory=Intervals)
    limits: Limits = field(default_factory=Limits)
    inputs: Inputs = field(default_factory=Inputs)
    ui: Ui = field(default_factory=Ui)
    # Hosted model names -> catalog models, first matching glob wins: {"gpt-*": "@default"}.
    # Only names the catalog does not know are mapped. Env: BROKER_MODEL_MAP as a JSON object.
    model_map: Mapping[str, str] = field(default_factory=dict)
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
    "BROKER_GPU_VENDOR": ("gpu", "vendor"),
    "BROKER_GPU_INDEX": ("gpu", "index"),
    "BROKER_CHAT_WAIT_S": ("timeouts", "chat_wait_s"),
    "BROKER_INPUT_MAX_BYTES": ("inputs", "max_bytes"),
    "BROKER_INPUT_URLS": ("inputs", "allow_urls"),
    "BROKER_INPUT_DIR": ("inputs", "staging_dir"),
    "BROKER_MODEL_MAP": ("model_map",),
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


def _coerce(name: str, value: Any, annotation: str) -> Any:
    if name == "unit":
        return None if value in (None, "") else unit_ref(value)
    if name == "model_map":
        return parse_map(json.loads(value) if isinstance(value, str) else value or {})
    if name == "allowed_units":
        return None if value is None else tuple(unit_ref(u) for u in value)
    if name == "index":
        return check_index(value)
    if annotation == "bool":
        return value.strip().lower() in TRUE_WORDS if isinstance(value, str) else bool(value)
    if annotation.startswith("tuple") and isinstance(value, list):
        return tuple(value)
    if annotation in SCALARS and value is not None:
        value = SCALARS[annotation](float(value) if annotation == "int" else value)
    if isinstance(value, str) and name.endswith("url"):
        return value.rstrip("/")
    return value
