"""Loading the deployment settings: one YAML file plus environment overrides.

Precedence is environment > file > the defaults (settingsschema.py, whose classes are
re-exported here). The file is $BROKER_CONFIG, else DEFAULT_PATH; a missing default file
means "all defaults" (systemd driver on this machine, ComfyUI and the API on localhost). An
unknown key is an error, so a typo never silently falls back to a default. Secrets (the API
token, model-server keys) are environment-only.
"""
from __future__ import annotations

import dataclasses
import json
import os
from collections.abc import Mapping
from typing import Any

import yaml

from .gpu import check_index
from .modelmap import parse_map
from .settingsschema import Comfy as Comfy
from .settingsschema import Driver as Driver
from .settingsschema import Inputs as Inputs
from .settingsschema import Network as Network
from .settingsschema import Server as Server
from .settingsschema import Settings as Settings
from .settingsschema import Ui as Ui
from .tuning import Gpu as Gpu
from .tuning import Intervals as Intervals
from .tuning import Limits as Limits
from .tuning import Timeouts as Timeouts
from .units import unit_ref

DEFAULT_PATH, CONFIG_ENV = "/etc/gpu-broker/config.yaml", "BROKER_CONFIG"
TRUE_WORDS = frozenset({"1", "true", "yes", "on"})
SCALARS: Mapping[str, type] = {"int": int, "float": float, "str": str}


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
