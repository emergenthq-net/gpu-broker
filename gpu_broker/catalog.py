"""The model catalog: what can be requested, how it runs, and the GPU policy defaults.

The catalog file is trusted configuration — it is the only place backend URLs come from.
The API may add entries (unknown repositories queued for download) but those never carry
an endpoint, so a request can never make the broker contact a new address.
"""
from __future__ import annotations

import os
import re
import threading
from typing import Any, NotRequired, TypedDict
from urllib.parse import urlsplit

import yaml

from .constants import BROKER_FIELDS, OPENAI_JSON_PATHS, ModelStatus, ResidencyMode, Runner
from .units import UnitRef, unit_ref

URL_SCHEMES = frozenset({"http", "https"})
TMP_SUFFIX = ".tmp"
HTTP_PATH = re.compile(r"^/[A-Za-z0-9._~/%:@+-]*$")

class Source(TypedDict, total=False):
    hf: str            # Hugging Face repo id
    gh: str            # GitHub URL
    slug: str          # directory under the models root
    include: list[str]  # file globs to fetch (Hugging Face only)


class Model(TypedDict, total=False):
    kind: str
    runner: str
    status: str
    unit: Any                  # unit spec, see units.py
    endpoint: str              # OpenAI-compatible base URL (llm_unit)
    served_name: str
    residency: str             # unit (default) | vllm_sleep | ollama
    health_path: str          # GET readiness path; default /health
    metrics_path: str         # optional observability path, e.g. /metrics
    api_paths: list[str]      # optional explicit subset of supported broker JSON paths
    auth_env: str              # env var holding the server's API key
    slots: int                 # concurrent calls the server accepts
    reserved_interactive: int  # of those, slots background work may never take
    variants: dict[str, dict[str, Any]]   # extra model ids: same server, request overrides
    template: str              # ComfyUI graph builder (comfy)
    params: dict[str, Any]
    comfy_template: str        # ComfyUI UI workflow the dashboard opens
    open_url: str              # front end opened instead of ComfyUI
    session_only: bool
    vram_mib: int
    caps: list[str]
    quality: int
    aliases: list[str]
    source: Source
    downloaded: bool


class Defaults(TypedDict):
    resident: str
    idle_restore_s: float
    session_idle_s: float
    session_yield_s: float
    session_max_s: float
    vram_total_mib: int
    vram_reserve_mib: int
    background_requesters: NotRequired[list[str]]   # callers treated as background by default


class CatalogData(TypedDict):
    defaults: Defaults
    models: dict[str, Model]


def validate(data: CatalogData) -> None:
    """Reject entries that would hand a driver or an HTTP client something unchecked."""
    if data["defaults"]["resident"] not in data["models"]:
        raise ValueError(f"defaults.resident {data['defaults']['resident']!r} is not a catalog model")
    names = set(data["models"])
    for key, m in data["models"].items():
        if m.get("runner") not in {r.value for r in Runner}:
            raise ValueError(f"{key}: unknown runner {m.get('runner')!r}")
        if m.get("status") not in {s.value for s in ModelStatus}:
            raise ValueError(f"{key}: unknown status {m.get('status')!r}")
        residency = m.get("residency", ResidencyMode.UNIT)
        if residency not in {mode.value for mode in ResidencyMode}:
            raise ValueError(f"{key}: unknown residency mode {residency!r}")
        if residency != ResidencyMode.UNIT and m.get("runner") != Runner.LLM_UNIT:
            raise ValueError(f"{key}: residency mode {residency!r} requires runner llm_unit")
        if residency != ResidencyMode.UNIT and not all(k in m for k in ("unit", "endpoint", "served_name")):
            raise ValueError(f"{key}: residency mode {residency!r} requires unit, endpoint and served_name")
        if "unit" in m:
            unit_ref(m["unit"])
        slots = int(m.get("slots", 1))
        if not 0 <= int(m.get("reserved_interactive", 0)) < slots:
            raise ValueError(f"{key}: reserved_interactive must be at least 0 and below slots ({slots})")
        for vid, overrides in (m.get("variants") or {}).items():
            if vid in names:
                raise ValueError(f"{key}: variant id {vid!r} is already a model or variant id")
            names.add(vid)
            if bad := set(overrides) & BROKER_FIELDS:
                raise ValueError(f"{key}: variant {vid!r} may not set broker fields {sorted(bad)}")
        for url_key in ("endpoint", "open_url"):
            if url_key in m and urlsplit(m[url_key]).scheme not in URL_SCHEMES:
                raise ValueError(f"{key}: {url_key} must be an http(s) URL")
        for path_key in ("health_path", "metrics_path"):
            if path_key in m and not HTTP_PATH.fullmatch(m[path_key]):
                raise ValueError(f"{key}: {path_key} must be an absolute HTTP path")
        if bad_paths := set(m.get("api_paths", [])) - OPENAI_JSON_PATHS:
            raise ValueError(f"{key}: unsupported api_paths {sorted(bad_paths)}")


class Catalog:
    """The loaded catalog plus the lock that serialises writes back to its file."""

    def __init__(self, path: str, data: CatalogData | None = None) -> None:
        self.path = path
        if data is None:
            with open(path) as f:
                data = yaml.safe_load(f)
            if not isinstance(data, dict):
                raise ValueError(f"{path}: not a catalog")
        validate(data)
        self.data: CatalogData = data
        self._lock = threading.Lock()

    @property
    def models(self) -> dict[str, Model]:
        return self.data["models"]

    @property
    def defaults(self) -> Defaults:
        return self.data["defaults"]

    def variant(self, name: str) -> tuple[str, dict[str, Any]] | None:
        """(parent model key, request overrides) if `name` is a model variant id."""
        for key, m in self.models.items():
            overrides = (m.get("variants") or {}).get(name)
            if overrides is not None:
                return key, overrides
        return None

    def llm_units(self) -> list[tuple[str, Model]]:
        return [(k, m) for k, m in self.models.items() if m.get("runner") == Runner.LLM_UNIT]

    def units(self) -> list[UnitRef]:
        return [unit_ref(m["unit"]) for m in self.models.values() if "unit" in m]

    def register(self, key: str, model: Model) -> None:
        """Add an entry for a newly requested repository (never with an endpoint)."""
        if "endpoint" in model or "unit" in model:
            raise ValueError("registered models cannot carry an endpoint or unit")
        with self._lock:
            self.models.setdefault(key, model)
        self.save()

    def mark_downloaded(self, key: str) -> None:
        """A downloaded model with a known template becomes ready; otherwise it stays as is
        until someone wires a runner for it."""
        with self._lock:
            m = self.models.get(key)
            if m is None:
                return
            m["downloaded"] = True
            if m.get("template") and m.get("status") == ModelStatus.DOWNLOADABLE:
                m["status"] = ModelStatus.READY.value
        self.save()

    def save(self) -> None:
        with self._lock:
            tmp = self.path + TMP_SUFFIX
            with open(tmp, "w") as f:
                yaml.safe_dump(self.data, f, sort_keys=False)
            os.replace(tmp, self.path)
