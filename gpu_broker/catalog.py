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

from . import inputs, templates
from .constants import BROKER_FIELDS, HTTP_SCHEMES, IMAGE_SLOTS, ModelStatus, Runner
from .units import UnitRef, unit_ref

RECIPE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,40}$")   # same grammar as drivers/validate.py and the host script
PARAM = re.compile(r"^[a-z][a-z0-9_]{0,40}$")
TMP_SUFFIX = ".tmp"


class Source(TypedDict, total=False):
    hf: str            # Hugging Face repo id
    gh: str            # GitHub URL
    slug: str          # directory under the models root
    include: list[str]  # file globs to fetch (Hugging Face only)


class ExecSpec(TypedDict, total=False):
    recipe: str          # a recipe the driver's host defines (argv, paths, outputs, its own timeout)
    timeout_s: float     # how long the broker waits for it; keep above the recipe's timeout
    params: list[str]    # request keys passed to the recipe as params.json (scalars only)
    choices: dict[str, list[Any]]   # param -> the only values accepted (checked at submit)


class Model(TypedDict, total=False):
    kind: str
    runner: str
    status: str
    unit: Any                  # unit spec, see units.py
    endpoint: str              # OpenAI-compatible base URL (llm_unit)
    served_name: str
    auth_env: str              # env var holding the server's API key
    slots: int                 # concurrent calls the server accepts
    reserved_interactive: int  # of those, slots background work may never take
    variants: dict[str, dict[str, Any]]   # extra model ids: same server, request overrides
    template: str              # ComfyUI graph builder (comfy)
    defaults: dict[str, Any]   # per-model request defaults (request > these > template DEFAULTS)
    params: dict[str, Any]
    comfy_template: str        # ComfyUI UI workflow the dashboard opens
    open_url: str              # front end opened instead of ComfyUI
    session_only: bool
    vram_mib: int
    caps: list[str]
    image_caps: list[str]      # caps that hold only when the job carries an input (e.g. i2v on a t2v model)
    inputs: dict[str, str]     # input files it takes: {slot: required|optional|one_of}, see inputs.py
    exec: ExecSpec             # runner exec: which host recipe runs it
    notes: str                 # free text for operators (why it is not wired, caveats)
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
        _validate_inputs(key, m)
        templates.check_defaults(key, m)
        if m.get("runner") == Runner.EXEC:
            _validate_exec(key, m.get("exec") or {})
        for url_key in ("endpoint", "open_url"):
            if url_key in m and urlsplit(m[url_key]).scheme not in HTTP_SCHEMES:
                raise ValueError(f"{key}: {url_key} must be an http(s) URL")


def _validate_inputs(key: str, m: Model) -> None:
    """Only runners that consume staged files take inputs: ComfyUI graphs (single images) and
    exec recipes. Anything else would stage files no one hands on."""
    spec = m.get("inputs") or {}
    inputs.validate_spec(key, spec)
    if spec:
        comfy = m.get("runner") == Runner.COMFY and "template" in m
        if not (comfy or m.get("runner") == Runner.EXEC):
            raise ValueError(f"{key}: only ComfyUI models with a template and exec models take inputs")
        if comfy and (other := sorted(set(spec) - set(IMAGE_SLOTS))):   # (so frame ranges are exec-only)
            raise ValueError(f"{key}: ComfyUI templates take only {list(IMAGE_SLOTS)}, not {other}")
    if m.get("image_caps") and not spec:
        raise ValueError(f"{key}: image_caps needs `inputs` (they apply only when a job carries one)")


def _validate_exec(key: str, spec: ExecSpec) -> None:
    if not RECIPE.match(str(spec.get("recipe", ""))):
        raise ValueError(f"{key}: runner exec needs exec.recipe, a recipe name like 'sharp'")
    timeout = spec.get("timeout_s")
    if not isinstance(timeout, (int, float)) or isinstance(timeout, bool) or timeout <= 0:
        raise ValueError(f"{key}: exec.timeout_s must be a positive number of seconds")
    if not all(isinstance(p, str) and PARAM.match(p) for p in spec.get("params", [])):
        raise ValueError(f"{key}: exec.params must be lowercase request key names")
    choices = spec.get("choices", {})
    if not isinstance(choices, dict) or not set(choices) <= set(spec.get("params", [])):
        raise ValueError(f"{key}: exec.choices must map names listed in exec.params to their allowed values")
    for p, allowed in choices.items():
        if not (isinstance(allowed, list) and allowed and all(isinstance(v, (int, float, str)) for v in allowed)):
            raise ValueError(f"{key}: exec.choices.{p} must be a non-empty list of numbers or strings")


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
        """A downloaded model with a template becomes ready; else it waits for a runner."""
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
