"""What a valid catalog is: the types of its entries and the rules every entry must pass.

The rules reject anything that would hand a driver or an HTTP client an unchecked value: an
unknown runner or status, a malformed unit, a non-http(s) URL, inputs a runner cannot hand on,
or an exec entry without a well-formed recipe name and timeout. catalog.py loads and saves the
file; it runs `validate` on every load.
"""
from __future__ import annotations

import re
from typing import Any, NotRequired, TypedDict
from urllib.parse import urlsplit

from . import inputs, templates
from .constants import BROKER_FIELDS, HTTP_SCHEMES, IMAGE_SLOTS, ModelStatus, Runner
from .units import unit_ref

RECIPE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,40}$")   # same grammar as drivers/validate.py and the host script
PARAM = re.compile(r"^[a-z][a-z0-9_]{0,40}$")


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
    health_path: str           # what answers 200 once the server is ready (default /health; Ollama: /api/version)
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
        if "health_path" in m and not _is_path(m["health_path"]):
            raise ValueError(f"{key}: health_path must be a path on the endpoint, like /health")
        for url_key in ("endpoint", "open_url"):
            if url_key in m and urlsplit(m[url_key]).scheme not in HTTP_SCHEMES:
                raise ValueError(f"{key}: {url_key} must be an http(s) URL")


def _is_path(v: Any) -> bool:
    """A path on the model's own endpoint: absolute, not `//host`, no query or fragment."""
    return isinstance(v, str) and v.startswith("/") and not v.startswith("//") and not set("?#\\") & set(v)


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
