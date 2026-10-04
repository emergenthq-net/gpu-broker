"""Codex CLI: a `gpu-broker` profile, `$CODEX_HOME/gpu-broker.config.toml` (default ~/.codex).

Codex 0.95 and later speak only the OpenAI Responses API, so the provider says
`wire_api = "responses"` (the broker serves /v1/responses). Since Codex 0.134 a profile is
its own file, `<name>.config.toml`, applied on top of config.toml by `codex --profile <name>`
(`-p`); the old `[profiles.<name>]` tables in config.toml are refused. The profile carries
the model, the provider and the key, and is kept private (mode 600). The key is the broker
key (`experimental_bearer_token`, so no environment variable is needed), unless the broker
passes OpenAI keys through to a cloud provider (upstreams.py) and OPENAI_API_KEY is set: then
Codex keeps sending that key (`env_key`) and the broker key rides in `http_headers` as
`x-gpu-broker-key`, so routed models are answered by OpenAI on the user's account first. config.toml is never edited, so plain
`codex` keeps its own provider and disconnect simply deletes the file.

Left alone, with the reason: a profile file of the same name that connect did not write,
and a config.toml that defines a `gpu-broker` provider or legacy profile of its own (Codex
would merge or refuse them).
"""
from __future__ import annotations

import json
import tomllib
from pathlib import Path

from . import edits
from .core import BROKER_KEY_HEADER, OWN_KEY_OPT, PROVIDER_ID, FileChange, Plan, Target, is_client_key, skipped
from .upstreams import DUAL, SINGLE

NAME, LABEL = "codex", "Codex CLI"
HOME_ENV = "CODEX_HOME"
DEFAULT_DIR = Path(".codex")
CONFIG = "config.toml"
PROFILE = f"{PROVIDER_ID}.config.toml"
OWN_KEY_ENV = "OPENAI_API_KEY"
KEY_MODE = f"{DUAL} when the broker passes OpenAI keys through and {OWN_KEY_ENV} is set, else {SINGLE}"


def folder(t: Target) -> Path:
    return Path(t.env[HOME_ENV]) if t.env.get(HOME_ENV) else t.home / DEFAULT_DIR


def detect(t: Target) -> tuple[bool, str]:
    d = folder(t)
    return (True, str(d / PROFILE)) if d.is_dir() else (False, f"{d} not found")


def _q(value: str) -> str:
    return json.dumps(value)   # a JSON string is a valid TOML basic string


def own_key(t: Target) -> str | None:
    """Why Codex cannot keep its own OpenAI key here, or None when it can."""
    if not t.options.get(OWN_KEY_OPT["openai"]):
        return "the broker does not pass OpenAI keys through to a cloud provider"
    value = t.env.get(OWN_KEY_ENV, "")
    if not value or is_client_key(value):
        return f"{OWN_KEY_ENV} is not set to an OpenAI key"
    return None


def profile(t: Target, own: bool) -> str:
    keys = ([f"env_key = {_q(OWN_KEY_ENV)}", f"http_headers = {{ {_q(BROKER_KEY_HEADER)} = {_q(t.key)} }}"] if own
            else [f"experimental_bearer_token = {_q(t.key)}"])
    return "\n".join([
        f"# {edits.NOTE}", f"# use it with `codex --profile {PROVIDER_ID}`",
        f"model = {_q(t.model)}", f"model_provider = {_q(PROVIDER_ID)}", "",
        f"[model_providers.{PROVIDER_ID}]", f"name = {_q(PROVIDER_ID)}", f"base_url = {_q(t.openai_base)}",
        *keys, 'wire_api = "responses"', ""])


def conflict(d: Path) -> str | None:
    """Why the profile cannot be written here, or None."""
    own = d / PROFILE
    if own.exists() and edits.NOTE not in own.read_text(edits.ENCODING, errors="replace"):
        return f"{own} exists and was not written by gpu-broker connect"
    base = d / CONFIG
    if not base.exists():
        return None
    try:
        cfg = tomllib.loads(base.read_text(edits.ENCODING))
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as e:
        return f"{base} is not valid TOML ({e}); Codex would not start either"
    if cfg.get("profile") == PROVIDER_ID or PROVIDER_ID in (cfg.get("profiles") or {}):
        return f"{base} has a legacy '{PROVIDER_ID}' profile, which Codex refuses next to a profile file"
    if PROVIDER_ID in (cfg.get("model_providers") or {}):
        return f"{base} defines its own '{PROVIDER_ID}' provider"
    return None


def plan(t: Target) -> Plan:
    found, why = detect(t)
    if not found:
        return skipped(NAME, why)
    d = folder(t)
    if clash := conflict(d):
        return skipped(NAME, clash)
    single = own_key(t)
    change = FileChange(d / PROFILE, profile(t, single is None).encode(edits.ENCODING), {"kind": edits.OWN_FILE},
                        force_mode=True)
    keys = (f"the profile sends your {OWN_KEY_ENV} and the broker key as x-gpu-broker-key" if single is None
            else f"the profile sends only the broker key ({single})")
    return Plan(NAME, [change], notes=[f"run `codex --profile {PROVIDER_ID}` (Codex 0.134 or later); "
                                       f"plain `codex` keeps its own provider; {keys}"],
                keys=DUAL if single is None else SINGLE)
