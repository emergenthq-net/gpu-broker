"""Hosted model names (`gpt-4o`, `claude-sonnet-4-5`, ...) mapped onto catalog models.

An app written for a hosted API asks for the hosted model by name. `model_map` in the config
says which catalog model answers instead: `{pattern: target}`, glob patterns, first match
wins, the target `@default` meaning the catalog's resident LLM. A name the catalog already
knows (key, alias, served name, HF id, variant) is never mapped, and a name no pattern
matches keeps the resolver's normal behaviour. The mapping is always reported as a
substitution note, so a caller can see what really ran.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fnmatch import fnmatchcase
from typing import Any

from .constants import DEFAULT_TARGET


@dataclass(frozen=True)
class Mapped:
    target: str   # the catalog name that will be resolved instead
    note: str     # why, for x_broker.substitution and the job record


def parse_map(value: Any) -> dict[str, str]:
    """A config value (YAML mapping or decoded JSON) checked into {pattern: target}."""
    if not (isinstance(value, Mapping) and all(isinstance(k, str) and isinstance(v, str) for k, v in value.items())):
        raise ValueError("model_map must map name patterns to catalog model names (strings)")
    return dict(value)


def map_name(model_map: Mapping[str, str], known: bool, name: str, default: str) -> Mapped | None:
    """The target for `name`, or None. `known` = the catalog already has `name`; `default` is
    the resident LLM that `@default` stands for. Matching ignores case."""
    if known:
        return None
    for pattern, target in model_map.items():
        if fnmatchcase(name.lower(), pattern.lower()):
            key = default if target == DEFAULT_TARGET else target
            return Mapped(key, f"'{name}' mapped to '{key}' by model_map pattern '{pattern}'")
    return None


def joined(*notes: str | None) -> str | None:
    """Several substitution notes as one, or None when there are none."""
    return "; ".join(n for n in notes if n) or None
