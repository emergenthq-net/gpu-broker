"""Driver-neutral names for the things a driver starts and stops.

A catalog or config entry may name a unit three ways:

    unit: llama-server                        # a systemd unit or container on this machine
    unit: {name: llama-server, target: 101}   # `target` = where it lives (a Proxmox container id)
    unit: {ct: 101, name: llama-server}       # older spelling of the same; `ct` means `target`

Everything is validated here, once, because unit names end up as subprocess arguments.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

UNIT_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.@:-]{0,127}$")
TARGET = re.compile(r"^[0-9]{1,9}$")
KEY_SEP = ":"  # "<target>:<name>", the allowlist key; matches the host script's ALLOW_UNITS format


@dataclass(frozen=True)
class UnitRef:
    name: str
    target: str | None = None

    @property
    def key(self) -> str:
        return self.name if self.target is None else f"{self.target}{KEY_SEP}{self.name}"


def unit_ref(spec: Any) -> UnitRef:
    """Parse and validate a unit spec; raises ValueError on anything that is not a clean name."""
    if isinstance(spec, UnitRef):
        return spec
    if isinstance(spec, str):
        name, target = spec, None
    elif isinstance(spec, dict) and spec.get("name"):
        name = str(spec["name"])
        raw = spec.get("target", spec.get("ct"))
        target = None if raw is None else str(raw)
    else:
        raise ValueError(f"bad unit spec {spec!r}: need a name")
    if not UNIT_NAME.match(name):
        raise ValueError(f"bad unit name {name!r}")
    if target is not None and not TARGET.match(target):
        raise ValueError(f"bad unit target {target!r}: expected a numeric container id")
    return UnitRef(name, target)
