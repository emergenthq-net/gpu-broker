"""The loaded model catalog: lookups over it, entries the API registers, and writes back to its file.

The catalog file is trusted configuration — it is the only place backend URLs come from.
The API may add entries (unknown repositories queued for download) but those never carry
an endpoint, so a request can never make the broker contact a new address. What a valid
entry is lives in catalogschema.py; its names are re-exported here.
"""
from __future__ import annotations

import os
import threading
from typing import Any

import yaml

from .catalogschema import PARAM as PARAM
from .catalogschema import RECIPE as RECIPE
from .catalogschema import CatalogData as CatalogData
from .catalogschema import Defaults as Defaults
from .catalogschema import ExecSpec as ExecSpec
from .catalogschema import Model as Model
from .catalogschema import Source as Source
from .catalogschema import validate as validate
from .constants import ModelStatus, Runner
from .units import UnitRef, unit_ref

TMP_SUFFIX = ".tmp"


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
