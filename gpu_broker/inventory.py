"""Catalog-derived capability inventory for API/UI introspection."""
from __future__ import annotations

from collections import Counter
from typing import Any

from .catalog import Catalog
from .resolve import budget, runnable


def summarize(catalog: Catalog) -> dict[str, Any]:
    kinds: Counter[str] = Counter()
    runners: Counter[str] = Counter()
    statuses: Counter[str] = Counter()
    residency: Counter[str] = Counter()
    caps: Counter[str] = Counter()
    runnable_models: list[str] = []

    for key, model in catalog.models.items():
        kinds[str(model.get("kind", "unknown"))] += 1
        runners[str(model.get("runner", "unknown"))] += 1
        statuses[str(model.get("status", "unknown"))] += 1
        residency[str(model.get("residency", "unit"))] += 1
        caps.update(str(cap) for cap in model.get("caps", []))
        if runnable(catalog.data, key, session=bool(model.get("session_only"))):
            runnable_models.append(key)

    return {
        "models": len(catalog.models),
        "runnable": len(runnable_models),
        "runnable_models": sorted(runnable_models),
        "kinds": dict(sorted(kinds.items())),
        "runners": dict(sorted(runners.items())),
        "statuses": dict(sorted(statuses.items())),
        "residency": dict(sorted(residency.items())),
        "capabilities": dict(sorted(caps.items())),
        "vram_budget_mib": budget(catalog.data),
    }
