"""`gpu-broker replay`: run an event log through one or more policies and print the comparison."""
from __future__ import annotations

import json
from collections.abc import Sequence

from ..catalog import Catalog
from ..policy import gpu_cost, make
from ..tuning import Scheduling
from . import trace as tr
from .report import report, text
from .sim import Sim


def run(events_path: str, catalog_path: str, policies: Sequence[str], as_json: bool, rules: Scheduling | None = None) -> int:
    """`rules`: the `scheduler` settings to replay with (aging and eviction bounds)."""
    rules = rules or Scheduling()
    catalog = Catalog(catalog_path)
    with open(events_path) as f:
        trace = tr.build(tr.read(f), catalog)
    run_s: dict[str, list[float]] = {}
    for j in trace.jobs:
        run_s.setdefault(j.model, []).append(j.run_s)
    cost = gpu_cost(catalog.models, run_s)   # the live broker measures the same thing from its store
    results = []
    for name in policies:
        res = Sim(trace, catalog, make(name, cost, rules.max_wait_s), rules.evict_wait_s).run()
        results.append(report(trace, res, name))
    print(json.dumps(results, indent=1) if as_json else "\n\n".join(text(r) for r in results))
    return 0
