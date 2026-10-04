"""What a replay did: waits per requester (or class), residency churn, and the same numbers for
what the real broker did, so a policy can be compared with the log and with another policy."""
from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping
from typing import Any

from ..metrics import percentile
from .sim import Result
from .trace import Trace

ABA_WINDOW_S = 900       # A → B → A within this counts as thrash
DAY_S = 86400
OTHER = "(other)"
TOP_REQUESTERS = 6       # the rest are pooled under OTHER


def rate(trace: Trace, ends: list[float]) -> float | None:
    """Jobs finished per hour, from the start of the log to the group's last end."""
    span = max(ends, default=trace.start) - trace.start
    return round(len(ends) * 3600 / span, 2) if span > 0 else None


def behind_others(trace: Trace, res: Result) -> dict[str, float]:
    """Each job's wait minus the time its own requester's other jobs were running meanwhile:
    the part of the wait a queue policy can shorten (a batch waits behind itself whatever the order)."""
    runs: dict[str, list[tuple[float, float]]] = {}
    for j in trace.jobs:
        if j.id in res.starts:
            runs.setdefault(j.requester, []).append((res.starts[j.id], res.ends.get(j.id, res.end)))
    out = {}
    for j in trace.jobs:
        start = res.starts.get(j.id, res.end)
        came = start - res.waits[j.id]
        own = sum(max(0.0, min(e, start) - max(s, came)) for s, e in runs.get(j.requester, []) if s != start or e != res.ends.get(j.id))
        out[j.id] = max(0.0, res.waits[j.id] - own)
    return out


def waits(values: list[float]) -> dict[str, Any]:
    return {"n": len(values), "p50": percentile(values, 0.5), "p90": percentile(values, 0.9),
            "p99": percentile(values, 0.99), "max": percentile(values, 1.0)}


def churn(residency: list[tuple[float, str]], span_s: float, switch_s: Mapping[str, float]) -> dict[str, Any]:
    aba = sum(1 for (t0, a), (_, b), (t2, c) in zip(residency, residency[1:], residency[2:], strict=False)
              if a == c != b and t2 - t0 < ABA_WINDOW_S)
    days = max(span_s / DAY_S, 1 / 24)
    return {"switches": len(residency), "per_day": round(len(residency) / days, 1), "aba_15m": aba,
            "switch_s": round(sum(switch_s[m] for _, m in residency), 1)}


def group_of(trace: Trace, by: Callable[[Any], str] | None) -> Callable[[Any], str]:
    if by is not None:
        return by
    top = {r for r, _ in Counter(j.requester for j in trace.jobs).most_common(TOP_REQUESTERS)}
    return lambda j: j.requester if j.requester in top else OTHER


def report(trace: Trace, res: Result, policy: str, by: Callable[[Any], str] | None = None) -> dict[str, Any]:
    key = group_of(trace, by)
    sim: dict[str, list[float]] = {}
    seen: dict[str, list[float]] = {}
    ends: dict[str, list[float]] = {}
    others: dict[str, list[float]] = {}
    behind = behind_others(trace, res)
    for j in trace.jobs:
        sim.setdefault(key(j), []).append(res.waits[j.id])
        others.setdefault(key(j), []).append(behind[j.id])
        ends.setdefault(key(j), []).append(res.ends.get(j.id, res.end))
        if j.observed_wait is not None:
            seen.setdefault(key(j), []).append(j.observed_wait)
    span = trace.end - trace.start
    return {"policy": policy, "jobs": len(trace.jobs), "lost": trace.lost,
            "chained": sum(j.after is not None for j in trace.jobs), "direct": trace.direct, "unknown_model": trace.unknown_model, "span_h": round(span / 3600, 1),
            "wait": {g: waits(v) for g, v in sorted(sim.items())},
            "per_h": {g: rate(trace, v) for g, v in sorted(ends.items())},
            "behind_others": {g: waits(v) for g, v in sorted(others.items())},
            "observed_wait": {g: waits(v) for g, v in sorted(seen.items())},
            "residency": churn(res.residency, span, trace.switch_s),
            "observed_residency": churn(trace.residency or [], span, trace.switch_s)}


def text(r: Mapping[str, Any]) -> str:
    out = [f"policy {r['policy']}: {r['jobs']} queued jobs over {r['span_h']} h "
           f"({r['chained']} sent on a previous end, {r["lost"]} lost to restarts not replayed, "
           f"{r['direct']} direct chats and {r['unknown_model']} jobs for models not in the catalog skipped)",
           f"{'group':<24}{'n':>7}{'p50':>9}{'p90':>9}{'p99':>9}{'max':>9}{'jobs/h':>9}{'oth p90':>9}   (wait in line, s; observed p90)"]
    for g, w in r["wait"].items():
        seen = r["observed_wait"].get(g, {}).get("p90")
        out.append(f"{g[:23]:<24}{w['n']:>7}" + "".join(f"{_fmt(w[k]):>9}" for k in ("p50", "p90", "p99", "max"))
                   + f"{_fmt(r['per_h'].get(g)):>9}{_fmt(r['behind_others'].get(g, {}).get('p90')):>9}   ({_fmt(seen)})")
    for label, c in (("residency", r["residency"]), ("observed", r["observed_residency"])):
        out.append(f"{label + ':':<11}{c['switches']} switches ({c['per_day']}/day), {c['aba_15m']} A-B-A within 15 min, "
                   f"{c['switch_s']} s switching")
    return "\n".join(out)


def _fmt(v: float | None) -> str:
    return "-" if v is None else f"{v:.1f}"
