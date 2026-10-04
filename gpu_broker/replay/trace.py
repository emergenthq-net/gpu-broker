"""An `events.jsonl` → the jobs that went through the queue, and what switching cost.

Per job: arrival (`job.received`), requester, catalog model (the substitution if there was
one, else the requested name looked up in the catalog), run time (`job.running` → its end) and
the wait it really had (`job.queued` → `job.switching`/`job.running`). Jobs that never ended in
the log (lost to a restart) are counted, not replayed: their caller sent them again. A job that
ended without running (failed while switching) replays with no run time. Direct chats never
entered the queue and are counted, not replayed.

Closed loops: a job that arrives within `CHAIN_S` of the end of its requester's previous job is
taken to be caused by it (a client that keeps N calls in flight sends the next when one
returns). The replay then sends it that long after the previous job ends in the simulation,
not at its logged time, so a faster or slower policy changes when the client asks again. A switch to
a model costs the median of its measured `job.switching` → `job.running` times (falling back to
its `residency.ready` load time, then to the median over all models).
"""
from __future__ import annotations

import bisect
import json
import statistics
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any

from ..catalog import Catalog
from ..classes import Classes
from ..constants import Event, JobState, Runner
from ..resolve import lookup

DEFAULT_SWITCH_S = 10.0   # no switch measured anywhere in the log
CHAIN_S = 5.0             # sent this soon after the requester's previous job ended → caused by it
REAL_SWITCH_S = 1.0       # shorter switching → running is no switch, just the state step
STARTED = ("job." + JobState.SWITCHING, "job." + JobState.RUNNING)
ENDED = ("job." + JobState.DONE, "job." + JobState.FAILED)


@dataclass
class Job:
    id: str
    arrival: float
    requester: str
    model: str
    run_s: float = 0.0
    after: str | None = None   # the job whose end released this one (closed loop)
    gap: float = 0.0           # seconds after `after` ended
    observed_wait: float | None = None   # what the real broker did, for calibration
    interactive: bool = False   # class by requester, as the broker classifies a job without `x-priority`
    pooled: bool = False        # an LLM call (holds a pool slot)
    requeued: bool = False


@dataclass
class Trace:
    jobs: list[Job]
    switch_s: dict[str, float]
    start: float
    end: float
    direct: int = 0
    unknown_model: int = 0
    lost: int = 0
    residency: list[tuple[float, str]] | None = None   # what the real broker made resident, when


def read(lines: Iterable[str]) -> Iterator[dict[str, Any]]:
    for line in lines:
        if line.strip():
            yield json.loads(line)


def model_of(catalog: Catalog, name: str) -> str | None:
    if (key := lookup(catalog.data, name)) is not None:
        return key
    v = catalog.variant(name)
    return v[0] if v else None


def build(events: Iterable[dict[str, Any]], catalog: Catalog) -> Trace:
    rows: dict[str, dict[str, Any]] = {}
    ready: dict[str, list[float]] = {}
    resident: list[tuple[float, str]] = []
    first = last = None
    for e in events:
        ts, kind, jid = e["ts"], e["kind"], e.get("job_id")
        first, last = first if first is not None else ts, ts
        if kind == Event.RES_RESIDENT:
            resident.append((ts, e["model"]))
        if kind == Event.RES_READY and "load_s" in e:
            ready.setdefault(e["model"], []).append(float(e["load_s"]))
        if not jid:
            continue
        r = rows.setdefault(jid, {})
        if kind == "job." + JobState.RECEIVED:
            r.update(arrival=ts, requester=e.get("requester", ""), requested=e.get("requested", ""))
        elif kind == Event.JOB_SUBSTITUTED:
            r["resolved"] = e.get("resolved")
        elif kind == Event.JOB_DIRECT:
            r["direct"] = True
        elif kind == "job." + JobState.QUEUED:
            r.setdefault("queued", ts)
        elif kind in STARTED:
            r.setdefault(kind.removeprefix("job."), ts)
        elif kind in ENDED:
            r.setdefault("ended", ts)
    trace = _assemble(rows, ready, catalog, first or 0.0, last or 0.0)
    trace.residency = [r for i, r in enumerate(resident) if i == 0 or resident[i - 1][1] != r[1]]
    return trace


def _assemble(rows: dict[str, dict[str, Any]], ready: dict[str, list[float]], catalog: Catalog,
              start: float, end: float) -> Trace:
    trace = Trace([], {}, start, end)
    switches: dict[str, list[float]] = {}
    ends: dict[str, list[tuple[float, str]]] = {}
    for jid, r in rows.items():
        if "arrival" not in r:
            continue   # its `job.received` is before the log starts
        if r.get("direct"):
            trace.direct += 1
            continue
        model = r.get("resolved") or model_of(catalog, r["requested"])
        if model is None or "queued" not in r:
            trace.unknown_model += model is None
            continue
        if "ended" not in r:
            trace.lost += 1
            continue
        began = r.get(JobState.SWITCHING) or r.get(JobState.RUNNING)
        job = Job(jid, r["arrival"], r["requester"], model, observed_wait=began - r["queued"] if began else None,
                  interactive=Classes().interactive(catalog, "", r["requester"]),
                  pooled=catalog.models[model]["runner"] == Runner.LLM_UNIT)
        job.run_s = r["ended"] - r[JobState.RUNNING] if JobState.RUNNING in r else 0.0
        if JobState.SWITCHING in r and JobState.RUNNING in r and r[JobState.RUNNING] - r[JobState.SWITCHING] > REAL_SWITCH_S:
            switches.setdefault(model, []).append(r[JobState.RUNNING] - r[JobState.SWITCHING])
        trace.jobs.append(job)
        ends.setdefault(job.requester, []).append((r["ended"], jid))
    _chain(trace.jobs, ends)
    every = [x for xs in switches.values() for x in xs] or [x for xs in ready.values() for x in xs]
    fallback = statistics.median(every) if every else DEFAULT_SWITCH_S
    for key in catalog.models:
        measured = switches.get(key) or ready.get(key)
        trace.switch_s[key] = statistics.median(measured) if measured else fallback
    trace.jobs.sort(key=lambda j: j.arrival)
    return trace


def _chain(jobs: list[Job], ends: dict[str, list[tuple[float, str]]]) -> None:
    """Tie each job to an earlier end of its requester's jobs within CHAIN_S: the latest one not
    already taken, so one end releases at most one job and a client's concurrency stays what it was."""
    taken: set[str] = set()
    for xs in ends.values():
        xs.sort()
    for j in sorted(jobs, key=lambda j: j.arrival):
        xs = ends[j.requester]
        i = bisect.bisect_right(xs, (j.arrival, chr(0x10FFFF)))
        while i and j.arrival - xs[i - 1][0] <= CHAIN_S:
            end, prev = xs[i - 1]
            if prev not in taken and prev != j.id:
                taken.add(prev)
                j.after, j.gap = prev, j.arrival - end
                break
            i -= 1
