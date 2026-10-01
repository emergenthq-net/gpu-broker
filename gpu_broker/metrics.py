"""Live metrics: GPU samples from the driver, and per-job latency and LLM throughput.

GPU: the driver's `gpu_stream()` yields a sample line every couple of seconds; `GpuSampler`
keeps the last hour in memory and reconnects when the stream drops. VRAM is split by owner
group (a container id, systemd unit or Docker container), labelled by config `ui.groups`.

Jobs: phase timings come from the event log (received → switching → running → finished);
LLM throughput comes from the `timings` block llama.cpp-style servers add to a completion.
"""
from __future__ import annotations

import collections
import json
import threading
import time
from collections.abc import Callable, Iterable, Iterator
from typing import Any

from .constants import ERR_SHORT, JobState
from .store import Store

Sample = dict[str, Any]
SUMMARY_FIELDS = ("total_s", "queue_s", "switch_s", "run_s", "ttft_s", "gen_tps", "prompt_tps")
PERCENTILES = {"p50": 0.5, "p95": 0.95}
MS_PER_S = 1000
DECIMALS = 2
STREAM_ENDED = "stream ended"


def parse_sample(line: str, now: float) -> Sample | None:
    """`used,total,util,power,temp,sm|group:mib ...` → sample dict (None if malformed)."""
    head, _, procs = line.strip().partition("|")
    try:
        used, total, util, power, temp, sm = (float(x) for x in head.split(","))
    except ValueError:
        return None
    by_group: dict[str, int] = collections.Counter()
    for p in procs.split():
        group, _, mib = p.rpartition(":")
        if group and mib.isdigit():
            by_group[group] += int(mib)
    return {"t": now, "used_mib": int(used), "total_mib": int(total), "util_pct": int(util),
            "power_w": round(power, 1), "temp_c": int(temp), "sm_mhz": int(sm), "by_group": dict(by_group)}


class GpuSampler:
    def __init__(self, stream: Callable[[], Iterator[str]], keep: int, retry_s: float) -> None:
        self.stream, self.retry_s = stream, retry_s
        self._buf: collections.deque[Sample] = collections.deque(maxlen=keep)
        self._lock = threading.Lock()
        self.error = ""

    def loop(self, stop: threading.Event) -> None:
        while not stop.is_set():
            try:
                for line in self.stream():
                    if stop.is_set():
                        return
                    if s := parse_sample(line, time.time()):
                        with self._lock:
                            self._buf.append(s)
                        self.error = ""
                self.error = STREAM_ENDED
            except Exception as e:  # noqa: BLE001 — shown on the dashboard, then reconnect
                self.error = str(e)[:ERR_SHORT]
            stop.wait(self.retry_s)

    def since(self, t: float) -> list[Sample]:
        with self._lock:
            return [s for s in self._buf if s["t"] > t]


def job_metrics(store: Store, since_ts: float, lookback_s: float) -> list[dict[str, Any]]:
    """One row per finished job: phase durations in seconds and, for LLM jobs, throughput.
    Phase events are read from `lookback_s` before the window: a long job started earlier."""
    phases = store.phase_times(since_ts - lookback_s)
    rows = []
    for j in store.finished_jobs(since_ts):
        p = phases.get(j["id"], {})
        end: float = j["updated"]
        rec: float = p.get(JobState.RECEIVED) or j["created"]
        sw, run = p.get(JobState.SWITCHING), p.get(JobState.RUNNING)
        r: dict[str, Any] = {
            "id": j["id"], "model": j["resolved"], "state": j["state"], "t": end,
            "total_s": round(end - rec, DECIMALS), "queue_s": round((sw or run or end) - rec, DECIMALS),
            "switch_s": round(run - sw, DECIMALS) if sw and run else None,
            "run_s": round(end - run, DECIMALS) if run else None}
        if tm := _timings(j.get("result")):
            r.update(prompt_tps=round(tm.get("prompt_per_second", 0), 1), gen_tps=round(tm.get("predicted_per_second", 0), 1),
                     prompt_tokens=tm.get("prompt_n"), cached_tokens=tm.get("cache_n"), gen_tokens=tm.get("predicted_n"),
                     ttft_s=round((run or rec) - rec + tm.get("prompt_ms", 0) / MS_PER_S, DECIMALS))
        rows.append(r)
    return rows


def _timings(result: Any) -> dict[str, Any] | None:
    if isinstance(result, str):
        try:
            result = json.loads(result)
        except ValueError:
            return None
    tm = result.get("timings") if isinstance(result, dict) else None
    return tm if isinstance(tm, dict) else None


def percentile(values: Iterable[float | None], q: float) -> float | None:
    xs = sorted(x for x in values if x is not None)
    return round(xs[min(len(xs) - 1, int(q * len(xs)))], DECIMALS) if xs else None


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    done = [r for r in rows if r["state"] == JobState.DONE]
    out: dict[str, Any] = {f: {name: percentile((r.get(f) for r in done), q) for name, q in PERCENTILES.items()}
                           for f in SUMMARY_FIELDS}
    out["gen_tokens_total"] = sum(r.get("gen_tokens") or 0 for r in rows)
    out["jobs"] = len(rows)
    return out
