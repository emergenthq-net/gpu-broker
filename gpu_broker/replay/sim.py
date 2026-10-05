"""Simulated time through the scheduler's rules, under a queue policy.

The rules mirror `scheduler.py`: a job for the resident LLM takes a pool slot (background calls
may use `slots - reserved_interactive`); any other job first waits for the pool to drain, then
switches (the model's measured switch cost) and, unless it is an LLM call, runs on the GPU
thread; after `idle_restore_s` with nothing queued or in flight, the default model comes back.
A job chained to another (trace.Job.after) arrives `gap` seconds after that one ends here.
Which waiting job may start is `line.choose`, the live scheduler's own rule; a job that needs
the pool drained is taken at once and holds the GPU thread until it starts (as the live loop
does), so new pool calls cannot starve a switch.
Not simulated: restarts, quiesces, failed switches, direct chats (they never queue).
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field

from ..catalog import Catalog
from ..line import Snap, choose
from ..policy import DEFAULT_EVICT_WAIT_S, Policy
from .trace import Job, Trace

POOL, GPU = "pool", "gpu"


@dataclass
class Result:
    waits: dict[str, float] = field(default_factory=dict)             # job id → seconds in line
    ends: dict[str, float] = field(default_factory=dict)              # job id → when it ended
    starts: dict[str, float] = field(default_factory=dict)            # job id → when it left the line
    residency: list[tuple[float, str]] = field(default_factory=list)  # (time, model) per switch
    end: float = 0.0


class Sim:
    def __init__(self, trace: Trace, catalog: Catalog, policy: Policy, evict_wait_s: float = DEFAULT_EVICT_WAIT_S) -> None:
        self.trace, self.models, self.policy, self.evict_wait_s = trace, catalog.models, policy, evict_wait_s
        policy.clock = lambda: self.t   # waits and aging in simulated time
        self.want = catalog.defaults["resident"]
        self.idle_s = float(catalog.defaults["idle_restore_s"])
        self.resident = self.want
        self.t = trace.jobs[0].arrival if trace.jobs else trace.start
        self.arrivals: list[tuple[float, int, Job]] = []   # (sim arrival, order, job), heap
        self.released: dict[str, list[Job]] = {}           # job id → jobs its end releases
        for n, j in enumerate(trace.jobs):
            if j.after is None:
                heapq.heappush(self.arrivals, (j.arrival, n, j))
            else:
                self.released.setdefault(j.after, []).append(j)
        self.order = len(trace.jobs)
        self.pool: list[tuple[float, str]] = []   # (end, job id) of calls in flight, heap
        self.gpu_free, self.gpu_job = self.t, ""  # the GPU thread is busy until then (with that job)
        self.active = self.t                      # last activity, for the idle restore
        self.arrived: dict[str, float] = {}
        self.pending: Job | None = None           # taken by the GPU thread, waiting for the pool to drain
        self.out = Result()

    def start(self, job: Job, how: str) -> None:
        if job is self.pending:
            self.pending = None
        else:
            self.policy.remove(job)
        self.out.waits[job.id] = self.t - self.arrived[job.id]
        self.out.starts[job.id] = self.t
        if how == POOL:
            heapq.heappush(self.pool, (self.t + job.run_s, job.id))
            return
        switch = 0.0
        if job.model != self.resident:
            switch, self.resident = self.trace.switch_s[job.model], job.model
            self.out.residency.append((self.t, job.model))
        if job.pooled:
            self.gpu_free = self.t + switch
            heapq.heappush(self.pool, (self.gpu_free + job.run_s, job.id))
        else:
            self.gpu_free = self.active = self.t + switch + job.run_s
            self.gpu_job = job.id

    def finish(self, jid: str, at: float) -> None:
        self.out.ends[jid] = at
        self.active = max(self.active, at)
        for j in self.released.pop(jid, []):
            heapq.heappush(self.arrivals, (at + j.gap, self.order, j))
            self.order += 1

    def dispatch(self) -> None:
        while self.t >= self.gpu_free:
            if self.gpu_job:
                self.finish(self.gpu_job, self.gpu_free)
                self.gpu_job = ""   # the jobs it releases arrive `gap` later, via the main loop
            if self.pending is not None:
                if self.pool:
                    return
                self.start(self.pending, GPU)
                continue
            snap = Snap.of(self.resident, len(self.pool), self.models)
            job = choose(self.policy, snap, self.evict_wait_s)
            if job is None:
                self.restore()
                return
            if job.pooled and job.model == self.resident:
                if snap.blocked(job):   # FIFO: the head waits for a slot, and so does everything behind it
                    return
                self.start(job, POOL)
            elif self.pool:   # taken now; nothing else starts until the pool drains
                self.policy.remove(job)
                self.pending = job
                return
            else:
                self.start(job, GPU)

    def restore(self) -> None:
        if (not len(self.policy) and self.pending is None and not self.pool and self.resident != self.want
                and self.t - self.active >= self.idle_s):
            self.resident = self.want
            self.out.residency.append((self.t, self.want))
            self.gpu_free = self.active = self.t + self.trace.switch_s[self.want]

    def step(self) -> bool:
        while self.arrivals and self.arrivals[0][0] <= self.t:
            at, _, job = heapq.heappop(self.arrivals)
            self.arrived[job.id] = at
            self.policy.add(job)
        while self.pool and self.pool[0][0] <= self.t:
            end, jid = heapq.heappop(self.pool)
            self.finish(jid, end)
        self.dispatch()
        nxt = [self.arrivals[0][0]] if self.arrivals else []
        nxt += [self.pool[0][0]] if self.pool else []
        nxt += [self.gpu_free] if self.gpu_free > self.t or self.gpu_job else []
        if self.resident != self.want and not len(self.policy) and not self.pool and self.active + self.idle_s > self.t:
            nxt.append(self.active + self.idle_s)   # the idle restore is due then
        if not nxt:
            return False
        self.t = max(self.t, min(nxt))
        return True

    def run(self) -> Result:
        while self.step():
            pass
        self.out.end = self.t
        return self.out
