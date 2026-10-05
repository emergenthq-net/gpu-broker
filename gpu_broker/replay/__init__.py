"""Replay a broker's event log through a scheduling policy in simulated time.

`trace` turns an `events.jsonl` into the jobs that went through the line (arrival, requester,
model, measured run time) and per-model switch costs; `sim` runs them through a policy with the
scheduler's rules (pool slots, drain before a switch, idle restore); `report` prints waits per
requester and residency churn. Nothing is contacted: the log and the catalog are the inputs.
"""
