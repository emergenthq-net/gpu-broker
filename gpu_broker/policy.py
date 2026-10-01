"""Pure queue policy for choosing the next GPU job.

Execution and residency stay in scheduler.py/residency.py. This module only orders pending
jobs from declarative facts, so policy can evolve without changing the single-writer GPU
invariant.

"balanced" has three deterministic rules:
1. interactive > normal > background;
2. waiting promotes a job one band per `aging_s`, preventing starvation;
3. within the same effective band and age, work for the resident model goes first, avoiding
   a needless unload/reload cycle. Sequence number is the final stable tie-breaker.

"fifo" ignores all policy metadata and preserves strict submission order.
"""
from __future__ import annotations

from dataclasses import dataclass

from .constants import Priority

FIFO = "fifo"
BALANCED = "balanced"
POLICIES = frozenset({FIFO, BALANCED})
_PRIORITY = {Priority.INTERACTIVE: 0, Priority.NORMAL: 1, Priority.BACKGROUND: 2}


@dataclass(frozen=True)
class Candidate:
    jid: str
    sequence: int
    priority: Priority
    queued_at: float
    resolved: str | None


def normalize_priority(value: str | Priority | None, interactive: bool = False) -> Priority:
    if interactive:
        return Priority.INTERACTIVE
    if isinstance(value, Priority):
        return value
    try:
        return Priority((value or Priority.NORMAL).strip().lower())
    except ValueError as e:
        raise ValueError(f"unknown priority {value!r}; expected one of {[p.value for p in Priority]}") from e


def order(candidates: list[Candidate], policy: str, resident: str | None, now: float, aging_s: float) -> list[Candidate]:
    if policy not in POLICIES:
        raise ValueError(f"unknown scheduler policy {policy!r}; expected one of {sorted(POLICIES)}")
    if policy == FIFO:
        return sorted(candidates, key=lambda c: c.sequence)

    def key(c: Candidate) -> tuple[int, int, int, int]:
        waited = max(0.0, now - c.queued_at)
        promotions = int(waited // aging_s) if aging_s > 0 else 0
        effective = max(0, _PRIORITY[c.priority] - promotions)
        locality_miss = int(not resident or c.resolved != resident)
        return effective, -promotions, locality_miss, c.sequence

    return sorted(candidates, key=key)
