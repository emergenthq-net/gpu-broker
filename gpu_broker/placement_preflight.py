"""Pure, zero-I/O preflight for candidate accelerator placements.

This module does not provision hardware or grant execution authority.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import isfinite


class Refusal(StrEnum):
    MISSING_AUTHORITY = "missing_authority"
    UNVERIFIED_AUTHORITY = "unverified_authority"
    STALE_CAPABILITY = "stale_capability"
    INSUFFICIENT_MEMORY = "insufficient_memory"
    DATA_POLICY = "data_policy"
    UNKNOWN_COST = "unknown_cost"
    BUDGET_EXCEEDED = "budget_exceeded"
    INVALID_REQUIREMENT = "invalid_requirement"


@dataclass(frozen=True)
class Requirement:
    cone_id: str
    implementation_id: str
    memory_bytes: int
    max_cost_usd: float
    authority_ref: str | None
    allow_remote: bool


@dataclass(frozen=True)
class Capability:
    provider_id: str
    memory_bytes: int
    observed_at_s: float
    remote: bool


@dataclass(frozen=True)
class Preflight:
    accepted: bool
    refusal: Refusal | None


def evaluate(
    requirement: Requirement,
    capability: Capability,
    *,
    now_s: float,
    max_age_s: float,
    estimated_total_usd: float | None,
    authority_verified: bool = False,
) -> Preflight:
    """Refuse unknown, stale or unauthorized placements before any effects."""
    if (
        not requirement.cone_id
        or not requirement.implementation_id
        or not capability.provider_id
        or requirement.memory_bytes <= 0
        or capability.memory_bytes < 0
        or not isfinite(requirement.max_cost_usd)
        or requirement.max_cost_usd < 0
        or not isfinite(now_s)
        or not isfinite(max_age_s)
        or max_age_s < 0
        or not isfinite(capability.observed_at_s)
    ):
        return Preflight(False, Refusal.INVALID_REQUIREMENT)
    if not requirement.authority_ref:
        return Preflight(False, Refusal.MISSING_AUTHORITY)
    if not authority_verified:
        return Preflight(False, Refusal.UNVERIFIED_AUTHORITY)
    if capability.observed_at_s > now_s or now_s - capability.observed_at_s > max_age_s:
        return Preflight(False, Refusal.STALE_CAPABILITY)
    if requirement.memory_bytes > capability.memory_bytes:
        return Preflight(False, Refusal.INSUFFICIENT_MEMORY)
    if capability.remote and not requirement.allow_remote:
        return Preflight(False, Refusal.DATA_POLICY)
    if estimated_total_usd is None or not isfinite(estimated_total_usd) or estimated_total_usd < 0:
        return Preflight(False, Refusal.UNKNOWN_COST)
    if estimated_total_usd > requirement.max_cost_usd:
        return Preflight(False, Refusal.BUDGET_EXCEEDED)
    return Preflight(True, None)
