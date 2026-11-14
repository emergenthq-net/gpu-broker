"""Offline preflight regression fixtures."""
from dataclasses import replace

import pytest

from gpu_broker.placement_preflight import Capability, Refusal, Requirement, evaluate

BASE = Requirement("cone", "impl", 1024, 2.0, "authorized-ref", False)
LOCAL = Capability("local", 4096, 100.0, False)


def check(req: Requirement = BASE, cap: Capability = LOCAL, cost: float | None = 1.0):
    return evaluate(req, cap, now_s=110.0, max_age_s=20.0, estimated_total_usd=cost, authority_verified=True)


def test_unverified_authority_is_rejected() -> None:
    result = evaluate(BASE, LOCAL, now_s=110.0, max_age_s=20.0, estimated_total_usd=1.0)
    assert result.refusal == Refusal.UNVERIFIED_AUTHORITY
    assert not result.accepted


def test_valid_local_candidate() -> None:
    assert check().accepted


@pytest.mark.parametrize(
    ("req", "cap", "cost", "expected"),
    [
        (replace(BASE, authority_ref=None), LOCAL, 1.0, Refusal.MISSING_AUTHORITY),
        (BASE, replace(LOCAL, observed_at_s=1.0), 1.0, Refusal.STALE_CAPABILITY),
        (BASE, replace(LOCAL, observed_at_s=111.0), 1.0, Refusal.STALE_CAPABILITY),
        (replace(BASE, memory_bytes=8192), LOCAL, 1.0, Refusal.INSUFFICIENT_MEMORY),
        (BASE, replace(LOCAL, remote=True), 1.0, Refusal.DATA_POLICY),
        (BASE, LOCAL, None, Refusal.UNKNOWN_COST),
        (BASE, LOCAL, float("nan"), Refusal.UNKNOWN_COST),
        (BASE, LOCAL, 3.0, Refusal.BUDGET_EXCEEDED),
        (replace(BASE, cone_id=""), LOCAL, 1.0, Refusal.INVALID_REQUIREMENT),
        (replace(BASE, max_cost_usd=float("inf")), LOCAL, 1.0, Refusal.INVALID_REQUIREMENT),
    ],
)
def test_refusal(req: Requirement, cap: Capability, cost: float | None, expected: Refusal) -> None:
    result = check(req, cap, cost)
    assert not result.accepted
    assert result.refusal == expected


def test_remote_requires_explicit_permission() -> None:
    assert check(replace(BASE, allow_remote=True), replace(LOCAL, remote=True)).accepted


def test_budget_boundary() -> None:
    assert check(cost=2.0).accepted
    assert check(cost=2.01).refusal == Refusal.BUDGET_EXCEEDED
