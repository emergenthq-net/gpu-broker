"""Opaque server-side authorization handle for a future ARCOS effect gateway.

This is an offline contract only. A grant handle is NOT a bearer credential.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class GrantRefusal(StrEnum):
    INVALID_BINDING = "invalid_binding"
    EXPIRED = "expired"
    REPLAY = "replay"
    UNVERIFIED = "unverified"


@dataclass(frozen=True)
class ExecutionBinding:
    permit_id: str
    attempt_id: str
    cone_id: str
    implementation_id: str
    provider_id: str
    max_cost_microusd: int
    expires_at_epoch_s: int


@dataclass(frozen=True)
class VerifiedGrant:
    """Only a trusted server gateway may construct/return this decision."""

    binding: ExecutionBinding
    verified: bool
    consumed: bool


def check_grant(
    expected: ExecutionBinding, grant: VerifiedGrant | None, *, now_epoch_s: int,
) -> GrantRefusal | None:
    """Pure final check. Atomic consumption must happen in the trusted gateway."""
    if (
        not expected.permit_id or not expected.attempt_id or not expected.cone_id
        or not expected.implementation_id or not expected.provider_id
        or isinstance(expected.max_cost_microusd, bool)
        or not isinstance(expected.max_cost_microusd, int)
        or expected.max_cost_microusd < 0
        or isinstance(expected.expires_at_epoch_s, bool)
        or not isinstance(expected.expires_at_epoch_s, int)
        or expected.expires_at_epoch_s < 0
        or isinstance(now_epoch_s, bool) or not isinstance(now_epoch_s, int)
        or now_epoch_s < 0
    ):
        return GrantRefusal.INVALID_BINDING
    if grant is None or not grant.verified:
        return GrantRefusal.UNVERIFIED
    if grant.binding != expected:
        return GrantRefusal.INVALID_BINDING
    if now_epoch_s >= expected.expires_at_epoch_s:
        return GrantRefusal.EXPIRED
    if grant.consumed:
        return GrantRefusal.REPLAY
    return None
