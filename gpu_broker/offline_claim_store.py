"""Offline single-process claim-state model; NOT a production authorization store.

The caller must be an authenticated ARCOS service. No network or provider effects.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from threading import Lock

from .execution_grant import ExecutionBinding, GrantRefusal, VerifiedGrant, check_grant


class ClaimStatus(StrEnum):
    CLAIMED = "claimed"
    ALREADY_CLAIMED = "already_claimed"
    REFUSED = "refused"


@dataclass(frozen=True)
class ClaimOutcome:
    status: ClaimStatus
    refusal: GrantRefusal | None = None


class OfflineClaimStore:
    """Model atomic one-winner claims; no persistence or cross-process safety."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._claimed: set[tuple[str, str]] = set()

    def claim(
        self, expected: ExecutionBinding, grant: VerifiedGrant | None, *, now_epoch_s: int,
    ) -> ClaimOutcome:
        refusal = check_grant(expected, grant, now_epoch_s=now_epoch_s)
        if refusal is not None:
            return ClaimOutcome(ClaimStatus.REFUSED, refusal)
        key = (expected.permit_id, expected.attempt_id)
        with self._lock:
            if key in self._claimed:
                return ClaimOutcome(ClaimStatus.ALREADY_CLAIMED, GrantRefusal.REPLAY)
            self._claimed.add(key)
            return ClaimOutcome(ClaimStatus.CLAIMED)
