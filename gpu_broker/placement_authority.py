"""Server-owned authority verifier boundary for candidate placement.

This is a pure orchestration seam, not a credential validator or network client.
Never deserialize a verifier callable from a request or provider response.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .placement_preflight import Capability, Preflight, Requirement, evaluate


@dataclass(frozen=True)
class AuthorityDecision:
    """Decision from a trusted verifier bound to an exact execution request."""

    authority_ref: str
    cone_id: str
    implementation_id: str
    provider_id: str
    permitted: bool


AuthorityVerifier = Callable[[Requirement, Capability], AuthorityDecision | None]


def verified_preflight(
    requirement: Requirement,
    capability: Capability,
    *,
    verifier: AuthorityVerifier,
    now_s: float,
    max_age_s: float,
    estimated_total_usd: float | None,
) -> Preflight:
    """Use only a server-injected verifier; fail closed on errors or mismatches.

    This helper does not itself authenticate a grant. Its caller must provide
    a verifier that checks signature, scope, expiry, revocation and principal.
    """
    try:
        decision = verifier(requirement, capability)
    except Exception:  # noqa: BLE001 - untrusted verifier failure must always refuse placement
        decision = None
    verified = (
        decision is not None
        and decision.permitted is True
        and decision.authority_ref == requirement.authority_ref
        and decision.cone_id == requirement.cone_id
        and decision.implementation_id == requirement.implementation_id
        and decision.provider_id == capability.provider_id
    )
    return evaluate(
        requirement,
        capability,
        now_s=now_s,
        max_age_s=max_age_s,
        estimated_total_usd=estimated_total_usd,
        authority_verified=verified,
    )
