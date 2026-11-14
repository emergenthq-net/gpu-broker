"""Pure placement comparison; no provider calls or execution effects."""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from .placement_authority import AuthorityVerifier, verified_preflight
from .placement_preflight import Capability, Refusal, Requirement


@dataclass(frozen=True)
class PlacementOffer:
    capability: Capability
    estimated_total_usd: float | None
    estimated_latency_ms: float | None


@dataclass(frozen=True)
class PlacementResult:
    chosen: PlacementOffer | None
    refusals: tuple[tuple[str, Refusal], ...]


def choose_placement(
    requirement: Requirement,
    offers: tuple[PlacementOffer, ...],
    *,
    verifier: AuthorityVerifier,
    now_s: float,
    max_age_s: float,
) -> PlacementResult:
    """Rank only preflight-accepted offers; unknown latency is not ranked."""
    candidates: list[PlacementOffer] = []
    refusals: list[tuple[str, Refusal]] = []
    for offer in offers:
        decision = verified_preflight(
            requirement, offer.capability, verifier=verifier, now_s=now_s,
            max_age_s=max_age_s, estimated_total_usd=offer.estimated_total_usd,
        )
        if not decision.accepted:
            if decision.refusal is not None:
                refusals.append((offer.capability.provider_id, decision.refusal))
            continue
        latency = offer.estimated_latency_ms
        if latency is None or not isfinite(latency) or latency < 0:
            refusals.append((offer.capability.provider_id, Refusal.INVALID_REQUIREMENT))
            continue
        candidates.append(offer)
    candidates.sort(key=lambda o: (
        o.estimated_total_usd if o.estimated_total_usd is not None else float("inf"),
        o.estimated_latency_ms if o.estimated_latency_ms is not None else float("inf"),
        o.capability.provider_id,
    ))
    return PlacementResult(candidates[0] if candidates else None, tuple(refusals))
