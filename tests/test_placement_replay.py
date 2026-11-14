"""End-to-end offline replay of remote catalog to authorized placement.

All inputs are synthetic. No network, GPU, or billing effects.
"""
from dataclasses import replace

from gpu_broker.placement_authority import AuthorityDecision
from gpu_broker.placement_offers import quote_to_offer
from gpu_broker.placement_planner import choose_placement
from gpu_broker.placement_preflight import Refusal, Requirement
from gpu_broker.provider_quotes import parse_supercomputer_skus


def test_catalog_to_placement_replay() -> None:
    catalog = {"skus": [
        {"name": "gpu-a", "vram_gb": 80, "cost_per_hr": 2.0, "availability": "live"},
        {"name": "gpu-b", "vram_gb": 96, "cost_per_hr": 3.0, "availability": "live"},
    ]}
    quotes = parse_supercomputer_skus(catalog, observed_at_s=100)
    offers = tuple(
        quote_to_offer(q, duration_s=1800, estimated_latency_ms=2000, additional_cost_usd=0.1)
        for q in quotes
    )
    requirement = Requirement("cone", "impl", 1024, 2.0, "grant", True)

    def verifier(req, cap):
        return AuthorityDecision(req.authority_ref or "", req.cone_id, req.implementation_id, cap.provider_id, True)

    result = choose_placement(requirement, offers, verifier=verifier, now_s=110, max_age_s=20)
    assert result.chosen == offers[0]
    assert result.chosen.estimated_total_usd == 1.1
    assert result.refusals == ()

    blocked = choose_placement(replace(requirement, allow_remote=False), offers,
                               verifier=verifier, now_s=110, max_age_s=20)
    assert blocked.chosen is None
    assert len(blocked.refusals) == 2
    assert all(reason == Refusal.DATA_POLICY for _, reason in blocked.refusals)


def test_missing_non_compute_cost_blocks_selection() -> None:
    quote = parse_supercomputer_skus({"skus": [
        {"name": "gpu-a", "vram_gb": 80, "cost_per_hr": 2.0, "availability": "live"},
    ]}, observed_at_s=100)[0]
    offer = quote_to_offer(quote, duration_s=1800, estimated_latency_ms=2000, additional_cost_usd=None)
    requirement = Requirement("cone", "impl", 1024, 2.0, "grant", True)

    def verifier(req, cap):
        return AuthorityDecision(req.authority_ref or "", req.cone_id, req.implementation_id, cap.provider_id, True)

    result = choose_placement(requirement, (offer,), verifier=verifier, now_s=110, max_age_s=20)
    assert result.chosen is None
    assert result.refusals == (("supercomputer.sh", Refusal.UNKNOWN_COST),)
