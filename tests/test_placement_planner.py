"""Zero-spend placement ranking fixtures."""
from dataclasses import replace

from gpu_broker.placement_authority import AuthorityDecision
from gpu_broker.placement_planner import PlacementOffer, choose_placement
from gpu_broker.placement_preflight import Capability, Refusal, Requirement

REQ = Requirement("cone", "impl", 1024, 2.0, "grant", True)
LOCAL = Capability("local", 4096, 100.0, False)
REMOTE = Capability("remote", 8192, 100.0, True)


def verify(req: Requirement, cap: Capability) -> AuthorityDecision:
    return AuthorityDecision(req.authority_ref or "", req.cone_id, req.implementation_id, cap.provider_id, True)


def test_cheapest_eligible_offer_wins() -> None:
    offers = (PlacementOffer(LOCAL, 1.5, 10), PlacementOffer(REMOTE, 1.0, 100))
    result = choose_placement(REQ, offers, verifier=verify, now_s=110, max_age_s=20)
    assert result.chosen == offers[1]


def test_refuse_over_budget_and_unknown_latency() -> None:
    offers = (PlacementOffer(LOCAL, 3.0, 10), PlacementOffer(REMOTE, 1.0, None))
    result = choose_placement(REQ, offers, verifier=verify, now_s=110, max_age_s=20)
    assert result.chosen is None
    assert ("local", Refusal.BUDGET_EXCEEDED) in result.refusals
    assert ("remote", Refusal.INVALID_REQUIREMENT) in result.refusals


def test_remote_policy_exclusion() -> None:
    result = choose_placement(
        replace(REQ, allow_remote=False), (PlacementOffer(REMOTE, 0.1, 1),),
        verifier=verify, now_s=110, max_age_s=20,
    )
    assert result.chosen is None
    assert result.refusals == (("remote", Refusal.DATA_POLICY),)
