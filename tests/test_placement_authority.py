"""Offline verifier-boundary fixtures."""
from dataclasses import replace

import pytest

from gpu_broker.placement_authority import AuthorityDecision, verified_preflight
from gpu_broker.placement_preflight import Capability, Refusal, Requirement

REQ = Requirement("cone-a", "impl-a", 1024, 2.0, "grant-a", True)
CAP = Capability("provider-a", 4096, 100.0, True)
GRANT = AuthorityDecision("grant-a", "cone-a", "impl-a", "provider-a", True)


def run(verifier):
    return verified_preflight(
        REQ, CAP, verifier=verifier, now_s=110.0,
        max_age_s=20.0, estimated_total_usd=1.0,
    )


def test_matching_server_verifier() -> None:
    assert run(lambda _req, _cap: GRANT).accepted


@pytest.mark.parametrize(
    "decision",
    [
        None,
        replace(GRANT, permitted=False),
        replace(GRANT, authority_ref="other"),
        replace(GRANT, cone_id="other"),
        replace(GRANT, implementation_id="other"),
        replace(GRANT, provider_id="other"),
    ],
)
def test_denied_or_unbound_grant(decision: AuthorityDecision | None) -> None:
    assert run(lambda _req, _cap: decision).refusal == Refusal.UNVERIFIED_AUTHORITY


def test_verifier_failure_is_fail_closed() -> None:
    def broken(_req: Requirement, _cap: Capability) -> AuthorityDecision:
        raise RuntimeError("verifier unavailable")

    assert run(broken).refusal == Refusal.UNVERIFIED_AUTHORITY
