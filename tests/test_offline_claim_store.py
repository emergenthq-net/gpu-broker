"""Offline concurrency model; does not authenticate or persist permits."""
from concurrent.futures import ThreadPoolExecutor

from gpu_broker.execution_grant import ExecutionBinding, VerifiedGrant
from gpu_broker.offline_claim_store import ClaimStatus, OfflineClaimStore

BINDING = ExecutionBinding("permit", "attempt", "cone", "impl", "provider", 1000000, 200)


def test_concurrent_claims_exactly_one_winner() -> None:
    store = OfflineClaimStore()
    grant = VerifiedGrant(BINDING, True, False)
    with ThreadPoolExecutor(max_workers=16) as pool:
        results = list(pool.map(lambda _: store.claim(BINDING, grant, now_epoch_s=100), range(64)))
    assert sum(result.status == ClaimStatus.CLAIMED for result in results) == 1
    assert sum(result.status == ClaimStatus.ALREADY_CLAIMED for result in results) == 63


def test_refusal_does_not_consume_claim() -> None:
    store = OfflineClaimStore()
    assert store.claim(BINDING, None, now_epoch_s=100).status == ClaimStatus.REFUSED
    assert store.claim(BINDING, VerifiedGrant(BINDING, True, False), now_epoch_s=100).status == ClaimStatus.CLAIMED
