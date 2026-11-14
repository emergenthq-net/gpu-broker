"""Persistent claim model regression tests; no provider effects."""
from concurrent.futures import ThreadPoolExecutor

from gpu_broker.execution_grant import ExecutionBinding, VerifiedGrant
from gpu_broker.offline_claim_store import ClaimStatus
from gpu_broker.sqlite_claim_store import SqliteClaimStore


def test_reopen_does_not_reset_claim(tmp_path) -> None:
    path = tmp_path / "claims.db"
    binding = ExecutionBinding("permit", "attempt", "cone", "impl", "provider", 1000, 200)
    grant = VerifiedGrant(binding, True, False)
    assert SqliteClaimStore(path).claim(binding, grant, now_epoch_s=100).status == ClaimStatus.CLAIMED
    assert SqliteClaimStore(path).claim(binding, grant, now_epoch_s=100).status == ClaimStatus.ALREADY_CLAIMED


def test_competing_sqlite_connections_one_winner(tmp_path) -> None:
    path = tmp_path / "claims.db"
    binding = ExecutionBinding("permit", "attempt", "cone", "impl", "provider", 1000, 200)
    grant = VerifiedGrant(binding, True, False)
    SqliteClaimStore(path)

    def attempt(_: int) -> ClaimStatus:
        return SqliteClaimStore(path).claim(binding, grant, now_epoch_s=100).status

    with ThreadPoolExecutor(max_workers=8) as pool:
        statuses = list(pool.map(attempt, range(24)))
    assert statuses.count(ClaimStatus.CLAIMED) == 1
    assert statuses.count(ClaimStatus.ALREADY_CLAIMED) == 23
