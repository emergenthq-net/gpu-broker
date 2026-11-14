"""Offline concurrent budget reservation tests."""
from concurrent.futures import ThreadPoolExecutor

from gpu_broker.sqlite_budget_ledger import ReservationStatus, SqliteBudgetLedger


def test_competing_attempts_cannot_overspend(tmp_path) -> None:
    path = tmp_path / "budget.db"
    assert SqliteBudgetLedger(path).set_initial_limit("scope", 100)

    def reserve(i: int) -> ReservationStatus:
        return SqliteBudgetLedger(path).reserve("scope", f"attempt-{i}", 20)

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(reserve, range(12)))
    assert results.count(ReservationStatus.RESERVED) == 5
    assert results.count(ReservationStatus.INSUFFICIENT) == 7


def test_duplicate_attempt_cannot_double_reserve(tmp_path) -> None:
    ledger = SqliteBudgetLedger(tmp_path / "budget.db")
    assert ledger.set_initial_limit("scope", 100)
    assert ledger.reserve("scope", "attempt", 70) == ReservationStatus.RESERVED
    assert ledger.reserve("scope", "attempt", 1) == ReservationStatus.DUPLICATE
    assert ledger.reserve("scope", "another", 31) == ReservationStatus.INSUFFICIENT


def test_rejects_malformed_amounts_and_missing_scope(tmp_path) -> None:
    ledger = SqliteBudgetLedger(tmp_path / "budget.db")
    assert not ledger.set_initial_limit("scope", True)
    assert ledger.reserve("scope", "attempt", True) == ReservationStatus.INVALID
    assert ledger.reserve("missing", "attempt", 1) == ReservationStatus.INVALID
