"""SQLite-backed offline claim replay model, not an ARCOS authority gateway.

Requires a trusted caller and an already-verified grant. SQLite BEGIN IMMEDIATE
serializes competing writers on the same database file across processes.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

from .execution_grant import ExecutionBinding, VerifiedGrant, check_grant
from .offline_claim_store import ClaimOutcome, ClaimStatus


class SqliteClaimStore:
    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        with self._connect() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS claims (permit_id TEXT NOT NULL, "
                "attempt_id TEXT NOT NULL, PRIMARY KEY (permit_id, attempt_id))"
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.database_path, timeout=5.0, isolation_level=None)

    def claim(
        self, expected: ExecutionBinding, grant: VerifiedGrant | None, *, now_epoch_s: int,
    ) -> ClaimOutcome:
        refusal = check_grant(expected, grant, now_epoch_s=now_epoch_s)
        if refusal is not None:
            return ClaimOutcome(ClaimStatus.REFUSED, refusal)
        with self._connect() as conn:
            try:
                conn.execute("BEGIN IMMEDIATE")
                row = conn.execute(
                    "SELECT 1 FROM claims WHERE permit_id = ? AND attempt_id = ?",
                    (expected.permit_id, expected.attempt_id),
                ).fetchone()
                if row is not None:
                    conn.execute("ROLLBACK")
                    return ClaimOutcome(ClaimStatus.ALREADY_CLAIMED)
                conn.execute(
                    "INSERT INTO claims (permit_id, attempt_id) VALUES (?, ?)",
                    (expected.permit_id, expected.attempt_id),
                )
                conn.execute("COMMIT")
                return ClaimOutcome(ClaimStatus.CLAIMED)
            except BaseException:
                if conn.in_transaction:
                    conn.execute("ROLLBACK")
                raise
