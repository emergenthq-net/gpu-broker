"""Offline SQLite budget reservation model; not an ARCOS execution authority.

Atomic reservation across distinct attempts on a shared local SQLite database.
"""
from __future__ import annotations

import sqlite3
from enum import StrEnum
from pathlib import Path


class ReservationStatus(StrEnum):
    RESERVED = "reserved"
    DUPLICATE = "duplicate"
    INSUFFICIENT = "insufficient"
    INVALID = "invalid"


class SqliteBudgetLedger:
    def __init__(self, path: Path) -> None:
        self.path = path
        with self._connect() as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS budgets (scope TEXT PRIMARY KEY, limit_micro INTEGER NOT NULL CHECK(limit_micro >= 0))")
            conn.execute("CREATE TABLE IF NOT EXISTS reservations (attempt TEXT PRIMARY KEY, scope TEXT NOT NULL, micro INTEGER NOT NULL CHECK(micro >= 0))")

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=5.0, isolation_level=None)

    def set_initial_limit(self, scope: str, limit_micro: int) -> bool:
        if not scope or type(limit_micro) is not int or not 0 <= limit_micro <= 9223372036854775807:
            return False
        with self._connect() as conn:
            cursor = conn.execute("INSERT OR IGNORE INTO budgets (scope, limit_micro) VALUES (?, ?)", (scope, limit_micro))
            return cursor.rowcount == 1

    def reserve(self, scope: str, attempt: str, micro: int) -> ReservationStatus:
        if not scope or not attempt or type(micro) is not int or not 0 <= micro <= 9223372036854775807:
            return ReservationStatus.INVALID
        with self._connect() as conn:
            try:
                conn.execute("BEGIN IMMEDIATE")
                if conn.execute("SELECT 1 FROM reservations WHERE attempt = ?", (attempt,)).fetchone():
                    conn.execute("ROLLBACK")
                    return ReservationStatus.DUPLICATE
                budget = conn.execute("SELECT limit_micro FROM budgets WHERE scope = ?", (scope,)).fetchone()
                if budget is None:
                    conn.execute("ROLLBACK")
                    return ReservationStatus.INVALID
                used = conn.execute("SELECT micro FROM reservations WHERE scope = ?", (scope,)).fetchall()
                remaining = budget[0]
                for (charge,) in used:
                    remaining -= charge
                if micro > remaining:
                    conn.execute("ROLLBACK")
                    return ReservationStatus.INSUFFICIENT
                conn.execute("INSERT INTO reservations (attempt, scope, micro) VALUES (?, ?, ?)", (attempt, scope, micro))
                conn.execute("COMMIT")
                return ReservationStatus.RESERVED
            except BaseException:
                if conn.in_transaction:
                    conn.execute("ROLLBACK")
                raise
