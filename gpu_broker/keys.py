"""Per-client API keys: one named key per app or machine, so nobody copies the main token.

`gpu-broker connect` (or the dashboard) issues a key; only its SHA-256 is stored, so the
database never holds a usable key. A key is shown once, when it is issued, and can be
revoked; a revoked key is kept (with its name and dates) so the record stays. Keys reach the
model routes only (web/auth.py). `last_used` is written at most once a minute per key, so a
busy client does not turn every request into a database write.
"""
from __future__ import annotations

import hashlib
import os
import secrets
import sqlite3
import threading
import time
import uuid
from collections.abc import Callable
from typing import Any

from .connect.core import KEY_PREFIX, is_client_key  # recognisable in a config file or a leaked log line

KEY_BYTES = 32
ID_HEX = 12
SHOWN_CHARS = len(KEY_PREFIX) + 4   # how much of a key the list shows, to tell keys apart
NAME_MAX = 80
SCHEMA = """CREATE TABLE IF NOT EXISTS client_keys (
  id TEXT PRIMARY KEY, name TEXT, digest TEXT UNIQUE, shown TEXT,
  created REAL, last_used REAL, revoked REAL)"""
LIST_COLUMNS = "id, name, shown, created, last_used, revoked"
LAST_USED_EVERY_S = 60.0


def digest(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


class KeyStore:
    """Its own connection to the broker database (sqlite allows several), under its own lock."""

    def __init__(self, db_path: str, clock: Callable[[], float] = time.time) -> None:
        if os.path.dirname(db_path):
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._path, self._lock = db_path, threading.Lock()
        self._conn: sqlite3.Connection | None = None
        self._clock = clock
        self._seen: dict[str, float] = {}   # key id -> when last_used was last written

    @property
    def _db(self) -> sqlite3.Connection:
        """Opened on first use, and again after close(): an app lifespan may run more than once."""
        if self._conn is None:
            self._conn = sqlite3.connect(self._path, check_same_thread=False, isolation_level=None)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute(SCHEMA)
        return self._conn

    def issue(self, name: str) -> dict[str, Any]:
        """A new key named `name`; the only time the key itself is returned."""
        name = name.strip()[:NAME_MAX]
        if not name:
            raise ValueError("a key needs a name (the app or machine it is for)")
        key = KEY_PREFIX + secrets.token_urlsafe(KEY_BYTES)
        row = {"id": uuid.uuid4().hex[:ID_HEX], "name": name, "shown": key[:SHOWN_CHARS], "created": time.time()}
        with self._lock:
            self._db.execute("INSERT INTO client_keys (id, name, digest, shown, created) VALUES (?,?,?,?,?)",
                             (row["id"], name, digest(key), row["shown"], row["created"]))
        return {**row, "key": key}

    def check(self, key: str) -> str | None:
        """The name of a live key, or None."""
        found = self.identify(key)
        return found[1] if found else None

    def identify(self, key: str) -> tuple[str, str] | None:
        """(id, name) of a live key, or None. Names need not be unique; the id is. Lookup is by
        digest, so it leaks nothing about other keys."""
        if not is_client_key(key):
            return None
        with self._lock:
            r = self._db.execute("SELECT id, name FROM client_keys WHERE digest=? AND revoked IS NULL", (digest(key),)).fetchone()
            if r is None:
                return None
            now = self._clock()
            if now - self._seen.get(r["id"], -LAST_USED_EVERY_S) >= LAST_USED_EVERY_S:
                self._db.execute("UPDATE client_keys SET last_used=? WHERE id=?", (now, r["id"]))
                self._seen[r["id"]] = now
        return str(r["id"]), str(r["name"])

    def revoke(self, kid: str) -> bool:
        with self._lock:
            cur = self._db.execute("UPDATE client_keys SET revoked=? WHERE id=? AND revoked IS NULL", (time.time(), kid))
        return cur.rowcount == 1

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._db.execute(f"SELECT {LIST_COLUMNS} FROM client_keys ORDER BY created").fetchall()  # noqa: S608 — fixed columns
        return [dict(r) for r in rows]

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None
