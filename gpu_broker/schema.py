"""The SQLite schema, and the migrations that bring an older database up to it."""
from __future__ import annotations

import sqlite3

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
  id TEXT PRIMARY KEY, created REAL, updated REAL, requester TEXT,
  requested TEXT, resolved TEXT, substitution TEXT, state TEXT,
  payload TEXT, result TEXT, error TEXT, download TEXT, exec_recipe TEXT, owner TEXT);
CREATE TABLE IF NOT EXISTS events (
  seq INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, job_id TEXT, kind TEXT, data TEXT);
CREATE TABLE IF NOT EXISTS downloads (
  slug TEXT PRIMARY KEY, kind TEXT, ref TEXT, state TEXT, created REAL, updated REAL,
  catalog_key TEXT, error TEXT);
CREATE TABLE IF NOT EXISTS flags (name TEXT PRIMARY KEY, ts REAL, data TEXT);
"""
# jobs.exec_recipe: the recipe of an exec job, NOT_EXEC ('') for any other job this code
# created, NULL when the writer is unknown (an older or rolled-back broker wrote the row).
NOT_EXEC = ""
# jobs.owner: the id of the client key that submitted the job; NULL for the main token. MCP
# tools show a client key only its own jobs (names repeat; ids do not).
# Jobs left non-terminal by the previous process, with `direct` (a direct chat).
# Parameters: Event.JOB_DIRECT, then the IN list.
ORPHANS = ("SELECT id, requested, resolved, state, payload, exec_recipe, "
           "EXISTS(SELECT 1 FROM events WHERE job_id=jobs.id AND kind=?) AS direct "
           "FROM jobs WHERE state NOT IN ")


def migrate(db: sqlite3.Connection) -> None:
    db.executescript(SCHEMA)
    if "exec_recipe" not in {r[1] for r in db.execute("PRAGMA table_info(jobs)")}:
        db.execute("ALTER TABLE jobs ADD COLUMN exec_recipe TEXT")   # databases from before 2026-10-01
    if "owner" not in {r[1] for r in db.execute("PRAGMA table_info(jobs)")}:
        db.execute("ALTER TABLE jobs ADD COLUMN owner TEXT")         # databases from before 2026-10-03
