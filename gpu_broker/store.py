"""Jobs, events, downloads and flags. SQLite is the source of truth; events are also appended
to a JSONL file for log shippers. One connection is shared by every thread, so every
statement — reads included — runs under one lock: a sqlite3 connection is not safe for
interleaved use."""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import uuid
from typing import Any

from . import schema
from .constants import (
    ACTIVE_DOWNLOADS,
    DOWNLOAD_EVENT_PREFIX,
    JOB_EVENT_PREFIX,
    RESIDENCY_EVENT_PREFIX,
    STATS_EXTRA_EVENTS,
    TERMINAL,
    DownloadState,
    Event,
    JobState,
)

Row = dict[str, Any]

JOB_ID_HEX = 12
JSON_COLUMNS = ("payload", "result", "download")
UPDATABLE = frozenset({"resolved", "substitution", "state", "result", "error", "download", "payload"})
EVENT_FIELDS = ("resolved", "substitution", "error")   # job fields copied into its state event
PHASES = (JobState.RECEIVED, JobState.SWITCHING, JobState.RUNNING)
FINISHED = (JobState.DONE, JobState.FAILED)
ORPHANED = "orphaned by broker restart"
SQL_IN_MAX = 500   # ids per IN (...) list, well under SQLite's bound-variable limit
DECIMALS = 1


def _in(values: tuple[str, ...] | frozenset[str]) -> str:
    """`?,?,?` for an IN clause. The only thing ever interpolated into SQL here (S608 noqa's)."""
    return ",".join("?" * len(values))


def _decode(row: Row) -> Row:
    return {k: json.loads(v) if k in JSON_COLUMNS and v else v for k, v in row.items()}


class Store:
    def __init__(self, db_path: str, jsonl_path: str | None = None) -> None:
        for path in (db_path, jsonl_path):
            if path and os.path.dirname(path):
                os.makedirs(os.path.dirname(path), exist_ok=True)
        self._db = sqlite3.connect(db_path, check_same_thread=False, isolation_level=None)
        self._db.row_factory = sqlite3.Row
        schema.migrate(self._db)
        self._lock = threading.Lock()
        self._jsonl = jsonl_path or None
        # Notified after every job update (callers wait without polling); taken only after `_lock` is released.
        self.job_changed = threading.Condition()

    def _all(self, sql: str, args: tuple[Any, ...] = ()) -> list[Row]:
        with self._lock:
            return [dict(r) for r in self._db.execute(sql, args).fetchall()]

    def _exec(self, sql: str, args: tuple[Any, ...] = ()) -> int:
        with self._lock:
            return self._db.execute(sql, args).rowcount

    # ---- events ---------------------------------------------------------
    def event(self, kind: str, job_id: str | None = None, **data: Any) -> None:
        with self._lock:
            self._event(kind, job_id, data)

    def _event(self, kind: str, job_id: str | None, data: dict[str, Any]) -> None:
        """Log one event; the caller holds `_lock`."""
        ts = time.time()
        self._db.execute("INSERT INTO events(ts,job_id,kind,data) VALUES (?,?,?,?)", (ts, job_id, kind, json.dumps(data)))
        if self._jsonl:
            with open(self._jsonl, "a") as f:
                f.write(json.dumps({"ts": ts, "job_id": job_id, "kind": kind, **data}) + "\n")

    def events(self, since: int, limit: int) -> list[Row]:
        """Up to `limit` events after sequence number `since`; a negative `since` means the last
        |since| events (still capped at `limit`)."""
        if since < 0:
            rows = self._all("SELECT * FROM (SELECT * FROM events ORDER BY seq DESC LIMIT ?) ORDER BY seq",
                             (min(-since, limit),))
        else:
            rows = self._all("SELECT * FROM events WHERE seq>? ORDER BY seq LIMIT ?", (since, limit))
        return [{**r, "data": json.loads(r["data"])} for r in rows]

    # ---- jobs -----------------------------------------------------------
    def create_job(self, requester: str, requested: str, payload: dict[str, Any], exec_recipe: str = schema.NOT_EXEC,
                   owner: str | None = None) -> str:
        jid, now = uuid.uuid4().hex[:JOB_ID_HEX], time.time()
        with self._lock:
            self._db.execute("INSERT INTO jobs(id,created,updated,requester,requested,state,payload,exec_recipe,owner) VALUES (?,?,?,?,?,?,?,?,?)",
                             (jid, now, now, requester, requested, JobState.RECEIVED, json.dumps(payload), exec_recipe, owner))
            self._event(JOB_EVENT_PREFIX + JobState.RECEIVED, jid, {"requester": requester, "requested": requested})
        return jid

    def update_job(self, jid: str, **fields: Any) -> None:
        """Set job columns; a `state` change also logs `job.<state>` with the outcome fields."""
        if bad := set(fields) - UPDATABLE:
            raise ValueError(f"not an updatable job column: {sorted(bad)}")
        cols = {k: json.dumps(v) if k in JSON_COLUMNS and v is not None and not isinstance(v, str) else v
                for k, v in fields.items()}
        cols["updated"] = time.time()
        sets = ", ".join(f"{k}=?" for k in cols)
        # One lock for the row and its event, so whoever sees the new state also sees `job.<state>`.
        with self._lock:
            self._db.execute(f"UPDATE jobs SET {sets} WHERE id=?", (*cols.values(), jid))  # noqa: S608 — column names from UPDATABLE
            if "state" in fields:
                self._event(JOB_EVENT_PREFIX + fields["state"], jid, {k: fields[k] for k in EVENT_FIELDS if k in fields})
        self._changed()

    def _changed(self) -> None:
        with self.job_changed:
            self.job_changed.notify_all()

    def job(self, jid: str) -> Row | None:
        return next(map(_decode, self._all("SELECT * FROM jobs WHERE id=?", (jid,))), None)

    def jobs(self, limit: int) -> list[Row]:
        return self._all("SELECT id,created,updated,requester,requested,resolved,substitution,state,error "
                         "FROM jobs ORDER BY created DESC LIMIT ?", (limit,))

    def finished_jobs(self, since_ts: float) -> list[Row]:
        return self._all(f"SELECT id, created, updated, resolved, state, result FROM jobs "  # noqa: S608
                         f"WHERE updated>? AND state IN ({_in(FINISHED)}) ORDER BY updated", (since_ts, *FINISHED))

    def phase_times(self, since_ts: float) -> dict[str, dict[str, float]]:
        """{job id: {phase event kind: first timestamp}} for the received/switching/running phases."""
        kinds = tuple(JOB_EVENT_PREFIX + p for p in PHASES)
        rows = self._all(f"SELECT job_id, kind, MIN(ts) AS ts FROM events WHERE ts>? AND job_id IS NOT NULL "  # noqa: S608
                         f"AND kind IN ({_in(kinds)}) GROUP BY job_id, kind", (since_ts, *kinds))
        out: dict[str, dict[str, float]] = {}
        for r in rows:
            out.setdefault(r["job_id"], {})[r["kind"].removeprefix(JOB_EVENT_PREFIX)] = r["ts"]
        return out

    def fail_orphans(self) -> tuple[list[Row], list[str]]:
        """At startup nothing is in flight. Returns (jobs failed as orphans, as they were, with
        `direct`; ids of jobs still waiting in the queue, oldest first, which never started and
        are re-queued). Whatever had started (received, switching, running, a direct chat) failed."""
        rows = list(map(_decode, self._all(schema.ORPHANS + f"({_in(TERMINAL)}) ORDER BY created, rowid", (Event.JOB_DIRECT, *TERMINAL))))
        waiting = [j["id"] for j in rows if j["state"] == JobState.QUEUED and not j["direct"]]
        keep = set(waiting)
        lost = [j for j in rows if j["id"] not in keep]
        ids, now = [j["id"] for j in lost], time.time()
        for part in (tuple(ids[i:i + SQL_IN_MAX]) for i in range(0, len(ids), SQL_IN_MAX)):
            self._exec(f"UPDATE jobs SET state=?, error=?, updated=? WHERE id IN ({_in(part)})",  # noqa: S608
                       (JobState.FAILED, ORPHANED, now, *part))
        if lost:
            self.event(Event.ORPHANS_FAILED, count=len(lost))
            self._changed()
        return lost, waiting

    def stats(self, since_ts: float) -> Row:
        """Per-model outcome counts and latency, plus residency/error event counts."""
        rows = self._all("SELECT COALESCE(resolved, requested) AS model, state, COUNT(*) AS n, "
                         "AVG(updated-created) AS avg_s, MAX(updated-created) AS max_s "
                         "FROM jobs WHERE created>? GROUP BY model, state", (since_ts,))
        ev = self._all(f"SELECT kind, COUNT(*) AS n FROM events WHERE ts>? AND "  # noqa: S608
                       f"(kind LIKE ? OR kind IN ({_in(STATS_EXTRA_EVENTS)})) GROUP BY kind",
                       (since_ts, RESIDENCY_EVENT_PREFIX + "%", *STATS_EXTRA_EVENTS))
        models: dict[str, dict[str, Row]] = {}
        for r in rows:
            models.setdefault(r["model"], {})[r["state"]] = {
                "n": r["n"], "avg_s": round(r["avg_s"] or 0, DECIMALS), "max_s": round(r["max_s"] or 0, DECIMALS)}
        return {"since": since_ts, "models": models, "events": {r["kind"]: r["n"] for r in ev}}

    # ---- flags: state that must survive a restart (holds.py); None clears one
    def flag(self, name: str) -> Row | None:
        rows = self._all("SELECT ts, data FROM flags WHERE name=?", (name,))
        return {"since": rows[0]["ts"], **json.loads(rows[0]["data"])} if rows else None

    def set_flag(self, name: str, data: dict[str, Any] | None) -> None:
        self._exec(*(("DELETE FROM flags WHERE name=?", (name,)) if data is None else
                     ("INSERT OR REPLACE INTO flags(name,ts,data) VALUES (?,?,?)", (name, time.time(), json.dumps(data)))))

    # ---- downloads ------------------------------------------------------
    def upsert_download(self, slug: str, kind: str, ref: str, catalog_key: str | None) -> bool:
        """Record a download request; True if it is new (or a retry of a failed one)."""
        now = time.time()
        with self._lock:
            row = self._db.execute("SELECT state FROM downloads WHERE slug=?", (slug,)).fetchone()
            if row and row["state"] in ACTIVE_DOWNLOADS:
                return False
            self._db.execute("INSERT OR REPLACE INTO downloads(slug,kind,ref,state,created,updated,catalog_key) "
                             "VALUES (?,?,?,?,?,?,?)", (slug, kind, ref, DownloadState.QUEUED, now, now, catalog_key))
        self.event(Event.DOWNLOAD_QUEUED, slug=slug, ref=ref)
        return True

    def set_download(self, slug: str, state: DownloadState, error: str | None = None) -> None:
        self._exec("UPDATE downloads SET state=?, updated=?, error=? WHERE slug=?", (state, time.time(), error, slug))
        self.event(DOWNLOAD_EVENT_PREFIX + state, slug=slug, error=error)

    def downloads(self, limit: int) -> list[Row]:
        return self._all("SELECT * FROM downloads ORDER BY created DESC LIMIT ?", (limit,))

    def download(self, slug: str) -> Row | None:
        return next(iter(self._all("SELECT * FROM downloads WHERE slug=?", (slug,))), None)
