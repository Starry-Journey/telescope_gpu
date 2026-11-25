"""Small SQLite job queue for long-running 3A/PD/FEA jobs.

The Web API only submits/reads jobs; a worker (Windows OpticStudio, or a Linux
PD/FEA worker) claims and completes them.  SQLite keeps the MVP dependency-free
while giving durable status across restarts.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[2]
DB_PATH = Path(os.environ.get("TEL_JOB_DB", str(ROOT / "results" / "jobs.sqlite3")))


class JobStore:
    def __init__(self, path: Path = DB_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._con() as con:
            con.execute("""CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY, kind TEXT, status TEXT, worker TEXT,
                payload TEXT, result TEXT, error TEXT,
                created REAL, updated REAL)""")

    @contextmanager
    def _con(self):
        """One transaction on a fresh connection, closed afterwards.

        sqlite3's ``with connection`` only commits/rolls back; it leaves the
        handle open, which on Windows keeps temp/test DB files locked.  Close
        it so TemporaryDirectory cleanup cannot fail.
        """
        con = sqlite3.connect(self.path, timeout=10)
        try:
            with con:  # commit on success, roll back on exception
                yield con
        finally:
            con.close()

    def submit(self, kind: str, payload: dict) -> str:
        jid = uuid.uuid4().hex[:12]
        now = time.time()
        with self._lock, self._con() as con:
            con.execute("INSERT INTO jobs VALUES (?,?,?,?,?,?,?,?,?)",
                        (jid, kind, "queued", None, json.dumps(payload),
                         None, None, now, now))
        return jid

    def get(self, jid: str) -> Optional[dict]:
        with self._con() as con:
            row = con.execute("SELECT * FROM jobs WHERE id=?", (jid,)).fetchone()
        if not row:
            return None
        keys = ["id", "kind", "status", "worker", "payload", "result", "error", "created", "updated"]
        d = dict(zip(keys, row))
        d["payload"] = json.loads(d["payload"]) if d["payload"] else {}
        d["result"] = json.loads(d["result"]) if d["result"] else None
        return d

    def list(self, limit: int = 20) -> list[dict]:
        with self._con() as con:
            rows = con.execute("SELECT id FROM jobs ORDER BY created DESC LIMIT ?", (limit,)).fetchall()
        return [self.get(r[0]) for r in rows]

    def claim(self, kind: str, worker: str) -> Optional[dict]:
        """Atomically claim one queued job (P1-10).

        The old SELECT-then-UPDATE pair raced under multiple workers.  The
        claim is now one conditional UPDATE (guarded by status='queued') whose
        RETURNING row decides ownership, so exactly one claimer wins per job.
        """
        with self._lock, self._con() as con:
            cur = con.execute(
                "UPDATE jobs SET status='running', worker=?, updated=? "
                "WHERE id=(SELECT id FROM jobs WHERE kind=? AND status='queued' "
                "ORDER BY created LIMIT 1) AND status='queued' RETURNING id",
                (worker, time.time(), kind))
            row = cur.fetchone()
        return self.get(row[0]) if row else None

    def complete(self, jid: str, result: dict) -> None:
        with self._lock, self._con() as con:
            con.execute("UPDATE jobs SET status='done', result=?, updated=? WHERE id=?",
                        (json.dumps(result), time.time(), jid))

    def fail(self, jid: str, error: str) -> None:
        with self._lock, self._con() as con:
            con.execute("UPDATE jobs SET status='failed', error=?, updated=? WHERE id=?",
                        (error, time.time(), jid))


_store = None


def get_store():
    global _store
    if _store is None:
        _store = JobStore()
    return _store


def __getattr__(name):
    if name == "store":
        return get_store()
    raise AttributeError(name)
