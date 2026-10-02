"""SQLite persistence for optimisation runs, analyst decisions and security events.

SQLite keeps the prototype dependency-free; the schema maps 1:1 onto
PostgreSQL for a production deployment.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS security_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, host TEXT, attack_type TEXT,
    probability REAL, confidence REAL, source TEXT);
CREATE TABLE IF NOT EXISTS optimization_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, solver TEXT, params TEXT,
    risk_before REAL, risk_after REAL, objective REAL, feasible INTEGER, result TEXT);
CREATE TABLE IF NOT EXISTS analyst_decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, run_id INTEGER, action_id TEXT,
    decision TEXT, analyst TEXT);
"""


class Storage:
    def __init__(self, path: str | Path = ":memory:"):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.lock = threading.Lock()
        with self.lock:
            self.conn.executescript(SCHEMA)

    def _exec(self, sql: str, args=()) -> sqlite3.Cursor:
        with self.lock:
            cur = self.conn.execute(sql, args)
            self.conn.commit()
            return cur

    def log_event(self, host, attack_type, probability, confidence, source="") -> None:
        self._exec("INSERT INTO security_events (ts, host, attack_type, probability, confidence, source)"
                   " VALUES (?,?,?,?,?,?)", (time.time(), host, attack_type, probability, confidence, source))

    def save_run(self, solver: str, params: dict, report: dict) -> int:
        r = report["result"]
        cur = self._exec(
            "INSERT INTO optimization_runs (ts, solver, params, risk_before, risk_after, objective,"
            " feasible, result) VALUES (?,?,?,?,?,?,?,?)",
            (time.time(), solver, json.dumps(params), report["risk_before"], report["risk_after"],
             r["objective"], int(r["feasible"]), json.dumps(report)))
        return int(cur.lastrowid)

    def get_run(self, run_id: int) -> dict | None:
        row = self._exec("SELECT * FROM optimization_runs WHERE id=?", (run_id,)).fetchone()
        if row is None:
            return None
        d = dict(row)
        d["params"] = json.loads(d["params"])
        d["result"] = json.loads(d["result"])
        return d

    def list_runs(self, limit: int = 50) -> list[dict]:
        rows = self._exec("SELECT id, ts, solver, risk_before, risk_after, objective, feasible "
                          "FROM optimization_runs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

    def log_decision(self, run_id: int, action_id: str, decision: str, analyst: str = "analyst") -> None:
        self._exec("INSERT INTO analyst_decisions (ts, run_id, action_id, decision, analyst)"
                   " VALUES (?,?,?,?,?)", (time.time(), run_id, action_id, decision, analyst))

    def decisions(self, run_id: int | None = None) -> list[dict]:
        if run_id is None:
            rows = self._exec("SELECT * FROM analyst_decisions ORDER BY id DESC").fetchall()
        else:
            rows = self._exec("SELECT * FROM analyst_decisions WHERE run_id=? ORDER BY id",
                              (run_id,)).fetchall()
        return [dict(r) for r in rows]

    def events(self, limit: int = 100) -> list[dict]:
        rows = self._exec("SELECT * FROM security_events ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]
