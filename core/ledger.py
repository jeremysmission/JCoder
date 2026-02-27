"""
Experiment Ledger
-----------------
SQLite ledger for evolver runs. Stores structured metrics and
references to artifacts written to disk.

Every run is an immutable record: run_id, timestamp, config fingerprint,
git commit (if available), and a metrics JSON blob.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    created_ts: float
    label: str
    config_fingerprint: str
    git_commit: Optional[str]
    metrics_json: Dict[str, Any]


class ExperimentLedger:
    """Append-only audit trail for evolver experiments."""

    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path))

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                  run_id TEXT PRIMARY KEY,
                  created_ts REAL NOT NULL,
                  label TEXT NOT NULL,
                  config_fingerprint TEXT NOT NULL,
                  git_commit TEXT,
                  metrics_json TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def write_run(
        self,
        run_id: str,
        label: str,
        config_fingerprint: str,
        metrics: Dict[str, Any],
        git_commit: Optional[str] = None,
        created_ts: Optional[float] = None,
    ) -> RunRecord:
        created_ts = created_ts if created_ts is not None else time.time()
        rec = RunRecord(
            run_id=run_id,
            created_ts=created_ts,
            label=label,
            config_fingerprint=config_fingerprint,
            git_commit=git_commit,
            metrics_json=metrics,
        )
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO runs
                (run_id, created_ts, label, config_fingerprint, git_commit, metrics_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    rec.run_id,
                    rec.created_ts,
                    rec.label,
                    rec.config_fingerprint,
                    rec.git_commit,
                    json.dumps(rec.metrics_json, ensure_ascii=False),
                ),
            )
            conn.commit()
        return rec

    def list_runs(self, limit: int = 50) -> List[RunRecord]:
        """Return recent runs, newest first."""
        with self._connect() as conn:
            cur = conn.execute(
                "SELECT run_id, created_ts, label, config_fingerprint, "
                "git_commit, metrics_json FROM runs "
                "ORDER BY created_ts DESC LIMIT ?",
                (limit,),
            )
            rows = cur.fetchall()
        return [
            RunRecord(
                run_id=r[0],
                created_ts=r[1],
                label=r[2],
                config_fingerprint=r[3],
                git_commit=r[4],
                metrics_json=json.loads(r[5]),
            )
            for r in rows
        ]
