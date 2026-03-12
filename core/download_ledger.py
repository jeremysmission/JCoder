from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


_SCHEMA = """
CREATE TABLE IF NOT EXISTS download_runs (
    run_id TEXT PRIMARY KEY,
    started_at TEXT NOT NULL,
    finished_at TEXT DEFAULT '',
    root_dir TEXT NOT NULL,
    config_json TEXT DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS download_records (
    dest_rel TEXT NOT NULL,
    url TEXT NOT NULL,
    run_id TEXT NOT NULL,
    status TEXT NOT NULL,
    attempts INTEGER DEFAULT 0,
    resumed_from INTEGER DEFAULT 0,
    bytes_written INTEGER DEFAULT 0,
    expected_size INTEGER DEFAULT 0,
    sha256 TEXT DEFAULT '',
    expected_sha256 TEXT DEFAULT '',
    etag TEXT DEFAULT '',
    last_modified TEXT DEFAULT '',
    error TEXT DEFAULT '',
    extra_json TEXT DEFAULT '{}',
    updated_at TEXT NOT NULL,
    PRIMARY KEY (dest_rel, run_id)
);

CREATE INDEX IF NOT EXISTS idx_download_status ON download_records(status);
CREATE INDEX IF NOT EXISTS idx_download_url ON download_records(url);
CREATE INDEX IF NOT EXISTS idx_download_sha256 ON download_records(sha256);
"""


@dataclass
class DownloadRecord:
    dest_rel: str
    url: str
    status: str
    attempts: int
    resumed_from: int
    bytes_written: int
    expected_size: int
    sha256: str
    expected_sha256: str
    etag: str
    last_modified: str
    error: str
    extra: Dict[str, Any]
    updated_at: str


class DownloadLedger:
    """SQLite-backed ledger for robust download runs."""

    def __init__(self, db_path: str) -> None:
        self.db_path = str(Path(db_path))
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self._lock = threading.Lock()
        with self._lock:
            self.conn.executescript(_SCHEMA)
            self.conn.commit()

    def start_run(self, run_id: str, root_dir: str, config: Dict[str, Any]) -> None:
        with self._lock:
            self.conn.execute(
                "INSERT OR REPLACE INTO download_runs "
                "(run_id, started_at, root_dir, config_json) VALUES (?, ?, ?, ?)",
                (run_id, _utc_now(), root_dir, json.dumps(config, sort_keys=True)),
            )
            self.conn.commit()

    def finish_run(self, run_id: str) -> None:
        with self._lock:
            self.conn.execute(
                "UPDATE download_runs SET finished_at=? WHERE run_id=?",
                (_utc_now(), run_id),
            )
            self.conn.commit()

    def record(
        self,
        run_id: str,
        dest_rel: str,
        url: str,
        status: str,
        *,
        attempts: int = 0,
        resumed_from: int = 0,
        bytes_written: int = 0,
        expected_size: int = 0,
        sha256: str = "",
        expected_sha256: str = "",
        etag: str = "",
        last_modified: str = "",
        error: str = "",
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        payload = extra or {}
        with self._lock:
            self.conn.execute(
                "INSERT OR REPLACE INTO download_records "
                "(dest_rel, url, run_id, status, attempts, resumed_from, bytes_written, "
                "expected_size, sha256, expected_sha256, etag, last_modified, error, "
                "extra_json, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    dest_rel,
                    url,
                    run_id,
                    status,
                    attempts,
                    resumed_from,
                    bytes_written,
                    expected_size,
                    sha256,
                    expected_sha256,
                    etag,
                    last_modified,
                    error,
                    json.dumps(payload, sort_keys=True),
                    _utc_now(),
                ),
            )
            self.conn.commit()

    def latest_success(self, dest_rel: str) -> Optional[DownloadRecord]:
        with self._lock:
            row = self.conn.execute(
                "SELECT dest_rel, url, status, attempts, resumed_from, bytes_written, "
                "expected_size, sha256, expected_sha256, etag, last_modified, error, "
                "extra_json, updated_at "
                "FROM download_records WHERE dest_rel=? AND status='success' "
                "ORDER BY updated_at DESC LIMIT 1",
                (dest_rel,),
            ).fetchone()
        return self._row_to_record(row)

    def find_success_by_sha256(self, sha256: str) -> Optional[DownloadRecord]:
        if not sha256:
            return None
        with self._lock:
            row = self.conn.execute(
                "SELECT dest_rel, url, status, attempts, resumed_from, bytes_written, "
                "expected_size, sha256, expected_sha256, etag, last_modified, error, "
                "extra_json, updated_at "
                "FROM download_records WHERE sha256=? AND status='success' "
                "ORDER BY updated_at DESC LIMIT 1",
                (sha256,),
            ).fetchone()
        return self._row_to_record(row)

    def latest_run_id(self) -> Optional[str]:
        with self._lock:
            row = self.conn.execute(
                "SELECT run_id FROM download_runs ORDER BY started_at DESC LIMIT 1"
            ).fetchone()
        return row[0] if row else None

    def close(self) -> None:
        with self._lock:
            self.conn.commit()
            self.conn.close()

    @staticmethod
    def _row_to_record(row: Optional[tuple]) -> Optional[DownloadRecord]:
        if not row:
            return None
        return DownloadRecord(
            dest_rel=row[0],
            url=row[1],
            status=row[2],
            attempts=int(row[3] or 0),
            resumed_from=int(row[4] or 0),
            bytes_written=int(row[5] or 0),
            expected_size=int(row[6] or 0),
            sha256=row[7] or "",
            expected_sha256=row[8] or "",
            etag=row[9] or "",
            last_modified=row[10] or "",
            error=row[11] or "",
            extra=json.loads(row[12] or "{}"),
            updated_at=row[13] or "",
        )
