"""Reusable per-thread SQLite connection owner for core modules."""

from __future__ import annotations

import sqlite3
import threading
import time
import weakref
from pathlib import Path

_JOURNAL_MODE_INITIALIZED: set[tuple[str, str]] = set()
_JOURNAL_MODE_LOCK = threading.Lock()


class _SQLiteConnectionRegistry:
    def __init__(
        self,
        db_path: str | Path,
        timeout: float,
        *,
        journal_mode: str = "WAL",
        synchronous: str = "NORMAL",
    ) -> None:
        self._db_path = Path(db_path)
        self._timeout = timeout
        self._journal_mode = journal_mode
        self._synchronous = synchronous
        self._lock = threading.Lock()
        self._local = threading.local()
        self._all_conns: list[sqlite3.Connection] = []

    def connect(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            with self._lock:
                conn = getattr(self._local, "conn", None)
                if conn is None:
                    conn = sqlite3.connect(
                        str(self._db_path),
                        timeout=self._timeout,
                        check_same_thread=False,
                    )
                    db_key = str(self._db_path.resolve())
                    for attempt in range(20):
                        try:
                            journal_key = (db_key, self._journal_mode)
                            if self._journal_mode:
                                with _JOURNAL_MODE_LOCK:
                                    needs_journal_mode = (
                                        journal_key not in _JOURNAL_MODE_INITIALIZED
                                    )
                                if needs_journal_mode:
                                    conn.execute(
                                        f"PRAGMA journal_mode={self._journal_mode}"
                                    )
                                    with _JOURNAL_MODE_LOCK:
                                        _JOURNAL_MODE_INITIALIZED.add(journal_key)
                            if self._synchronous:
                                conn.execute(
                                    f"PRAGMA synchronous={self._synchronous}"
                                )
                            conn.execute("PRAGMA busy_timeout=30000")
                            break
                        except sqlite3.OperationalError as exc:
                            if "locked" in str(exc).lower() and attempt < 19:
                                time.sleep(0.02 * (attempt + 1))
                                continue
                            conn.close()
                            raise
                    self._local.conn = conn
                    self._all_conns.append(conn)
        return conn

    def close(self) -> None:
        with self._lock:
            conns = self._all_conns
            self._all_conns = []
            self._local = threading.local()
        for conn in conns:
            try:
                conn.close()
            except sqlite3.Error:
                pass


class SQLiteConnectionOwner:
    """Provide one SQLite connection per thread with shared cleanup."""

    def __init__(
        self,
        db_path: str | Path,
        *,
        timeout: float = 30.0,
        journal_mode: str = "WAL",
        synchronous: str = "NORMAL",
    ):
        self._registry = _SQLiteConnectionRegistry(
            db_path,
            timeout,
            journal_mode=journal_mode,
            synchronous=synchronous,
        )
        self._finalizer = weakref.finalize(
            self,
            _SQLiteConnectionRegistry.close,
            self._registry,
        )

    def connect(self) -> sqlite3.Connection:
        return self._registry.connect()

    def close(self) -> None:
        self._registry.close()
