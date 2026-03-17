"""Regression tests for per-thread SQLite connection reuse."""

from __future__ import annotations

import gc
import sqlite3
import threading
import time
import weakref

from core.sqlite_owner import SQLiteConnectionOwner
from core.telemetry import QueryEvent, TelemetryStore


def test_sqlite_owner_reuses_one_connection_per_thread(tmp_db_path, monkeypatch):
    db_path = tmp_db_path
    calls = {"count": 0}
    real_connect = sqlite3.connect

    def counting_connect(*args, **kwargs):
        calls["count"] += 1
        return real_connect(*args, **kwargs)

    monkeypatch.setattr("core.sqlite_owner.sqlite3.connect", counting_connect)
    owner = SQLiteConnectionOwner(db_path)
    connection_ids = []

    def worker():
        for _ in range(5):
            connection_ids.append(id(owner.connect()))

    threads = [threading.Thread(target=worker) for _ in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert calls["count"] == 3
    assert len(set(connection_ids)) == 3
    owner.close()


def test_sqlite_owner_close_allows_fresh_connection(tmp_db_path):
    owner = SQLiteConnectionOwner(tmp_db_path)
    first = owner.connect()
    owner.close()
    second = owner.connect()

    assert first is not second
    owner.close()


def test_sqlite_owner_does_not_keep_self_alive(tmp_db_path):
    owner = SQLiteConnectionOwner(tmp_db_path)
    owner.connect()
    ref = weakref.ref(owner)

    owner = None
    gc.collect()

    assert ref() is None


def test_telemetry_store_concurrent_writes_do_not_lock(tmp_path):
    store = TelemetryStore(str(tmp_path / "telemetry.db"))
    errors = []

    def worker(thread_id: int) -> None:
        try:
            for i in range(50):
                store.log(
                    QueryEvent(
                        query_id=f"{thread_id}-{i}",
                        query_text=f"query {thread_id}-{i}",
                        timestamp=time.time(),
                        confidence=0.5,
                    )
                )
        except Exception as exc:  # pragma: no cover - regression guard
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    stats = store.stats()
    store.close()

    assert not errors
    assert stats["total"] == 150


def test_telemetry_store_reopens_after_close(tmp_path):
    store = TelemetryStore(str(tmp_path / "telemetry.db"))
    store.log(QueryEvent(query_id="q1", query_text="one", timestamp=time.time(), confidence=0.1))
    store.close()
    store.log(QueryEvent(query_id="q2", query_text="two", timestamp=time.time(), confidence=0.2))

    stats = store.stats()
    store.close()

    assert stats["total"] == 2
