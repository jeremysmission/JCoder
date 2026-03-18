"""Extended tests for storage rotation (SQLite pruning) and telemetry.

Sprint R9 -- validates retention pruning, dry-run mode, protected-table
semantics, concurrent prune safety, event recording, metric aggregation,
feedback-loop scoring, session summaries, and JSON export.

All I/O is confined to tmp_path fixtures; nothing touches real databases.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

import pytest

from core.telemetry import QueryEvent, TelemetryStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_event(
    idx: int,
    *,
    confidence: float = 0.5,
    feedback: str | None = None,
    ts_offset: float = 0.0,
) -> QueryEvent:
    """Build a deterministic QueryEvent for testing."""
    return QueryEvent(
        query_id=f"q_{idx:05d}",
        query_text=f"test query {idx}",
        timestamp=1_700_000_000.0 + idx + ts_offset,
        retrieval_latency_ms=10.0 + idx,
        generation_latency_ms=20.0 + idx,
        chunk_ids=[f"c_{idx}_0", f"c_{idx}_1"],
        chunk_scores=[0.9, 0.7],
        source_files=[f"file_{idx}.py"],
        answer_snippet=f"answer {idx}",
        confidence=confidence,
        reflection_relevant=0.8,
        reflection_supported=0.7,
        reflection_useful=0.6,
        feedback=feedback,
    )


def _seed_store(store: TelemetryStore, n: int, **kw: Any) -> List[QueryEvent]:
    """Insert *n* events and return them."""
    events = [_make_event(i, **kw) for i in range(n)]
    for ev in events:
        store.log(ev)
    return events


# ---------------------------------------------------------------------------
# Storage Rotation -- prune_old
# ---------------------------------------------------------------------------

class TestPruneOldRows:
    """Prune old rows beyond retention limit."""

    def test_no_prune_when_under_limit(self, tmp_path: Path) -> None:
        store = TelemetryStore(str(tmp_path / "tel.db"))
        _seed_store(store, 5)
        deleted = store.prune_old(keep=10)
        assert deleted == 0
        assert store.stats()["total"] == 5

    def test_prune_removes_oldest(self, tmp_path: Path) -> None:
        store = TelemetryStore(str(tmp_path / "tel.db"))
        events = _seed_store(store, 20)
        deleted = store.prune_old(keep=5)
        assert deleted == 15
        remaining = store.recent(limit=100)
        assert len(remaining) == 5
        # The 5 most recent (highest timestamp) should survive
        surviving_ids = {e.query_id for e in remaining}
        expected_ids = {f"q_{i:05d}" for i in range(15, 20)}
        assert surviving_ids == expected_ids

    def test_prune_exact_boundary(self, tmp_path: Path) -> None:
        store = TelemetryStore(str(tmp_path / "tel.db"))
        _seed_store(store, 10)
        deleted = store.prune_old(keep=10)
        assert deleted == 0
        assert store.stats()["total"] == 10

    def test_prune_to_zero_keep(self, tmp_path: Path) -> None:
        """keep=0 falls back to MAX_ROWS, so nothing is pruned when < MAX_ROWS."""
        store = TelemetryStore(str(tmp_path / "tel.db"))
        _seed_store(store, 10)
        deleted = store.prune_old(keep=0)
        assert deleted == 0  # 10 < 100_000

    def test_prune_default_max_rows(self, tmp_path: Path) -> None:
        store = TelemetryStore(str(tmp_path / "tel.db"))
        assert store.MAX_ROWS == 100_000


# ---------------------------------------------------------------------------
# Rotation by date / size (timestamp ordering)
# ---------------------------------------------------------------------------

class TestRotationByDate:
    """Oldest-first deletion preserves chronological ordering."""

    def test_oldest_removed_first(self, tmp_path: Path) -> None:
        store = TelemetryStore(str(tmp_path / "tel.db"))
        # Insert with explicit timestamp gaps
        for i in range(10):
            ev = _make_event(i, ts_offset=i * 100)
            store.log(ev)
        store.prune_old(keep=3)
        remaining = store.recent(limit=100)
        ts_list = [e.timestamp for e in remaining]
        # Remaining should be the 3 newest
        assert len(ts_list) == 3
        assert all(t >= 1_700_000_000.0 + 7 + 700 for t in ts_list)


# ---------------------------------------------------------------------------
# Dry-run mode (simulated -- prune_old doesn't have dry-run, so we
# verify read-only stat check before and after)
# ---------------------------------------------------------------------------

class TestDryRunMode:
    """Verify we can inspect what *would* be pruned without deleting."""

    def test_stats_before_prune_is_nondestructive(self, tmp_path: Path) -> None:
        store = TelemetryStore(str(tmp_path / "tel.db"))
        _seed_store(store, 20)
        pre = store.stats()
        assert pre["total"] == 20
        # Reading stats must not change anything
        post = store.stats()
        assert post["total"] == 20

    def test_recent_query_is_nondestructive(self, tmp_path: Path) -> None:
        store = TelemetryStore(str(tmp_path / "tel.db"))
        _seed_store(store, 15)
        _ = store.recent(limit=5)
        assert store.stats()["total"] == 15


# ---------------------------------------------------------------------------
# Protected tables not pruned
# ---------------------------------------------------------------------------

class TestProtectedTables:
    """prune_old only touches query_events; other tables are untouched."""

    def test_custom_table_survives_prune(self, tmp_path: Path) -> None:
        store = TelemetryStore(str(tmp_path / "tel.db"))
        _seed_store(store, 20)
        # Create a side table that should never be touched
        with store._connect() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS protected_config "
                "(key TEXT PRIMARY KEY, value TEXT)"
            )
            conn.execute(
                "INSERT INTO protected_config VALUES ('version', '1.0')"
            )
            conn.commit()

        store.prune_old(keep=5)

        with store._connect() as conn:
            row = conn.execute(
                "SELECT value FROM protected_config WHERE key='version'"
            ).fetchone()
        assert row is not None and row[0] == "1.0"


# ---------------------------------------------------------------------------
# Concurrent prune safety
# ---------------------------------------------------------------------------

class TestConcurrentPruneSafety:
    """Multiple threads pruning simultaneously must not corrupt the DB."""

    def test_parallel_prunes(self, tmp_path: Path) -> None:
        db_path = str(tmp_path / "tel.db")
        store = TelemetryStore(db_path)
        _seed_store(store, 100)

        def _prune_worker(keep: int) -> int:
            # Each thread gets its own TelemetryStore (and connection)
            s = TelemetryStore(db_path)
            return s.prune_old(keep=keep)

        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(_prune_worker, 30) for _ in range(4)]
            results = [f.result() for f in as_completed(futures)]

        # After all prunes, at most 30 rows remain
        final = TelemetryStore(db_path)
        assert final.stats()["total"] <= 30

    def test_parallel_log_and_prune(self, tmp_path: Path) -> None:
        db_path = str(tmp_path / "tel.db")
        store = TelemetryStore(db_path)
        _seed_store(store, 50)

        errors: list[Exception] = []

        def _writer(start: int) -> None:
            s = TelemetryStore(db_path)
            for i in range(start, start + 20):
                try:
                    s.log(_make_event(1000 + i))
                except Exception as exc:
                    errors.append(exc)

        def _pruner() -> None:
            s = TelemetryStore(db_path)
            try:
                s.prune_old(keep=40)
            except Exception as exc:
                errors.append(exc)

        with ThreadPoolExecutor(max_workers=3) as pool:
            pool.submit(_writer, 0)
            pool.submit(_writer, 20)
            pool.submit(_pruner)

        assert errors == [], f"Concurrent errors: {errors}"


# ---------------------------------------------------------------------------
# Telemetry -- Event recording
# ---------------------------------------------------------------------------

class TestEventRecording:
    """Basic log / retrieve round-trip."""

    def test_log_and_retrieve(self, tmp_path: Path) -> None:
        store = TelemetryStore(str(tmp_path / "tel.db"))
        ev = _make_event(0)
        store.log(ev)
        rows = store.recent(limit=1)
        assert len(rows) == 1
        assert rows[0].query_id == "q_00000"
        assert rows[0].chunk_ids == ["c_0_0", "c_0_1"]

    def test_upsert_replaces(self, tmp_path: Path) -> None:
        store = TelemetryStore(str(tmp_path / "tel.db"))
        ev = _make_event(0, confidence=0.3)
        store.log(ev)
        ev2 = _make_event(0, confidence=0.9)
        store.log(ev2)
        rows = store.recent(limit=10)
        assert len(rows) == 1
        assert rows[0].confidence == 0.9

    def test_answer_snippet_truncated(self, tmp_path: Path) -> None:
        store = TelemetryStore(str(tmp_path / "tel.db"))
        ev = _make_event(0)
        ev.answer_snippet = "x" * 1000
        store.log(ev)
        row = store.recent(limit=1)[0]
        assert len(row.answer_snippet) == 500


# ---------------------------------------------------------------------------
# Telemetry -- Metric aggregation (stats)
# ---------------------------------------------------------------------------

class TestMetricAggregation:
    """stats() returns correct aggregates."""

    def test_stats_empty(self, tmp_path: Path) -> None:
        store = TelemetryStore(str(tmp_path / "tel.db"))
        s = store.stats()
        assert s == {"total": 0}

    def test_stats_populated(self, tmp_path: Path) -> None:
        store = TelemetryStore(str(tmp_path / "tel.db"))
        _seed_store(store, 10, confidence=0.8, feedback="good")
        s = store.stats()
        assert s["total"] == 10
        assert s["avg_confidence"] == 0.8
        assert s["feedback_good"] == 10
        assert s["feedback_bad"] == 0

    def test_stats_mixed_feedback(self, tmp_path: Path) -> None:
        store = TelemetryStore(str(tmp_path / "tel.db"))
        for i in range(6):
            store.log(_make_event(i, feedback="good"))
        for i in range(6, 10):
            store.log(_make_event(i, feedback="bad"))
        s = store.stats()
        assert s["feedback_good"] == 6
        assert s["feedback_bad"] == 4


# ---------------------------------------------------------------------------
# Telemetry -- Feedback loop scoring
# ---------------------------------------------------------------------------

class TestFeedbackLoopScoring:
    """set_feedback, low/high confidence queries, failed_queries."""

    def test_set_feedback(self, tmp_path: Path) -> None:
        store = TelemetryStore(str(tmp_path / "tel.db"))
        store.log(_make_event(0))
        store.set_feedback("q_00000", "bad")
        failed = store.failed_queries(limit=10)
        assert len(failed) == 1
        assert failed[0].feedback == "bad"

    def test_low_confidence_queries(self, tmp_path: Path) -> None:
        store = TelemetryStore(str(tmp_path / "tel.db"))
        for i in range(5):
            store.log(_make_event(i, confidence=0.1))
        for i in range(5, 10):
            store.log(_make_event(i, confidence=0.9))
        low = store.low_confidence_queries(threshold=0.3, limit=100)
        assert len(low) == 5
        assert all(e.confidence <= 0.3 for e in low)

    def test_high_confidence_queries(self, tmp_path: Path) -> None:
        store = TelemetryStore(str(tmp_path / "tel.db"))
        for i in range(5):
            store.log(_make_event(i, confidence=0.1))
        for i in range(5, 10):
            store.log(_make_event(i, confidence=0.9))
        high = store.high_confidence_queries(threshold=0.7, limit=100)
        assert len(high) == 5
        assert all(e.confidence >= 0.7 for e in high)


# ---------------------------------------------------------------------------
# Telemetry -- Session summary generation
# ---------------------------------------------------------------------------

class TestSessionSummary:
    """stats() acts as the session summary; verify structure."""

    def test_summary_keys(self, tmp_path: Path) -> None:
        store = TelemetryStore(str(tmp_path / "tel.db"))
        _seed_store(store, 5)
        s = store.stats()
        for key in (
            "total", "avg_confidence", "avg_retrieval_ms",
            "avg_generation_ms", "feedback_good", "feedback_bad",
        ):
            assert key in s

    def test_summary_latency_averages(self, tmp_path: Path) -> None:
        store = TelemetryStore(str(tmp_path / "tel.db"))
        _seed_store(store, 3)
        s = store.stats()
        # retrieval_latency_ms = 10+i -> mean(10,11,12) = 11.0
        assert s["avg_retrieval_ms"] == 11.0
        # generation_latency_ms = 20+i -> mean(20,21,22) = 21.0
        assert s["avg_generation_ms"] == 21.0


# ---------------------------------------------------------------------------
# Telemetry -- Export to JSON
# ---------------------------------------------------------------------------

class TestExportToJson:
    """Round-trip events through JSON serialisation."""

    def test_export_recent_to_json(self, tmp_path: Path) -> None:
        store = TelemetryStore(str(tmp_path / "tel.db"))
        _seed_store(store, 5)
        events = store.recent(limit=100)
        payload = json.dumps([_event_to_dict(e) for e in events])
        loaded = json.loads(payload)
        assert len(loaded) == 5
        assert loaded[0]["query_id"].startswith("q_")

    def test_export_stats_to_json(self, tmp_path: Path) -> None:
        store = TelemetryStore(str(tmp_path / "tel.db"))
        _seed_store(store, 3, feedback="good")
        payload = json.dumps(store.stats())
        loaded = json.loads(payload)
        assert loaded["total"] == 3

    def test_export_preserves_chunk_ids(self, tmp_path: Path) -> None:
        store = TelemetryStore(str(tmp_path / "tel.db"))
        ev = _make_event(0)
        store.log(ev)
        row = store.recent(limit=1)[0]
        d = _event_to_dict(row)
        assert d["chunk_ids"] == ["c_0_0", "c_0_1"]
        assert d["chunk_scores"] == [0.9, 0.7]


def _event_to_dict(ev: QueryEvent) -> Dict[str, Any]:
    """Lightweight serialiser matching dataclass fields."""
    from dataclasses import asdict
    return asdict(ev)
