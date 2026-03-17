"""Tests for persistent cross-session memory (Sprint 16)."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from core.persistent_memory import (
    MemoryEntry,
    MemorySearchResult,
    PatternMatch,
    PersistentMemory,
    SQLiteMemoryBackend,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def backend(tmp_path):
    b = SQLiteMemoryBackend(tmp_path / "test_memory.db")
    yield b
    b.close()


@pytest.fixture
def memory(tmp_path):
    m = PersistentMemory(db_path=tmp_path / "test_persistent.db")
    yield m
    m.close()


def _make_entry(session_id="s1", query="test query", response="test response",
                entry_type="interaction", tags=None, quality=0.5):
    return MemoryEntry(
        entry_id="",
        session_id=session_id,
        entry_type=entry_type,
        query=query,
        response=response,
        tags=tags or [],
        quality_score=quality,
    )


# ---------------------------------------------------------------------------
# MemoryEntry
# ---------------------------------------------------------------------------

class TestMemoryEntry:
    def test_create(self):
        e = MemoryEntry(
            entry_id="m1", session_id="s1", entry_type="interaction",
            query="How to sort?", response="Use sorted()",
        )
        assert e.entry_id == "m1"
        assert e.quality_score == 0.0

    def test_tag_str(self):
        e = MemoryEntry(
            entry_id="m1", session_id="s1", entry_type="interaction",
            query="q", response="r", tags=["python", "algo", "algo"],
        )
        assert e.tag_str == "algo,algo,python"

    def test_empty_tags(self):
        e = MemoryEntry(
            entry_id="m1", session_id="s1", entry_type="interaction",
            query="q", response="r",
        )
        assert e.tag_str == ""


# ---------------------------------------------------------------------------
# SQLiteMemoryBackend
# ---------------------------------------------------------------------------

class TestSQLiteBackend:
    def test_store_and_retrieve(self, backend):
        entry = _make_entry()
        eid = backend.store(entry)
        assert eid.startswith("mem_")

        results = backend.get_recent(limit=10)
        assert len(results) == 1
        assert results[0].query == "test query"

    def test_search_fts5(self, backend):
        backend.store(_make_entry(query="quicksort algorithm", response="Use partition"))
        backend.store(_make_entry(query="bubble sort", response="Compare adjacent"))

        results = backend.search("quicksort")
        assert len(results) == 1
        assert "quicksort" in results[0].entry.query

    def test_search_empty_query(self, backend):
        backend.store(_make_entry())
        results = backend.search("")
        assert results == []

    def test_search_special_chars(self, backend):
        backend.store(_make_entry(query="test query"))
        # Special chars should be stripped, not crash
        results = backend.search("test (query)")
        assert isinstance(results, list)

    def test_get_by_session(self, backend):
        backend.store(_make_entry(session_id="s1", query="q1"))
        backend.store(_make_entry(session_id="s2", query="q2"))
        backend.store(_make_entry(session_id="s1", query="q3"))

        s1 = backend.get_by_session("s1")
        assert len(s1) == 2
        assert all(e.session_id == "s1" for e in s1)

    def test_get_recent_ordering(self, backend):
        backend.store(_make_entry(query="first"))
        backend.store(_make_entry(query="second"))

        recent = backend.get_recent(limit=2)
        assert len(recent) == 2
        # Most recent first
        assert recent[0].query == "second"

    def test_stats(self, backend):
        backend.store(_make_entry(session_id="s1", quality=0.8))
        backend.store(_make_entry(session_id="s2", quality=0.6))

        s = backend.stats()
        assert s["total_entries"] == 2
        assert s["total_sessions"] == 2
        assert s["avg_quality"] == 0.7

    def test_stats_empty(self, backend):
        s = backend.stats()
        assert s["total_entries"] == 0
        assert s["total_sessions"] == 0

    def test_session_summary(self, backend):
        backend.store(_make_entry(session_id="s1", tags=["python"], quality=0.8))
        backend.store(_make_entry(session_id="s1", tags=["algo"], quality=0.6))

        summary = backend.summarize_session("s1")
        assert summary["entry_count"] == 2
        assert summary["avg_quality"] == 0.7

    def test_session_summaries_list(self, backend):
        backend.store(_make_entry(session_id="s1"))
        backend.store(_make_entry(session_id="s2"))
        backend.summarize_session("s1")
        backend.summarize_session("s2")

        summaries = backend.get_session_summaries()
        assert len(summaries) == 2

    def test_detect_patterns_requires_min_sessions(self, backend):
        # Only 1 session -- no patterns
        backend.store(_make_entry(session_id="s1", query="sort algorithm"))
        patterns = backend.detect_patterns(min_occurrences=3)
        assert patterns == []

    def test_detect_recurring_query(self, backend):
        # Same query across 3 sessions
        for i in range(3):
            backend.store(_make_entry(
                session_id=f"s{i}",
                query="how to implement binary search",
            ))

        patterns = backend.detect_patterns(min_occurrences=3)
        recurring = [p for p in patterns if p.pattern_type == "recurring_query"]
        assert len(recurring) >= 1

    def test_quality_pattern_detection(self, backend):
        # High quality pattern with same tags
        for i in range(5):
            backend.store(_make_entry(
                session_id=f"s{i}", tags=["python", "algo"],
                quality=0.9,
            ))

        patterns = backend.detect_patterns(min_occurrences=3)
        quality_patterns = [p for p in patterns
                           if p.pattern_type in ("success_strategy", "failure_mode")]
        assert len(quality_patterns) >= 1

    def test_limit_caps(self, backend):
        # Verify limit capping doesn't crash
        for i in range(5):
            backend.store(_make_entry(query=f"query {i}"))

        assert len(backend.get_recent(limit=3)) == 3
        assert len(backend.search("query", limit=2)) == 2
        assert len(backend.get_by_session("s1", limit=2)) == 2


# ---------------------------------------------------------------------------
# PersistentMemory (high-level API)
# ---------------------------------------------------------------------------

class TestPersistentMemory:
    def test_record_interaction(self, memory):
        eid = memory.record_interaction(
            session_id="s1",
            query="How to sort a list?",
            response="Use sorted() or list.sort()",
            quality=0.9,
            tags=["python", "sorting"],
        )
        assert eid.startswith("mem_")

    def test_record_insight(self, memory):
        eid = memory.record_insight(
            session_id="s1",
            insight="FTS5 queries are faster than LIKE for keyword search",
            source_query="search optimization",
            tags=["sqlite", "performance"],
        )
        assert eid.startswith("ins_")

    def test_record_error(self, memory):
        eid = memory.record_error(
            session_id="s1",
            query="run dangerous command",
            error="Permission denied: rm -rf /",
            tags=["safety"],
        )
        assert eid.startswith("err_")

    def test_recall(self, memory):
        memory.record_interaction("s1", "quicksort implementation", "Use partition")
        memory.record_interaction("s1", "bubble sort", "Compare adjacent elements")

        results = memory.recall("quicksort")
        assert len(results) >= 1
        assert "quicksort" in results[0].entry.query

    def test_recall_empty(self, memory):
        results = memory.recall("nonexistent topic")
        assert results == []

    def test_recall_session(self, memory):
        memory.record_interaction("s1", "q1", "r1")
        memory.record_interaction("s2", "q2", "r2")

        s1 = memory.recall_session("s1")
        assert len(s1) == 1
        assert s1[0].session_id == "s1"

    def test_recent(self, memory):
        memory.record_interaction("s1", "older", "r1")
        memory.record_interaction("s2", "newer", "r2")

        recent = memory.recent(limit=5)
        assert len(recent) == 2
        assert recent[0].query == "newer"

    def test_cross_session_recall(self, memory):
        """Gate: recall context from multiple sessions ago."""
        # Record across 5 sessions
        for i in range(5):
            memory.record_interaction(
                f"session_{i}",
                f"binary search implementation variant {i}",
                f"response for variant {i}",
                quality=0.8,
            )

        # Search should find entries from earliest session
        results = memory.recall("binary search")
        assert len(results) >= 5
        sessions_found = {r.entry.session_id for r in results}
        assert "session_0" in sessions_found  # 5 sessions ago

    def test_find_patterns(self, memory):
        # Create recurring pattern
        for i in range(4):
            memory.record_interaction(
                f"s{i}",
                "how do I write tests",
                f"Use pytest, variant {i}",
                tags=["testing"],
                quality=0.8,
            )

        patterns = memory.find_patterns(min_occurrences=3)
        assert len(patterns) >= 1

    def test_session_summary(self, memory):
        memory.record_interaction("s1", "q1", "r1", quality=0.9, tags=["python"])
        memory.record_interaction("s1", "q2", "r2", quality=0.7, tags=["algo"])

        summary = memory.summarize_session("s1")
        assert summary["entry_count"] == 2
        assert summary["avg_quality"] == 0.8

    def test_stats(self, memory):
        memory.record_interaction("s1", "q1", "r1")
        memory.record_interaction("s2", "q2", "r2")

        s = memory.stats()
        assert s["total_entries"] == 2
        assert s["total_sessions"] == 2

    def test_model_agnostic_storage(self, memory):
        """Verify storage is pure text -- no embeddings or model deps."""
        memory.record_interaction("s1", "test query", "test response")

        # Directly verify SQLite contains plain text
        results = memory.recall("test query")
        assert results[0].entry.response == "test response"
        # No vector/embedding fields in the result
        assert not hasattr(results[0].entry, "embedding")
