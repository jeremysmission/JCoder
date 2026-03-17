"""
Persistent Cross-Session Memory (Sprint 16)
---------------------------------------------
Model-agnostic memory layer that persists across sessions.

Every JCoder interaction is recorded and searchable. Cross-session
pattern detection identifies recurring themes, successful strategies,
and failure patterns. Designed as an adapter layer so LimitlessApp V2
can plug in as the backend later.

Storage: SQLite (text-based, survives model swaps).
Search: FTS5 keyword matching + metadata filtering.

Gate: JCoder recalls context from 5 sessions ago accurately.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

from core.sqlite_owner import SQLiteConnectionOwner

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class MemoryEntry:
    """A single memory record."""
    entry_id: str
    session_id: str
    entry_type: str  # "interaction", "insight", "pattern", "error"
    query: str
    response: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    quality_score: float = 0.0
    created_at: float = field(default_factory=time.time)

    @property
    def tag_str(self) -> str:
        return ",".join(sorted(self.tags))


@dataclass
class PatternMatch:
    """A detected cross-session pattern."""
    pattern_id: str
    pattern_type: str  # "recurring_query", "success_strategy", "failure_mode"
    description: str
    evidence: List[str]  # entry_ids that support this pattern
    confidence: float
    occurrences: int
    first_seen: float
    last_seen: float


@dataclass
class MemorySearchResult:
    """Result from searching persistent memory."""
    entry: MemoryEntry
    relevance: float


# ---------------------------------------------------------------------------
# Backend protocol (adapter interface for LimitlessApp V2)
# ---------------------------------------------------------------------------

class MemoryBackend(Protocol):
    """Protocol for pluggable memory backends."""

    def store(self, entry: MemoryEntry) -> str: ...

    def search(self, query: str, limit: int = 10) -> List[MemorySearchResult]: ...

    def get_by_session(self, session_id: str, limit: int = 100) -> List[MemoryEntry]: ...

    def get_recent(self, limit: int = 50) -> List[MemoryEntry]: ...

    def detect_patterns(self, min_occurrences: int = 3) -> List[PatternMatch]: ...

    def stats(self) -> Dict[str, Any]: ...

    def close(self) -> None: ...


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    entry_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    entry_type TEXT NOT NULL,
    query TEXT NOT NULL,
    response TEXT NOT NULL,
    metadata_json TEXT DEFAULT '{}',
    tags TEXT DEFAULT '',
    quality_score REAL DEFAULT 0.0,
    created_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_memories_session ON memories(session_id);
CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(entry_type);
CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at);

CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
    query, response, tags, content=memories,
    content_rowid=rowid
);

CREATE TABLE IF NOT EXISTS patterns (
    pattern_id TEXT PRIMARY KEY,
    pattern_type TEXT NOT NULL,
    description TEXT NOT NULL,
    evidence_json TEXT DEFAULT '[]',
    confidence REAL DEFAULT 0.0,
    occurrences INTEGER DEFAULT 0,
    first_seen REAL NOT NULL,
    last_seen REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS session_summaries (
    session_id TEXT PRIMARY KEY,
    summary TEXT NOT NULL,
    entry_count INTEGER DEFAULT 0,
    avg_quality REAL DEFAULT 0.0,
    tags_json TEXT DEFAULT '[]',
    created_at REAL NOT NULL
);
"""

_FTS_TRIGGERS = """
CREATE TRIGGER IF NOT EXISTS memory_ai AFTER INSERT ON memories BEGIN
    INSERT INTO memory_fts(rowid, query, response, tags)
    VALUES (new.rowid, new.query, new.response, new.tags);
END;

CREATE TRIGGER IF NOT EXISTS memory_ad AFTER DELETE ON memories BEGIN
    INSERT INTO memory_fts(memory_fts, rowid, query, response, tags)
    VALUES ('delete', old.rowid, old.query, old.response, old.tags);
END;
"""


# ---------------------------------------------------------------------------
# SQLite Backend (default, model-agnostic)
# ---------------------------------------------------------------------------

class SQLiteMemoryBackend:
    """Persistent memory backed by SQLite + FTS5.

    Text-only storage (no embeddings) -- survives model swaps.
    FTS5 provides fast keyword search across all interactions.
    """

    def __init__(self, db_path: str | Path):
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._owner = SQLiteConnectionOwner(self._db_path)
        self._init_schema()

    @property
    def _conn(self) -> sqlite3.Connection:
        return self._owner.connect()

    def _init_schema(self) -> None:
        conn = self._conn
        conn.executescript(_SCHEMA)
        conn.executescript(_FTS_TRIGGERS)
        conn.commit()

    def store(self, entry: MemoryEntry) -> str:
        """Store a memory entry. Returns entry_id."""
        if not entry.entry_id:
            entry.entry_id = f"mem_{uuid.uuid4().hex[:12]}"

        conn = self._conn
        conn.execute(
            "INSERT OR REPLACE INTO memories "
            "(entry_id, session_id, entry_type, query, response, "
            "metadata_json, tags, quality_score, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                entry.entry_id, entry.session_id, entry.entry_type,
                entry.query, entry.response,
                json.dumps(entry.metadata, default=str),
                entry.tag_str, entry.quality_score, entry.created_at,
            ),
        )
        conn.commit()
        log.debug("Stored memory %s (type=%s)", entry.entry_id, entry.entry_type)
        return entry.entry_id

    def search(self, query: str, limit: int = 10) -> List[MemorySearchResult]:
        """Search memories by keyword (FTS5)."""
        limit = min(limit, 10_000)
        # Sanitize query for FTS5
        safe_query = " ".join(
            w for w in query.split()
            if w and not any(c in w for c in "(){}[]\"'*:")
        )
        if not safe_query.strip():
            return []

        try:
            rows = self._conn.execute(
                "SELECT m.*, rank FROM memory_fts f "
                "JOIN memories m ON f.rowid = m.rowid "
                "WHERE memory_fts MATCH ? "
                "ORDER BY rank LIMIT ?",
                (safe_query, limit),
            ).fetchall()
        except sqlite3.OperationalError:
            log.debug("FTS5 search failed for query: %s", safe_query, exc_info=True)
            return []

        results = []
        for r in rows:
            entry = self._row_to_entry(r[:9])
            relevance = -r[9] if len(r) > 9 else 0.0  # FTS5 rank is negative
            results.append(MemorySearchResult(entry=entry, relevance=relevance))
        return results

    def get_by_session(self, session_id: str, limit: int = 100) -> List[MemoryEntry]:
        """Get all memories from a specific session."""
        limit = min(limit, 10_000)
        rows = self._conn.execute(
            "SELECT * FROM memories WHERE session_id=? "
            "ORDER BY created_at LIMIT ?",
            (session_id, limit),
        ).fetchall()
        return [self._row_to_entry(r) for r in rows]

    def get_recent(self, limit: int = 50) -> List[MemoryEntry]:
        """Get most recent memories across all sessions."""
        limit = min(limit, 10_000)
        rows = self._conn.execute(
            "SELECT * FROM memories ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [self._row_to_entry(r) for r in rows]

    def detect_patterns(self, min_occurrences: int = 3) -> List[PatternMatch]:
        """Detect recurring patterns across sessions.

        Analyzes:
        1. Recurring queries (same keywords across sessions)
        2. Success strategies (high-quality patterns)
        3. Failure modes (low-quality recurring patterns)
        """
        patterns = []
        patterns.extend(self._detect_recurring_queries(min_occurrences))
        patterns.extend(self._detect_quality_patterns(min_occurrences))
        return patterns

    def _detect_recurring_queries(self, min_occ: int) -> List[PatternMatch]:
        """Find queries that appear across multiple sessions."""
        rows = self._conn.execute(
            "SELECT query, COUNT(DISTINCT session_id) as sessions, "
            "COUNT(*) as total, MIN(created_at), MAX(created_at), "
            "GROUP_CONCAT(entry_id, ',') "
            "FROM memories WHERE entry_type='interaction' "
            "GROUP BY LOWER(SUBSTR(query, 1, 100)) "
            "HAVING sessions >= ? "
            "ORDER BY sessions DESC LIMIT 50",
            (min_occ,),
        ).fetchall()

        patterns = []
        for r in rows:
            evidence = r[5].split(",")[:10] if r[5] else []
            patterns.append(PatternMatch(
                pattern_id=f"recur_{uuid.uuid4().hex[:8]}",
                pattern_type="recurring_query",
                description=f"Query appears in {r[1]} sessions ({r[2]} total): {r[0][:100]}",
                evidence=evidence,
                confidence=min(1.0, r[1] / 10.0),
                occurrences=r[2],
                first_seen=r[3],
                last_seen=r[4],
            ))
        return patterns

    def _detect_quality_patterns(self, min_occ: int) -> List[PatternMatch]:
        """Find patterns in high/low quality interactions."""
        patterns = []

        # High quality tags
        rows = self._conn.execute(
            "SELECT tags, AVG(quality_score) as avg_q, COUNT(*) as cnt "
            "FROM memories WHERE tags != '' AND quality_score > 0 "
            "GROUP BY tags HAVING cnt >= ? "
            "ORDER BY avg_q DESC LIMIT 20",
            (min_occ,),
        ).fetchall()

        for r in rows:
            ptype = "success_strategy" if r[1] >= 0.7 else "failure_mode"
            patterns.append(PatternMatch(
                pattern_id=f"qual_{uuid.uuid4().hex[:8]}",
                pattern_type=ptype,
                description=f"Tags [{r[0]}]: avg quality {r[1]:.2f} across {r[2]} entries",
                evidence=[],
                confidence=min(1.0, r[2] / 10.0),
                occurrences=r[2],
                first_seen=0.0,
                last_seen=0.0,
            ))

        return patterns

    def summarize_session(self, session_id: str) -> Dict[str, Any]:
        """Generate and store a session summary."""
        entries = self.get_by_session(session_id)
        if not entries:
            return {}

        tag_counts = Counter()
        total_quality = 0.0
        for e in entries:
            tag_counts.update(e.tags)
            total_quality += e.quality_score

        avg_quality = total_quality / len(entries) if entries else 0.0
        top_tags = [t for t, _ in tag_counts.most_common(10)]

        summary = {
            "session_id": session_id,
            "entry_count": len(entries),
            "avg_quality": round(avg_quality, 3),
            "top_tags": top_tags,
            "first_entry": entries[0].created_at,
            "last_entry": entries[-1].created_at,
        }

        conn = self._conn
        conn.execute(
            "INSERT OR REPLACE INTO session_summaries "
            "(session_id, summary, entry_count, avg_quality, tags_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                session_id,
                json.dumps(summary, default=str),
                len(entries), avg_quality,
                json.dumps(top_tags), entries[0].created_at,
            ),
        )
        conn.commit()
        return summary

    def get_session_summaries(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent session summaries."""
        rows = self._conn.execute(
            "SELECT * FROM session_summaries ORDER BY created_at DESC LIMIT ?",
            (min(limit, 1000),),
        ).fetchall()
        return [
            {
                "session_id": r[0], "summary": r[1],
                "entry_count": r[2], "avg_quality": r[3],
                "tags": json.loads(r[4] or "[]"), "created_at": r[5],
            }
            for r in rows
        ]

    def stats(self) -> Dict[str, Any]:
        """Aggregate statistics."""
        conn = self._conn
        row = conn.execute(
            "SELECT COUNT(*), COUNT(DISTINCT session_id), "
            "AVG(quality_score), MIN(created_at), MAX(created_at) "
            "FROM memories"
        ).fetchone()

        type_counts = dict(conn.execute(
            "SELECT entry_type, COUNT(*) FROM memories GROUP BY entry_type"
        ).fetchall())

        return {
            "total_entries": row[0] or 0,
            "total_sessions": row[1] or 0,
            "avg_quality": round(row[2] or 0.0, 3),
            "oldest": row[3] or 0.0,
            "newest": row[4] or 0.0,
            "by_type": type_counts,
        }

    MAX_ENTRIES = 50_000

    def prune_old(self, keep: int = 0) -> int:
        """Delete oldest entries beyond *keep* (defaults to MAX_ENTRIES).

        Returns the number of rows deleted.
        """
        keep = keep or self.MAX_ENTRIES
        conn = self._conn
        cur = conn.execute(
            "DELETE FROM memories WHERE entry_id NOT IN "
            "(SELECT entry_id FROM memories "
            "ORDER BY created_at DESC LIMIT ?)",
            (keep,),
        )
        conn.commit()
        return cur.rowcount

    def close(self) -> None:
        self._owner.close()

    @staticmethod
    def _row_to_entry(row: tuple) -> MemoryEntry:
        return MemoryEntry(
            entry_id=row[0],
            session_id=row[1],
            entry_type=row[2],
            query=row[3],
            response=row[4],
            metadata=json.loads(row[5] or "{}"),
            tags=[t for t in (row[6] or "").split(",") if t],
            quality_score=row[7] or 0.0,
            created_at=row[8] or 0.0,
        )


# ---------------------------------------------------------------------------
# PersistentMemory -- high-level API wrapping a backend
# ---------------------------------------------------------------------------

class PersistentMemory:
    """Cross-session persistent memory with pattern detection.

    Wraps a MemoryBackend (default: SQLiteMemoryBackend) and provides
    high-level methods for recording interactions, recalling context,
    and detecting patterns.

    When LimitlessApp V2 is ready, swap the backend without changing
    any calling code.
    """

    def __init__(
        self,
        backend: Optional[SQLiteMemoryBackend] = None,
        db_path: str | Path = "_persistent_memory/memory.db",
    ):
        self._backend = backend or SQLiteMemoryBackend(db_path)

    def record_interaction(
        self,
        session_id: str,
        query: str,
        response: str,
        quality: float = 0.0,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Record a query/response interaction."""
        entry = MemoryEntry(
            entry_id=f"mem_{uuid.uuid4().hex[:12]}",
            session_id=session_id,
            entry_type="interaction",
            query=query,
            response=response,
            quality_score=quality,
            tags=tags or [],
            metadata=metadata or {},
        )
        return self._backend.store(entry)

    def record_insight(
        self,
        session_id: str,
        insight: str,
        source_query: str = "",
        tags: Optional[List[str]] = None,
    ) -> str:
        """Record a learned insight or pattern."""
        entry = MemoryEntry(
            entry_id=f"ins_{uuid.uuid4().hex[:12]}",
            session_id=session_id,
            entry_type="insight",
            query=source_query,
            response=insight,
            quality_score=1.0,
            tags=tags or ["insight"],
        )
        return self._backend.store(entry)

    def record_error(
        self,
        session_id: str,
        query: str,
        error: str,
        tags: Optional[List[str]] = None,
    ) -> str:
        """Record an error for future avoidance."""
        entry = MemoryEntry(
            entry_id=f"err_{uuid.uuid4().hex[:12]}",
            session_id=session_id,
            entry_type="error",
            query=query,
            response=error,
            quality_score=0.0,
            tags=tags or ["error"],
        )
        return self._backend.store(entry)

    def recall(self, query: str, limit: int = 10) -> List[MemorySearchResult]:
        """Search persistent memory for relevant context."""
        return self._backend.search(query, limit=limit)

    def recall_session(self, session_id: str) -> List[MemoryEntry]:
        """Get all memories from a specific session."""
        return self._backend.get_by_session(session_id)

    def recent(self, limit: int = 20) -> List[MemoryEntry]:
        """Get recent memories across sessions."""
        return self._backend.get_recent(limit=limit)

    def find_patterns(self, min_occurrences: int = 3) -> List[PatternMatch]:
        """Detect cross-session patterns."""
        return self._backend.detect_patterns(min_occurrences=min_occurrences)

    def summarize_session(self, session_id: str) -> Dict[str, Any]:
        """Generate and store a session summary."""
        return self._backend.summarize_session(session_id)

    def stats(self) -> Dict[str, Any]:
        """Get aggregate statistics."""
        return self._backend.stats()

    def close(self) -> None:
        self._backend.close()
