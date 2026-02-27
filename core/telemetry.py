"""
Telemetry
---------
SQLite-backed logging for every query, retrieval, and outcome.
Foundation for all self-learning feedback loops.

Records: query text, retrieved chunk IDs, scores, answer snippet,
latency, and optional user feedback signal. Enables:
- Failure analysis (which queries consistently fail?)
- Confidence calibration (are high-score retrievals actually good?)
- Curriculum generation (hard queries become training data)
- Drift detection (is retrieval quality degrading over time?)
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class QueryEvent:
    """Single query through the pipeline."""
    query_id: str
    query_text: str
    timestamp: float
    retrieval_latency_ms: float = 0.0
    generation_latency_ms: float = 0.0
    chunk_ids: List[str] = field(default_factory=list)
    chunk_scores: List[float] = field(default_factory=list)
    source_files: List[str] = field(default_factory=list)
    answer_snippet: str = ""
    confidence: float = 0.0
    reflection_relevant: float = 0.0
    reflection_supported: float = 0.0
    reflection_useful: float = 0.0
    feedback: Optional[str] = None  # "good" | "bad" | None


class TelemetryStore:
    """Append-only query telemetry for self-learning feedback loops."""

    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path))

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS query_events (
                    query_id TEXT PRIMARY KEY,
                    query_text TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    retrieval_latency_ms REAL,
                    generation_latency_ms REAL,
                    chunk_ids_json TEXT,
                    chunk_scores_json TEXT,
                    source_files_json TEXT,
                    answer_snippet TEXT,
                    confidence REAL,
                    reflection_relevant REAL,
                    reflection_supported REAL,
                    reflection_useful REAL,
                    feedback TEXT
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_events_ts
                ON query_events(timestamp)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_events_confidence
                ON query_events(confidence)
            """)
            conn.commit()

    def log(self, event: QueryEvent) -> None:
        """Write a single query event."""
        with self._connect() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO query_events
                (query_id, query_text, timestamp, retrieval_latency_ms,
                 generation_latency_ms, chunk_ids_json, chunk_scores_json,
                 source_files_json, answer_snippet, confidence,
                 reflection_relevant, reflection_supported, reflection_useful,
                 feedback)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                event.query_id, event.query_text, event.timestamp,
                event.retrieval_latency_ms, event.generation_latency_ms,
                json.dumps(event.chunk_ids),
                json.dumps(event.chunk_scores),
                json.dumps(event.source_files),
                event.answer_snippet[:500],
                event.confidence,
                event.reflection_relevant,
                event.reflection_supported,
                event.reflection_useful,
                event.feedback,
            ))
            conn.commit()

    def set_feedback(self, query_id: str, feedback: str) -> None:
        """Update feedback for a logged query (user signal)."""
        with self._connect() as conn:
            conn.execute(
                "UPDATE query_events SET feedback=? WHERE query_id=?",
                (feedback, query_id),
            )
            conn.commit()

    def low_confidence_queries(self, threshold: float = 0.3,
                                limit: int = 100) -> List[QueryEvent]:
        """Return queries where confidence was below threshold."""
        return self._query_by_confidence("<=", threshold, limit)

    def high_confidence_queries(self, threshold: float = 0.7,
                                 limit: int = 100) -> List[QueryEvent]:
        """Return queries where confidence was above threshold."""
        return self._query_by_confidence(">=", threshold, limit)

    def failed_queries(self, limit: int = 100) -> List[QueryEvent]:
        """Return queries with negative feedback."""
        with self._connect() as conn:
            cur = conn.execute(
                "SELECT * FROM query_events WHERE feedback='bad' "
                "ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            )
            return [self._row_to_event(r) for r in cur.fetchall()]

    def recent(self, limit: int = 50) -> List[QueryEvent]:
        """Return most recent query events."""
        with self._connect() as conn:
            cur = conn.execute(
                "SELECT * FROM query_events ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            )
            return [self._row_to_event(r) for r in cur.fetchall()]

    def stats(self) -> Dict[str, Any]:
        """Aggregate stats for monitoring."""
        with self._connect() as conn:
            cur = conn.execute("""
                SELECT
                    COUNT(*) as total,
                    AVG(confidence) as avg_confidence,
                    AVG(retrieval_latency_ms) as avg_retrieval_ms,
                    AVG(generation_latency_ms) as avg_gen_ms,
                    SUM(CASE WHEN feedback='good' THEN 1 ELSE 0 END) as good,
                    SUM(CASE WHEN feedback='bad' THEN 1 ELSE 0 END) as bad
                FROM query_events
            """)
            row = cur.fetchone()
        if not row or row[0] == 0:
            return {"total": 0}
        return {
            "total": row[0],
            "avg_confidence": round(row[1] or 0, 3),
            "avg_retrieval_ms": round(row[2] or 0, 1),
            "avg_generation_ms": round(row[3] or 0, 1),
            "feedback_good": row[4] or 0,
            "feedback_bad": row[5] or 0,
        }

    def _query_by_confidence(self, op: str, threshold: float,
                              limit: int) -> List[QueryEvent]:
        with self._connect() as conn:
            cur = conn.execute(
                f"SELECT * FROM query_events WHERE confidence {op} ? "
                "ORDER BY timestamp DESC LIMIT ?",
                (threshold, limit),
            )
            return [self._row_to_event(r) for r in cur.fetchall()]

    @staticmethod
    def _row_to_event(row) -> QueryEvent:
        return QueryEvent(
            query_id=row[0],
            query_text=row[1],
            timestamp=row[2],
            retrieval_latency_ms=row[3] or 0.0,
            generation_latency_ms=row[4] or 0.0,
            chunk_ids=json.loads(row[5]) if row[5] else [],
            chunk_scores=json.loads(row[6]) if row[6] else [],
            source_files=json.loads(row[7]) if row[7] else [],
            answer_snippet=row[8] or "",
            confidence=row[9] or 0.0,
            reflection_relevant=row[10] or 0.0,
            reflection_supported=row[11] or 0.0,
            reflection_useful=row[12] or 0.0,
            feedback=row[13],
        )
