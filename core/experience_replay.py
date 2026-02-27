"""
Experience Replay (Zero-Cost Self-Improvement)
------------------------------------------------
Stores successful query-answer trajectories and injects them as
in-context examples for similar future queries. No training,
no weight updates -- pure prompt engineering with memory.

Based on: Self-Improving Agent research (2025) showing experience
replay lifted performance from 73% to 93% on code tasks.

The key insight: successful past interactions are the best
few-shot examples because they're from YOUR codebase, YOUR
query patterns, and YOUR preferred answer style.

Storage: SQLite (consistent with other JCoder modules).
Retrieval: Keyword overlap scoring (fast, no embedding needed).
"""

from __future__ import annotations

import json
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class Experience:
    """A stored successful interaction."""
    exp_id: str
    query: str
    answer: str
    source_files: List[str]
    confidence: float
    timestamp: float
    use_count: int = 0


class ExperienceStore:
    """
    Stores and retrieves successful past interactions for few-shot injection.

    Only stores experiences that were marked as high-quality:
    - Confidence above threshold
    - Explicitly positive feedback
    - Answer contains code (not just "I don't know")
    """

    MAX_EXPERIENCES = 1000  # Keep store bounded

    def __init__(self, db_path: str, min_confidence: float = 0.6):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.min_confidence = min_confidence
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path))

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS experiences (
                    exp_id TEXT PRIMARY KEY,
                    query TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    source_files_json TEXT,
                    confidence REAL,
                    timestamp REAL,
                    use_count INTEGER DEFAULT 0,
                    keywords TEXT
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_exp_confidence
                ON experiences(confidence DESC)
            """)
            conn.commit()

    def store(self, exp_id: str, query: str, answer: str,
              source_files: List[str], confidence: float) -> bool:
        """
        Store an experience if it meets quality criteria.
        Returns True if stored, False if rejected.
        """
        if confidence < self.min_confidence:
            return False
        if len(answer.strip()) < 50:
            return False
        # Must contain some code-like content
        if not re.search(r"[a-zA-Z_]\w*\(|def |class |import |return ", answer):
            return False

        keywords = self._extract_keywords(query)

        with self._connect() as conn:
            # Enforce max size
            conn.execute("""
                DELETE FROM experiences WHERE exp_id IN (
                    SELECT exp_id FROM experiences
                    ORDER BY confidence ASC, timestamp ASC
                    LIMIT max(0, (SELECT count(*) FROM experiences) - ?)
                )
            """, (self.MAX_EXPERIENCES - 1,))

            conn.execute("""
                INSERT OR REPLACE INTO experiences
                (exp_id, query, answer, source_files_json, confidence,
                 timestamp, use_count, keywords)
                VALUES (?, ?, ?, ?, ?, ?, 0, ?)
            """, (
                exp_id, query, answer[:2000],
                json.dumps(source_files),
                confidence, time.time(), keywords,
            ))
            conn.commit()
        return True

    def retrieve(self, query: str, top_k: int = 3) -> List[Experience]:
        """
        Find the most relevant past experiences for a query.
        Uses keyword overlap + confidence scoring.
        """
        query_keywords = set(self._extract_keywords(query).split())
        if not query_keywords:
            return []

        with self._connect() as conn:
            cur = conn.execute(
                "SELECT exp_id, query, answer, source_files_json, "
                "confidence, timestamp, use_count, keywords "
                "FROM experiences ORDER BY confidence DESC LIMIT 100"
            )
            rows = cur.fetchall()

        # Score by keyword overlap
        scored: List[Tuple[float, Experience]] = []
        for row in rows:
            exp_keywords = set((row[7] or "").split())
            if not exp_keywords:
                continue
            overlap = len(query_keywords & exp_keywords)
            total = len(query_keywords | exp_keywords)
            jaccard = overlap / total if total > 0 else 0.0
            # Combine keyword match with confidence
            score = 0.6 * jaccard + 0.4 * (row[4] or 0.0)

            if score > 0.1:  # minimum relevance threshold
                exp = Experience(
                    exp_id=row[0], query=row[1], answer=row[2],
                    source_files=json.loads(row[3]) if row[3] else [],
                    confidence=row[4] or 0.0,
                    timestamp=row[5] or 0.0,
                    use_count=row[6] or 0,
                )
                scored.append((score, exp))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = [exp for _, exp in scored[:top_k]]

        # Increment use counts
        if results:
            with self._connect() as conn:
                for exp in results:
                    conn.execute(
                        "UPDATE experiences SET use_count = use_count + 1 "
                        "WHERE exp_id = ?", (exp.exp_id,))
                conn.commit()

        return results

    def format_as_examples(self, experiences: List[Experience],
                           max_chars: int = 2000) -> str:
        """Format experiences as few-shot examples for prompt injection."""
        if not experiences:
            return ""

        parts = ["Here are examples of successful past interactions:\n"]
        total = 0

        for exp in experiences:
            example = (
                f"Q: {exp.query}\n"
                f"A: {exp.answer[:500]}\n"
                f"Sources: {', '.join(exp.source_files[:3])}\n"
                f"---\n"
            )
            if total + len(example) > max_chars:
                break
            parts.append(example)
            total += len(example)

        return "\n".join(parts)

    def stats(self) -> Dict:
        """Return store statistics."""
        with self._connect() as conn:
            cur = conn.execute("""
                SELECT COUNT(*), AVG(confidence), SUM(use_count),
                       AVG(use_count)
                FROM experiences
            """)
            row = cur.fetchone()
        if not row or row[0] == 0:
            return {"total": 0}
        return {
            "total": row[0],
            "avg_confidence": round(row[1] or 0, 3),
            "total_uses": row[2] or 0,
            "avg_uses": round(row[3] or 0, 1),
        }

    @staticmethod
    def _extract_keywords(text: str) -> str:
        """Extract meaningful keywords from text."""
        stop = {
            "what", "how", "does", "the", "this", "that", "is", "are",
            "was", "were", "do", "can", "could", "would", "should",
            "where", "when", "why", "which", "who", "a", "an", "in",
            "on", "at", "to", "for", "of", "with", "from", "by", "and",
            "or", "not", "it", "its", "my", "your", "me", "i", "we",
        }
        words = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", text.lower())
        keywords = [w for w in words if w not in stop and len(w) > 2]
        return " ".join(sorted(set(keywords)))
