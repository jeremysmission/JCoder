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
    q_value: float = 0.0
    pass_count: int = 0
    fail_count: int = 0
    p2value: float = 0.0


class ExperienceStore:
    """
    Stores and retrieves successful past interactions for few-shot injection.

    Only stores experiences that were marked as high-quality:
    - Confidence above threshold
    - Explicitly positive feedback
    - Answer contains code (not just "I don't know")
    """

    MAX_EXPERIENCES = 1000  # Keep store bounded

    def __init__(
        self,
        db_path: str,
        min_confidence: float = 0.6,
        q_learning_rate: float = 0.2,
        q_value_weight: float = 0.3,
        p2value_alpha: float = 0.5,
        near_miss_boost: float = 1.3,
    ):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.min_confidence = min_confidence
        self.q_learning_rate = q_learning_rate
        self.q_value_weight = q_value_weight
        self.p2value_alpha = p2value_alpha
        self.near_miss_boost = near_miss_boost
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
                    keywords TEXT,
                    q_value REAL DEFAULT 0.0
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_exp_confidence
                ON experiences(confidence DESC)
            """)
            conn.commit()
            cols = {
                row[1]
                for row in conn.execute("PRAGMA table_info(experiences)").fetchall()
            }
            if "q_value" not in cols:
                conn.execute(
                    "ALTER TABLE experiences ADD COLUMN q_value REAL DEFAULT 0.0"
                )
            if "pass_count" not in cols:
                conn.execute(
                    "ALTER TABLE experiences ADD COLUMN pass_count INTEGER DEFAULT 0"
                )
            if "fail_count" not in cols:
                conn.execute(
                    "ALTER TABLE experiences ADD COLUMN fail_count INTEGER DEFAULT 0"
                )
            if "p2value" not in cols:
                conn.execute(
                    "ALTER TABLE experiences ADD COLUMN p2value REAL DEFAULT 0.0"
                )
                conn.commit()

    def compute_p2value(
        self,
        confidence: float,
        pass_count: int = 0,
        fail_count: int = 0,
    ) -> float:
        """Compute P2Value priority score blending confidence with test pass rate.

        When no test results exist (pass_count + fail_count == 0), returns
        raw confidence.  Otherwise blends using ``p2value_alpha``:

            base = alpha * confidence + (1 - alpha) * pass_rate

        If ``fail_count == 1`` (near-miss), applies ``near_miss_boost``.
        Result is capped at 1.0.
        """
        total = pass_count + fail_count
        if total == 0:
            return float(confidence)
        pass_rate = pass_count / total
        base = self.p2value_alpha * confidence + (1.0 - self.p2value_alpha) * pass_rate
        if fail_count == 1:
            base *= self.near_miss_boost
        return min(1.0, max(0.0, base))

    def store(self, exp_id: str, query: str, answer: str,
              source_files: List[str], confidence: float,
              pass_count: int = 0, fail_count: int = 0) -> bool:
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
        p2value = self.compute_p2value(confidence, pass_count, fail_count)

        with self._connect() as conn:
            # Enforce max size -- evict lowest p2value first
            conn.execute("""
                DELETE FROM experiences WHERE exp_id IN (
                    SELECT exp_id FROM experiences
                    ORDER BY p2value ASC, confidence ASC, timestamp ASC
                    LIMIT max(0, (SELECT count(*) FROM experiences) - ?)
                )
            """, (self.MAX_EXPERIENCES - 1,))

            conn.execute("""
                INSERT OR REPLACE INTO experiences
                (exp_id, query, answer, source_files_json, confidence,
                 timestamp, use_count, keywords, q_value,
                 pass_count, fail_count, p2value)
                VALUES (?, ?, ?, ?, ?, ?, 0, ?, 0.0, ?, ?, ?)
            """, (
                exp_id, query, answer[:2000],
                json.dumps(source_files),
                confidence, time.time(), keywords,
                pass_count, fail_count, p2value,
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
                "confidence, timestamp, use_count, keywords, q_value, "
                "pass_count, fail_count, p2value "
                "FROM experiences ORDER BY p2value DESC, confidence DESC LIMIT 100"
            )
            rows = cur.fetchall()

        # Score by keyword overlap + p2value
        scored: List[Tuple[float, Experience]] = []
        for row in rows:
            exp_keywords = set((row[7] or "").split())
            if not exp_keywords:
                continue
            overlap = len(query_keywords & exp_keywords)
            total = len(query_keywords | exp_keywords)
            jaccard = overlap / total if total > 0 else 0.0
            p2v = row[11] or 0.0
            # Combine keyword match with p2value (replaces raw confidence)
            quality = p2v if p2v > 0 else (row[4] or 0.0)
            score = (
                (1.0 - self.q_value_weight) * (0.6 * jaccard + 0.4 * quality)
                + self.q_value_weight * max(0.0, min(1.0, row[8] or 0.0))
            )

            if score > 0.1:  # minimum relevance threshold
                exp = Experience(
                    exp_id=row[0], query=row[1], answer=row[2],
                    source_files=json.loads(row[3]) if row[3] else [],
                    confidence=row[4] or 0.0,
                    timestamp=row[5] or 0.0,
                    use_count=row[6] or 0,
                    q_value=row[8] or 0.0,
                    pass_count=row[9] or 0,
                    fail_count=row[10] or 0,
                    p2value=p2v,
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
                       AVG(use_count), AVG(q_value), AVG(p2value),
                       SUM(CASE WHEN fail_count = 1 THEN 1 ELSE 0 END)
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
            "avg_q_value": round(row[4] or 0, 3),
            "avg_p2value": round(row[5] or 0, 3),
            "near_miss_count": row[6] or 0,
        }

    def update_q_value(self, exp_id: str, reward: float) -> None:
        """Apply a simple Bellman-style update toward the observed reward."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT q_value FROM experiences WHERE exp_id = ?",
                (exp_id,),
            ).fetchone()
            if row is None:
                return
            current = float(row[0] or 0.0)
            updated = current + self.q_learning_rate * (float(reward) - current)
            conn.execute(
                "UPDATE experiences SET q_value = ? WHERE exp_id = ?",
                (updated, exp_id),
            )
            conn.commit()

    def replay_blend(
        self,
        new_experiences: List[Experience],
        replay_ratio: float = 0.3,
        max_total: int = 10,
    ) -> List[Experience]:
        """Blend fresh experiences with top stored ones."""
        if max_total <= 0:
            return []

        replay_count = max(0, min(max_total, int(round(max_total * replay_ratio))))
        new_count = max_total - replay_count
        replay = self.retrieve(
            " ".join(exp.query for exp in new_experiences),
            top_k=replay_count,
        )
        # Fall back to top-scoring stored experiences when keyword match is sparse
        if len(replay) < replay_count:
            replay = self._top_experiences(replay_count)
        combined = list(new_experiences)[:new_count] + replay[:replay_count]
        return combined[:max_total]

    def _top_experiences(self, top_k: int) -> List[Experience]:
        """Return the highest-scoring experiences regardless of keyword match."""
        with self._connect() as conn:
            cur = conn.execute(
                "SELECT exp_id, query, answer, source_files_json, "
                "confidence, timestamp, use_count, keywords, q_value, "
                "pass_count, fail_count, p2value "
                "FROM experiences ORDER BY p2value DESC, confidence DESC LIMIT ?",
                (top_k,),
            )
            rows = cur.fetchall()
        return [
            Experience(
                exp_id=r[0], query=r[1], answer=r[2],
                source_files=json.loads(r[3]) if r[3] else [],
                confidence=r[4] or 0.0, timestamp=r[5] or 0.0,
                use_count=r[6] or 0, q_value=r[8] or 0.0,
                pass_count=r[9] or 0, fail_count=r[10] or 0,
                p2value=r[11] or 0.0,
            )
            for r in rows
        ]

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
