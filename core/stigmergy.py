"""
Stigmergic Retrieval Booster (Ant Colony for RAG)
---------------------------------------------------
Indirect coordination through environment modification. Chunks
that lead to successful answers accumulate "pheromone" that boosts
their retrieval scores for similar future queries.

Based on:
- Ant Colony Optimization (Dorigo, 1992): Pheromone-based path optimization
- Stigmergy in Multi-Agent Systems (Frontiers AI, 2025): LLM-powered swarms
- LLM-Assisted Iterative Evolution with Swarm Intelligence (2025)

Key insight: Standard retrieval treats every query independently.
Stigmergy creates a feedback loop where past successes improve
future retrievals WITHOUT any model changes or prompt engineering.

Algorithm:
1. When retrieval + generation succeeds: deposit positive pheromone
   on the chunks that contributed to the answer
2. When retrieval + generation fails: deposit negative pheromone
3. Pheromone evaporates over time (prevents stale trails)
4. During retrieval: boost chunk scores by pheromone strength

Zero LLM cost per query. Pure SQLite bookkeeping.
"""

from __future__ import annotations

import math
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class PheromoneConfig:
    """Pheromone system configuration."""
    positive_deposit: float = 1.0    # deposit on success
    negative_deposit: float = -0.3   # deposit on failure
    evaporation_rate: float = 0.05   # fraction lost per hour
    min_pheromone: float = 0.01      # cleanup threshold
    max_pheromone: float = 10.0      # ceiling to prevent runaway
    boost_weight: float = 0.15       # how much pheromone affects final score


class StigmergicBooster:
    """
    Pheromone-based retrieval score booster.

    Chunks that contributed to good answers get boosted.
    Chunks that led to bad answers get penalized.
    All trails evaporate over time to prevent staleness.

    Zero LLM cost. Pure SQLite.
    """

    def __init__(
        self,
        db_path: str = "_stigmergy/pheromones.db",
        config: Optional[PheromoneConfig] = None,
    ):
        self.config = config or PheromoneConfig()
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self._db_path))

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS pheromones (
                    chunk_id TEXT NOT NULL,
                    query_type TEXT NOT NULL,
                    strength REAL DEFAULT 1.0,
                    success_count INTEGER DEFAULT 0,
                    fail_count INTEGER DEFAULT 0,
                    last_updated REAL,
                    PRIMARY KEY (chunk_id, query_type)
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_pheromone_strength
                ON pheromones(strength DESC)
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trail_log (
                    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chunk_id TEXT,
                    query_type TEXT,
                    delta REAL,
                    was_success INTEGER,
                    timestamp REAL
                )
            """)
            conn.commit()

    def deposit(
        self,
        chunk_ids: List[str],
        query_type: str,
        success: bool,
    ) -> None:
        """
        Deposit pheromone on chunks after a query.

        Args:
            chunk_ids: IDs of chunks used in retrieval
            query_type: Category of query (e.g., "lookup", "debug", "explain")
            success: Whether the answer was good (True) or bad (False)
        """
        delta = (
            self.config.positive_deposit if success
            else self.config.negative_deposit
        )
        now = time.time()

        with self._connect() as conn:
            for cid in chunk_ids:
                conn.execute("""
                    INSERT INTO pheromones
                    (chunk_id, query_type, strength, success_count,
                     fail_count, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(chunk_id, query_type) DO UPDATE SET
                        strength = MIN(?, MAX(0, strength + ?)),
                        success_count = success_count + ?,
                        fail_count = fail_count + ?,
                        last_updated = ?
                """, (
                    cid, query_type,
                    max(0, delta), 1 if success else 0,
                    0 if success else 1, now,
                    # ON CONFLICT params:
                    self.config.max_pheromone, delta,
                    1 if success else 0,
                    0 if success else 1,
                    now,
                ))

                # Log trail
                conn.execute("""
                    INSERT INTO trail_log
                    (chunk_id, query_type, delta, was_success, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                """, (cid, query_type, delta, 1 if success else 0, now))

            conn.commit()

    def boost_scores(
        self,
        chunk_scores: List[Tuple[str, float]],
        query_type: str,
    ) -> List[Tuple[str, float]]:
        """
        Boost retrieval scores using pheromone trails.

        Args:
            chunk_scores: List of (chunk_id, base_score) from retrieval
            query_type: Category of query

        Returns:
            List of (chunk_id, boosted_score) with pheromone enhancement
        """
        if not chunk_scores:
            return chunk_scores

        chunk_ids = [cid for cid, _ in chunk_scores]
        pheromone_map = self._get_pheromones(chunk_ids, query_type)

        boosted = []
        for cid, base_score in chunk_scores:
            pheromone = pheromone_map.get(cid, 0.0)
            # Sigmoid-normalized boost: prevents extreme values
            if pheromone > 0:
                boost = self.config.boost_weight * (
                    2.0 / (1.0 + math.exp(-pheromone)) - 1.0
                )
            else:
                boost = 0.0
            boosted.append((cid, base_score + boost))

        # Re-sort by boosted score
        boosted.sort(key=lambda x: x[1], reverse=True)
        return boosted

    def evaporate(self) -> int:
        """
        Apply pheromone evaporation. Call periodically (e.g., hourly).

        Returns number of trails cleaned up.
        """
        now = time.time()
        with self._connect() as conn:
            # Evaporate based on time elapsed
            conn.execute("""
                UPDATE pheromones SET
                    strength = strength * POWER(
                        ?,
                        (? - last_updated) / 3600.0
                    ),
                    last_updated = ?
                WHERE strength > ?
            """, (
                1.0 - self.config.evaporation_rate,
                now, now,
                self.config.min_pheromone,
            ))

            # Cleanup dead trails
            cur = conn.execute(
                "DELETE FROM pheromones WHERE strength <= ?",
                (self.config.min_pheromone,),
            )
            cleaned = cur.rowcount
            conn.commit()
        return cleaned

    def hot_chunks(self, query_type: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Return chunks with strongest pheromone for a query type."""
        with self._connect() as conn:
            cur = conn.execute(
                "SELECT chunk_id, strength, success_count, fail_count "
                "FROM pheromones WHERE query_type = ? "
                "ORDER BY strength DESC LIMIT ?",
                (query_type, limit),
            )
            return [
                {
                    "chunk_id": r[0], "strength": round(r[1], 3),
                    "successes": r[2], "failures": r[3],
                    "success_rate": (
                        round(r[2] / (r[2] + r[3]), 3)
                        if (r[2] + r[3]) > 0 else 0.0
                    ),
                }
                for r in cur.fetchall()
            ]

    def stats(self) -> Dict[str, Any]:
        """Return pheromone system statistics."""
        with self._connect() as conn:
            cur = conn.execute("""
                SELECT query_type, COUNT(*), AVG(strength),
                       MAX(strength), SUM(success_count), SUM(fail_count)
                FROM pheromones GROUP BY query_type
            """)
            by_type = [
                {
                    "query_type": r[0], "trails": r[1],
                    "avg_strength": round(r[2], 3),
                    "max_strength": round(r[3], 3),
                    "total_successes": r[4],
                    "total_failures": r[5],
                }
                for r in cur.fetchall()
            ]

            total = (conn.execute(
                "SELECT COUNT(*) FROM pheromones"
            ).fetchone() or (0,))[0]
            log_count = (conn.execute(
                "SELECT COUNT(*) FROM trail_log"
            ).fetchone() or (0,))[0]

        return {
            "total_trails": total,
            "total_deposits": log_count,
            "by_type": by_type,
        }

    def _get_pheromones(
        self,
        chunk_ids: List[str],
        query_type: str,
    ) -> Dict[str, float]:
        """Get current pheromone strengths for a set of chunks."""
        now = time.time()
        result = {}
        with self._connect() as conn:
            for cid in chunk_ids:
                cur = conn.execute(
                    "SELECT strength, last_updated FROM pheromones "
                    "WHERE chunk_id = ? AND query_type = ?",
                    (cid, query_type),
                )
                row = cur.fetchone()
                if row:
                    # Apply real-time evaporation
                    hours = (now - row[1]) / 3600.0
                    evaporated = row[0] * (
                        (1.0 - self.config.evaporation_rate) ** hours
                    )
                    result[cid] = max(0.0, evaporated)
        return result
