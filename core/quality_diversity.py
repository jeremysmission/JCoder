"""
Quality-Diversity Archive (MAP-Elites Pattern)
-------------------------------------------------
Instead of keeping only the single best config/prompt/answer,
maintains a diverse archive of high-quality solutions across
different behavioral niches.

Based on:
- MAP-Elites (2015): Illuminating search spaces of solutions
- Quality-Diversity Optimization (2020): Diverse high-performing solutions
- CMA-ME (GECCO 2020): Combining CMA-ES with MAP-Elites archives

Key insight: A single "best" configuration doesn't exist. Different
types of queries need different configurations. MAP-Elites discovers
the best configuration for EACH niche.

Behavior space dimensions:
1. Query complexity (simple <-> hard)
2. Answer type (lookup <-> reasoning)
3. Retrieval confidence (low <-> high)

Each cell in the archive stores the best-performing configuration
for that specific niche. This gives the system N specialized
configurations instead of one generalist.
"""

from __future__ import annotations

import copy
import json
import math
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ArchiveCell:
    """A single cell in the MAP-Elites archive."""
    niche_key: str  # e.g. "complex_reasoning_highconf"
    config_json: str  # serialized configuration
    fitness: float  # quality score
    behavior: Dict[str, float]  # behavior descriptor values
    visits: int = 0
    last_updated: float = 0.0
    improvement_count: int = 0  # how many times this cell was improved


@dataclass
class QDSolution:
    """A candidate solution with its behavior descriptor."""
    config: Dict[str, Any]
    fitness: float
    behavior: Dict[str, float]  # values for each behavior dimension
    niche_key: str = ""


# ---------------------------------------------------------------------------
# Behavior descriptor computation
# ---------------------------------------------------------------------------

def compute_behavior(
    query_complexity: float,
    answer_type: str,
    retrieval_confidence: float,
) -> Dict[str, float]:
    """
    Compute behavior descriptor from query characteristics.

    Returns a dict with normalized (0-1) values for each dimension.
    """
    # Discretize answer type to numeric
    type_map = {
        "lookup": 0.0,
        "explain": 0.25,
        "debug": 0.5,
        "reasoning": 0.75,
        "design": 1.0,
    }
    answer_val = type_map.get(answer_type, 0.5)

    return {
        "complexity": max(0.0, min(1.0, query_complexity)),
        "answer_type": answer_val,
        "retrieval_conf": max(0.0, min(1.0, retrieval_confidence)),
    }


def niche_key(behavior: Dict[str, float], resolution: int = 4) -> str:
    """
    Map continuous behavior to discrete niche key.

    resolution=4 means each dimension is divided into 4 bins,
    giving 4^3 = 64 total niches.
    """
    parts = []
    for dim in sorted(behavior.keys()):
        val = behavior[dim]
        bucket = min(resolution - 1, int(val * resolution))
        parts.append(f"{dim}={bucket}")
    return "|".join(parts)


# ---------------------------------------------------------------------------
# MAP-Elites Archive
# ---------------------------------------------------------------------------

class QualityDiversityArchive:
    """
    MAP-Elites archive: maintains the best solution per behavioral niche.

    Unlike standard optimization (one best solution), this finds the
    best solution for EACH type of problem. Result: a gallery of
    specialized configurations.
    """

    def __init__(
        self,
        db_path: str = "_qd_archive/archive.db",
        resolution: int = 4,
    ):
        """
        Args:
            db_path: SQLite path for persistent archive
            resolution: Grid resolution per behavior dimension
                        4 = 64 niches (4^3), 5 = 125 niches, etc.
        """
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self.resolution = resolution
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self._db_path))

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS archive (
                    niche_key TEXT PRIMARY KEY,
                    config_json TEXT NOT NULL,
                    fitness REAL NOT NULL,
                    behavior_json TEXT NOT NULL,
                    visits INTEGER DEFAULT 0,
                    last_updated REAL,
                    improvement_count INTEGER DEFAULT 0
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS history (
                    entry_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    niche_key TEXT,
                    fitness REAL,
                    config_json TEXT,
                    timestamp REAL
                )
            """)
            conn.commit()

    def add(self, solution: QDSolution) -> bool:
        """
        Try to add a solution to the archive.

        Returns True if the solution was accepted (either filled an empty
        niche or improved an existing one).
        """
        key = niche_key(solution.behavior, self.resolution)
        solution.niche_key = key

        with self._connect() as conn:
            # Check existing
            cur = conn.execute(
                "SELECT fitness, visits FROM archive WHERE niche_key = ?",
                (key,),
            )
            row = cur.fetchone()

            if row is None:
                # Empty niche -- always accept
                conn.execute("""
                    INSERT INTO archive
                    (niche_key, config_json, fitness, behavior_json,
                     visits, last_updated, improvement_count)
                    VALUES (?, ?, ?, ?, 1, ?, 0)
                """, (
                    key, json.dumps(solution.config), solution.fitness,
                    json.dumps(solution.behavior), time.time(),
                ))
                self._log_history(conn, key, solution)
                conn.commit()
                return True

            existing_fitness = row[0]
            existing_visits = row[1]

            if solution.fitness > existing_fitness:
                # Better solution found for this niche
                conn.execute("""
                    UPDATE archive SET
                        config_json = ?,
                        fitness = ?,
                        behavior_json = ?,
                        visits = ?,
                        last_updated = ?,
                        improvement_count = improvement_count + 1
                    WHERE niche_key = ?
                """, (
                    json.dumps(solution.config), solution.fitness,
                    json.dumps(solution.behavior),
                    existing_visits + 1, time.time(), key,
                ))
                self._log_history(conn, key, solution)
                conn.commit()
                return True

            # Inferior solution -- just increment visits
            conn.execute(
                "UPDATE archive SET visits = visits + 1 WHERE niche_key = ?",
                (key,),
            )
            conn.commit()
            return False

    def lookup(self, behavior: Dict[str, float]) -> Optional[Dict[str, Any]]:
        """
        Find the best config for a given behavior descriptor.
        Returns the config dict or None if the niche is empty.
        """
        key = niche_key(behavior, self.resolution)

        with self._connect() as conn:
            cur = conn.execute(
                "SELECT config_json, fitness FROM archive WHERE niche_key = ?",
                (key,),
            )
            row = cur.fetchone()

        if row:
            return {
                "config": json.loads(row[0]),
                "fitness": row[1],
                "niche": key,
            }

        # Fallback: find nearest occupied niche
        return self._nearest_niche(behavior)

    def _nearest_niche(self, behavior: Dict[str, float]) -> Optional[Dict[str, Any]]:
        """Find the nearest occupied niche by behavior distance."""
        with self._connect() as conn:
            cur = conn.execute(
                "SELECT niche_key, config_json, fitness, behavior_json FROM archive"
            )
            rows = cur.fetchall()

        if not rows:
            return None

        best_dist = float("inf")
        best_row = None

        for row in rows:
            stored_behavior = json.loads(row[3])
            dist = sum(
                (behavior.get(k, 0.0) - stored_behavior.get(k, 0.0)) ** 2
                for k in behavior
            )
            if dist < best_dist:
                best_dist = dist
                best_row = row

        if best_row:
            return {
                "config": json.loads(best_row[1]),
                "fitness": best_row[2],
                "niche": best_row[0],
                "distance": round(math.sqrt(best_dist), 4),
            }
        return None

    def coverage(self) -> Dict[str, Any]:
        """Report archive coverage statistics."""
        max_niches = self.resolution ** 3  # 3 behavior dimensions
        with self._connect() as conn:
            cur = conn.execute("""
                SELECT COUNT(*), AVG(fitness), MAX(fitness),
                       SUM(visits), SUM(improvement_count)
                FROM archive
            """)
            row = cur.fetchone()

        if not row or row[0] == 0:
            return {
                "filled": 0,
                "max_niches": max_niches,
                "coverage_pct": 0.0,
            }

        return {
            "filled": row[0],
            "max_niches": max_niches,
            "coverage_pct": round(100.0 * row[0] / max_niches, 1),
            "avg_fitness": round(row[1], 3),
            "max_fitness": round(row[2], 3),
            "total_visits": row[3],
            "total_improvements": row[4],
        }

    def top_configs(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Return top-performing configs across all niches."""
        with self._connect() as conn:
            cur = conn.execute(
                "SELECT niche_key, config_json, fitness, behavior_json, "
                "visits, improvement_count FROM archive "
                "ORDER BY fitness DESC LIMIT ?",
                (limit,),
            )
            return [
                {
                    "niche": r[0],
                    "config": json.loads(r[1]),
                    "fitness": r[2],
                    "behavior": json.loads(r[3]),
                    "visits": r[4],
                    "improvements": r[5],
                }
                for r in cur.fetchall()
            ]

    def underexplored_niches(self, min_visits: int = 3) -> List[str]:
        """Find niches that haven't been explored enough."""
        with self._connect() as conn:
            cur = conn.execute(
                "SELECT niche_key FROM archive WHERE visits < ? "
                "ORDER BY visits ASC",
                (min_visits,),
            )
            return [r[0] for r in cur.fetchall()]

    def export_gallery(self) -> Dict[str, Any]:
        """Export the full archive as a gallery for visualization."""
        with self._connect() as conn:
            cur = conn.execute(
                "SELECT niche_key, config_json, fitness, behavior_json, "
                "visits, improvement_count FROM archive ORDER BY niche_key"
            )
            cells = [
                {
                    "niche": r[0],
                    "config": json.loads(r[1]),
                    "fitness": r[2],
                    "behavior": json.loads(r[3]),
                    "visits": r[4],
                    "improvements": r[5],
                }
                for r in cur.fetchall()
            ]

        return {
            "coverage": self.coverage(),
            "cells": cells,
            "exported_at": time.time(),
        }

    def _log_history(self, conn: sqlite3.Connection,
                      key: str, solution: QDSolution) -> None:
        conn.execute("""
            INSERT INTO history (niche_key, fitness, config_json, timestamp)
            VALUES (?, ?, ?, ?)
        """, (key, solution.fitness, json.dumps(solution.config), time.time()))
