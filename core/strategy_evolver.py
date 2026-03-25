"""
Retrieval Strategy Evolver
--------------------------
Learns which retrieval strategy works best for each question type.

Instead of using the same hybrid search for every query, this module
maintains a population of retrieval strategies and evolves them based
on actual performance feedback.

Strategies encode:
  - Which FTS5 databases to prioritize
  - How many results to fetch from dense vs sparse
  - Whether to boost domain-specific terms
  - Fusion method (RRF vs DBSF)
  - Reranking aggressiveness

The evolver:
  1. Classifies incoming queries by type (code, concept, debug, etc.)
  2. Looks up the best-performing strategy for that type
  3. Executes retrieval with that strategy
  4. Records the outcome for future evolution

This is Darwinian retrieval — strategies that produce good answers
reproduce, strategies that fail die off.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class RetrievalStrategy:
    """A single retrieval strategy configuration."""
    strategy_id: str
    query_type: str  # "code", "concept", "debug", "system", "general"
    fts5_top_k: int = 10
    faiss_top_k: int = 10
    fusion_method: str = "rrf"  # "rrf" or "dbsf"
    rrf_k: int = 60
    domain_boost: bool = True
    rerank_pool_multiplier: int = 3
    min_score_threshold: float = 0.0
    generation: int = 0
    parent_id: str = ""
    avg_score: float = 0.0
    use_count: int = 0
    win_count: int = 0

    @property
    def win_rate(self) -> float:
        return self.win_count / max(self.use_count, 1)


_QUERY_TYPE_KEYWORDS = {
    "code": {"implement", "write", "function", "class", "method", "code",
             "snippet", "example", "def", "return"},
    "concept": {"explain", "what", "how", "why", "difference", "between",
                "compare", "concept", "principle"},
    "debug": {"error", "bug", "fix", "crash", "exception", "traceback",
              "fail", "broken", "wrong", "issue"},
    "system": {"architecture", "design", "pattern", "system", "pipeline",
               "infrastructure", "deploy", "configure"},
}


def classify_query(query: str) -> str:
    """Classify a query into a type based on keyword analysis."""
    lower = query.lower()
    scores = {}
    for qtype, keywords in _QUERY_TYPE_KEYWORDS.items():
        scores[qtype] = sum(1 for kw in keywords if kw in lower)
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "general"


def _make_id(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:12]


class StrategyEvolver:
    """Evolves retrieval strategies based on performance feedback."""

    POOL_SIZE = 20  # strategies per query type
    MUTATION_RATE = 0.3
    TOURNAMENT_SIZE = 3

    def __init__(self, db_path: str = "data/strategy_evolver.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        self._ensure_seed_strategies()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path))

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS strategies (
                    strategy_id TEXT PRIMARY KEY,
                    query_type TEXT,
                    config TEXT,
                    generation INTEGER DEFAULT 0,
                    avg_score REAL DEFAULT 0.0,
                    use_count INTEGER DEFAULT 0,
                    win_count INTEGER DEFAULT 0,
                    created_at REAL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS outcomes (
                    outcome_id TEXT PRIMARY KEY,
                    strategy_id TEXT,
                    query_type TEXT,
                    query_hash TEXT,
                    score REAL,
                    timestamp REAL
                )
            """)

    def _ensure_seed_strategies(self) -> None:
        """Seed initial strategies for each query type."""
        with self._connect() as conn:
            count = conn.execute("SELECT COUNT(*) FROM strategies").fetchone()[0]
            if count > 0:
                return

            for qtype in ["code", "concept", "debug", "system", "general"]:
                # Seed 4 diverse strategies per type
                seeds = [
                    RetrievalStrategy(
                        strategy_id=_make_id(f"{qtype}_balanced"),
                        query_type=qtype, fts5_top_k=10, faiss_top_k=10,
                        fusion_method="rrf", domain_boost=True,
                    ),
                    RetrievalStrategy(
                        strategy_id=_make_id(f"{qtype}_dense_heavy"),
                        query_type=qtype, fts5_top_k=5, faiss_top_k=20,
                        fusion_method="dbsf", domain_boost=False,
                    ),
                    RetrievalStrategy(
                        strategy_id=_make_id(f"{qtype}_sparse_heavy"),
                        query_type=qtype, fts5_top_k=20, faiss_top_k=5,
                        fusion_method="rrf", domain_boost=True,
                    ),
                    RetrievalStrategy(
                        strategy_id=_make_id(f"{qtype}_aggressive_rerank"),
                        query_type=qtype, fts5_top_k=15, faiss_top_k=15,
                        fusion_method="rrf", rerank_pool_multiplier=5,
                    ),
                ]
                for s in seeds:
                    conn.execute(
                        "INSERT OR IGNORE INTO strategies "
                        "(strategy_id, query_type, config, generation, created_at) "
                        "VALUES (?, ?, ?, ?, ?)",
                        (s.strategy_id, s.query_type, json.dumps(asdict(s)),
                         0, time.time()),
                    )

    def select_strategy(self, query: str) -> RetrievalStrategy:
        """Select the best strategy for a query using tournament selection."""
        qtype = classify_query(query)

        with self._connect() as conn:
            rows = conn.execute(
                "SELECT config FROM strategies WHERE query_type = ? "
                "ORDER BY avg_score DESC LIMIT ?",
                (qtype, self.POOL_SIZE),
            ).fetchall()

        if not rows:
            return RetrievalStrategy(
                strategy_id="default", query_type=qtype,
            )

        # Tournament selection: pick best from random subset
        candidates = [json.loads(r[0]) for r in rows]
        tournament = random.sample(
            candidates, min(self.TOURNAMENT_SIZE, len(candidates)),
        )
        best = max(tournament, key=lambda c: c.get("avg_score", 0))
        return RetrievalStrategy(**{
            k: v for k, v in best.items()
            if k in RetrievalStrategy.__dataclass_fields__
        })

    def record_outcome(
        self, strategy_id: str, query: str, score: float,
    ) -> None:
        """Record the outcome of using a strategy."""
        qtype = classify_query(query)
        query_hash = _make_id(query)

        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO outcomes "
                "(outcome_id, strategy_id, query_type, query_hash, score, timestamp) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (_make_id(f"{strategy_id}_{query_hash}"),
                 strategy_id, qtype, query_hash, score, time.time()),
            )

            # Update strategy stats
            conn.execute(
                "UPDATE strategies SET "
                "use_count = use_count + 1, "
                "win_count = win_count + CASE WHEN ? >= 0.5 THEN 1 ELSE 0 END, "
                "avg_score = (avg_score * use_count + ?) / (use_count + 1) "
                "WHERE strategy_id = ?",
                (score, score, strategy_id),
            )

    def evolve(self) -> int:
        """Run one generation of evolution. Returns number of new strategies."""
        with self._connect() as conn:
            # Get all strategies grouped by type
            rows = conn.execute(
                "SELECT config FROM strategies ORDER BY avg_score DESC"
            ).fetchall()

        if len(rows) < 4:
            return 0

        strategies = [json.loads(r[0]) for r in rows]
        by_type = {}
        for s in strategies:
            by_type.setdefault(s["query_type"], []).append(s)

        new_count = 0
        for qtype, pool in by_type.items():
            if len(pool) < 2:
                continue

            # Kill bottom 25%
            keep = pool[:max(2, int(len(pool) * 0.75))]

            # Mutate top performers
            for parent in keep[:3]:
                child = dict(parent)
                child["strategy_id"] = _make_id(
                    f"{qtype}_{time.time()}_{random.random()}"
                )
                child["generation"] = parent.get("generation", 0) + 1
                child["parent_id"] = parent["strategy_id"]
                child["avg_score"] = 0.0
                child["use_count"] = 0
                child["win_count"] = 0

                # Random mutations
                if random.random() < self.MUTATION_RATE:
                    child["fts5_top_k"] = max(3, child["fts5_top_k"]
                                              + random.choice([-3, -1, 1, 3, 5]))
                if random.random() < self.MUTATION_RATE:
                    child["faiss_top_k"] = max(3, child["faiss_top_k"]
                                               + random.choice([-3, -1, 1, 3, 5]))
                if random.random() < self.MUTATION_RATE:
                    child["fusion_method"] = random.choice(["rrf", "dbsf"])
                if random.random() < self.MUTATION_RATE:
                    child["domain_boost"] = not child.get("domain_boost", True)
                if random.random() < self.MUTATION_RATE:
                    child["rrf_k"] = random.choice([30, 45, 60, 80, 100])

                with self._connect() as conn:
                    conn.execute(
                        "INSERT OR IGNORE INTO strategies "
                        "(strategy_id, query_type, config, generation, created_at) "
                        "VALUES (?, ?, ?, ?, ?)",
                        (child["strategy_id"], qtype, json.dumps(child),
                         child["generation"], time.time()),
                    )
                new_count += 1

        return new_count
