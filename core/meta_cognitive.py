"""
Meta-Cognitive Controller (Learning to Learn)
-----------------------------------------------
The "brain of the brain" -- decides WHICH self-learning strategy to
apply for each query. Learns from telemetry which strategies work
best for which types of queries.

Based on:
- MAML++ / Reptile: Learn good initialization for fast adaptation
- Cascading Bandits: Multi-armed bandit for strategy selection
- Thompson Sampling: Bayesian exploration of strategy effectiveness

This is the key differentiator: most RAG systems use a fixed pipeline.
Meta-cognitive control means the pipeline adapts per-query based on
what has historically worked for similar queries.

Strategies it can select:
1. Standard (fast, direct retrieval + generation)
2. Corrective (CRAG with reformulation)
3. Best-of-N (generate multiple, verify, pick best)
4. Cascade (route to different model tiers by complexity)
5. Reflective (generate + self-assess + regenerate if poor)

Selection uses Thompson Sampling with Beta distributions:
- Each (strategy, query_type) pair has a Beta(alpha, beta) prior
- Alpha incremented on success, Beta on failure
- Thompson sampling explores underexplored strategies automatically
"""

from __future__ import annotations

import json
import math
import random
import re
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Query type classification
# ---------------------------------------------------------------------------

@dataclass
class QuerySignature:
    """Fingerprint of a query for strategy matching."""
    query_type: str  # "lookup" | "reasoning" | "design" | "debug" | "explain"
    complexity: float  # 0.0-1.0
    length_bucket: str  # "short" | "medium" | "long"
    has_code: bool
    multi_part: bool


def classify_query(query: str) -> QuerySignature:
    """Classify a query into a type signature for strategy matching."""
    q_lower = query.lower()
    words = q_lower.split()
    n_words = len(words)

    # Query type
    if any(w in q_lower for w in ["where is", "find", "show me", "what is", "list"]):
        qtype = "lookup"
    elif any(w in q_lower for w in ["why", "how does", "explain", "what happens"]):
        qtype = "explain"
    elif any(w in q_lower for w in ["fix", "bug", "error", "wrong", "broken", "crash"]):
        qtype = "debug"
    elif any(w in q_lower for w in ["design", "architect", "implement", "create", "build"]):
        qtype = "design"
    elif any(w in q_lower for w in ["compare", "versus", "trade", "should i", "better"]):
        qtype = "reasoning"
    else:
        qtype = "explain"

    # Complexity
    complexity = min(1.0, max(0.0, (n_words - 5) / 40.0))
    complex_words = sum(1 for w in ["refactor", "optimize", "concurrent", "distributed",
                                     "migrate", "security"] if w in q_lower)
    complexity = min(1.0, complexity + complex_words * 0.15)

    # Length bucket
    if n_words <= 8:
        length = "short"
    elif n_words <= 25:
        length = "medium"
    else:
        length = "long"

    # Code detection
    has_code = bool(re.search(r"[a-zA-Z_]\w*\.\w+|def\s+\w+|class\s+\w+|```", query))

    # Multi-part
    multi_part = bool(re.search(r"\band\b|\balso\b|,.*\?", q_lower))

    return QuerySignature(
        query_type=qtype,
        complexity=complexity,
        length_bucket=length,
        has_code=has_code,
        multi_part=multi_part,
    )


# ---------------------------------------------------------------------------
# Strategy arm (Thompson Sampling)
# ---------------------------------------------------------------------------

@dataclass
class StrategyArm:
    """A single strategy option with Beta distribution prior."""
    name: str
    alpha: float = 1.0  # successes + prior
    beta: float = 1.0   # failures + prior
    total_uses: int = 0
    total_reward: float = 0.0
    avg_latency_ms: float = 0.0

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    def sample(self, rng: random.Random) -> float:
        """Thompson sampling: draw from Beta(alpha, beta)."""
        return rng.betavariate(self.alpha, self.beta)

    def update(self, reward: float) -> None:
        """Update Beta distribution with observed reward (0.0-1.0)."""
        self.alpha += reward
        self.beta += (1.0 - reward)
        self.total_uses += 1
        self.total_reward += reward


# ---------------------------------------------------------------------------
# Meta-Cognitive Controller
# ---------------------------------------------------------------------------

STRATEGIES = ["standard", "corrective", "best_of_n", "cascade", "reflective"]


class MetaCognitiveController:
    """
    Learns which strategy to apply per query type using Thompson Sampling.

    The controller maintains Beta distributions for each (strategy, query_type)
    pair. Over time, it learns that e.g. "debug" queries work best with
    "corrective" retrieval, while "lookup" queries are fastest with "standard".
    """

    def __init__(
        self,
        db_path: str = "_meta_cog/controller.db",
        exploration_bonus: float = 0.1,
        seed: int = 42,
    ):
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self.exploration_bonus = exploration_bonus
        self.rng = random.Random(seed)
        self._arms: Dict[str, Dict[str, StrategyArm]] = {}
        self._init_db()
        self._load_arms()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self._db_path))

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS strategy_arms (
                    query_type TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    alpha REAL DEFAULT 1.0,
                    beta REAL DEFAULT 1.0,
                    total_uses INTEGER DEFAULT 0,
                    total_reward REAL DEFAULT 0.0,
                    avg_latency_ms REAL DEFAULT 0.0,
                    PRIMARY KEY (query_type, strategy)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS decisions (
                    decision_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    query_text TEXT,
                    query_type TEXT,
                    chosen_strategy TEXT,
                    reward REAL,
                    latency_ms REAL,
                    sampled_scores_json TEXT
                )
            """)
            conn.commit()

    def _load_arms(self) -> None:
        """Load persisted arm stats from SQLite."""
        with self._connect() as conn:
            cur = conn.execute("SELECT * FROM strategy_arms")
            for row in cur.fetchall():
                qtype, strategy = row[0], row[1]
                if qtype not in self._arms:
                    self._arms[qtype] = {}
                self._arms[qtype][strategy] = StrategyArm(
                    name=strategy,
                    alpha=row[2], beta=row[3],
                    total_uses=row[4], total_reward=row[5],
                    avg_latency_ms=row[6],
                )

    def _get_arm(self, query_type: str, strategy: str) -> StrategyArm:
        """Get or create an arm for a (query_type, strategy) pair."""
        if query_type not in self._arms:
            self._arms[query_type] = {}
        if strategy not in self._arms[query_type]:
            self._arms[query_type][strategy] = StrategyArm(name=strategy)
        return self._arms[query_type][strategy]

    def select_strategy(self, query: str) -> Tuple[str, QuerySignature]:
        """
        Select the best strategy for a query using Thompson Sampling.

        Returns (strategy_name, query_signature).
        """
        sig = classify_query(query)

        # Thompson sample from each strategy's arm
        samples: Dict[str, float] = {}
        for strategy in STRATEGIES:
            arm = self._get_arm(sig.query_type, strategy)
            ts_score = arm.sample(self.rng)

            # Exploration bonus for underexplored strategies
            if arm.total_uses < 5:
                ts_score += self.exploration_bonus

            samples[strategy] = ts_score

        # Select highest Thompson sample
        best_strategy = max(samples, key=lambda s: samples[s])

        # Log the decision
        self._log_decision(query, sig, best_strategy, samples)

        return best_strategy, sig

    def report_outcome(
        self,
        query: str,
        strategy: str,
        reward: float,
        latency_ms: float = 0.0,
    ) -> None:
        """
        Report the outcome of a strategy execution.

        Args:
            query: The original query
            strategy: Which strategy was used
            reward: 0.0-1.0 quality score (from reflection or feedback)
            latency_ms: How long the strategy took
        """
        sig = classify_query(query)
        arm = self._get_arm(sig.query_type, strategy)
        arm.update(reward)

        # Update running average latency
        if arm.total_uses > 0:
            arm.avg_latency_ms = (
                (arm.avg_latency_ms * (arm.total_uses - 1) + latency_ms)
                / arm.total_uses
            )

        self._persist_arm(sig.query_type, arm)

    def strategy_report(self) -> Dict[str, Any]:
        """Generate a human-readable report of learned strategy preferences."""
        report = {}
        for qtype, arms in self._arms.items():
            report[qtype] = {}
            for strategy, arm in arms.items():
                report[qtype][strategy] = {
                    "mean_reward": round(arm.mean, 3),
                    "total_uses": arm.total_uses,
                    "avg_latency_ms": round(arm.avg_latency_ms, 1),
                    "confidence": round(
                        1.0 - 1.0 / (1.0 + arm.alpha + arm.beta), 3),
                }
        return report

    def best_strategy_per_type(self) -> Dict[str, str]:
        """Return the current best strategy for each known query type."""
        result = {}
        for qtype, arms in self._arms.items():
            if arms:
                best = max(arms.values(), key=lambda a: a.mean)
                result[qtype] = best.name
        return result

    def _log_decision(
        self,
        query: str,
        sig: QuerySignature,
        chosen: str,
        samples: Dict[str, float],
    ) -> None:
        try:
            with self._connect() as conn:
                conn.execute("""
                    INSERT INTO decisions
                    (timestamp, query_text, query_type, chosen_strategy,
                     sampled_scores_json)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    time.time(), query[:500], sig.query_type, chosen,
                    json.dumps({k: round(v, 4) for k, v in samples.items()}),
                ))
                conn.commit()
        except Exception:
            pass

    def _persist_arm(self, query_type: str, arm: StrategyArm) -> None:
        try:
            with self._connect() as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO strategy_arms
                    (query_type, strategy, alpha, beta, total_uses,
                     total_reward, avg_latency_ms)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    query_type, arm.name, arm.alpha, arm.beta,
                    arm.total_uses, arm.total_reward, arm.avg_latency_ms,
                ))
                conn.commit()
        except Exception:
            pass
