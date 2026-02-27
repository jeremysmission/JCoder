"""
Active Learner (Query-by-Committee + Uncertainty Sampling)
------------------------------------------------------------
Decides WHAT the system should learn next by identifying the
queries where improvement effort will have the highest impact.

Based on:
- Query-by-Committee (Seung et al., 1992): Committee disagreement
- Expected Model Change (Cohn et al., 1996): Sample where gradients are largest
- Fisher Information Active Reward Modeling (2025): Focus on decision boundaries
- Mind the Gap (2024): Solver-Verifier gap drives self-improvement

Three selection strategies:

1. Uncertainty Sampling: Queries where the system is least confident
   (highest entropy across multiple generations)

2. Committee Disagreement: Queries where different retrieval/prompting
   strategies give the most divergent answers

3. Frontier Detection: Queries near the boundary between "system can
   answer" and "system cannot answer" (maximum learning signal)

These three signals are combined into a single "learning value" score.
The system should focus self-improvement on high-learning-value queries.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


@dataclass
class LearningCandidate:
    """A query scored for learning value."""
    query: str
    uncertainty: float  # 0.0-1.0 (higher = more uncertain)
    disagreement: float  # 0.0-1.0 (higher = more disagreement)
    frontier_score: float  # 0.0-1.0 (higher = closer to capability boundary)
    learning_value: float  # combined score
    answers: List[str] = field(default_factory=list)
    strategy_answers: Dict[str, str] = field(default_factory=dict)


class ActiveLearner:
    """
    Identifies the highest-value queries for self-improvement.

    Uses three complementary signals to find queries where learning
    effort will have maximum impact. Results feed into:
    - Prompt Evolver (focus evolution on hard queries)
    - Experience Replay (prioritize storing borderline successes)
    - Self-Bench Generator (generate more questions like hard ones)
    - Adversarial Self-Play (calibrate difficulty targeting)
    """

    def __init__(
        self,
        generate_fn: Callable[[str, float], str],
        db_path: str = "_active_learn/learner.db",
        n_samples: int = 5,
        n_strategies: int = 3,
    ):
        """
        Args:
            generate_fn: Function(query, temperature) -> answer_text
                         Used for uncertainty sampling.
            db_path: SQLite path for candidate tracking
            n_samples: Number of samples for uncertainty estimation
            n_strategies: Number of strategies for committee voting
        """
        self.generate_fn = generate_fn
        self.n_samples = n_samples
        self.n_strategies = n_strategies
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        self._strategy_fns: List[Callable[[str], str]] = []

    def register_strategy(self, fn: Callable[[str], str]) -> None:
        """Register a strategy function for committee disagreement."""
        self._strategy_fns.append(fn)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self._db_path))

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS candidates (
                    query_hash TEXT PRIMARY KEY,
                    query TEXT NOT NULL,
                    uncertainty REAL,
                    disagreement REAL,
                    frontier_score REAL,
                    learning_value REAL,
                    answers_json TEXT,
                    is_resolved INTEGER DEFAULT 0,
                    scored_at REAL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_candidates_value
                ON candidates(learning_value DESC)
            """)
            conn.commit()

    def score_queries(
        self,
        queries: List[str],
        confidence_scores: Optional[List[float]] = None,
    ) -> List[LearningCandidate]:
        """
        Score a batch of queries for learning value.

        Args:
            queries: Queries to evaluate
            confidence_scores: Optional existing confidence scores
                               (e.g., from reflection engine)
        Returns:
            List of LearningCandidate sorted by learning_value (desc)
        """
        candidates = []
        conf_scores = confidence_scores or [0.5] * len(queries)

        for i, query in enumerate(queries):
            # 1. Uncertainty sampling
            uncertainty, answers = self._estimate_uncertainty(query)

            # 2. Committee disagreement
            disagreement, strategy_answers = self._committee_vote(query)

            # 3. Frontier detection
            frontier = self._frontier_score(conf_scores[i])

            # Combined learning value
            learning_value = (
                0.35 * uncertainty +
                0.35 * disagreement +
                0.30 * frontier
            )

            cand = LearningCandidate(
                query=query,
                uncertainty=uncertainty,
                disagreement=disagreement,
                frontier_score=frontier,
                learning_value=learning_value,
                answers=answers,
                strategy_answers=strategy_answers,
            )
            candidates.append(cand)
            self._persist_candidate(cand)

        candidates.sort(key=lambda c: c.learning_value, reverse=True)
        return candidates

    def top_learning_opportunities(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Return the highest-value unresolved learning candidates."""
        with self._connect() as conn:
            cur = conn.execute(
                "SELECT query, uncertainty, disagreement, frontier_score, "
                "learning_value FROM candidates WHERE is_resolved = 0 "
                "ORDER BY learning_value DESC LIMIT ?",
                (limit,),
            )
            return [
                {
                    "query": r[0],
                    "uncertainty": round(r[1], 3),
                    "disagreement": round(r[2], 3),
                    "frontier": round(r[3], 3),
                    "learning_value": round(r[4], 3),
                }
                for r in cur.fetchall()
            ]

    def mark_resolved(self, query: str) -> None:
        """Mark a query as resolved (learned from)."""
        qhash = hashlib.sha256(query.encode()).hexdigest()[:16]
        with self._connect() as conn:
            conn.execute(
                "UPDATE candidates SET is_resolved = 1 WHERE query_hash = ?",
                (qhash,),
            )
            conn.commit()

    def _estimate_uncertainty(self, query: str) -> Tuple[float, List[str]]:
        """
        Estimate uncertainty by generating multiple answers at varying
        temperatures and measuring entropy of the answer distribution.
        """
        answers = []
        temps = [0.1, 0.3, 0.5, 0.7, 0.9][:self.n_samples]

        for temp in temps:
            try:
                answer = self.generate_fn(query, temp)
                answers.append(answer)
            except Exception:
                answers.append("")

        if not answers:
            return 1.0, answers

        # Compute entropy of answer distribution
        # Normalize answers to detect meaningful variation
        normalized = [self._normalize_answer(a) for a in answers]
        entropy = self._shannon_entropy(normalized)

        return entropy, answers

    def _committee_vote(self, query: str) -> Tuple[float, Dict[str, str]]:
        """
        Run query through multiple strategies and measure disagreement.
        """
        if not self._strategy_fns:
            return 0.5, {}

        strategy_answers = {}
        for i, fn in enumerate(self._strategy_fns[:self.n_strategies]):
            try:
                answer = fn(query)
                strategy_answers[f"strategy_{i}"] = answer
            except Exception:
                strategy_answers[f"strategy_{i}"] = ""

        if len(strategy_answers) < 2:
            return 0.5, strategy_answers

        # Measure disagreement
        answers = list(strategy_answers.values())
        normalized = [self._normalize_answer(a) for a in answers]
        disagreement = self._shannon_entropy(normalized)

        return disagreement, strategy_answers

    @staticmethod
    def _frontier_score(confidence: float) -> float:
        """
        Score how close a query is to the capability frontier.
        Maximum at confidence ~0.5 (boundary between can/can't answer).
        Uses a bell curve: 4 * p * (1-p), peaks at p=0.5.
        """
        return 4.0 * confidence * (1.0 - confidence)

    @staticmethod
    def _normalize_answer(answer: str) -> str:
        """Normalize for comparison: lowercase, strip whitespace, first 200 chars."""
        return answer.lower().strip()[:200]

    @staticmethod
    def _shannon_entropy(items: List[str]) -> float:
        """Shannon entropy normalized to [0, 1]."""
        if not items:
            return 0.0
        counts = Counter(items)
        total = len(items)
        n_unique = len(counts)
        if n_unique <= 1:
            return 0.0
        entropy = 0.0
        for count in counts.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)
        # Normalize by max possible entropy
        max_entropy = math.log2(n_unique)
        return entropy / max_entropy if max_entropy > 0 else 0.0

    def _persist_candidate(self, cand: LearningCandidate) -> None:
        qhash = hashlib.sha256(cand.query.encode()).hexdigest()[:16]
        try:
            with self._connect() as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO candidates
                    (query_hash, query, uncertainty, disagreement,
                     frontier_score, learning_value, answers_json, scored_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    qhash, cand.query[:500], cand.uncertainty,
                    cand.disagreement, cand.frontier_score,
                    cand.learning_value,
                    json.dumps(cand.answers[:3]),
                    time.time(),
                ))
                conn.commit()
        except Exception:
            pass

    def stats(self) -> Dict[str, Any]:
        with self._connect() as conn:
            total = conn.execute(
                "SELECT COUNT(*) FROM candidates"
            ).fetchone()[0]
            unresolved = conn.execute(
                "SELECT COUNT(*) FROM candidates WHERE is_resolved = 0"
            ).fetchone()[0]
            avg_value = conn.execute(
                "SELECT AVG(learning_value) FROM candidates"
            ).fetchone()[0]
        return {
            "total_scored": total,
            "unresolved": unresolved,
            "avg_learning_value": round(avg_value or 0, 3),
        }
