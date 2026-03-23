"""
Continual Learning Controller (Anti-Forgetting)
--------------------------------------------------
Prevents catastrophic forgetting during self-improvement by tracking
capability baselines and rolling back changes that cause regression.

Based on:
- EWC (Elastic Weight Consolidation, 2017): Protect important parameters
- Progressive Neural Networks (2016): Add capacity without forgetting
- PackNet (2018): Iterative pruning for multi-task learning
- CLEAR Benchmark (2022): Continual learning evaluation framework

Adapted for RAG systems (no weight updates -- config/prompt evolution):

1. Capability Baselines: Track performance on a fixed test set per capability
   (retrieval accuracy, refusal accuracy, latency, etc.)

2. Regression Detection: After any self-improvement step, re-run baselines.
   If ANY capability drops below threshold, roll back the change.

3. Capability Expansion: When a new capability is added (e.g., new query type),
   verify it doesn't degrade existing capabilities.

4. Memory Consolidation: Periodically merge experience replay entries,
   prune low-quality configs from the QD archive, and compact telemetry.

Safety: This module is the LAST LINE OF DEFENSE against self-improvement
loops that inadvertently break working functionality.
"""

from __future__ import annotations

import copy
import json
import logging
import sqlite3
import time

log = logging.getLogger(__name__)
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


@dataclass
class CapabilityBaseline:
    """Baseline performance for a single capability."""
    name: str  # e.g., "factual_accuracy", "refusal_rate", "retrieval_precision"
    score: float  # 0.0-1.0 baseline score
    threshold: float  # minimum acceptable score (usually baseline - margin)
    test_queries: List[str]  # fixed queries to evaluate this capability
    measured_at: float = 0.0


@dataclass
class RegressionCheck:
    """Result of checking for regression after a change."""
    passed: bool
    failed_capabilities: List[str]
    scores: Dict[str, float]  # capability -> current score
    baselines: Dict[str, float]  # capability -> baseline score
    checked_at: float = 0.0


@dataclass
class ConsolidationResult:
    """Result of a memory consolidation cycle."""
    experiences_pruned: int = 0
    configs_pruned: int = 0
    telemetry_compacted: int = 0
    duration_ms: float = 0.0


class ContinualLearner:
    """
    Guards against catastrophic forgetting during self-improvement.

    Usage:
        learner = ContinualLearner(eval_fn, db_path)
        learner.set_baseline("factual_accuracy", 0.92, queries)
        learner.set_baseline("refusal_rate", 0.98, queries)

        # Before applying a self-improvement:
        check = learner.check_regression(candidate_config)
        if check.passed:
            apply(candidate_config)
        else:
            rollback()
    """

    def __init__(
        self,
        eval_fn: Callable[[str, List[str]], float],
        db_path: str = "_continual/learner.db",
        regression_margin: float = 0.05,
    ):
        """
        Args:
            eval_fn: Function(capability_name, test_queries) -> score (0.0-1.0)
                     Evaluates a capability on its test queries under current config.
            db_path: SQLite path for baseline tracking
            regression_margin: How much decline is acceptable (0.05 = 5%)
        """
        self.eval_fn = eval_fn
        self.margin = regression_margin
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self._db_path))

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS baselines (
                    name TEXT PRIMARY KEY,
                    score REAL NOT NULL,
                    threshold REAL NOT NULL,
                    test_queries_json TEXT NOT NULL,
                    measured_at REAL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS regression_checks (
                    check_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    passed INTEGER,
                    failed_capabilities_json TEXT,
                    scores_json TEXT,
                    baselines_json TEXT,
                    checked_at REAL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS config_snapshots (
                    snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    config_json TEXT,
                    reason TEXT,
                    created_at REAL
                )
            """)
            conn.commit()

    def set_baseline(
        self,
        name: str,
        score: float,
        test_queries: List[str],
        threshold: Optional[float] = None,
    ) -> CapabilityBaseline:
        """
        Register or update a capability baseline.

        Args:
            name: Capability name (e.g., "factual_accuracy")
            score: Current baseline score (0.0-1.0)
            test_queries: Fixed queries for evaluating this capability
            threshold: Minimum acceptable score. Defaults to score - margin.
        """
        if threshold is None:
            threshold = max(0.0, score - self.margin)

        baseline = CapabilityBaseline(
            name=name,
            score=score,
            threshold=threshold,
            test_queries=test_queries,
            measured_at=time.time(),
        )

        with self._connect() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO baselines
                (name, score, threshold, test_queries_json, measured_at)
                VALUES (?, ?, ?, ?, ?)
            """, (
                name, score, threshold,
                json.dumps(test_queries), time.time(),
            ))
            conn.commit()

        return baseline

    def get_baselines(self) -> List[CapabilityBaseline]:
        """Return all registered capability baselines."""
        with self._connect() as conn:
            cur = conn.execute("SELECT * FROM baselines ORDER BY name")
            return [
                CapabilityBaseline(
                    name=r[0], score=r[1], threshold=r[2],
                    test_queries=json.loads(r[3]),
                    measured_at=r[4] or 0.0,
                )
                for r in cur.fetchall()
            ]

    def check_regression(
        self,
        config: Optional[Dict[str, Any]] = None,
    ) -> RegressionCheck:
        """
        Check all capability baselines for regression.

        If config is provided, snapshot it for potential rollback.
        Returns RegressionCheck with pass/fail and details.
        """
        baselines = self.get_baselines()
        if not baselines:
            return RegressionCheck(
                passed=True,
                failed_capabilities=[],
                scores={}, baselines={},
                checked_at=time.time(),
            )

        # Snapshot config for rollback
        if config:
            self._snapshot_config(config, "pre_regression_check")

        failed = []
        scores = {}
        baseline_scores = {}

        for bl in baselines:
            try:
                current_score = self.eval_fn(bl.name, bl.test_queries)
            except Exception as exc:
                log.debug("Baseline eval failed: %s", exc)
                current_score = 0.0

            scores[bl.name] = round(current_score, 4)
            baseline_scores[bl.name] = bl.score

            if current_score < bl.threshold:
                failed.append(bl.name)

        passed = len(failed) == 0
        check = RegressionCheck(
            passed=passed,
            failed_capabilities=failed,
            scores=scores,
            baselines=baseline_scores,
            checked_at=time.time(),
        )

        # Log check
        self._log_check(check)
        return check

    def update_baselines(self) -> Dict[str, float]:
        """
        Re-measure all baselines using current system state.
        Only RAISES baselines, never lowers them (ratchet up).
        """
        baselines = self.get_baselines()
        updates = {}

        for bl in baselines:
            try:
                current = self.eval_fn(bl.name, bl.test_queries)
            except Exception as exc:
                log.debug("Expansion eval failed: %s", exc)
                continue

            if current > bl.score:
                # Ratchet up: new baseline is higher
                self.set_baseline(
                    bl.name, current, bl.test_queries,
                    threshold=max(bl.threshold, current - self.margin),
                )
                updates[bl.name] = current

        return updates

    def rollback_to_last(self) -> Optional[Dict[str, Any]]:
        """Return the most recent config snapshot for rollback."""
        with self._connect() as conn:
            cur = conn.execute(
                "SELECT config_json FROM config_snapshots "
                "ORDER BY created_at DESC LIMIT 1"
            )
            row = cur.fetchone()
        if row:
            return json.loads(row[0])
        return None

    def consolidate(
        self,
        experience_prune_fn: Optional[Callable[[], int]] = None,
        config_prune_fn: Optional[Callable[[], int]] = None,
        telemetry_compact_fn: Optional[Callable[[], int]] = None,
    ) -> ConsolidationResult:
        """
        Run memory consolidation cycle.

        Each prune/compact function is optional. They should:
        - Remove low-quality entries
        - Merge redundant data
        - Return count of items affected
        """
        t0 = time.time()
        result = ConsolidationResult()

        if experience_prune_fn:
            try:
                result.experiences_pruned = experience_prune_fn()
            except Exception as exc:
                log.debug("Consolidation step failed: %s", exc)

        if config_prune_fn:
            try:
                result.configs_pruned = config_prune_fn()
            except Exception as exc:
                log.debug("Consolidation step failed: %s", exc)

        if telemetry_compact_fn:
            try:
                result.telemetry_compacted = telemetry_compact_fn()
            except Exception as exc:
                log.debug("Consolidation step failed: %s", exc)

        result.duration_ms = (time.time() - t0) * 1000
        return result

    def health_report(self) -> Dict[str, Any]:
        """Generate a health report of the continual learning system."""
        baselines = self.get_baselines()

        with self._connect() as conn:
            total_checks = (conn.execute(
                "SELECT COUNT(*) FROM regression_checks"
            ).fetchone() or (0,))[0]
            failed_checks = (conn.execute(
                "SELECT COUNT(*) FROM regression_checks WHERE passed = 0"
            ).fetchone() or (0,))[0]
            snapshots = (conn.execute(
                "SELECT COUNT(*) FROM config_snapshots"
            ).fetchone() or (0,))[0]

        return {
            "capabilities_tracked": len(baselines),
            "baselines": {
                bl.name: {
                    "score": bl.score,
                    "threshold": bl.threshold,
                    "test_queries": len(bl.test_queries),
                }
                for bl in baselines
            },
            "total_regression_checks": total_checks,
            "failed_checks": failed_checks,
            "failure_rate": (
                round(failed_checks / total_checks, 3)
                if total_checks > 0 else 0.0
            ),
            "config_snapshots": snapshots,
        }

    def _snapshot_config(self, config: Dict[str, Any], reason: str) -> None:
        try:
            with self._connect() as conn:
                conn.execute("""
                    INSERT INTO config_snapshots (config_json, reason, created_at)
                    VALUES (?, ?, ?)
                """, (json.dumps(config, default=str), reason, time.time()))
                conn.commit()
        except Exception as exc:
            log.debug("Continual learner: %s", exc)

    def _log_check(self, check: RegressionCheck) -> None:
        try:
            with self._connect() as conn:
                conn.execute("""
                    INSERT INTO regression_checks
                    (passed, failed_capabilities_json, scores_json,
                     baselines_json, checked_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    1 if check.passed else 0,
                    json.dumps(check.failed_capabilities),
                    json.dumps(check.scores),
                    json.dumps(check.baselines),
                    check.checked_at,
                ))
                conn.commit()
        except Exception as exc:
            log.debug("Continual learner: %s", exc)
