"""
Recursive Meta-Learning (Sprint 24)
-------------------------------------
The evolution system learns to optimize itself. Tracks which data
sources, mutation strategies, and configurations lead to accepted
evolutions, then auto-tunes the pipeline accordingly.

Levels:
  Level 1: Evolution optimizes agent configs (Sprint 21)
  Level 2: Meta-learning optimizes the evolution system itself
  Level 3: Meta-research optimizes the research pipeline

Architecture:
  1. SourceValueTracker  -- records which sources lead to accepted evolutions
  2. StrategyTracker     -- records which mutation strategies succeed
  3. MetaLearner         -- feedback loop from ledger to pipeline priorities
  4. AutoPrioritizer     -- amplifies high-value, deprioritizes low-value sources

Gate: Demonstrated improvement in evolution acceptance rate over 4 weeks.
"""

from __future__ import annotations

import json
import logging
import math
import time
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from core.sqlite_owner import SQLiteConnectionOwner

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class SourceRecord:
    """Tracks a data source's contribution to evolution outcomes."""
    source_id: str
    source_type: str  # "dataset", "paper", "code_repo", "docs", "se_site"
    name: str
    total_uses: int = 0
    accepted_uses: int = 0
    rejected_uses: int = 0
    error_uses: int = 0
    cumulative_improvement: float = 0.0
    last_used: float = 0.0
    priority: float = 1.0  # 0.0 = deprioritized, 1.0 = normal, 2.0 = amplified


@dataclass
class StrategyRecord:
    """Tracks a mutation strategy's success rate."""
    strategy_id: str
    name: str
    total_attempts: int = 0
    accepted: int = 0
    rejected: int = 0
    errors: int = 0
    avg_improvement: float = 0.0
    last_used: float = 0.0
    weight: float = 1.0  # selection weight for future use


@dataclass
class MetaLearningCycle:
    """Record of one meta-learning optimization pass."""
    cycle_id: str
    timestamp: float
    sources_evaluated: int = 0
    strategies_evaluated: int = 0
    sources_amplified: List[str] = field(default_factory=list)
    sources_deprioritized: List[str] = field(default_factory=list)
    strategies_boosted: List[str] = field(default_factory=list)
    strategies_penalized: List[str] = field(default_factory=list)
    acceptance_rate_before: float = 0.0
    acceptance_rate_after: float = 0.0


@dataclass
class WeeklySnapshot:
    """Weekly acceptance rate snapshot for trend analysis."""
    week_num: int
    timestamp: float
    total_evolutions: int = 0
    accepted: int = 0
    acceptance_rate: float = 0.0
    top_sources: List[str] = field(default_factory=list)
    top_strategies: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_META_SCHEMA = """
CREATE TABLE IF NOT EXISTS source_values (
    source_id TEXT PRIMARY KEY,
    source_type TEXT NOT NULL,
    name TEXT NOT NULL,
    total_uses INTEGER DEFAULT 0,
    accepted_uses INTEGER DEFAULT 0,
    rejected_uses INTEGER DEFAULT 0,
    error_uses INTEGER DEFAULT 0,
    cumulative_improvement REAL DEFAULT 0.0,
    last_used REAL DEFAULT 0.0,
    priority REAL DEFAULT 1.0
);

CREATE TABLE IF NOT EXISTS strategy_records (
    strategy_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    total_attempts INTEGER DEFAULT 0,
    accepted INTEGER DEFAULT 0,
    rejected INTEGER DEFAULT 0,
    errors INTEGER DEFAULT 0,
    avg_improvement REAL DEFAULT 0.0,
    last_used REAL DEFAULT 0.0,
    weight REAL DEFAULT 1.0
);

CREATE TABLE IF NOT EXISTS meta_cycles (
    cycle_id TEXT PRIMARY KEY,
    timestamp REAL NOT NULL,
    sources_evaluated INTEGER DEFAULT 0,
    strategies_evaluated INTEGER DEFAULT 0,
    amplified_json TEXT DEFAULT '[]',
    deprioritized_json TEXT DEFAULT '[]',
    boosted_json TEXT DEFAULT '[]',
    penalized_json TEXT DEFAULT '[]',
    acceptance_rate_before REAL DEFAULT 0,
    acceptance_rate_after REAL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS weekly_snapshots (
    week_num INTEGER PRIMARY KEY,
    timestamp REAL NOT NULL,
    total_evolutions INTEGER DEFAULT 0,
    accepted INTEGER DEFAULT 0,
    acceptance_rate REAL DEFAULT 0.0,
    top_sources_json TEXT DEFAULT '[]',
    top_strategies_json TEXT DEFAULT '[]'
);
"""


# ---------------------------------------------------------------------------
# Source Value Tracker
# ---------------------------------------------------------------------------

class SourceValueTracker:
    """Tracks which data sources contribute to evolution success."""

    def __init__(self, db_path: str | Path):
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._owner = SQLiteConnectionOwner(self._db_path)
        conn = self._owner.connect()
        conn.executescript(_META_SCHEMA)
        conn.commit()

    @property
    def _conn(self):
        return self._owner.connect()

    def register_source(self, source_id: str, source_type: str, name: str) -> None:
        """Register a new data source."""
        conn = self._conn
        conn.execute(
            "INSERT OR IGNORE INTO source_values "
            "(source_id, source_type, name, last_used) VALUES (?, ?, ?, ?)",
            (source_id, source_type, name, time.time()),
        )
        conn.commit()

    def record_outcome(
        self,
        source_id: str,
        decision: str,
        improvement: float = 0.0,
    ) -> None:
        """Record an evolution outcome attributed to a source."""
        conn = self._conn
        now = time.time()

        if decision == "accepted":
            conn.execute(
                "UPDATE source_values SET total_uses = total_uses + 1, "
                "accepted_uses = accepted_uses + 1, "
                "cumulative_improvement = cumulative_improvement + ?, "
                "last_used = ? WHERE source_id = ?",
                (improvement, now, source_id),
            )
        elif decision == "rejected":
            conn.execute(
                "UPDATE source_values SET total_uses = total_uses + 1, "
                "rejected_uses = rejected_uses + 1, "
                "last_used = ? WHERE source_id = ?",
                (now, source_id),
            )
        elif decision == "error":
            conn.execute(
                "UPDATE source_values SET total_uses = total_uses + 1, "
                "error_uses = error_uses + 1, "
                "last_used = ? WHERE source_id = ?",
                (now, source_id),
            )
        conn.commit()

    def set_priority(self, source_id: str, priority: float) -> None:
        """Set source priority (0=deprioritized, 1=normal, 2=amplified)."""
        priority = max(0.0, min(2.0, priority))
        conn = self._conn
        conn.execute(
            "UPDATE source_values SET priority = ? WHERE source_id = ?",
            (priority, source_id),
        )
        conn.commit()

    def get_sources(self, min_uses: int = 0) -> List[SourceRecord]:
        """Get all tracked sources."""
        rows = self._conn.execute(
            "SELECT * FROM source_values WHERE total_uses >= ? "
            "ORDER BY priority DESC, accepted_uses DESC LIMIT 500",
            (min_uses,),
        ).fetchall()
        return [self._row_to_source(r) for r in rows]

    def get_top_sources(self, limit: int = 10) -> List[SourceRecord]:
        """Get sources ranked by acceptance rate."""
        rows = self._conn.execute(
            "SELECT * FROM source_values WHERE total_uses >= 3 "
            "ORDER BY (CAST(accepted_uses AS REAL) / MAX(total_uses, 1)) DESC "
            "LIMIT ?",
            (min(limit, 100),),
        ).fetchall()
        return [self._row_to_source(r) for r in rows]

    def get_low_value_sources(self, max_rate: float = 0.2) -> List[SourceRecord]:
        """Get sources with low acceptance rates."""
        rows = self._conn.execute(
            "SELECT * FROM source_values WHERE total_uses >= 3 "
            "AND (CAST(accepted_uses AS REAL) / MAX(total_uses, 1)) <= ? "
            "ORDER BY total_uses DESC LIMIT 100",
            (max_rate,),
        ).fetchall()
        return [self._row_to_source(r) for r in rows]

    def close(self) -> None:
        self._owner.close()

    @staticmethod
    def _row_to_source(row: tuple) -> SourceRecord:
        return SourceRecord(
            source_id=row[0], source_type=row[1], name=row[2],
            total_uses=row[3], accepted_uses=row[4],
            rejected_uses=row[5], error_uses=row[6],
            cumulative_improvement=row[7], last_used=row[8],
            priority=row[9],
        )


# ---------------------------------------------------------------------------
# Strategy Tracker
# ---------------------------------------------------------------------------

class StrategyTracker:
    """Tracks mutation strategy effectiveness."""

    def __init__(self, db_path: str | Path):
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._owner = SQLiteConnectionOwner(self._db_path)
        conn = self._owner.connect()
        conn.executescript(_META_SCHEMA)
        conn.commit()

    @property
    def _conn(self):
        return self._owner.connect()

    def register_strategy(self, strategy_id: str, name: str) -> None:
        """Register a mutation strategy."""
        conn = self._conn
        conn.execute(
            "INSERT OR IGNORE INTO strategy_records "
            "(strategy_id, name, last_used) VALUES (?, ?, ?)",
            (strategy_id, name, time.time()),
        )
        conn.commit()

    def record_outcome(
        self,
        strategy_id: str,
        decision: str,
        improvement: float = 0.0,
    ) -> None:
        """Record a strategy outcome."""
        conn = self._conn
        now = time.time()

        if decision == "accepted":
            # Update running average improvement
            row = conn.execute(
                "SELECT total_attempts, avg_improvement FROM strategy_records "
                "WHERE strategy_id = ?", (strategy_id,),
            ).fetchone()
            if row:
                n = row[0]
                old_avg = row[1]
                new_avg = (old_avg * n + improvement) / (n + 1) if n > 0 else improvement
            else:
                new_avg = improvement

            conn.execute(
                "UPDATE strategy_records SET total_attempts = total_attempts + 1, "
                "accepted = accepted + 1, avg_improvement = ?, "
                "last_used = ? WHERE strategy_id = ?",
                (new_avg, now, strategy_id),
            )
        elif decision == "rejected":
            conn.execute(
                "UPDATE strategy_records SET total_attempts = total_attempts + 1, "
                "rejected = rejected + 1, last_used = ? WHERE strategy_id = ?",
                (now, strategy_id),
            )
        elif decision == "error":
            conn.execute(
                "UPDATE strategy_records SET total_attempts = total_attempts + 1, "
                "errors = errors + 1, last_used = ? WHERE strategy_id = ?",
                (now, strategy_id),
            )
        conn.commit()

    def set_weight(self, strategy_id: str, weight: float) -> None:
        """Set strategy selection weight."""
        weight = max(0.1, min(5.0, weight))
        conn = self._conn
        conn.execute(
            "UPDATE strategy_records SET weight = ? WHERE strategy_id = ?",
            (weight, strategy_id),
        )
        conn.commit()

    def get_strategies(self) -> List[StrategyRecord]:
        """Get all tracked strategies."""
        rows = self._conn.execute(
            "SELECT * FROM strategy_records ORDER BY weight DESC LIMIT 200",
        ).fetchall()
        return [self._row_to_strategy(r) for r in rows]

    def select_strategy(self) -> Optional[StrategyRecord]:
        """Select a strategy weighted by success rate."""
        strategies = self.get_strategies()
        if not strategies:
            return None

        total_weight = sum(s.weight for s in strategies)
        if total_weight <= 0:
            return strategies[0]

        # Weighted random selection (deterministic for testing via seed)
        import random
        r = random.random() * total_weight
        cumulative = 0.0
        for s in strategies:
            cumulative += s.weight
            if r <= cumulative:
                return s
        return strategies[-1]

    def close(self) -> None:
        self._owner.close()

    @staticmethod
    def _row_to_strategy(row: tuple) -> StrategyRecord:
        return StrategyRecord(
            strategy_id=row[0], name=row[1],
            total_attempts=row[2], accepted=row[3],
            rejected=row[4], errors=row[5],
            avg_improvement=row[6], last_used=row[7],
            weight=row[8],
        )


# ---------------------------------------------------------------------------
# Auto-Prioritizer
# ---------------------------------------------------------------------------

class AutoPrioritizer:
    """Automatically adjusts source priorities and strategy weights.

    Rules:
    - Sources with acceptance rate > 0.5 get amplified (priority 1.5-2.0)
    - Sources with acceptance rate < 0.2 get deprioritized (priority 0.3-0.5)
    - Strategies with acceptance rate > 0.4 get boosted (weight 1.5-3.0)
    - Strategies with acceptance rate < 0.1 get penalized (weight 0.3-0.5)
    """

    AMPLIFY_THRESHOLD = 0.5
    DEPRIORITIZE_THRESHOLD = 0.2
    BOOST_THRESHOLD = 0.4
    PENALIZE_THRESHOLD = 0.1
    MIN_SAMPLES = 3

    def __init__(
        self,
        source_tracker: SourceValueTracker,
        strategy_tracker: StrategyTracker,
    ):
        self._sources = source_tracker
        self._strategies = strategy_tracker

    def run(self) -> MetaLearningCycle:
        """Run one auto-prioritization pass."""
        cycle = MetaLearningCycle(
            cycle_id=f"meta_{uuid.uuid4().hex[:8]}",
            timestamp=time.time(),
        )

        # Prioritize sources
        sources = self._sources.get_sources(min_uses=self.MIN_SAMPLES)
        cycle.sources_evaluated = len(sources)

        for src in sources:
            rate = src.accepted_uses / max(src.total_uses, 1)
            if rate >= self.AMPLIFY_THRESHOLD:
                new_priority = 1.0 + rate  # 1.5 - 2.0
                self._sources.set_priority(src.source_id, new_priority)
                cycle.sources_amplified.append(src.source_id)
            elif rate <= self.DEPRIORITIZE_THRESHOLD:
                new_priority = max(0.3, rate + 0.1)
                self._sources.set_priority(src.source_id, new_priority)
                cycle.sources_deprioritized.append(src.source_id)

        # Adjust strategy weights
        strategies = self._strategies.get_strategies()
        cycle.strategies_evaluated = len(strategies)

        for strat in strategies:
            if strat.total_attempts < self.MIN_SAMPLES:
                continue
            rate = strat.accepted / max(strat.total_attempts, 1)
            if rate >= self.BOOST_THRESHOLD:
                new_weight = 1.0 + rate * 2.0  # 1.8 - 3.0
                self._strategies.set_weight(strat.strategy_id, new_weight)
                cycle.strategies_boosted.append(strat.strategy_id)
            elif rate <= self.PENALIZE_THRESHOLD:
                new_weight = max(0.3, rate + 0.2)
                self._strategies.set_weight(strat.strategy_id, new_weight)
                cycle.strategies_penalized.append(strat.strategy_id)

        return cycle


# ---------------------------------------------------------------------------
# Recursive Meta-Learner
# ---------------------------------------------------------------------------

class RecursiveMetaLearner:
    """Top-level meta-learning engine.

    Connects source tracking, strategy tracking, and auto-prioritization
    into a feedback loop that improves evolution acceptance rates over time.
    """

    def __init__(
        self,
        db_dir: str | Path = "_meta_learning",
        source_tracker: Optional[SourceValueTracker] = None,
        strategy_tracker: Optional[StrategyTracker] = None,
    ):
        self._db_dir = Path(db_dir)
        self._db_dir.mkdir(parents=True, exist_ok=True)

        self._sources = source_tracker or SourceValueTracker(
            self._db_dir / "sources.db"
        )
        self._strategies = strategy_tracker or StrategyTracker(
            self._db_dir / "strategies.db"
        )
        self._prioritizer = AutoPrioritizer(self._sources, self._strategies)
        self._cycles: List[MetaLearningCycle] = []
        self._snapshots: List[WeeklySnapshot] = []

    def record_evolution(
        self,
        source_ids: List[str],
        strategy_id: str,
        decision: str,
        improvement: float = 0.0,
    ) -> None:
        """Record an evolution outcome for meta-learning."""
        for sid in source_ids:
            self._sources.record_outcome(sid, decision, improvement)
        self._strategies.record_outcome(strategy_id, decision, improvement)

    def optimize(self) -> MetaLearningCycle:
        """Run one meta-learning optimization cycle."""
        cycle = self._prioritizer.run()
        self._cycles.append(cycle)
        log.info(
            "Meta-learning cycle: %d sources evaluated, %d amplified, %d deprioritized",
            cycle.sources_evaluated,
            len(cycle.sources_amplified),
            len(cycle.sources_deprioritized),
        )
        return cycle

    def snapshot_week(self, week_num: int, total: int, accepted: int) -> WeeklySnapshot:
        """Record a weekly acceptance rate snapshot."""
        rate = accepted / max(total, 1)
        top_sources = [s.source_id for s in self._sources.get_top_sources(5)]
        strategies = self._strategies.get_strategies()
        top_strats = [s.strategy_id for s in strategies[:5]]

        snapshot = WeeklySnapshot(
            week_num=week_num,
            timestamp=time.time(),
            total_evolutions=total,
            accepted=accepted,
            acceptance_rate=round(rate, 3),
            top_sources=top_sources,
            top_strategies=top_strats,
        )
        self._snapshots.append(snapshot)
        return snapshot

    def trend(self) -> Dict[str, Any]:
        """Analyze acceptance rate trend across weeks."""
        if len(self._snapshots) < 2:
            return {
                "weeks": len(self._snapshots),
                "improving": False,
                "trend": 0.0,
                "snapshots": [
                    {"week": s.week_num, "rate": s.acceptance_rate}
                    for s in self._snapshots
                ],
            }

        rates = [s.acceptance_rate for s in self._snapshots]
        # Simple trend: compare second half to first half
        mid = len(rates) // 2
        first_half_avg = sum(rates[:mid]) / max(mid, 1)
        second_half_avg = sum(rates[mid:]) / max(len(rates) - mid, 1)

        return {
            "weeks": len(self._snapshots),
            "improving": second_half_avg > first_half_avg,
            "trend": round(second_half_avg - first_half_avg, 4),
            "first_half_avg": round(first_half_avg, 4),
            "second_half_avg": round(second_half_avg, 4),
            "snapshots": [
                {"week": s.week_num, "rate": s.acceptance_rate}
                for s in self._snapshots
            ],
        }

    def register_source(self, source_id: str, source_type: str, name: str) -> None:
        """Register a data source for tracking."""
        self._sources.register_source(source_id, source_type, name)

    def register_strategy(self, strategy_id: str, name: str) -> None:
        """Register a mutation strategy for tracking."""
        self._strategies.register_strategy(strategy_id, name)

    def get_source_rankings(self) -> List[SourceRecord]:
        """Get sources ranked by value."""
        return self._sources.get_top_sources()

    def get_strategy_rankings(self) -> List[StrategyRecord]:
        """Get strategies ranked by weight."""
        return self._strategies.get_strategies()

    def stats(self) -> Dict[str, Any]:
        """Aggregate meta-learning statistics."""
        sources = self._sources.get_sources()
        strategies = self._strategies.get_strategies()

        return {
            "total_sources": len(sources),
            "total_strategies": len(strategies),
            "optimization_cycles": len(self._cycles),
            "weekly_snapshots": len(self._snapshots),
            "trend": self.trend(),
        }

    def close(self) -> None:
        self._sources.close()
        self._strategies.close()
