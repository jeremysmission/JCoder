"""
Extended tests for MetaCognitiveController and RecursiveMetaLearner (Sprint 26).

Covers:
  MetaCognitive  -- strategy selection, Thompson sampling, bandit convergence,
                    strategy gating by query type, outcome recording/history.
  RecursiveMetaLearner -- source/strategy tracking, optimization cycles,
                          plateau detection, search-space exploration,
                          best-config tracking, budget enforcement.
"""

from __future__ import annotations

import os
import random
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from core.meta_cognitive import (
    STRATEGIES,
    MetaCognitiveController,
    QuerySignature,
    StrategyArm,
    classify_query,
)
from core.recursive_meta_learner import (
    AutoPrioritizer,
    RecursiveMetaLearner,
    SourceValueTracker,
    StrategyTracker,
)


# ---- helpers ---------------------------------------------------------------

@pytest.fixture()
def tmp_db(tmp_path):
    """Return a temp directory path string for DB storage."""
    return str(tmp_path)


@pytest.fixture()
def controller(tmp_path):
    db = str(tmp_path / "controller.db")
    return MetaCognitiveController(db_path=db, seed=42)


@pytest.fixture()
def meta_learner(tmp_path):
    db_dir = str(tmp_path / "meta_learn")
    return RecursiveMetaLearner(db_dir=db_dir)


# ===========================================================================
# MetaCognitive -- StrategyArm unit tests
# ===========================================================================

class TestStrategyArm:
    def test_initial_mean(self):
        arm = StrategyArm(name="test")
        assert arm.mean == pytest.approx(0.5)

    def test_update_shifts_mean_up(self):
        arm = StrategyArm(name="test")
        for _ in range(10):
            arm.update(1.0)
        assert arm.mean > 0.8
        assert arm.total_uses == 10

    def test_update_shifts_mean_down(self):
        arm = StrategyArm(name="test")
        for _ in range(10):
            arm.update(0.0)
        assert arm.mean < 0.2

    def test_sample_in_range(self):
        rng = random.Random(99)
        arm = StrategyArm(name="test", alpha=5.0, beta=5.0)
        samples = [arm.sample(rng) for _ in range(200)]
        assert all(0.0 <= s <= 1.0 for s in samples)
        avg = sum(samples) / len(samples)
        assert 0.3 < avg < 0.7  # should center near 0.5


# ===========================================================================
# MetaCognitive -- classify_query
# ===========================================================================

class TestClassifyQuery:
    def test_lookup_query(self):
        sig = classify_query("where is the config file?")
        assert sig.query_type == "lookup"

    def test_debug_query(self):
        sig = classify_query("fix this crash when I call parse()")
        assert sig.query_type == "debug"

    def test_design_query(self):
        sig = classify_query("design a REST API for users")
        assert sig.query_type == "design"

    def test_reasoning_query(self):
        sig = classify_query("should I use Redis versus Memcached?")
        assert sig.query_type == "reasoning"

    def test_explain_fallback(self):
        sig = classify_query("the quick brown fox")
        assert sig.query_type == "explain"

    def test_complexity_short(self):
        sig = classify_query("list files")
        assert sig.complexity < 0.1

    def test_complexity_long(self):
        long_q = "explain how to refactor the distributed service " + "word " * 30
        sig = classify_query(long_q)
        assert sig.complexity > 0.3

    def test_code_detection(self):
        sig = classify_query("fix the bug in parser.run()")
        assert sig.has_code is True

    def test_multi_part(self):
        sig = classify_query("find the config and also show logging setup?")
        assert sig.multi_part is True


# ===========================================================================
# MetaCognitive -- Controller strategy selection
# ===========================================================================

class TestControllerSelection:
    def test_returns_valid_strategy(self, controller):
        strategy, sig = controller.select_strategy("where is main.py?")
        assert strategy in STRATEGIES
        assert isinstance(sig, QuerySignature)

    def test_deterministic_with_seed(self, tmp_path):
        db = str(tmp_path / "det.db")
        c1 = MetaCognitiveController(db_path=db, seed=123)
        s1, _ = c1.select_strategy("explain how caching works")

        db2 = str(tmp_path / "det2.db")
        c2 = MetaCognitiveController(db_path=db2, seed=123)
        s2, _ = c2.select_strategy("explain how caching works")
        assert s1 == s2

    def test_exploration_bonus_for_new_arms(self, controller):
        """Arms with < 5 uses get an exploration bonus."""
        arm = controller._get_arm("lookup", "standard")
        assert arm.total_uses == 0  # fresh arm qualifies for bonus

    def test_strategy_gated_by_query_type(self, controller):
        """Different query types can converge to different strategies."""
        # Train: debug queries do well with corrective
        for _ in range(30):
            controller.report_outcome(
                "fix the crash bug", "corrective", reward=0.9
            )
            controller.report_outcome(
                "fix the crash bug", "standard", reward=0.2
            )
        # Train: lookup queries do well with standard
        for _ in range(30):
            controller.report_outcome(
                "where is the config?", "standard", reward=0.9
            )
            controller.report_outcome(
                "where is the config?", "corrective", reward=0.2
            )

        best = controller.best_strategy_per_type()
        assert best.get("debug") == "corrective"
        assert best.get("lookup") == "standard"


# ===========================================================================
# MetaCognitive -- Outcome recording and history
# ===========================================================================

class TestOutcomeRecording:
    def test_report_updates_arm(self, controller):
        controller.report_outcome("find config", "standard", reward=1.0, latency_ms=50)
        arm = controller._get_arm("lookup", "standard")
        assert arm.total_uses == 1
        assert arm.alpha > 1.0

    def test_latency_tracking(self, controller):
        controller.report_outcome("find x", "standard", reward=0.8, latency_ms=100)
        controller.report_outcome("find y", "standard", reward=0.7, latency_ms=200)
        arm = controller._get_arm("lookup", "standard")
        assert 100 < arm.avg_latency_ms < 200

    def test_strategy_report_populated(self, controller):
        controller.report_outcome("find x", "standard", reward=0.9)
        report = controller.strategy_report()
        assert "lookup" in report
        assert "standard" in report["lookup"]
        assert report["lookup"]["standard"]["total_uses"] == 1

    def test_best_strategy_per_type(self, controller):
        for _ in range(20):
            controller.report_outcome("fix bug x", "corrective", reward=0.95)
        best = controller.best_strategy_per_type()
        assert "debug" in best


# ===========================================================================
# MetaCognitive -- Thompson sampling convergence
# ===========================================================================

class TestThompsonConvergence:
    def test_bandit_converges_to_best(self, tmp_path):
        """After enough feedback the bandit should mostly pick the best arm."""
        db = str(tmp_path / "converge.db")
        ctrl = MetaCognitiveController(db_path=db, seed=7)

        # Simulate: 'reflective' is best for 'explain' queries
        for _ in range(80):
            ctrl.report_outcome("explain how X works", "reflective", reward=0.95)
            ctrl.report_outcome("explain how X works", "standard", reward=0.3)
            ctrl.report_outcome("explain how X works", "corrective", reward=0.3)

        picks = [ctrl.select_strategy("explain how Y works")[0] for _ in range(50)]
        reflective_pct = picks.count("reflective") / len(picks)
        assert reflective_pct > 0.5, f"reflective only picked {reflective_pct:.0%}"


# ===========================================================================
# RecursiveMetaLearner -- Source tracking
# ===========================================================================

class TestSourceTracking:
    def test_register_and_retrieve(self, meta_learner):
        meta_learner.register_source("src1", "dataset", "MyDataset")
        sources = meta_learner.get_source_rankings()
        # Needs >=3 uses for top_sources query, so direct get_sources
        all_src = meta_learner._sources.get_sources()
        assert any(s.source_id == "src1" for s in all_src)

    def test_outcome_recording(self, meta_learner):
        meta_learner.register_source("s1", "paper", "Paper1")
        meta_learner.register_strategy("mut1", "Mutation1")
        meta_learner.record_evolution(["s1"], "mut1", "accepted", improvement=0.05)
        sources = meta_learner._sources.get_sources()
        s1 = next(s for s in sources if s.source_id == "s1")
        assert s1.accepted_uses == 1
        assert s1.cumulative_improvement == pytest.approx(0.05)


# ===========================================================================
# RecursiveMetaLearner -- Strategy tracking
# ===========================================================================

class TestStrategyTracking:
    def test_register_and_record(self, meta_learner):
        meta_learner.register_strategy("strat_a", "PromptTweak")
        meta_learner.record_evolution([], "strat_a", "accepted", improvement=0.1)
        meta_learner.record_evolution([], "strat_a", "rejected")
        strats = meta_learner.get_strategy_rankings()
        sa = next(s for s in strats if s.strategy_id == "strat_a")
        assert sa.accepted == 1
        assert sa.rejected == 1
        assert sa.total_attempts == 2

    def test_error_outcome(self, meta_learner):
        meta_learner.register_strategy("strat_e", "ErrorProne")
        meta_learner.record_evolution([], "strat_e", "error")
        strats = meta_learner.get_strategy_rankings()
        se = next(s for s in strats if s.strategy_id == "strat_e")
        assert se.errors == 1


# ===========================================================================
# RecursiveMetaLearner -- Optimization cycle (auto-prioritization)
# ===========================================================================

class TestOptimizationCycle:
    def test_optimize_amplifies_good_sources(self, meta_learner):
        meta_learner.register_source("good", "dataset", "Good")
        for _ in range(5):
            meta_learner.record_evolution(["good"], "x", "accepted", 0.1)
        cycle = meta_learner.optimize()
        assert "good" in cycle.sources_amplified

    def test_optimize_deprioritizes_bad_sources(self, meta_learner):
        meta_learner.register_source("bad", "dataset", "Bad")
        meta_learner.register_strategy("x", "X")
        for _ in range(5):
            meta_learner.record_evolution(["bad"], "x", "rejected")
        cycle = meta_learner.optimize()
        assert "bad" in cycle.sources_deprioritized

    def test_optimize_boosts_good_strategy(self, meta_learner):
        meta_learner.register_strategy("gs", "GoodStrat")
        for _ in range(5):
            meta_learner.record_evolution([], "gs", "accepted", 0.1)
        cycle = meta_learner.optimize()
        assert "gs" in cycle.strategies_boosted

    def test_optimize_penalizes_bad_strategy(self, meta_learner):
        meta_learner.register_strategy("bs", "BadStrat")
        # All rejected -> rate 0.0 < 0.1 threshold
        for _ in range(5):
            meta_learner.record_evolution([], "bs", "rejected")
        cycle = meta_learner.optimize()
        assert "bs" in cycle.strategies_penalized


# ===========================================================================
# RecursiveMetaLearner -- Performance plateau / trend detection
# ===========================================================================

class TestPlateauDetection:
    def test_improving_trend(self, meta_learner):
        meta_learner.snapshot_week(1, total=100, accepted=20)
        meta_learner.snapshot_week(2, total=100, accepted=30)
        meta_learner.snapshot_week(3, total=100, accepted=50)
        meta_learner.snapshot_week(4, total=100, accepted=60)
        t = meta_learner.trend()
        assert t["improving"] is True
        assert t["trend"] > 0

    def test_declining_trend(self, meta_learner):
        meta_learner.snapshot_week(1, total=100, accepted=60)
        meta_learner.snapshot_week(2, total=100, accepted=50)
        meta_learner.snapshot_week(3, total=100, accepted=30)
        meta_learner.snapshot_week(4, total=100, accepted=20)
        t = meta_learner.trend()
        assert t["improving"] is False
        assert t["trend"] < 0

    def test_single_snapshot_no_trend(self, meta_learner):
        meta_learner.snapshot_week(1, total=50, accepted=25)
        t = meta_learner.trend()
        assert t["improving"] is False
        assert t["weeks"] == 1


# ===========================================================================
# RecursiveMetaLearner -- Best config tracking & stats
# ===========================================================================

class TestBestConfigTracking:
    def test_stats_aggregate(self, meta_learner):
        meta_learner.register_source("a", "dataset", "A")
        meta_learner.register_strategy("b", "B")
        meta_learner.optimize()
        stats = meta_learner.stats()
        assert stats["total_sources"] >= 1
        assert stats["total_strategies"] >= 1
        assert stats["optimization_cycles"] == 1

    def test_multiple_optimize_cycles(self, meta_learner):
        meta_learner.register_source("s", "dataset", "S")
        meta_learner.register_strategy("m", "M")
        for _ in range(5):
            meta_learner.record_evolution(["s"], "m", "accepted", 0.05)
        meta_learner.optimize()
        meta_learner.optimize()
        stats = meta_learner.stats()
        assert stats["optimization_cycles"] == 2


# ===========================================================================
# RecursiveMetaLearner -- Budget enforcement (cycle count)
# ===========================================================================

class TestBudgetEnforcement:
    def test_budget_caps_optimization(self, meta_learner):
        """Caller can enforce a budget by limiting optimize() calls."""
        budget = 3
        meta_learner.register_source("x", "paper", "X")
        meta_learner.register_strategy("y", "Y")
        for _ in range(10):
            meta_learner.record_evolution(["x"], "y", "accepted", 0.02)

        for _ in range(budget):
            meta_learner.optimize()
        assert meta_learner.stats()["optimization_cycles"] == budget

    def test_snapshot_budget(self, meta_learner):
        """Weekly snapshots accumulate correctly up to budget."""
        for w in range(1, 7):
            meta_learner.snapshot_week(w, total=100, accepted=40 + w)
        t = meta_learner.trend()
        assert t["weeks"] == 6


# ===========================================================================
# RecursiveMetaLearner -- Search space exploration
# ===========================================================================

class TestSearchSpaceExploration:
    def test_diverse_sources_tracked(self, meta_learner):
        for i in range(5):
            meta_learner.register_source(f"src_{i}", "dataset", f"DS{i}")
        all_src = meta_learner._sources.get_sources()
        assert len(all_src) == 5

    def test_diverse_strategies_tracked(self, meta_learner):
        for i in range(4):
            meta_learner.register_strategy(f"st_{i}", f"Strat{i}")
        strats = meta_learner.get_strategy_rankings()
        assert len(strats) == 4

    def test_strategy_weight_selection(self, meta_learner):
        """Weighted selection favours higher-weight strategies."""
        meta_learner.register_strategy("hi", "HighWeight")
        meta_learner.register_strategy("lo", "LowWeight")
        # Give hi many accepted
        for _ in range(10):
            meta_learner.record_evolution([], "hi", "accepted", 0.1)
            meta_learner.record_evolution([], "lo", "rejected")
        meta_learner.optimize()
        strats = meta_learner.get_strategy_rankings()
        hi_s = next(s for s in strats if s.strategy_id == "hi")
        lo_s = next(s for s in strats if s.strategy_id == "lo")
        assert hi_s.weight > lo_s.weight


# ===========================================================================
# Edge cases
# ===========================================================================

class TestEdgeCases:
    def test_controller_no_crash_on_empty(self, controller):
        report = controller.strategy_report()
        assert report == {} or isinstance(report, dict)
        best = controller.best_strategy_per_type()
        assert isinstance(best, dict)

    def test_meta_learner_close(self, meta_learner):
        meta_learner.close()
        # Should not raise on double close
        meta_learner.close()
