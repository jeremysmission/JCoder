"""
Tests for core.meta_cognitive -- Thompson sampling routing.
No external deps; pure SQLite + in-memory.
"""

from __future__ import annotations

import pytest

from core.meta_cognitive import (
    QuerySignature,
    StrategyArm,
    MetaCognitiveController,
    classify_query,
    STRATEGIES,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ctrl(tmp_path, **kwargs):
    db = str(tmp_path / "metacog.db")
    return MetaCognitiveController(db_path=db, **kwargs)


# ---------------------------------------------------------------------------
# QuerySignature
# ---------------------------------------------------------------------------

class TestQuerySignature:

    def test_lookup(self):
        sig = classify_query("where is the sort function?")
        assert sig.query_type == "lookup"

    def test_debug(self):
        sig = classify_query("fix the crash in parser.py")
        assert sig.query_type == "debug"

    def test_design(self):
        sig = classify_query("design a caching layer for the API")
        assert sig.query_type == "design"

    def test_reasoning(self):
        sig = classify_query("should I use Redis versus Memcached?")
        assert sig.query_type == "reasoning"

    def test_explain(self):
        sig = classify_query("how does the auth middleware work?")
        assert sig.query_type == "explain"

    def test_complexity_scaling(self):
        short = classify_query("find foo")
        long_q = classify_query(
            "refactor the distributed caching layer to optimize "
            "concurrent access patterns and migrate to the new security framework"
        )
        assert long_q.complexity > short.complexity

    def test_code_detection(self):
        sig = classify_query("explain MyClass.process_data")
        assert sig.has_code is True

    def test_multi_part(self):
        sig = classify_query("explain sorting and also show the tests?")
        assert sig.multi_part is True


# ---------------------------------------------------------------------------
# StrategyArm
# ---------------------------------------------------------------------------

class TestStrategyArm:

    def test_defaults(self):
        arm = StrategyArm(name="standard")
        assert arm.mean == 0.5

    def test_update_with_reward(self):
        arm = StrategyArm(name="standard")
        arm.update(reward=0.8)
        assert arm.alpha == 1.8
        assert arm.beta == 1.2
        assert arm.total_uses == 1

    def test_sample_in_range(self):
        import random
        arm = StrategyArm(name="corrective")
        rng = random.Random(42)
        s = arm.sample(rng)
        assert 0.0 <= s <= 1.0


# ---------------------------------------------------------------------------
# Controller construction
# ---------------------------------------------------------------------------

class TestConstruction:

    def test_default_exploration_bonus(self, tmp_path):
        ctrl = _ctrl(tmp_path)
        assert ctrl.exploration_bonus == 0.1


# ---------------------------------------------------------------------------
# Strategy selection
# ---------------------------------------------------------------------------

class TestSelectStrategy:

    def test_returns_valid_strategy(self, tmp_path):
        ctrl = _ctrl(tmp_path)
        strategy, sig = ctrl.select_strategy("where is the sort function?")
        assert strategy in STRATEGIES
        assert sig.query_type == "lookup"

    def test_exploration_bonus(self, tmp_path):
        """Underexplored strategies should get exploration bonus."""
        ctrl = _ctrl(tmp_path, exploration_bonus=0.5)
        # All strategies start with 0 uses, so all get bonus
        strategy, _ = ctrl.select_strategy("explain auth flow")
        assert strategy in STRATEGIES

    def test_learns_from_outcomes(self, tmp_path):
        """Strategy with high reward should eventually be preferred."""
        ctrl = _ctrl(tmp_path, seed=42)
        # Train: standard always gets high reward for lookup queries
        for _ in range(20):
            ctrl.report_outcome("find foo", "standard", reward=0.95)
            ctrl.report_outcome("find foo", "best_of_n", reward=0.3)

        # Now standard should be preferred for lookup
        selections = [ctrl.select_strategy("find bar")[0] for _ in range(50)]
        assert selections.count("standard") > selections.count("best_of_n")


# ---------------------------------------------------------------------------
# Outcome reporting
# ---------------------------------------------------------------------------

class TestReportOutcome:

    def test_report_updates_arm(self, tmp_path):
        ctrl = _ctrl(tmp_path)
        ctrl.report_outcome("find foo", "standard", reward=0.9, latency_ms=100)
        report = ctrl.strategy_report()
        assert "lookup" in report

    def test_persistence(self, tmp_path):
        db = str(tmp_path / "persist.db")
        ctrl = MetaCognitiveController(db_path=db)
        ctrl.report_outcome("find foo", "standard", reward=0.9)

        # Reload from same db
        ctrl2 = MetaCognitiveController(db_path=db)
        report = ctrl2.strategy_report()
        assert "lookup" in report
        assert "standard" in report["lookup"]


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

class TestReports:

    def test_strategy_report_format(self, tmp_path):
        ctrl = _ctrl(tmp_path)
        ctrl.report_outcome("find x", "standard", reward=0.8)
        report = ctrl.strategy_report()
        entry = report["lookup"]["standard"]
        assert "mean_reward" in entry
        assert "confidence" in entry

    def test_best_strategy_per_type(self, tmp_path):
        ctrl = _ctrl(tmp_path)
        for _ in range(10):
            ctrl.report_outcome("find x", "standard", reward=0.95)
            ctrl.report_outcome("find x", "best_of_n", reward=0.4)
        best = ctrl.best_strategy_per_type()
        assert best.get("lookup") == "standard"
