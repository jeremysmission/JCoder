"""
Tests for core.meta_cognitive -- Cost-aware Thompson sampling routing.
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
    DEFAULT_STRATEGY_COSTS,
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
        assert arm.avg_cost == 0.0

    def test_update_with_reward(self):
        arm = StrategyArm(name="standard")
        arm.update(reward=0.8)
        assert arm.alpha == 1.8
        assert arm.beta == 1.2
        assert arm.total_uses == 1

    def test_update_with_cost(self):
        arm = StrategyArm(name="best_of_n")
        arm.update(reward=0.9, cost=3.0)
        assert arm.avg_cost == 3.0
        arm.update(reward=0.7, cost=5.0)
        # Running avg: (3.0 * 1 + 5.0) / 2 = 4.0
        assert abs(arm.avg_cost - 4.0) < 0.01

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

    def test_default_cost_weight(self, tmp_path):
        ctrl = _ctrl(tmp_path)
        assert ctrl.cost_weight == 0.0

    def test_custom_cost_weight(self, tmp_path):
        ctrl = _ctrl(tmp_path, cost_weight=0.3)
        assert ctrl.cost_weight == 0.3

    def test_cost_weight_clamped(self, tmp_path):
        ctrl = _ctrl(tmp_path, cost_weight=5.0)
        assert ctrl.cost_weight == 1.0
        ctrl2 = _ctrl(tmp_path, cost_weight=-1.0)
        assert ctrl2.cost_weight == 0.0

    def test_custom_strategy_costs(self, tmp_path):
        custom = {"standard": 0.5, "best_of_n": 10.0}
        ctrl = _ctrl(tmp_path, strategy_costs=custom)
        assert ctrl.strategy_costs["standard"] == 0.5


# ---------------------------------------------------------------------------
# Strategy selection (quality only -- backward compat)
# ---------------------------------------------------------------------------

class TestSelectStrategyQualityOnly:

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
# Cost-aware selection (BARP pattern)
# ---------------------------------------------------------------------------

class TestCostAwareSelection:

    def test_cheap_preferred_with_cost_weight(self, tmp_path):
        """With cost_weight > 0, cheaper strategies should be preferred
        when quality is similar."""
        ctrl = _ctrl(tmp_path, cost_weight=0.5, seed=42)

        # Train: both strategies have similar quality
        for _ in range(20):
            ctrl.report_outcome("find foo", "standard", reward=0.8, cost=1.0)
            ctrl.report_outcome("find foo", "best_of_n", reward=0.82, cost=5.0)

        # With cost penalty, standard should win despite slightly lower quality
        selections = [ctrl.select_strategy("find bar")[0] for _ in range(50)]
        assert selections.count("standard") > selections.count("best_of_n"), (
            f"standard={selections.count('standard')}, "
            f"best_of_n={selections.count('best_of_n')}"
        )

    def test_no_cost_penalty_when_weight_zero(self, tmp_path):
        """With cost_weight=0, cost should not affect selection."""
        ctrl = _ctrl(tmp_path, cost_weight=0.0, seed=42)

        # Train: best_of_n has higher quality but higher cost
        for _ in range(30):
            ctrl.report_outcome("explain auth", "standard", reward=0.5, cost=1.0)
            ctrl.report_outcome("explain auth", "best_of_n", reward=0.95, cost=10.0)

        # best_of_n should win since cost is ignored
        selections = [ctrl.select_strategy("explain auth flow")[0] for _ in range(50)]
        assert selections.count("best_of_n") > selections.count("standard")

    def test_cost_weight_adjusts_tradeoff(self, tmp_path):
        """Higher cost_weight should shift preference toward cheaper strategies."""
        # Low cost weight
        ctrl_low = _ctrl(tmp_path, cost_weight=0.1, seed=42)
        for _ in range(20):
            ctrl_low.report_outcome("find x", "standard", reward=0.7, cost=1.0)
            ctrl_low.report_outcome("find x", "reflective", reward=0.8, cost=3.0)
        sel_low = [ctrl_low.select_strategy("find y")[0] for _ in range(100)]

        # High cost weight (separate db)
        db2 = str(tmp_path / "metacog2.db")
        ctrl_high = MetaCognitiveController(
            db_path=db2, cost_weight=0.8, seed=42)
        for _ in range(20):
            ctrl_high.report_outcome("find x", "standard", reward=0.7, cost=1.0)
            ctrl_high.report_outcome("find x", "reflective", reward=0.8, cost=3.0)
        sel_high = [ctrl_high.select_strategy("find y")[0] for _ in range(100)]

        # Higher cost weight should mean more standard selections
        assert sel_high.count("standard") >= sel_low.count("standard")


# ---------------------------------------------------------------------------
# Outcome reporting with cost
# ---------------------------------------------------------------------------

class TestReportOutcome:

    def test_report_updates_arm(self, tmp_path):
        ctrl = _ctrl(tmp_path)
        ctrl.report_outcome("find foo", "standard", reward=0.9, latency_ms=100, cost=1.5)
        report = ctrl.strategy_report()
        assert "lookup" in report
        assert report["lookup"]["standard"]["avg_cost"] == 1.5

    def test_cost_accumulates(self, tmp_path):
        ctrl = _ctrl(tmp_path)
        ctrl.report_outcome("find foo", "standard", reward=0.9, cost=2.0)
        ctrl.report_outcome("find bar", "standard", reward=0.8, cost=4.0)
        report = ctrl.strategy_report()
        # avg_cost should be ~3.0
        assert abs(report["lookup"]["standard"]["avg_cost"] - 3.0) < 0.01

    def test_persistence_with_cost(self, tmp_path):
        db = str(tmp_path / "persist.db")
        ctrl = MetaCognitiveController(db_path=db)
        ctrl.report_outcome("find foo", "standard", reward=0.9, cost=2.5)
        ctrl.close()

        # Reload
        ctrl2 = MetaCognitiveController(db_path=db)
        report = ctrl2.strategy_report()
        assert abs(report["lookup"]["standard"]["avg_cost"] - 2.5) < 0.01


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

class TestReports:

    def test_strategy_report_format(self, tmp_path):
        ctrl = _ctrl(tmp_path)
        ctrl.report_outcome("find x", "standard", reward=0.8, cost=1.0)
        report = ctrl.strategy_report()
        entry = report["lookup"]["standard"]
        assert "mean_reward" in entry
        assert "avg_cost" in entry
        assert "confidence" in entry

    def test_cost_report_uses_estimates_when_no_data(self, tmp_path):
        ctrl = _ctrl(tmp_path)
        report = ctrl.cost_report()
        assert report["standard"]["avg_cost"] == DEFAULT_STRATEGY_COSTS["standard"]
        assert report["best_of_n"]["avg_cost"] == DEFAULT_STRATEGY_COSTS["best_of_n"]

    def test_cost_report_uses_observed_when_available(self, tmp_path):
        ctrl = _ctrl(tmp_path)
        ctrl.report_outcome("find x", "standard", reward=0.8, cost=1.5)
        report = ctrl.cost_report()
        assert report["standard"]["avg_cost"] == 1.5

    def test_best_strategy_per_type(self, tmp_path):
        ctrl = _ctrl(tmp_path)
        for _ in range(10):
            ctrl.report_outcome("find x", "standard", reward=0.95)
            ctrl.report_outcome("find x", "best_of_n", reward=0.4)
        best = ctrl.best_strategy_per_type()
        assert best.get("lookup") == "standard"


# ---------------------------------------------------------------------------
# GATE TEST: Cost-aware routing
# ---------------------------------------------------------------------------

class TestGateCostAwareRouting:
    """
    Gate test for Sprint 20: with cost_weight > 0, a cheap strategy
    should be preferred for easy queries even when a slightly higher
    quality strategy exists at much higher cost.
    """

    def test_cost_routing_prefers_cheap_for_easy_queries(self, tmp_path):
        """
        Train all strategies. Standard is cheap+decent, best_of_n is
        expensive+slightly better, others are mediocre.
        With cost_weight=0.5, cheap standard should beat expensive best_of_n.
        """
        ctrl = _ctrl(tmp_path, cost_weight=0.5, seed=42)

        # Train all 5 strategies so exploration bonus is consumed
        for _ in range(30):
            ctrl.report_outcome("find function", "standard",
                                reward=0.80, cost=1.0)
            ctrl.report_outcome("find function", "best_of_n",
                                reward=0.85, cost=5.0)
            ctrl.report_outcome("find function", "corrective",
                                reward=0.40, cost=2.0)
            ctrl.report_outcome("find function", "cascade",
                                reward=0.40, cost=1.5)
            ctrl.report_outcome("find function", "reflective",
                                reward=0.40, cost=3.0)

        selections = [ctrl.select_strategy("find method")[0] for _ in range(100)]
        standard_count = selections.count("standard")
        best_of_n_count = selections.count("best_of_n")

        assert standard_count > best_of_n_count, (
            f"Standard should beat best_of_n with cost penalty, "
            f"got standard={standard_count}, best_of_n={best_of_n_count}"
        )
