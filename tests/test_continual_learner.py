"""
Tests for core.continual_learner -- ContinualLearner anti-forgetting guard.
Uses a real temp SQLite database; no mocking of the learner itself.
"""

from __future__ import annotations

import os
import tempfile
from typing import Dict, List
from unittest.mock import MagicMock

import pytest

from core.continual_learner import (
    CapabilityBaseline,
    ContinualLearner,
    ConsolidationResult,
    RegressionCheck,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_db(tmp_path):
    """Return a temp DB path for the learner."""
    return str(tmp_path / "learner.db")


def _constant_eval(score: float):
    """Return an eval_fn that always returns `score`."""
    return lambda name, queries: score


def _per_capability_eval(scores: Dict[str, float]):
    """Return an eval_fn that returns scores keyed by capability name."""
    return lambda name, queries: scores.get(name, 0.0)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:

    def test_creates_db(self, tmp_db):
        """ContinualLearner creates the SQLite database on init."""
        learner = ContinualLearner(eval_fn=_constant_eval(1.0), db_path=tmp_db)
        assert os.path.exists(tmp_db)

    def test_creates_parent_dirs(self, tmp_path):
        """Nested DB path directories are created automatically."""
        db = str(tmp_path / "deep" / "nested" / "learner.db")
        learner = ContinualLearner(eval_fn=_constant_eval(1.0), db_path=db)
        assert os.path.exists(db)

    def test_custom_margin(self, tmp_db):
        """Regression margin is configurable."""
        learner = ContinualLearner(
            eval_fn=_constant_eval(1.0), db_path=tmp_db,
            regression_margin=0.10,
        )
        assert learner.margin == 0.10


# ---------------------------------------------------------------------------
# set_baseline / get_baselines
# ---------------------------------------------------------------------------

class TestBaselines:

    def test_set_and_get(self, tmp_db):
        """set_baseline stores, get_baselines retrieves."""
        learner = ContinualLearner(eval_fn=_constant_eval(1.0), db_path=tmp_db)
        learner.set_baseline("accuracy", 0.90, ["q1", "q2"])
        baselines = learner.get_baselines()
        assert len(baselines) == 1
        assert baselines[0].name == "accuracy"
        assert baselines[0].score == 0.90
        assert baselines[0].test_queries == ["q1", "q2"]

    def test_default_threshold(self, tmp_db):
        """Threshold defaults to score - margin."""
        learner = ContinualLearner(
            eval_fn=_constant_eval(1.0), db_path=tmp_db,
            regression_margin=0.05,
        )
        bl = learner.set_baseline("accuracy", 0.90, ["q1"])
        assert bl.threshold == pytest.approx(0.85)

    def test_custom_threshold(self, tmp_db):
        """Custom threshold overrides the default."""
        learner = ContinualLearner(eval_fn=_constant_eval(1.0), db_path=tmp_db)
        bl = learner.set_baseline("accuracy", 0.90, ["q1"], threshold=0.80)
        assert bl.threshold == 0.80

    def test_threshold_floor_at_zero(self, tmp_db):
        """Threshold never goes below 0.0."""
        learner = ContinualLearner(
            eval_fn=_constant_eval(1.0), db_path=tmp_db,
            regression_margin=0.50,
        )
        bl = learner.set_baseline("accuracy", 0.10, ["q1"])
        assert bl.threshold == 0.0

    def test_upsert_baseline(self, tmp_db):
        """Setting the same baseline name overwrites the previous one."""
        learner = ContinualLearner(eval_fn=_constant_eval(1.0), db_path=tmp_db)
        learner.set_baseline("accuracy", 0.80, ["q1"])
        learner.set_baseline("accuracy", 0.95, ["q1", "q2"])
        baselines = learner.get_baselines()
        assert len(baselines) == 1
        assert baselines[0].score == 0.95

    def test_multiple_baselines(self, tmp_db):
        """Multiple distinct baselines are tracked independently."""
        learner = ContinualLearner(eval_fn=_constant_eval(1.0), db_path=tmp_db)
        learner.set_baseline("accuracy", 0.90, ["q1"])
        learner.set_baseline("refusal", 0.95, ["q2"])
        learner.set_baseline("latency", 0.80, ["q3"])
        baselines = learner.get_baselines()
        assert len(baselines) == 3
        names = {b.name for b in baselines}
        assert names == {"accuracy", "latency", "refusal"}


# ---------------------------------------------------------------------------
# check_regression
# ---------------------------------------------------------------------------

class TestCheckRegression:

    def test_passes_when_above_threshold(self, tmp_db):
        """Regression check passes when all scores are above threshold."""
        learner = ContinualLearner(
            eval_fn=_constant_eval(0.92), db_path=tmp_db,
            regression_margin=0.05,
        )
        learner.set_baseline("accuracy", 0.90, ["q1"])
        check = learner.check_regression()
        assert check.passed is True
        assert check.failed_capabilities == []

    def test_fails_when_below_threshold(self, tmp_db):
        """Regression check fails when any score drops below threshold."""
        learner = ContinualLearner(
            eval_fn=_constant_eval(0.80), db_path=tmp_db,
            regression_margin=0.05,
        )
        learner.set_baseline("accuracy", 0.90, ["q1"])  # threshold=0.85
        check = learner.check_regression()
        assert check.passed is False
        assert "accuracy" in check.failed_capabilities

    def test_empty_baselines_passes(self, tmp_db):
        """With no baselines registered, regression check always passes."""
        learner = ContinualLearner(eval_fn=_constant_eval(0.0), db_path=tmp_db)
        check = learner.check_regression()
        assert check.passed is True

    def test_partial_failure(self, tmp_db):
        """Only the failing capability is listed, passing ones are fine."""
        scores = {"accuracy": 0.92, "refusal": 0.70}
        learner = ContinualLearner(
            eval_fn=_per_capability_eval(scores), db_path=tmp_db,
            regression_margin=0.05,
        )
        learner.set_baseline("accuracy", 0.90, ["q1"])  # threshold=0.85 -- 0.92 passes
        learner.set_baseline("refusal", 0.95, ["q2"])   # threshold=0.90 -- 0.70 fails
        check = learner.check_regression()
        assert check.passed is False
        assert "refusal" in check.failed_capabilities
        assert "accuracy" not in check.failed_capabilities

    def test_config_snapshot_on_check(self, tmp_db):
        """When config is provided, it's snapshotted for rollback."""
        learner = ContinualLearner(
            eval_fn=_constant_eval(1.0), db_path=tmp_db,
        )
        learner.set_baseline("accuracy", 0.90, ["q1"])
        config = {"temperature": 0.1, "top_k": 5}
        learner.check_regression(config=config)

        rolled_back = learner.rollback_to_last()
        assert rolled_back == config

    def test_eval_exception_scores_zero(self, tmp_db):
        """If eval_fn throws, the score is treated as 0.0."""
        def _crashing_eval(name, queries):
            raise RuntimeError("eval crash")

        learner = ContinualLearner(eval_fn=_crashing_eval, db_path=tmp_db)
        learner.set_baseline("accuracy", 0.90, ["q1"])
        check = learner.check_regression()
        assert check.passed is False
        assert check.scores["accuracy"] == 0.0


# ---------------------------------------------------------------------------
# update_baselines (ratchet)
# ---------------------------------------------------------------------------

class TestUpdateBaselines:

    def test_ratchets_up(self, tmp_db):
        """Baseline score is raised when current performance is higher."""
        learner = ContinualLearner(
            eval_fn=_constant_eval(0.95), db_path=tmp_db,
        )
        learner.set_baseline("accuracy", 0.90, ["q1"])
        updates = learner.update_baselines()
        assert "accuracy" in updates
        assert updates["accuracy"] == 0.95
        # Verify persisted
        bl = learner.get_baselines()[0]
        assert bl.score == 0.95

    def test_never_lowers(self, tmp_db):
        """Baseline score is NOT lowered when current performance drops."""
        learner = ContinualLearner(
            eval_fn=_constant_eval(0.80), db_path=tmp_db,
        )
        learner.set_baseline("accuracy", 0.90, ["q1"])
        updates = learner.update_baselines()
        assert updates == {}
        bl = learner.get_baselines()[0]
        assert bl.score == 0.90


# ---------------------------------------------------------------------------
# rollback_to_last
# ---------------------------------------------------------------------------

class TestRollback:

    def test_returns_most_recent(self, tmp_db):
        """rollback_to_last returns the most recently snapshotted config."""
        import time
        learner = ContinualLearner(eval_fn=_constant_eval(1.0), db_path=tmp_db)
        learner.set_baseline("accuracy", 0.90, ["q1"])
        learner.check_regression(config={"v": 1})
        time.sleep(0.05)  # ensure distinct timestamps
        learner.check_regression(config={"v": 2})
        rolled = learner.rollback_to_last()
        assert rolled["v"] == 2

    def test_empty_returns_none(self, tmp_db):
        """With no snapshots, rollback_to_last returns None."""
        learner = ContinualLearner(eval_fn=_constant_eval(1.0), db_path=tmp_db)
        assert learner.rollback_to_last() is None


# ---------------------------------------------------------------------------
# consolidate
# ---------------------------------------------------------------------------

class TestConsolidate:

    def test_all_prune_fns_called(self, tmp_db):
        """All provided prune/compact functions are called."""
        learner = ContinualLearner(eval_fn=_constant_eval(1.0), db_path=tmp_db)
        result = learner.consolidate(
            experience_prune_fn=lambda: 5,
            config_prune_fn=lambda: 3,
            telemetry_compact_fn=lambda: 100,
        )
        assert result.experiences_pruned == 5
        assert result.configs_pruned == 3
        assert result.telemetry_compacted == 100
        assert result.duration_ms >= 0

    def test_missing_fns_skipped(self, tmp_db):
        """Missing prune functions are skipped gracefully."""
        learner = ContinualLearner(eval_fn=_constant_eval(1.0), db_path=tmp_db)
        result = learner.consolidate(experience_prune_fn=lambda: 2)
        assert result.experiences_pruned == 2
        assert result.configs_pruned == 0

    def test_exception_in_prune_fn_isolated(self, tmp_db):
        """If a prune function throws, others still run."""
        learner = ContinualLearner(eval_fn=_constant_eval(1.0), db_path=tmp_db)

        def _crash():
            raise RuntimeError("prune crash")

        result = learner.consolidate(
            experience_prune_fn=_crash,
            config_prune_fn=lambda: 7,
        )
        assert result.experiences_pruned == 0  # crashed
        assert result.configs_pruned == 7      # still ran


# ---------------------------------------------------------------------------
# health_report
# ---------------------------------------------------------------------------

class TestHealthReport:

    def test_empty_report(self, tmp_db):
        """Health report with no baselines or checks."""
        learner = ContinualLearner(eval_fn=_constant_eval(1.0), db_path=tmp_db)
        report = learner.health_report()
        assert report["capabilities_tracked"] == 0
        assert report["total_regression_checks"] == 0
        assert report["failure_rate"] == 0.0

    def test_populated_report(self, tmp_db):
        """Health report reflects baselines and checks."""
        learner = ContinualLearner(
            eval_fn=_constant_eval(0.92), db_path=tmp_db,
        )
        learner.set_baseline("accuracy", 0.90, ["q1"])
        learner.set_baseline("refusal", 0.95, ["q2"])
        learner.check_regression()

        report = learner.health_report()
        assert report["capabilities_tracked"] == 2
        assert report["total_regression_checks"] == 1
        assert "accuracy" in report["baselines"]
        assert report["baselines"]["accuracy"]["score"] == 0.90


# ---------------------------------------------------------------------------
# Persistence across instances
# ---------------------------------------------------------------------------

class TestPersistence:

    def test_baselines_survive_restart(self, tmp_db):
        """Baselines persist across ContinualLearner instances."""
        learner1 = ContinualLearner(eval_fn=_constant_eval(1.0), db_path=tmp_db)
        learner1.set_baseline("accuracy", 0.90, ["q1", "q2"])

        learner2 = ContinualLearner(eval_fn=_constant_eval(1.0), db_path=tmp_db)
        baselines = learner2.get_baselines()
        assert len(baselines) == 1
        assert baselines[0].name == "accuracy"

    def test_regression_history_survives_restart(self, tmp_db):
        """Regression check history persists."""
        learner1 = ContinualLearner(eval_fn=_constant_eval(0.92), db_path=tmp_db)
        learner1.set_baseline("accuracy", 0.90, ["q1"])
        learner1.check_regression()

        learner2 = ContinualLearner(eval_fn=_constant_eval(1.0), db_path=tmp_db)
        report = learner2.health_report()
        assert report["total_regression_checks"] == 1
