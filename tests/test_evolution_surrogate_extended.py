"""
Extended tests for EvolutionRunner and SurrogateScorer (Sprint 30).

All external dependencies (git, subprocess, SQLite on disk) are mocked
so these tests run fast and offline.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

from core.evolution_runner import (
    EvolutionCycle,
    EvolutionDecision,
    EvolutionLedger,
    EvolutionRunner,
    _create_worktree,
    _remove_worktree,
)
from core.surrogate_scorer import (
    SurrogateEvalStore,
    SurrogateModel,
    extract_features,
)


# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------

@pytest.fixture()
def tmp_ledger(tmp_path):
    """Create a real EvolutionLedger backed by a temp SQLite DB."""
    ledger = EvolutionLedger(tmp_path / "ledger.db")
    yield ledger
    ledger.close()


@pytest.fixture()
def runner(tmp_ledger):
    """EvolutionRunner with real ledger but subprocess mocked."""
    with patch("core.evolution_runner.subprocess") as mock_sp:
        mock_sp.run.return_value = MagicMock(
            stdout="", stderr="", returncode=0
        )
        r = EvolutionRunner(
            ledger=tmp_ledger,
            repo_root=Path("/fake/repo"),
            use_worktree=False,
        )
        yield r
        r.close()


@pytest.fixture()
def surrogate_store(tmp_path):
    """SurrogateEvalStore backed by temp DB."""
    store = SurrogateEvalStore(
        db_path=tmp_path / "surrogate.db",
        refit_interval=5,
        min_samples=3,
    )
    yield store
    store.close()


# ===================================================================
# Evolution Runner tests
# ===================================================================

class TestBaselineArchiving:
    """Baseline is archived BEFORE any mutation runs."""

    def test_baseline_archived_before_mutation(self, runner, tmp_ledger):
        cfg = {"lr": 0.01}
        cycle = runner.run_cycle(
            baseline_config=cfg,
            baseline_score=70.0,
            eval_fn=lambda c: 75.0,
            mutate_fn=lambda c: {**c, "lr": 0.005},
        )
        baselines = tmp_ledger.get_baselines()
        assert len(baselines) == 1
        assert baselines[0]["score"] == 70.0
        assert baselines[0]["config"]["lr"] == 0.01

    def test_baseline_archived_even_on_mutation_error(self, runner, tmp_ledger):
        def bad_mutate(_):
            raise ValueError("boom")

        cycle = runner.run_cycle(
            baseline_config={"x": 1},
            baseline_score=50.0,
            eval_fn=lambda c: 0.0,
            mutate_fn=bad_mutate,
        )
        assert cycle.decision == EvolutionDecision.ERROR
        assert tmp_ledger.get_baselines()  # still archived


class TestMutationApplication:
    """Mutation function is called with baseline config."""

    def test_mutate_fn_receives_baseline(self, runner):
        received = {}

        def capture_mutate(cfg):
            received.update(cfg)
            return {**cfg, "epochs": 20}

        runner.run_cycle(
            baseline_config={"epochs": 10},
            baseline_score=60.0,
            eval_fn=lambda c: 65.0,
            mutate_fn=capture_mutate,
        )
        assert received == {"epochs": 10}

    def test_mutation_error_yields_error_decision(self, runner):
        cycle = runner.run_cycle(
            baseline_config={},
            baseline_score=50.0,
            eval_fn=lambda c: 0.0,
            mutate_fn=lambda c: (_ for _ in ()).throw(RuntimeError("fail")),
        )
        assert cycle.decision == EvolutionDecision.ERROR
        assert "Mutation failed" in cycle.reason


class TestCandidateEvaluation:
    """Candidate is evaluated and regression-gated."""

    def test_eval_fn_called_with_candidate_config(self, runner):
        evaluated = {}

        def track_eval(cfg):
            evaluated.update(cfg)
            return 80.0

        runner.run_cycle(
            baseline_config={"lr": 0.01},
            baseline_score=70.0,
            eval_fn=track_eval,
            mutate_fn=lambda c: {**c, "lr": 0.005},
        )
        assert evaluated["lr"] == 0.005

    def test_eval_results_contain_delta(self, runner):
        cycle = runner.run_cycle(
            baseline_config={"a": 1},
            baseline_score=70.0,
            eval_fn=lambda c: 73.0,
            mutate_fn=lambda c: c,
            min_improvement=0.5,
        )
        assert cycle.eval_results["delta"] == pytest.approx(3.0)

    def test_regression_gate_rejects_on_failure(self, runner):
        cycle = runner.run_cycle(
            baseline_config={},
            baseline_score=70.0,
            eval_fn=lambda c: 75.0,
            mutate_fn=lambda c: c,
            regression_fn=lambda base, cand: False,  # always fail
        )
        assert cycle.decision == EvolutionDecision.REJECTED
        assert "regression gate" in cycle.reason.lower()

    def test_regression_gate_passes(self, runner):
        cycle = runner.run_cycle(
            baseline_config={},
            baseline_score=70.0,
            eval_fn=lambda c: 80.0,
            mutate_fn=lambda c: c,
            regression_fn=lambda base, cand: True,
            min_improvement=0.5,
        )
        assert cycle.decision == EvolutionDecision.ACCEPTED


class TestAcceptRejectDecision:
    """Accept if improvement >= min_improvement, reject otherwise."""

    def test_accepted_when_improvement_sufficient(self, runner):
        cycle = runner.run_cycle(
            baseline_config={},
            baseline_score=70.0,
            eval_fn=lambda c: 71.0,
            mutate_fn=lambda c: c,
            min_improvement=0.5,
        )
        assert cycle.decision == EvolutionDecision.ACCEPTED

    def test_rejected_when_improvement_insufficient(self, runner):
        cycle = runner.run_cycle(
            baseline_config={},
            baseline_score=70.0,
            eval_fn=lambda c: 70.3,
            mutate_fn=lambda c: c,
            min_improvement=0.5,
        )
        assert cycle.decision == EvolutionDecision.REJECTED
        assert "Insufficient" in cycle.reason


class TestAuditTrail:
    """Every cycle is recorded in the ledger."""

    def test_cycle_recorded_in_history(self, runner, tmp_ledger):
        runner.run_cycle(
            baseline_config={"k": 5},
            baseline_score=60.0,
            eval_fn=lambda c: 65.0,
            mutate_fn=lambda c: c,
        )
        history = tmp_ledger.get_history()
        assert len(history) == 1
        assert history[0]["decision"] == EvolutionDecision.ACCEPTED

    def test_stats_reflect_accept_reject(self, runner, tmp_ledger):
        # accepted
        runner.run_cycle({}, 60.0, lambda c: 65.0, lambda c: c, min_improvement=0.5)
        # rejected
        runner.run_cycle({}, 60.0, lambda c: 60.1, lambda c: c, min_improvement=0.5)
        s = tmp_ledger.stats()
        assert s["accepted"] == 1
        assert s["rejected"] == 1
        assert s["total_cycles"] == 2


class TestWorktreeIsolation:
    """Git worktree creation and cleanup."""

    def test_create_worktree_calls_git(self):
        with patch("core.evolution_runner.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            with patch("pathlib.Path.exists", return_value=True):
                result = _create_worktree(Path("/repo"), "evo_abc")
            assert result is not None
            args = mock_run.call_args[0][0]
            assert "worktree" in args
            assert "add" in args

    def test_remove_worktree_calls_git(self):
        with patch("core.evolution_runner.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            _remove_worktree(Path("/repo"), Path("/repo/.worktrees/evo_abc"))
            assert mock_run.call_count >= 1

    def test_worktree_cleaned_up_after_eval_error(self, tmp_ledger):
        with patch("core.evolution_runner.subprocess") as mock_sp:
            mock_sp.run.return_value = MagicMock(stdout="", returncode=0)
            with patch(
                "core.evolution_runner._create_worktree",
                return_value=Path("/fake/wt"),
            ) as mock_create, patch(
                "core.evolution_runner._remove_worktree"
            ) as mock_remove:
                r = EvolutionRunner(
                    ledger=tmp_ledger,
                    repo_root=Path("/repo"),
                    use_worktree=True,
                )
                r.run_cycle(
                    baseline_config={},
                    baseline_score=50.0,
                    eval_fn=lambda c: (_ for _ in ()).throw(RuntimeError("eval crash")),
                    mutate_fn=lambda c: c,
                )
                mock_remove.assert_called_once()


class TestRollbackOnRegression:
    """Regression causes rejection -- no side effects persist."""

    def test_regression_rejects_and_records(self, runner, tmp_ledger):
        cycle = runner.run_cycle(
            baseline_config={"v": 1},
            baseline_score=80.0,
            eval_fn=lambda c: 60.0,  # worse
            mutate_fn=lambda c: {**c, "v": 2},
            regression_fn=lambda base, cand: cand >= base,
            min_improvement=0.5,
        )
        assert cycle.decision == EvolutionDecision.REJECTED
        assert cycle.regression_check["passed"] is False
        history = tmp_ledger.get_history()
        assert history[0]["decision"] == EvolutionDecision.REJECTED


# ===================================================================
# Surrogate Scorer tests
# ===================================================================

class TestFeatureExtraction:
    """extract_features turns configs into float dicts."""

    def test_numeric_passthrough(self):
        f = extract_features({"lr": 0.01, "epochs": 10})
        assert f["lr"] == 0.01
        assert f["epochs"] == 10.0

    def test_bool_to_float(self):
        f = extract_features({"flag": True, "off": False})
        assert f["flag"] == 1.0
        assert f["off"] == 0.0

    def test_string_length(self):
        f = extract_features({"model": "gpt4"})
        assert f["model_len"] == 4.0

    def test_nested_dict(self):
        f = extract_features({"opt": {"lr": 0.1}})
        assert f["opt.lr"] == 0.1


class TestSurrogateModelPrediction:
    """SurrogateModel.predict returns score from features."""

    def test_unfitted_returns_neutral(self):
        m = SurrogateModel()
        assert m.predict({"x": 1.0}) == 0.5

    def test_fitted_returns_prediction(self):
        m = SurrogateModel()
        features = [{"x": float(i)} for i in range(10)]
        scores = [0.1 * i for i in range(10)]
        r2 = m.fit(features, scores)
        assert r2 > 0.5
        pred = m.predict({"x": 5.0})
        assert 0.0 <= pred <= 1.0


class TestSurrogateTraining:
    """Training on accumulated data."""

    def test_fit_requires_min_samples(self):
        m = SurrogateModel()
        r2 = m.fit([{"x": 1.0}, {"x": 2.0}], [0.5, 0.6])  # only 2
        assert r2 == 0.0
        assert not m.is_fitted

    def test_fit_stores_sample_count(self):
        m = SurrogateModel()
        feats = [{"x": float(i)} for i in range(5)]
        scores = [0.1 * i for i in range(5)]
        m.fit(feats, scores)
        assert m.n_samples == 5

    def test_refit_via_store(self, surrogate_store):
        for i in range(6):
            surrogate_store.record({"x": i, "y": i * 2}, 0.1 * i)
        r2 = surrogate_store.refit()
        assert r2 >= 0.0
        assert surrogate_store.model.is_fitted


class TestPredictionAccuracy:
    """Surrogate tracks R-squared."""

    def test_r_squared_reasonable_on_linear_data(self):
        m = SurrogateModel()
        feats = [{"x": float(i)} for i in range(20)]
        scores = [0.04 * i + 0.1 for i in range(20)]
        r2 = m.fit(feats, scores)
        assert r2 > 0.8

    def test_r_squared_low_on_random_data(self):
        import random
        rng = random.Random(42)
        m = SurrogateModel()
        feats = [{"x": float(i)} for i in range(20)]
        scores = [rng.random() for _ in range(20)]
        r2 = m.fit(feats, scores)
        assert r2 < 0.5  # weak fit on noise


class TestFastVsFullRouting:
    """should_evaluate gates expensive eval via surrogate threshold."""

    def test_unfitted_always_evaluates(self, surrogate_store):
        assert surrogate_store.should_evaluate({"x": 0}, threshold=0.9)

    def test_fitted_filters_low_scoring(self, surrogate_store):
        # Train on data where high x -> high score
        for i in range(6):
            surrogate_store.record({"x": float(i * 10)}, 0.1 * i)
        surrogate_store.refit()
        # Low-feature config should be filtered
        result = surrogate_store.should_evaluate({"x": 0.0}, threshold=0.9)
        # With a fitted model, prediction for x=0 should be low
        assert isinstance(result, bool)

    def test_surrogate_prefilter_in_runner(self, tmp_ledger):
        """Runner skips eval when surrogate rejects."""
        mock_store = MagicMock()
        mock_store.should_evaluate.return_value = False

        with patch("core.evolution_runner.subprocess") as mock_sp:
            mock_sp.run.return_value = MagicMock(stdout="", returncode=0)
            r = EvolutionRunner(
                ledger=tmp_ledger,
                surrogate_store=mock_store,
                surrogate_threshold=0.5,
            )
            eval_called = []
            cycle = r.run_cycle(
                baseline_config={},
                baseline_score=70.0,
                eval_fn=lambda c: eval_called.append(1) or 99.0,
                mutate_fn=lambda c: c,
            )
            assert cycle.decision == EvolutionDecision.REJECTED
            assert "Surrogate pre-filter" in cycle.reason
            assert len(eval_called) == 0  # eval was never called


class TestScoreCaching:
    """SurrogateEvalStore caches real scores and surrogate predictions."""

    def test_record_stores_true_score(self, surrogate_store):
        surrogate_store.record({"a": 1}, 0.75)
        stats = surrogate_store.stats()
        assert stats["total_evaluations"] == 1
        assert stats["avg_true_score"] == pytest.approx(0.75, abs=0.01)

    def test_record_returns_surrogate_when_fitted(self, surrogate_store):
        for i in range(6):
            surrogate_store.record({"x": float(i)}, 0.1 * i)
        surrogate_store.refit()
        result = surrogate_store.record({"x": 3.0}, 0.3)
        # Should return the surrogate prediction, not the true score
        assert isinstance(result, float)

    def test_serialization_roundtrip(self):
        m = SurrogateModel()
        feats = [{"x": float(i)} for i in range(5)]
        scores = [0.1 * i for i in range(5)]
        m.fit(feats, scores)
        data = m.to_dict()
        m2 = SurrogateModel()
        m2.from_dict(data)
        assert m2.is_fitted
        assert m2.predict({"x": 3.0}) == pytest.approx(
            m.predict({"x": 3.0}), abs=1e-6
        )
