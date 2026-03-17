"""Tests for Weekly Software Evolution Engine (Sprint 21)."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict

import pytest

from core.evolution_runner import (
    EvolutionCycle,
    EvolutionDecision,
    EvolutionLedger,
    EvolutionRunner,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ledger(tmp_path):
    led = EvolutionLedger(tmp_path / "test_ledger.db")
    yield led
    led.close()


@pytest.fixture
def runner(tmp_path):
    r = EvolutionRunner(
        ledger_path=tmp_path / "test_runner.db",
        repo_root=tmp_path,
        use_worktree=False,
    )
    yield r
    r.close()


def _baseline_config() -> Dict[str, Any]:
    return {"top_k": 5, "temperature": 0.7, "model": "phi4:14b"}


# ---------------------------------------------------------------------------
# EvolutionDecision
# ---------------------------------------------------------------------------

class TestEvolutionDecision:
    def test_accepted_value(self):
        assert EvolutionDecision.ACCEPTED == "accepted"

    def test_rejected_value(self):
        assert EvolutionDecision.REJECTED == "rejected"

    def test_error_value(self):
        assert EvolutionDecision.ERROR == "error"


# ---------------------------------------------------------------------------
# EvolutionCycle dataclass
# ---------------------------------------------------------------------------

class TestEvolutionCycle:
    def test_create_minimal(self):
        c = EvolutionCycle(cycle_id="evo_test1", started_at=time.time())
        assert c.cycle_id == "evo_test1"
        assert c.completed_at == 0.0
        assert c.decision == ""

    def test_create_full(self):
        c = EvolutionCycle(
            cycle_id="evo_full",
            started_at=1000.0,
            completed_at=1005.0,
            baseline_score=80.0,
            candidate_score=85.0,
            decision=EvolutionDecision.ACCEPTED,
            reason="Improvement: +5.00",
            baseline_config={"k": 5},
            candidate_config={"k": 10},
            eval_results={"delta": 5.0},
            regression_check={"passed": True},
        )
        assert c.candidate_score == 85.0
        assert c.decision == "accepted"

    def test_default_dicts(self):
        c = EvolutionCycle(cycle_id="evo_d", started_at=0.0)
        assert c.baseline_config == {}
        assert c.candidate_config == {}
        assert c.eval_results == {}
        assert c.regression_check == {}


# ---------------------------------------------------------------------------
# EvolutionLedger
# ---------------------------------------------------------------------------

class TestEvolutionLedger:
    def test_record_and_retrieve(self, ledger):
        cycle = EvolutionCycle(
            cycle_id="evo_abc",
            started_at=1000.0,
            completed_at=1005.0,
            baseline_score=70.0,
            candidate_score=75.0,
            decision=EvolutionDecision.ACCEPTED,
            reason="test",
        )
        ledger.record_cycle(cycle)

        history = ledger.get_history(limit=10)
        assert len(history) == 1
        assert history[0]["cycle_id"] == "evo_abc"
        assert history[0]["decision"] == "accepted"

    def test_archive_baseline(self, ledger):
        bid = ledger.archive_baseline("evo_1", {"k": 5}, 80.0)
        assert bid.startswith("base_")

        baselines = ledger.get_baselines()
        assert len(baselines) == 1
        assert baselines[0]["config"] == {"k": 5}
        assert baselines[0]["score"] == 80.0

    def test_multiple_cycles_ordering(self, ledger):
        for i in range(5):
            cycle = EvolutionCycle(
                cycle_id=f"evo_{i}",
                started_at=1000.0 + i,
                completed_at=1005.0 + i,
                decision=EvolutionDecision.ACCEPTED if i % 2 == 0 else EvolutionDecision.REJECTED,
            )
            ledger.record_cycle(cycle)

        history = ledger.get_history(limit=3)
        assert len(history) == 3
        # Most recent first
        assert history[0]["cycle_id"] == "evo_4"

    def test_stats_empty(self, ledger):
        s = ledger.stats()
        assert s["total_cycles"] == 0
        assert s["accepted"] == 0
        assert s["acceptance_rate"] == 0.0

    def test_stats_populated(self, ledger):
        decisions = ["accepted", "accepted", "rejected", "error", "accepted"]
        for i, d in enumerate(decisions):
            cycle = EvolutionCycle(
                cycle_id=f"evo_{i}",
                started_at=1000.0 + i,
                decision=d,
            )
            ledger.record_cycle(cycle)

        s = ledger.stats()
        assert s["total_cycles"] == 5
        assert s["accepted"] == 3
        assert s["rejected"] == 1
        assert s["errors"] == 1
        assert s["acceptance_rate"] == 0.6

    def test_history_limit_cap(self, ledger):
        # Request huge limit -- capped internally
        history = ledger.get_history(limit=999999)
        assert isinstance(history, list)

    def test_baselines_limit_cap(self, ledger):
        baselines = ledger.get_baselines(limit=999999)
        assert isinstance(baselines, list)

    def test_record_cycle_with_json_fields(self, ledger):
        cycle = EvolutionCycle(
            cycle_id="evo_json",
            started_at=1000.0,
            baseline_config={"nested": {"key": "value"}},
            candidate_config={"list": [1, 2, 3]},
            eval_results={"score": 95.5},
            regression_check={"passed": True, "details": "all good"},
        )
        ledger.record_cycle(cycle)

        history = ledger.get_history()
        assert len(history) == 1
        assert history[0]["cycle_id"] == "evo_json"


# ---------------------------------------------------------------------------
# EvolutionRunner
# ---------------------------------------------------------------------------

class TestEvolutionRunner:
    def test_accepted_cycle(self, runner):
        """Candidate beats baseline by enough -- accepted."""
        result = runner.run_cycle(
            baseline_config=_baseline_config(),
            baseline_score=70.0,
            eval_fn=lambda cfg: 75.0,
            mutate_fn=lambda cfg: {**cfg, "top_k": 10},
            min_improvement=0.5,
        )
        assert result.decision == EvolutionDecision.ACCEPTED
        assert result.candidate_score == 75.0
        assert result.completed_at > 0

    def test_rejected_insufficient_improvement(self, runner):
        """Candidate improves but not enough -- rejected."""
        result = runner.run_cycle(
            baseline_config=_baseline_config(),
            baseline_score=70.0,
            eval_fn=lambda cfg: 70.1,
            mutate_fn=lambda cfg: {**cfg, "top_k": 10},
            min_improvement=0.5,
        )
        assert result.decision == EvolutionDecision.REJECTED
        assert "Insufficient" in result.reason

    def test_rejected_regression(self, runner):
        """Candidate fails regression gate -- rejected."""
        result = runner.run_cycle(
            baseline_config=_baseline_config(),
            baseline_score=70.0,
            eval_fn=lambda cfg: 80.0,
            mutate_fn=lambda cfg: {**cfg, "top_k": 10},
            regression_fn=lambda base, cand: False,  # Always fails
            min_improvement=0.5,
        )
        assert result.decision == EvolutionDecision.REJECTED
        assert "regression" in result.reason.lower()

    def test_regression_passes(self, runner):
        """Candidate passes regression gate and improvement -- accepted."""
        result = runner.run_cycle(
            baseline_config=_baseline_config(),
            baseline_score=70.0,
            eval_fn=lambda cfg: 80.0,
            mutate_fn=lambda cfg: {**cfg, "top_k": 10},
            regression_fn=lambda base, cand: True,
            min_improvement=0.5,
        )
        assert result.decision == EvolutionDecision.ACCEPTED

    def test_mutation_error(self, runner):
        """Mutation function throws -- error recorded."""
        result = runner.run_cycle(
            baseline_config=_baseline_config(),
            baseline_score=70.0,
            eval_fn=lambda cfg: 80.0,
            mutate_fn=lambda cfg: (_ for _ in ()).throw(ValueError("bad mutation")),
            min_improvement=0.5,
        )
        assert result.decision == EvolutionDecision.ERROR
        assert "Mutation failed" in result.reason

    def test_eval_error(self, runner):
        """Eval function throws -- error recorded."""
        def bad_eval(cfg):
            raise RuntimeError("eval crashed")

        result = runner.run_cycle(
            baseline_config=_baseline_config(),
            baseline_score=70.0,
            eval_fn=bad_eval,
            mutate_fn=lambda cfg: {**cfg, "top_k": 10},
            min_improvement=0.5,
        )
        assert result.decision == EvolutionDecision.ERROR
        assert "Evaluation failed" in result.reason

    def test_baseline_archived(self, runner):
        """Baseline is archived before mutation."""
        runner.run_cycle(
            baseline_config=_baseline_config(),
            baseline_score=70.0,
            eval_fn=lambda cfg: 75.0,
            mutate_fn=lambda cfg: {**cfg, "top_k": 10},
        )

        baselines = runner.get_baselines()
        assert len(baselines) == 1
        assert baselines[0]["score"] == 70.0

    def test_history_recorded(self, runner):
        """Cycle is recorded in history."""
        runner.run_cycle(
            baseline_config=_baseline_config(),
            baseline_score=70.0,
            eval_fn=lambda cfg: 75.0,
            mutate_fn=lambda cfg: {**cfg, "top_k": 10},
        )

        history = runner.get_history()
        assert len(history) == 1
        assert history[0]["decision"] == "accepted"

    def test_stats_after_cycles(self, runner):
        """Stats reflect multiple cycles."""
        # Run 3 cycles: 2 accepted, 1 rejected
        for score in [75.0, 80.0, 70.1]:
            runner.run_cycle(
                baseline_config=_baseline_config(),
                baseline_score=70.0,
                eval_fn=lambda cfg, s=score: s,
                mutate_fn=lambda cfg: {**cfg, "top_k": 10},
                min_improvement=0.5,
            )

        stats = runner.stats()
        assert stats["total_cycles"] == 3
        assert stats["accepted"] == 2
        assert stats["rejected"] == 1
        assert stats["baselines_archived"] == 3

    def test_eval_results_recorded(self, runner):
        """Eval results capture baseline, candidate, and delta."""
        result = runner.run_cycle(
            baseline_config=_baseline_config(),
            baseline_score=70.0,
            eval_fn=lambda cfg: 85.0,
            mutate_fn=lambda cfg: {**cfg, "top_k": 10},
        )
        assert result.eval_results["baseline"] == 70.0
        assert result.eval_results["candidate"] == 85.0
        assert result.eval_results["delta"] == 15.0

    def test_regression_check_recorded(self, runner):
        """Regression check results are captured."""
        result = runner.run_cycle(
            baseline_config=_baseline_config(),
            baseline_score=70.0,
            eval_fn=lambda cfg: 85.0,
            mutate_fn=lambda cfg: {**cfg, "top_k": 10},
            regression_fn=lambda base, cand: True,
        )
        assert result.regression_check["passed"] is True
        assert result.regression_check["baseline"] == 70.0
        assert result.regression_check["candidate"] == 85.0

    def test_cycle_id_format(self, runner):
        """Cycle IDs follow evo_ prefix convention."""
        result = runner.run_cycle(
            baseline_config=_baseline_config(),
            baseline_score=70.0,
            eval_fn=lambda cfg: 75.0,
            mutate_fn=lambda cfg: {**cfg, "top_k": 10},
        )
        assert result.cycle_id.startswith("evo_")

    def test_no_worktree_by_default(self, runner):
        """Worktree path is empty when use_worktree=False."""
        result = runner.run_cycle(
            baseline_config=_baseline_config(),
            baseline_score=70.0,
            eval_fn=lambda cfg: 75.0,
            mutate_fn=lambda cfg: {**cfg, "top_k": 10},
        )
        assert result.worktree_path == ""


# ---------------------------------------------------------------------------
# Surrogate integration
# ---------------------------------------------------------------------------

class TestSurrogateIntegration:
    """Tests for surrogate pre-filter wired into EvolutionRunner."""

    def test_surrogate_rejects_bad_candidate(self, tmp_path):
        """Surrogate that always says 'no' should prevent eval_fn call."""
        class FakeSurrogate:
            def should_evaluate(self, config, threshold):
                return False  # always reject
            def record(self, config, score):
                pass

        eval_called = []
        def tracking_eval(cfg):
            eval_called.append(True)
            return 90.0

        r = EvolutionRunner(
            ledger_path=tmp_path / "test.db",
            repo_root=tmp_path,
            surrogate_store=FakeSurrogate(),
            surrogate_threshold=0.3,
        )
        result = r.run_cycle(
            baseline_config=_baseline_config(),
            baseline_score=70.0,
            eval_fn=tracking_eval,
            mutate_fn=lambda cfg: {**cfg, "top_k": 10},
        )
        assert result.decision == EvolutionDecision.REJECTED
        assert "Surrogate" in result.reason
        assert len(eval_called) == 0  # eval_fn never called
        r.close()

    def test_surrogate_passes_good_candidate(self, tmp_path):
        """Surrogate that says 'yes' should let eval_fn run."""
        class FakeSurrogate:
            def should_evaluate(self, config, threshold):
                return True
            def record(self, config, score):
                self.last_recorded = score

        surrogate = FakeSurrogate()
        r = EvolutionRunner(
            ledger_path=tmp_path / "test.db",
            repo_root=tmp_path,
            surrogate_store=surrogate,
            surrogate_threshold=0.3,
        )
        result = r.run_cycle(
            baseline_config=_baseline_config(),
            baseline_score=70.0,
            eval_fn=lambda cfg: 80.0,
            mutate_fn=lambda cfg: {**cfg, "top_k": 10},
            min_improvement=0.5,
        )
        assert result.decision == EvolutionDecision.ACCEPTED
        assert result.candidate_score == 80.0
        assert surrogate.last_recorded == 80.0
        r.close()

    def test_no_surrogate_runs_normally(self, tmp_path):
        """Without surrogate, evolution runs as before."""
        r = EvolutionRunner(
            ledger_path=tmp_path / "test.db",
            repo_root=tmp_path,
        )
        result = r.run_cycle(
            baseline_config=_baseline_config(),
            baseline_score=70.0,
            eval_fn=lambda cfg: 75.0,
            mutate_fn=lambda cfg: {**cfg, "top_k": 10},
            min_improvement=0.5,
        )
        assert result.decision == EvolutionDecision.ACCEPTED
        r.close()

    def test_real_surrogate_store_integration(self, tmp_path):
        """Wire a real SurrogateEvalStore into the runner."""
        from core.surrogate_scorer import SurrogateEvalStore

        store = SurrogateEvalStore(
            db_path=tmp_path / "surrogate.db",
            refit_interval=100,
            min_samples=3,
        )

        r = EvolutionRunner(
            ledger_path=tmp_path / "test.db",
            repo_root=tmp_path,
            surrogate_store=store,
            surrogate_threshold=0.3,
        )

        # Unfitted surrogate should let everything through
        result = r.run_cycle(
            baseline_config=_baseline_config(),
            baseline_score=70.0,
            eval_fn=lambda cfg: 75.0,
            mutate_fn=lambda cfg: {**cfg, "top_k": 10},
            min_improvement=0.5,
        )
        assert result.decision == EvolutionDecision.ACCEPTED
        assert store.stats()["total_evaluations"] == 1
        r.close()
        store.close()
