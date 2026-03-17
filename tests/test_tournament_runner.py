"""Tests for VM Tournament Mode (Sprint 22)."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict

import pytest

from core.tournament_runner import (
    CloneResult,
    TournamentLedger,
    TournamentResult,
    TournamentRound,
    TournamentRunner,
)
from core.evolution_runner import EvolutionCycle, EvolutionDecision


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tourn_ledger(tmp_path):
    led = TournamentLedger(tmp_path / "test_tourn.db")
    yield led
    led.close()


@pytest.fixture
def runner(tmp_path):
    r = TournamentRunner(
        ledger_dir=tmp_path / "evo",
        repo_root=tmp_path,
        max_workers=4,
    )
    yield r
    r.close()


def _baseline() -> Dict[str, Any]:
    return {"top_k": 5, "temperature": 0.7}


# ---------------------------------------------------------------------------
# CloneResult / TournamentRound data classes
# ---------------------------------------------------------------------------

class TestCloneResult:
    def test_create(self):
        cycle = EvolutionCycle(cycle_id="evo_1", started_at=1000.0)
        cr = CloneResult(clone_id="clone_000", cycle=cycle)
        assert cr.clone_id == "clone_000"
        assert cr.rank == 0

    def test_with_rank(self):
        cycle = EvolutionCycle(cycle_id="evo_2", started_at=1000.0)
        cr = CloneResult(clone_id="clone_001", cycle=cycle, rank=3)
        assert cr.rank == 3


class TestTournamentRound:
    def test_empty_round(self):
        r = TournamentRound(round_num=1)
        assert r.matchups == []
        assert r.survivors == []
        assert r.eliminated == []

    def test_populated_round(self):
        r = TournamentRound(
            round_num=2,
            matchups=[{"a": "c1", "b": "c2", "winner": "c1"}],
            survivors=["c1"],
            eliminated=["c2"],
        )
        assert len(r.matchups) == 1
        assert r.survivors == ["c1"]


class TestTournamentResult:
    def test_defaults(self):
        tr = TournamentResult(
            tournament_id="tourn_test", started_at=1000.0,
        )
        assert tr.num_clones == 0
        assert tr.champion_id == ""
        assert tr.rounds == []


# ---------------------------------------------------------------------------
# TournamentLedger
# ---------------------------------------------------------------------------

class TestTournamentLedger:
    def test_record_and_retrieve(self, tourn_ledger):
        result = TournamentResult(
            tournament_id="tourn_abc",
            started_at=1000.0,
            completed_at=1005.0,
            num_clones=10,
            champion_id="clone_003",
            champion_score=85.0,
            baseline_score=70.0,
            decision="accepted",
            reason="test",
        )
        cycle = EvolutionCycle(
            cycle_id="evo_1", started_at=1000.0,
            candidate_score=85.0, decision="accepted",
            candidate_config={"k": 10},
        )
        result.clone_results = [
            CloneResult(clone_id="clone_003", cycle=cycle, rank=1)
        ]

        tourn_ledger.record_tournament(result)

        history = tourn_ledger.get_history()
        assert len(history) == 1
        assert history[0]["tournament_id"] == "tourn_abc"
        assert history[0]["champion_id"] == "clone_003"
        assert history[0]["num_clones"] == 10

    def test_get_clone_results(self, tourn_ledger):
        cycle = EvolutionCycle(
            cycle_id="evo_x", started_at=1000.0,
            candidate_score=80.0, candidate_config={"k": 8},
        )
        result = TournamentResult(
            tournament_id="tourn_xyz",
            started_at=1000.0,
            clone_results=[
                CloneResult(clone_id="c1", cycle=cycle, rank=1),
            ],
        )
        tourn_ledger.record_tournament(result)

        clones = tourn_ledger.get_clone_results("tourn_xyz")
        assert len(clones) == 1
        assert clones[0]["clone_id"] == "c1"

    def test_stats_empty(self, tourn_ledger):
        s = tourn_ledger.stats()
        assert s["total_tournaments"] == 0

    def test_stats_populated(self, tourn_ledger):
        for i, d in enumerate(["accepted", "rejected", "accepted"]):
            result = TournamentResult(
                tournament_id=f"tourn_{i}",
                started_at=1000.0 + i,
                num_clones=10,
                decision=d,
            )
            tourn_ledger.record_tournament(result)

        s = tourn_ledger.stats()
        assert s["total_tournaments"] == 3
        assert s["accepted"] == 2


# ---------------------------------------------------------------------------
# TournamentRunner
# ---------------------------------------------------------------------------

class TestTournamentRunner:
    def test_accepted_tournament(self, runner):
        """Champion beats baseline -- accepted."""
        scores = iter([75.0, 80.0, 72.0, 85.0, 78.0])

        result = runner.run_tournament(
            num_clones=5,
            baseline_config=_baseline(),
            baseline_score=70.0,
            eval_fn=lambda cfg: next(scores),
            mutate_fn=lambda cfg: {**cfg, "top_k": cfg.get("top_k", 5) + 1},
            min_improvement=0.5,
        )
        assert result.decision == EvolutionDecision.ACCEPTED
        assert result.champion_score == 85.0
        assert result.num_clones == 5
        assert len(result.clone_results) == 5

    def test_rejected_no_improvement(self, runner):
        """All clones below threshold -- rejected."""
        result = runner.run_tournament(
            num_clones=4,
            baseline_config=_baseline(),
            baseline_score=70.0,
            eval_fn=lambda cfg: 70.1,
            mutate_fn=lambda cfg: {**cfg, "top_k": 10},
            min_improvement=1.0,
        )
        assert result.decision == EvolutionDecision.REJECTED
        assert "required" in result.reason

    def test_tournament_bracket_2_clones(self, runner):
        """Minimal 2-clone tournament."""
        scores = iter([75.0, 80.0])

        result = runner.run_tournament(
            num_clones=2,
            baseline_config=_baseline(),
            baseline_score=70.0,
            eval_fn=lambda cfg: next(scores),
            mutate_fn=lambda cfg: {**cfg, "top_k": 10},
        )
        assert result.decision == EvolutionDecision.ACCEPTED
        assert result.champion_score == 80.0
        assert len(result.rounds) >= 1

    def test_tournament_bracket_10_clones(self, runner):
        """Full 10-clone tournament with bracket elimination."""
        counter = {"n": 0}

        def scored_eval(cfg):
            counter["n"] += 1
            return 70.0 + counter["n"] * 2.0

        result = runner.run_tournament(
            num_clones=10,
            baseline_config=_baseline(),
            baseline_score=70.0,
            eval_fn=scored_eval,
            mutate_fn=lambda cfg: {**cfg, "top_k": cfg.get("top_k", 5) + 1},
            min_improvement=0.5,
        )
        assert result.decision == EvolutionDecision.ACCEPTED
        assert result.num_clones == 10
        assert len(result.clone_results) == 10
        assert len(result.rounds) >= 1
        assert result.completed_at > 0

    def test_ranking(self, runner):
        """Clones are ranked by score."""
        counter = {"n": 0}

        def scored_eval(cfg):
            counter["n"] += 1
            return 70.0 + counter["n"]

        result = runner.run_tournament(
            num_clones=5,
            baseline_config=_baseline(),
            baseline_score=70.0,
            eval_fn=scored_eval,
            mutate_fn=lambda cfg: {**cfg, "top_k": 10},
        )

        ranks = [cr.rank for cr in result.clone_results]
        assert sorted(ranks) == [1, 2, 3, 4, 5]
        # Rank 1 has the highest score
        rank1 = [cr for cr in result.clone_results if cr.rank == 1][0]
        for cr in result.clone_results:
            assert rank1.cycle.candidate_score >= cr.cycle.candidate_score

    def test_all_error_clones(self, runner):
        """All clones error out -- tournament errors."""
        def bad_eval(cfg):
            raise RuntimeError("boom")

        result = runner.run_tournament(
            num_clones=3,
            baseline_config=_baseline(),
            baseline_score=70.0,
            eval_fn=bad_eval,
            mutate_fn=lambda cfg: {**cfg, "top_k": 10},
        )
        assert result.decision == EvolutionDecision.ERROR
        assert "All clones failed" in result.reason

    def test_clone_count_capped(self, runner):
        """Clone count is capped at 100."""
        result = runner.run_tournament(
            num_clones=200,
            baseline_config=_baseline(),
            baseline_score=70.0,
            eval_fn=lambda cfg: 75.0,
            mutate_fn=lambda cfg: {**cfg, "top_k": 10},
        )
        assert result.num_clones == 100

    def test_clone_count_min(self, runner):
        """Clone count minimum is 2."""
        result = runner.run_tournament(
            num_clones=1,
            baseline_config=_baseline(),
            baseline_score=70.0,
            eval_fn=lambda cfg: 75.0,
            mutate_fn=lambda cfg: {**cfg, "top_k": 10},
        )
        assert result.num_clones == 2

    def test_history_recorded(self, runner):
        """Tournament is recorded in history."""
        runner.run_tournament(
            num_clones=3,
            baseline_config=_baseline(),
            baseline_score=70.0,
            eval_fn=lambda cfg: 75.0,
            mutate_fn=lambda cfg: {**cfg, "top_k": 10},
        )

        history = runner.get_history()
        assert len(history) == 1

    def test_stats(self, runner):
        """Stats reflect tournaments."""
        runner.run_tournament(
            num_clones=4,
            baseline_config=_baseline(),
            baseline_score=70.0,
            eval_fn=lambda cfg: 75.0,
            mutate_fn=lambda cfg: {**cfg, "top_k": 10},
        )

        stats = runner.stats()
        assert stats["total_tournaments"] == 1

    def test_regression_fn_forwarded(self, runner):
        """Regression function is forwarded to each clone."""
        result = runner.run_tournament(
            num_clones=3,
            baseline_config=_baseline(),
            baseline_score=70.0,
            eval_fn=lambda cfg: 80.0,
            mutate_fn=lambda cfg: {**cfg, "top_k": 10},
            regression_fn=lambda base, cand: False,  # All fail regression
        )
        # All clones rejected by regression
        viable = [
            cr for cr in result.clone_results
            if cr.cycle.decision != EvolutionDecision.ERROR
        ]
        for cr in viable:
            assert cr.cycle.decision == EvolutionDecision.REJECTED

    def test_tournament_id_format(self, runner):
        """Tournament IDs follow tourn_ prefix."""
        result = runner.run_tournament(
            num_clones=2,
            baseline_config=_baseline(),
            baseline_score=70.0,
            eval_fn=lambda cfg: 75.0,
            mutate_fn=lambda cfg: {**cfg, "top_k": 10},
        )
        assert result.tournament_id.startswith("tourn_")

    def test_champion_config_captured(self, runner):
        """Champion config is captured in result."""
        result = runner.run_tournament(
            num_clones=2,
            baseline_config=_baseline(),
            baseline_score=70.0,
            eval_fn=lambda cfg: 75.0,
            mutate_fn=lambda cfg: {**cfg, "top_k": 99},
        )
        assert result.champion_config.get("top_k") == 99
