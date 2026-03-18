"""Extended tests for VM Tournament Mode (Sprint 22).

Covers: initialization, round-robin matchups, scoring/ranking, elimination,
champion selection, config mutation, worktree isolation, regression gate,
audit trail, and edge cases -- all fully mocked.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

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
# Helpers
# ---------------------------------------------------------------------------

def _baseline() -> Dict[str, Any]:
    return {"top_k": 5, "temperature": 0.7}


def _make_clone(clone_id: str, score: float, decision: str = "accepted",
                config: Dict[str, Any] | None = None) -> CloneResult:
    """Build a CloneResult with the given score."""
    return CloneResult(
        clone_id=clone_id,
        cycle=EvolutionCycle(
            cycle_id=f"evo_{clone_id}",
            started_at=1000.0,
            completed_at=1001.0,
            candidate_score=score,
            decision=decision,
            candidate_config=config or {"top_k": 10},
        ),
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tourn_ledger(tmp_path):
    led = TournamentLedger(tmp_path / "ext_tourn.db")
    yield led
    led.close()


@pytest.fixture
def runner(tmp_path):
    r = TournamentRunner(
        ledger_dir=tmp_path / "evo_ext",
        repo_root=tmp_path,
        max_workers=2,
    )
    yield r
    r.close()


# ---------------------------------------------------------------------------
# 1. Tournament initialization with N candidates
# ---------------------------------------------------------------------------

class TestInitialization:
    def test_n_candidates_spawned(self, runner):
        """All N candidates are spawned and returned."""
        n = 6
        result = runner.run_tournament(
            num_clones=n,
            baseline_config=_baseline(),
            baseline_score=50.0,
            eval_fn=lambda cfg: 55.0,
            mutate_fn=lambda cfg: {**cfg, "top_k": 20},
        )
        assert result.num_clones == n
        assert len(result.clone_results) == n

    def test_tournament_id_unique(self, runner):
        """Two tournaments get distinct IDs."""
        ids = set()
        for _ in range(3):
            r = runner.run_tournament(
                num_clones=2,
                baseline_config=_baseline(),
                baseline_score=50.0,
                eval_fn=lambda cfg: 55.0,
                mutate_fn=lambda cfg: {**cfg, "top_k": 10},
            )
            ids.add(r.tournament_id)
        assert len(ids) == 3

    def test_started_at_populated(self, runner):
        result = runner.run_tournament(
            num_clones=2,
            baseline_config=_baseline(),
            baseline_score=50.0,
            eval_fn=lambda cfg: 55.0,
            mutate_fn=lambda cfg: cfg,
        )
        assert result.started_at > 0
        assert result.completed_at >= result.started_at


# ---------------------------------------------------------------------------
# 2. Round-robin / bracket matchup generation
# ---------------------------------------------------------------------------

class TestMatchupGeneration:
    def test_bracket_produces_matchups(self, runner):
        """Bracket rounds contain matchup dicts with a/b/winner keys."""
        counter = {"n": 0}
        def scored(cfg):
            counter["n"] += 1
            return 50.0 + counter["n"] * 3.0

        result = runner.run_tournament(
            num_clones=4,
            baseline_config=_baseline(),
            baseline_score=50.0,
            eval_fn=scored,
            mutate_fn=lambda cfg: {**cfg, "top_k": 10},
        )
        assert len(result.rounds) >= 1
        for rnd in result.rounds:
            for m in rnd.matchups:
                assert "a" in m and "b" in m and "winner" in m

    def test_odd_clone_gets_bye(self, runner):
        """With an odd number of viable clones the last one auto-advances."""
        counter = {"n": 0}
        def scored(cfg):
            counter["n"] += 1
            return 50.0 + counter["n"]

        result = runner.run_tournament(
            num_clones=5,
            baseline_config=_baseline(),
            baseline_score=50.0,
            eval_fn=scored,
            mutate_fn=lambda cfg: {**cfg, "top_k": 10},
        )
        # At least one round must have more survivors than matchups
        has_bye = any(
            len(rnd.survivors) > len(rnd.matchups) for rnd in result.rounds
        )
        assert has_bye

    def test_bracket_internal_directly(self):
        """Directly test _tournament_bracket with crafted clones."""
        tr = TournamentRunner.__new__(TournamentRunner)
        clones = [_make_clone(f"c{i}", score=float(i * 10)) for i in range(4)]
        champion, rounds = tr._tournament_bracket(clones)
        assert champion.clone_id == "c3"  # highest score
        assert len(rounds) >= 1


# ---------------------------------------------------------------------------
# 3. Scoring and ranking after rounds
# ---------------------------------------------------------------------------

class TestScoringAndRanking:
    def test_ranks_assigned_1_to_n(self, runner):
        counter = {"n": 0}
        def scored(cfg):
            counter["n"] += 1
            return 50.0 + counter["n"]

        result = runner.run_tournament(
            num_clones=4,
            baseline_config=_baseline(),
            baseline_score=50.0,
            eval_fn=scored,
            mutate_fn=lambda cfg: {**cfg, "top_k": 10},
        )
        ranks = sorted(cr.rank for cr in result.clone_results)
        assert ranks == [1, 2, 3, 4]

    def test_rank1_has_highest_score(self, runner):
        counter = {"n": 0}
        def scored(cfg):
            counter["n"] += 1
            return 50.0 + counter["n"] * 5.0

        result = runner.run_tournament(
            num_clones=5,
            baseline_config=_baseline(),
            baseline_score=50.0,
            eval_fn=scored,
            mutate_fn=lambda cfg: cfg,
        )
        top = max(result.clone_results, key=lambda c: c.cycle.candidate_score)
        assert top.rank == 1


# ---------------------------------------------------------------------------
# 4. Elimination of worst performers
# ---------------------------------------------------------------------------

class TestElimination:
    def test_eliminated_list_populated(self, runner):
        counter = {"n": 0}
        def scored(cfg):
            counter["n"] += 1
            return 50.0 + counter["n"]

        result = runner.run_tournament(
            num_clones=4,
            baseline_config=_baseline(),
            baseline_score=50.0,
            eval_fn=scored,
            mutate_fn=lambda cfg: cfg,
        )
        total_eliminated = sum(len(r.eliminated) for r in result.rounds)
        assert total_eliminated >= 1

    def test_survivors_shrink_each_round(self, runner):
        counter = {"n": 0}
        def scored(cfg):
            counter["n"] += 1
            return 50.0 + counter["n"]

        result = runner.run_tournament(
            num_clones=8,
            baseline_config=_baseline(),
            baseline_score=50.0,
            eval_fn=scored,
            mutate_fn=lambda cfg: cfg,
        )
        if len(result.rounds) >= 2:
            assert (len(result.rounds[0].survivors)
                    >= len(result.rounds[1].survivors))


# ---------------------------------------------------------------------------
# 5. Champion selection
# ---------------------------------------------------------------------------

class TestChampionSelection:
    def test_champion_is_highest_scorer(self, runner):
        scores = [60.0, 90.0, 70.0, 80.0]
        it = iter(scores)
        result = runner.run_tournament(
            num_clones=4,
            baseline_config=_baseline(),
            baseline_score=50.0,
            eval_fn=lambda cfg: next(it),
            mutate_fn=lambda cfg: cfg,
        )
        assert result.champion_score == 90.0

    def test_champion_accepted_above_threshold(self, runner):
        result = runner.run_tournament(
            num_clones=3,
            baseline_config=_baseline(),
            baseline_score=50.0,
            eval_fn=lambda cfg: 60.0,
            mutate_fn=lambda cfg: cfg,
            min_improvement=5.0,
        )
        assert result.decision == EvolutionDecision.ACCEPTED

    def test_champion_rejected_below_threshold(self, runner):
        result = runner.run_tournament(
            num_clones=3,
            baseline_config=_baseline(),
            baseline_score=50.0,
            eval_fn=lambda cfg: 50.3,
            mutate_fn=lambda cfg: cfg,
            min_improvement=1.0,
        )
        assert result.decision == EvolutionDecision.REJECTED


# ---------------------------------------------------------------------------
# 6. Config mutation between rounds
# ---------------------------------------------------------------------------

class TestConfigMutation:
    def test_mutate_fn_called_per_clone(self, runner):
        call_count = {"n": 0}
        def counting_mutate(cfg):
            call_count["n"] += 1
            return {**cfg, "top_k": cfg["top_k"] + call_count["n"]}

        runner.run_tournament(
            num_clones=4,
            baseline_config=_baseline(),
            baseline_score=50.0,
            eval_fn=lambda cfg: 55.0,
            mutate_fn=counting_mutate,
        )
        assert call_count["n"] == 4

    def test_champion_config_reflects_mutation(self, runner):
        result = runner.run_tournament(
            num_clones=2,
            baseline_config=_baseline(),
            baseline_score=50.0,
            eval_fn=lambda cfg: 60.0,
            mutate_fn=lambda cfg: {**cfg, "top_k": 42},
        )
        assert result.champion_config["top_k"] == 42


# ---------------------------------------------------------------------------
# 7. Git worktree isolation per candidate (mocked)
# ---------------------------------------------------------------------------

class TestWorktreeIsolation:
    @patch("core.evolution_runner._create_worktree", return_value=Path("/tmp/wt"))
    @patch("core.evolution_runner._remove_worktree")
    def test_worktree_created_when_enabled(self, mock_remove, mock_create,
                                           tmp_path):
        """When use_worktree=True on the EvolutionRunner, worktree helpers
        are invoked. TournamentRunner itself sets use_worktree=False by
        default, so we verify the flag propagation at the runner level."""
        from core.evolution_runner import EvolutionRunner, EvolutionLedger
        ledger = EvolutionLedger(tmp_path / "wt_test.db")
        evo = EvolutionRunner(
            ledger=ledger, repo_root=tmp_path, use_worktree=True,
        )
        cycle = evo.run_cycle(
            baseline_config=_baseline(),
            baseline_score=50.0,
            eval_fn=lambda cfg: 55.0,
            mutate_fn=lambda cfg: cfg,
        )
        assert mock_create.called
        assert mock_remove.called
        ledger.close()

    def test_tournament_clones_default_no_worktree(self, runner):
        """TournamentRunner spawns clones with use_worktree=False."""
        with patch("core.evolution_runner._create_worktree") as mock_create:
            runner.run_tournament(
                num_clones=2,
                baseline_config=_baseline(),
                baseline_score=50.0,
                eval_fn=lambda cfg: 55.0,
                mutate_fn=lambda cfg: cfg,
            )
            mock_create.assert_not_called()


# ---------------------------------------------------------------------------
# 8. Regression gate (candidate must pass tests)
# ---------------------------------------------------------------------------

class TestRegressionGate:
    def test_regression_pass_allows_acceptance(self, runner):
        result = runner.run_tournament(
            num_clones=3,
            baseline_config=_baseline(),
            baseline_score=50.0,
            eval_fn=lambda cfg: 60.0,
            mutate_fn=lambda cfg: cfg,
            regression_fn=lambda base, cand: True,
            min_improvement=0.5,
        )
        assert result.decision == EvolutionDecision.ACCEPTED

    def test_regression_fail_rejects_all(self, runner):
        result = runner.run_tournament(
            num_clones=3,
            baseline_config=_baseline(),
            baseline_score=50.0,
            eval_fn=lambda cfg: 60.0,
            mutate_fn=lambda cfg: cfg,
            regression_fn=lambda base, cand: False,
        )
        viable = [
            cr for cr in result.clone_results
            if cr.cycle.decision != EvolutionDecision.ERROR
        ]
        for cr in viable:
            assert cr.cycle.decision == EvolutionDecision.REJECTED


# ---------------------------------------------------------------------------
# 9. Immutable audit trail logging
# ---------------------------------------------------------------------------

class TestAuditTrail:
    def test_tournament_persisted_in_ledger(self, tourn_ledger):
        result = TournamentResult(
            tournament_id="tourn_audit1",
            started_at=1000.0,
            completed_at=1010.0,
            num_clones=4,
            champion_id="c2",
            champion_score=88.0,
            baseline_score=70.0,
            decision="accepted",
            reason="improved",
            rounds=[TournamentRound(
                round_num=1,
                matchups=[{"a": "c1", "b": "c2", "winner": "c2"}],
                survivors=["c2"],
                eliminated=["c1"],
            )],
            clone_results=[_make_clone("c2", 88.0)],
        )
        tourn_ledger.record_tournament(result)
        history = tourn_ledger.get_history()
        assert history[0]["tournament_id"] == "tourn_audit1"
        assert history[0]["champion_score"] == 88.0

    def test_clone_results_persisted(self, tourn_ledger):
        clones = [_make_clone("cx", 77.0), _make_clone("cy", 82.0)]
        result = TournamentResult(
            tournament_id="tourn_audit2",
            started_at=1000.0,
            clone_results=clones,
        )
        tourn_ledger.record_tournament(result)
        rows = tourn_ledger.get_clone_results("tourn_audit2")
        assert len(rows) == 2
        ids = {r["clone_id"] for r in rows}
        assert ids == {"cx", "cy"}

    def test_multiple_tournaments_ordered(self, tourn_ledger):
        for i in range(3):
            tourn_ledger.record_tournament(TournamentResult(
                tournament_id=f"tourn_ord_{i}",
                started_at=1000.0 + i,
            ))
        history = tourn_ledger.get_history()
        assert len(history) == 3
        # Most recent first
        assert history[0]["tournament_id"] == "tourn_ord_2"

    def test_stats_reflect_decisions(self, tourn_ledger):
        for i, d in enumerate(["accepted", "rejected", "accepted", "error"]):
            tourn_ledger.record_tournament(TournamentResult(
                tournament_id=f"tourn_stat_{i}",
                started_at=1000.0 + i,
                num_clones=5,
                decision=d,
            ))
        s = tourn_ledger.stats()
        assert s["total_tournaments"] == 4
        assert s["accepted"] == 2
        assert s["avg_clones"] == 5.0


# ---------------------------------------------------------------------------
# 10. Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_single_candidate_clamped_to_two(self, runner):
        """num_clones=1 is clamped to 2."""
        result = runner.run_tournament(
            num_clones=1,
            baseline_config=_baseline(),
            baseline_score=50.0,
            eval_fn=lambda cfg: 55.0,
            mutate_fn=lambda cfg: cfg,
        )
        assert result.num_clones == 2
        assert len(result.clone_results) == 2

    def test_all_candidates_equal_score(self, runner):
        """When every clone scores identically, one is still chosen."""
        result = runner.run_tournament(
            num_clones=6,
            baseline_config=_baseline(),
            baseline_score=50.0,
            eval_fn=lambda cfg: 60.0,
            mutate_fn=lambda cfg: cfg,
        )
        assert result.champion_score == 60.0
        assert result.champion_id != ""
        assert result.decision == EvolutionDecision.ACCEPTED

    def test_zero_rounds_all_error(self, runner):
        """If all clones error, no bracket rounds are produced."""
        def boom(cfg):
            raise RuntimeError("fail")

        result = runner.run_tournament(
            num_clones=4,
            baseline_config=_baseline(),
            baseline_score=50.0,
            eval_fn=boom,
            mutate_fn=lambda cfg: cfg,
        )
        assert result.decision == EvolutionDecision.ERROR
        assert result.rounds == []

    def test_large_clone_count_capped(self, runner):
        result = runner.run_tournament(
            num_clones=999,
            baseline_config=_baseline(),
            baseline_score=50.0,
            eval_fn=lambda cfg: 55.0,
            mutate_fn=lambda cfg: cfg,
        )
        assert result.num_clones == 100

    def test_bracket_single_viable_after_errors(self, runner):
        """Only one clone succeeds; it becomes champion by default."""
        call_count = {"n": 0}
        def one_succeeds(cfg):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return 80.0
            raise RuntimeError("fail")

        result = runner.run_tournament(
            num_clones=3,
            baseline_config=_baseline(),
            baseline_score=50.0,
            eval_fn=one_succeeds,
            mutate_fn=lambda cfg: cfg,
        )
        # One viable clone should still produce a result (not ERROR)
        viable = [
            cr for cr in result.clone_results
            if cr.cycle.decision != EvolutionDecision.ERROR
        ]
        assert len(viable) >= 1
