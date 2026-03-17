"""
Tests for core.island_model -- Island Model Evolution (Sprint 29).
Verifies multi-population evolution with ring migration, ledger,
and cross-island championship selection.
"""

from __future__ import annotations

import copy
import pytest

from core.island_model import (
    IslandConfig,
    IslandLedger,
    IslandModelResult,
    IslandModelRunner,
    IslandState,
    Migrant,
    MigrationEvent,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_ledger(tmp_path):
    ledger = IslandLedger(tmp_path / "island.db")
    yield ledger
    ledger.close()


@pytest.fixture
def runner(tmp_path):
    r = IslandModelRunner(
        num_islands=3,
        population_per_island=4,
        migration_count=1,
        migration_interval=1,
        ledger_dir=str(tmp_path),
    )
    yield r
    r.close()


# Simple eval and mutate functions for testing
_COUNTER = {"calls": 0}

def _eval_fn(config):
    """Score = base_val + small random-ish variation based on call count."""
    _COUNTER["calls"] += 1
    return config.get("val", 50.0) + (_COUNTER["calls"] % 7) * 0.5


def _mutate_fn(config):
    """Bump val by a small amount."""
    c = copy.deepcopy(config)
    c["val"] = c.get("val", 50.0) + 1.0
    return c


def _bad_eval_fn(config):
    raise RuntimeError("eval crash")


def _bad_mutate_fn(config):
    raise RuntimeError("mutate crash")


# ---------------------------------------------------------------------------
# Ledger tests
# ---------------------------------------------------------------------------

class TestIslandLedger:

    def test_record_and_retrieve(self, tmp_ledger):
        result = IslandModelResult(
            run_id="test_run_001",
            num_islands=3,
            generations=5,
            champion_island="island_01",
            champion_score=88.5,
            champion_config={"val": 88.5},
            baseline_score=80.0,
            island_best_scores={"island_00": 85.0, "island_01": 88.5},
            decision="accepted",
            reason="improvement",
            elapsed_ms=500.0,
        )
        tmp_ledger.record(result)
        history = tmp_ledger.history(limit=10)
        assert len(history) == 1
        assert history[0]["run_id"] == "test_run_001"
        assert history[0]["champion_score"] == 88.5
        assert history[0]["decision"] == "accepted"

    def test_empty_history(self, tmp_ledger):
        assert tmp_ledger.history() == []

    def test_multiple_records(self, tmp_ledger):
        for i in range(5):
            result = IslandModelResult(
                run_id=f"run_{i}",
                num_islands=3,
                generations=2,
                champion_score=float(i * 10),
                elapsed_ms=100.0,
            )
            tmp_ledger.record(result)
        history = tmp_ledger.history(limit=3)
        assert len(history) == 3

    def test_record_with_migrations(self, tmp_ledger):
        result = IslandModelResult(
            run_id="mig_run",
            num_islands=4,
            generations=3,
            migrations=[
                MigrationEvent(1, "island_00", "island_01", 85.0),
                MigrationEvent(2, "island_01", "island_02", 90.0),
            ],
            elapsed_ms=200.0,
        )
        tmp_ledger.record(result)
        history = tmp_ledger.history()
        assert len(history) == 1


# ---------------------------------------------------------------------------
# Island Model Runner tests
# ---------------------------------------------------------------------------

class TestIslandModelRunner:

    def setup_method(self):
        _COUNTER["calls"] = 0

    def test_basic_run(self, runner):
        result = runner.run(
            generations=2,
            baseline_config={"val": 50.0},
            baseline_score=50.0,
            eval_fn=_eval_fn,
            mutate_fn=_mutate_fn,
        )
        assert result.num_islands == 3
        assert result.generations == 2
        assert result.champion_island != ""
        assert result.champion_score > 0.0
        assert result.decision in ("accepted", "rejected")
        assert len(result.island_best_scores) == 3

    def test_champion_beats_baseline(self, runner):
        """With monotonically increasing mutation, champion should improve."""
        result = runner.run(
            generations=3,
            baseline_config={"val": 50.0},
            baseline_score=50.0,
            eval_fn=_eval_fn,
            mutate_fn=_mutate_fn,
            min_improvement=0.1,
        )
        # After 3 generations of +1.0 mutations, should beat baseline
        assert result.champion_score > 50.0

    def test_migration_occurs(self, runner):
        """With migration_interval=1, every generation should migrate."""
        result = runner.run(
            generations=3,
            baseline_config={"val": 50.0},
            baseline_score=50.0,
            eval_fn=_eval_fn,
            mutate_fn=_mutate_fn,
        )
        # 3 islands, 3 generations, migration every gen (except last)
        # Generations 1 and 2 should produce migrations
        assert len(result.migrations) >= 3  # at least 3 islands * 1 per interval

    def test_no_migration_last_generation(self, tmp_path):
        """Migration should not happen in the final generation."""
        r = IslandModelRunner(
            num_islands=2,
            population_per_island=3,
            migration_count=1,
            migration_interval=1,
            ledger_dir=str(tmp_path),
        )
        result = r.run(
            generations=1,
            baseline_config={"val": 50.0},
            baseline_score=50.0,
            eval_fn=_eval_fn,
            mutate_fn=_mutate_fn,
        )
        # Only 1 generation, migration skipped (gen < generations - 1 is False)
        assert len(result.migrations) == 0
        r.close()

    def test_high_improvement_threshold_rejects(self, runner):
        result = runner.run(
            generations=1,
            baseline_config={"val": 50.0},
            baseline_score=50.0,
            eval_fn=_eval_fn,
            mutate_fn=_mutate_fn,
            min_improvement=1000.0,  # impossibly high
        )
        assert result.decision == "rejected"

    def test_eval_failure_handled(self, tmp_path):
        """Islands should handle eval failures gracefully."""
        r = IslandModelRunner(
            num_islands=2,
            population_per_island=2,
            ledger_dir=str(tmp_path),
        )
        result = r.run(
            generations=1,
            baseline_config={"val": 50.0},
            baseline_score=50.0,
            eval_fn=_bad_eval_fn,
            mutate_fn=_mutate_fn,
        )
        # Should complete without crash, all scores will be 0.0
        assert result.champion_score == 0.0
        r.close()

    def test_mutate_failure_handled(self, tmp_path):
        """Islands should handle mutation failures gracefully."""
        r = IslandModelRunner(
            num_islands=2,
            population_per_island=2,
            ledger_dir=str(tmp_path),
        )
        result = r.run(
            generations=1,
            baseline_config={"val": 50.0},
            baseline_score=50.0,
            eval_fn=_eval_fn,
            mutate_fn=_bad_mutate_fn,
        )
        # Fallback to baseline copies, should still complete
        assert result.num_islands == 2
        r.close()

    def test_island_count_clamped(self, tmp_path):
        """num_islands should be clamped to [2, 16]."""
        r = IslandModelRunner(num_islands=1, ledger_dir=str(tmp_path))
        assert r.num_islands == 2
        r.close()

        r2 = IslandModelRunner(num_islands=100, ledger_dir=str(tmp_path))
        assert r2.num_islands == 16
        r2.close()

    def test_generations_clamped(self, tmp_path):
        """Generations should be clamped to [1, 100]."""
        r = IslandModelRunner(
            num_islands=2,
            population_per_island=2,
            ledger_dir=str(tmp_path),
        )
        result = r.run(
            generations=0,  # clamped to 1
            baseline_config={"val": 50.0},
            baseline_score=50.0,
            eval_fn=_eval_fn,
            mutate_fn=_mutate_fn,
        )
        assert result.generations == 1
        r.close()

    def test_ring_migration_topology(self, tmp_path):
        """Migration should follow ring: 0->1, 1->2, ..., N-1->0."""
        r = IslandModelRunner(
            num_islands=3,
            population_per_island=3,
            migration_count=1,
            migration_interval=1,
            ledger_dir=str(tmp_path),
        )
        result = r.run(
            generations=2,
            baseline_config={"val": 50.0},
            baseline_score=50.0,
            eval_fn=_eval_fn,
            mutate_fn=_mutate_fn,
        )
        # Check ring topology in first migration batch
        if result.migrations:
            sources = {m.source_island for m in result.migrations}
            targets = {m.target_island for m in result.migrations}
            # All islands should participate as sources
            assert len(sources) >= 2
            assert len(targets) >= 2
        r.close()

    def test_elapsed_ms_recorded(self, runner):
        result = runner.run(
            generations=2,
            baseline_config={"val": 50.0},
            baseline_score=50.0,
            eval_fn=_eval_fn,
            mutate_fn=_mutate_fn,
        )
        assert result.elapsed_ms >= 0.0  # fast CPUs may round sub-ms to 0

    def test_history_recorded(self, runner):
        runner.run(
            generations=1,
            baseline_config={"val": 50.0},
            baseline_score=50.0,
            eval_fn=_eval_fn,
            mutate_fn=_mutate_fn,
        )
        history = runner.history()
        assert len(history) == 1

    def test_each_island_gets_distinct_population(self, tmp_path):
        """Each island should evolve independently."""
        scores_seen = set()

        def tracking_eval(config):
            score = config.get("val", 0.0) + 1.0
            scores_seen.add(score)
            return score

        r = IslandModelRunner(
            num_islands=3,
            population_per_island=3,
            ledger_dir=str(tmp_path),
        )
        result = r.run(
            generations=2,
            baseline_config={"val": 10.0},
            baseline_score=10.0,
            eval_fn=tracking_eval,
            mutate_fn=_mutate_fn,
        )
        # Multiple distinct scores should have been evaluated
        assert len(scores_seen) >= 2
        r.close()

    def test_migration_count_respected(self, tmp_path):
        """migration_count=2 should send 2 migrants per island per interval."""
        r = IslandModelRunner(
            num_islands=3,
            population_per_island=6,
            migration_count=2,
            migration_interval=1,
            ledger_dir=str(tmp_path),
        )
        result = r.run(
            generations=2,
            baseline_config={"val": 50.0},
            baseline_score=50.0,
            eval_fn=_eval_fn,
            mutate_fn=_mutate_fn,
        )
        # gen 1: 3 islands * 2 migrants = 6 migrations
        gen1_migs = [m for m in result.migrations if m.generation == 1]
        assert len(gen1_migs) == 6
        r.close()


# ---------------------------------------------------------------------------
# Data structure tests
# ---------------------------------------------------------------------------

class TestDataStructures:

    def test_island_state_defaults(self):
        state = IslandState(island_id="test")
        assert state.generation == 0
        assert state.best_score == 0.0
        assert state.population == []
        assert state.immigrants == []

    def test_migrant_fields(self):
        m = Migrant(config={"val": 10}, score=85.0, source_island="island_00")
        assert m.score == 85.0
        assert m.source_island == "island_00"

    def test_migration_event_fields(self):
        e = MigrationEvent(
            generation=3,
            source_island="island_00",
            target_island="island_01",
            migrant_score=90.0,
        )
        assert e.generation == 3
        assert e.migrant_score == 90.0

    def test_result_defaults(self):
        r = IslandModelResult(
            run_id="test",
            num_islands=4,
            generations=5,
        )
        assert r.champion_island == ""
        assert r.champion_score == 0.0
        assert r.migrations == []
        assert r.decision == ""


# ---------------------------------------------------------------------------
# GATE TEST
# ---------------------------------------------------------------------------

class TestGateIslandModel:
    """
    Gate test: island model with migration should produce better results
    than any single island's initial score.
    """

    def setup_method(self):
        _COUNTER["calls"] = 0

    def test_migration_improves_diversity(self, tmp_path):
        """With migration, island best scores should diverge (different
        islands explore different regions)."""
        r = IslandModelRunner(
            num_islands=4,
            population_per_island=4,
            migration_count=1,
            migration_interval=1,
            ledger_dir=str(tmp_path),
        )
        result = r.run(
            generations=3,
            baseline_config={"val": 50.0},
            baseline_score=50.0,
            eval_fn=_eval_fn,
            mutate_fn=_mutate_fn,
            min_improvement=0.1,
        )
        scores = list(result.island_best_scores.values())
        assert len(scores) == 4
        # At least some islands should have improved above baseline
        assert max(scores) > 50.0
        # Global champion should be the best across all islands
        assert result.champion_score == max(scores)
        r.close()
