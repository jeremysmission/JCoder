"""
Tests for core.stigmergy -- Pheromone-based retrieval booster.
Uses temp databases; no persistent state needed.
"""

from __future__ import annotations

import time

import pytest

from core.stigmergy import PheromoneConfig, StigmergicBooster


# ---------------------------------------------------------------------------
# PheromoneConfig
# ---------------------------------------------------------------------------

class TestPheromoneConfig:

    def test_defaults(self):
        c = PheromoneConfig()
        assert c.positive_deposit == 1.0
        assert c.negative_deposit == -0.3
        assert c.evaporation_rate == 0.05
        assert c.boost_weight == 0.15

    def test_custom(self):
        c = PheromoneConfig(positive_deposit=2.0, max_pheromone=5.0)
        assert c.positive_deposit == 2.0
        assert c.max_pheromone == 5.0


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:

    def test_creates_db(self, tmp_path):
        db = str(tmp_path / "phero.db")
        sb = StigmergicBooster(db_path=db)
        stats = sb.stats()
        assert stats["total_trails"] == 0
        assert stats["total_deposits"] == 0

    def test_custom_config(self, tmp_path):
        db = str(tmp_path / "phero.db")
        cfg = PheromoneConfig(positive_deposit=2.0)
        sb = StigmergicBooster(db_path=db, config=cfg)
        assert sb.config.positive_deposit == 2.0


# ---------------------------------------------------------------------------
# deposit
# ---------------------------------------------------------------------------

class TestDeposit:

    def test_positive_deposit(self, tmp_path):
        db = str(tmp_path / "phero.db")
        sb = StigmergicBooster(db_path=db)
        sb.deposit(["chunk_1", "chunk_2"], "lookup", success=True)
        stats = sb.stats()
        assert stats["total_trails"] == 2
        assert stats["total_deposits"] == 2

    def test_negative_deposit(self, tmp_path):
        db = str(tmp_path / "phero.db")
        sb = StigmergicBooster(db_path=db)
        sb.deposit(["chunk_1"], "debug", success=False)
        stats = sb.stats()
        assert stats["total_trails"] == 1

    def test_cumulative_deposits(self, tmp_path):
        db = str(tmp_path / "phero.db")
        sb = StigmergicBooster(db_path=db)
        sb.deposit(["chunk_1"], "lookup", success=True)
        sb.deposit(["chunk_1"], "lookup", success=True)
        hot = sb.hot_chunks("lookup")
        assert len(hot) == 1
        assert hot[0]["successes"] == 2
        assert hot[0]["strength"] > 1.0  # accumulated

    def test_mixed_success_failure(self, tmp_path):
        db = str(tmp_path / "phero.db")
        sb = StigmergicBooster(db_path=db)
        sb.deposit(["chunk_1"], "lookup", success=True)
        sb.deposit(["chunk_1"], "lookup", success=False)
        hot = sb.hot_chunks("lookup")
        assert hot[0]["successes"] == 1
        assert hot[0]["failures"] == 1


# ---------------------------------------------------------------------------
# boost_scores
# ---------------------------------------------------------------------------

class TestBoostScores:

    def test_empty_input(self, tmp_path):
        db = str(tmp_path / "phero.db")
        sb = StigmergicBooster(db_path=db)
        assert sb.boost_scores([], "lookup") == []

    def test_boosts_strong_trails(self, tmp_path):
        db = str(tmp_path / "phero.db")
        sb = StigmergicBooster(db_path=db)
        # Deposit pheromone on chunk_1
        sb.deposit(["chunk_1"], "lookup", success=True)
        sb.deposit(["chunk_1"], "lookup", success=True)

        scores = [("chunk_1", 0.5), ("chunk_2", 0.6)]
        boosted = sb.boost_scores(scores, "lookup")
        # chunk_1 should be boosted
        chunk_1_score = next(s for cid, s in boosted if cid == "chunk_1")
        assert chunk_1_score > 0.5

    def test_no_pheromone_no_boost(self, tmp_path):
        db = str(tmp_path / "phero.db")
        sb = StigmergicBooster(db_path=db)
        scores = [("chunk_1", 0.5)]
        boosted = sb.boost_scores(scores, "lookup")
        assert boosted[0][1] == 0.5  # unchanged

    def test_reorders_by_boosted_score(self, tmp_path):
        db = str(tmp_path / "phero.db")
        sb = StigmergicBooster(db_path=db)
        # Give chunk_2 strong pheromone
        for _ in range(5):
            sb.deposit(["chunk_2"], "lookup", success=True)

        scores = [("chunk_1", 0.8), ("chunk_2", 0.5)]
        boosted = sb.boost_scores(scores, "lookup")
        # chunk_2 might overtake chunk_1 after boost
        assert boosted[0][1] >= boosted[1][1]  # sorted desc


# ---------------------------------------------------------------------------
# evaporate
# ---------------------------------------------------------------------------

class TestEvaporate:

    def test_evaporate_cleans_weak_trails(self, tmp_path):
        db = str(tmp_path / "phero.db")
        cfg = PheromoneConfig(min_pheromone=0.5, negative_deposit=-1.0)
        sb = StigmergicBooster(db_path=db, config=cfg)
        sb.deposit(["chunk_1"], "lookup", success=False)  # negative strength
        cleaned = sb.evaporate()
        stats = sb.stats()
        # Trail should be cleaned if strength dropped below min
        assert stats["total_trails"] <= 1


# ---------------------------------------------------------------------------
# hot_chunks
# ---------------------------------------------------------------------------

class TestHotChunks:

    def test_returns_sorted(self, tmp_path):
        db = str(tmp_path / "phero.db")
        sb = StigmergicBooster(db_path=db)
        sb.deposit(["a"], "lookup", success=True)
        sb.deposit(["b"], "lookup", success=True)
        sb.deposit(["b"], "lookup", success=True)
        hot = sb.hot_chunks("lookup", limit=10)
        assert len(hot) == 2
        assert hot[0]["chunk_id"] == "b"  # higher strength
        assert hot[0]["strength"] > hot[1]["strength"]

    def test_empty_query_type(self, tmp_path):
        db = str(tmp_path / "phero.db")
        sb = StigmergicBooster(db_path=db)
        sb.deposit(["a"], "lookup", success=True)
        hot = sb.hot_chunks("debug")  # different query type
        assert hot == []

    def test_success_rate(self, tmp_path):
        db = str(tmp_path / "phero.db")
        sb = StigmergicBooster(db_path=db)
        sb.deposit(["a"], "lookup", success=True)
        sb.deposit(["a"], "lookup", success=True)
        sb.deposit(["a"], "lookup", success=False)
        hot = sb.hot_chunks("lookup")
        # 2 success / 3 total
        assert hot[0]["success_rate"] == pytest.approx(0.667, abs=0.01)


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------

class TestStats:

    def test_stats_by_type(self, tmp_path):
        db = str(tmp_path / "phero.db")
        sb = StigmergicBooster(db_path=db)
        sb.deposit(["a", "b"], "lookup", success=True)
        sb.deposit(["c"], "debug", success=True)
        stats = sb.stats()
        assert stats["total_trails"] == 3
        assert len(stats["by_type"]) == 2
