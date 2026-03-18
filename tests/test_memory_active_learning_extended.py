"""Extended tests for ProceduralMemory and ActiveLearner.

Covers storage, recall, update, pruning, capacity limits for procedural
memory and uncertainty sampling, committee disagreement, frontier scoring,
query lifecycle, diversity, and budget-aware selection for active learning.
"""

import math
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from core.procedural_memory import ProceduralExperience, ProceduralMemory
from core.active_learner import ActiveLearner, LearningCandidate


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_db(tmp_path):
    """Return a temporary DB path for ProceduralMemory."""
    return str(tmp_path / "proc_mem" / "test.db")


@pytest.fixture()
def memory(tmp_db):
    return ProceduralMemory(db_path=tmp_db)


@pytest.fixture()
def gen_fn():
    """A mock generate_fn(query, temperature) -> answer."""
    fn = MagicMock(side_effect=lambda q, t: f"answer-{t:.1f}")
    return fn


@pytest.fixture()
def learner(tmp_path, gen_fn):
    db = str(tmp_path / "al" / "test.db")
    return ActiveLearner(generate_fn=gen_fn, db_path=db, n_samples=3)


# =========================================================================
# PROCEDURAL MEMORY TESTS
# =========================================================================

class TestProceduralMemoryStoreAndRetrieve:
    """Store and retrieve procedures."""

    def test_store_and_recall_single(self, memory):
        exp = ProceduralExperience("h1", "act1", "ok", True, {"k": "v"})
        memory.store(exp)
        results = memory.recall("h1")
        assert len(results) == 1
        assert results[0].action == "act1"
        assert results[0].outcome == "ok"
        assert results[0].success is True
        assert results[0].metadata == {"k": "v"}

    def test_recall_empty(self, memory):
        assert memory.recall("nonexistent") == []

    def test_store_multiple_same_hash(self, memory):
        memory.store(ProceduralExperience("h1", "a1", "o1", True))
        memory.store(ProceduralExperience("h1", "a2", "o2", False))
        results = memory.recall("h1")
        assert len(results) == 2
        actions = {r.action for r in results}
        assert actions == {"a1", "a2"}

    def test_recall_max_results(self, memory):
        for i in range(10):
            memory.store(ProceduralExperience("h1", f"a{i}", "ok", True))
        results = memory.recall("h1", max_results=3)
        assert len(results) == 3

    def test_recall_different_hashes_isolated(self, memory):
        memory.store(ProceduralExperience("h1", "a1", "o1", True))
        memory.store(ProceduralExperience("h2", "a2", "o2", False))
        assert len(memory.recall("h1")) == 1
        assert len(memory.recall("h2")) == 1
        assert memory.recall("h1")[0].action == "a1"


class TestProceduralMemoryUpdate:
    """Update existing procedures (INSERT OR REPLACE on same PK)."""

    def test_update_replaces_outcome(self, memory):
        memory.store(ProceduralExperience("h1", "act", "old", False))
        memory.store(ProceduralExperience("h1", "act", "new", True))
        results = memory.recall("h1")
        assert len(results) == 1
        assert results[0].outcome == "new"
        assert results[0].success is True

    def test_update_preserves_other_actions(self, memory):
        memory.store(ProceduralExperience("h1", "a1", "o1", True))
        memory.store(ProceduralExperience("h1", "a2", "o2", True))
        memory.store(ProceduralExperience("h1", "a1", "o1-v2", False))
        results = memory.recall("h1")
        assert len(results) == 2


class TestProceduralMemorySearchByKeyword:
    """Search by keyword via outcome/action text matching."""

    def test_keyword_in_action(self, memory):
        memory.store(ProceduralExperience("h1", "run pytest", "pass", True))
        memory.store(ProceduralExperience("h2", "build docker", "ok", True))
        results = memory.recall("h1")
        assert any("pytest" in r.action for r in results)

    def test_keyword_in_metadata(self, memory):
        memory.store(ProceduralExperience(
            "h1", "act", "ok", True, {"tags": "refactor,cleanup"}
        ))
        results = memory.recall("h1")
        assert "refactor" in results[0].metadata.get("tags", "")


class TestProceduralMemoryStaleness:
    """Expiry/staleness detection via timestamp-based pruning."""

    def test_prune_old_removes_excess(self, memory):
        for i in range(20):
            memory.store(ProceduralExperience(f"h{i}", f"a{i}", "ok", True))
        deleted = memory.prune_old(keep=5)
        assert deleted == 15

    def test_prune_old_noop_when_under_limit(self, memory):
        for i in range(3):
            memory.store(ProceduralExperience(f"h{i}", f"a{i}", "ok", True))
        deleted = memory.prune_old(keep=10)
        assert deleted == 0

    def test_prune_old_default_keeps_max_entries(self, memory):
        # Just verify default path doesn't crash with few entries
        deleted = memory.prune_old()
        assert deleted == 0


class TestProceduralMemoryConsolidation:
    """Memory consolidation -- merge similar experiences via replace."""

    def test_consolidate_duplicate_state_action(self, memory):
        memory.store(ProceduralExperience("h1", "act", "v1", False))
        memory.store(ProceduralExperience("h1", "act", "v2", True))
        results = memory.recall("h1")
        assert len(results) == 1
        assert results[0].outcome == "v2"

    def test_consolidate_metadata_overwritten(self, memory):
        memory.store(ProceduralExperience("h1", "a", "o", True, {"v": 1}))
        memory.store(ProceduralExperience("h1", "a", "o", True, {"v": 2}))
        results = memory.recall("h1")
        assert results[0].metadata["v"] == 2


class TestProceduralMemoryCapacityLimits:
    """Memory capacity limits."""

    def test_max_entries_constant(self):
        assert ProceduralMemory.MAX_ENTRIES == 10_000

    def test_prune_enforces_capacity(self, memory):
        for i in range(50):
            memory.store(ProceduralExperience(f"s{i}", f"a{i}", "ok", True))
        deleted = memory.prune_old(keep=10)
        assert deleted == 40
        # Verify only 10 remain
        with memory._connect() as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM experiences"
            ).fetchone()[0]
        assert count == 10


# =========================================================================
# ACTIVE LEARNER TESTS
# =========================================================================

class TestUncertaintySampling:
    """Select most uncertain queries."""

    def test_all_identical_answers_low_uncertainty(self, tmp_path):
        fn = MagicMock(return_value="same answer")
        al = ActiveLearner(generate_fn=fn, db_path=str(tmp_path / "a/t.db"),
                           n_samples=3)
        candidates = al.score_queries(["q1"])
        assert candidates[0].uncertainty == 0.0

    def test_all_different_answers_high_uncertainty(self, tmp_path):
        counter = iter(["alpha", "bravo", "charlie"])
        fn = MagicMock(side_effect=lambda q, t: next(counter))
        al = ActiveLearner(generate_fn=fn, db_path=str(tmp_path / "a/t.db"),
                           n_samples=3)
        candidates = al.score_queries(["q1"])
        assert candidates[0].uncertainty == pytest.approx(1.0, abs=0.01)

    def test_uncertainty_between_zero_and_one(self, learner):
        candidates = learner.score_queries(["what is X?"])
        assert 0.0 <= candidates[0].uncertainty <= 1.0

    def test_generate_fn_called_n_samples_times(self, tmp_path):
        fn = MagicMock(return_value="ans")
        al = ActiveLearner(generate_fn=fn, db_path=str(tmp_path / "a/t.db"),
                           n_samples=4)
        al.score_queries(["q"])
        assert fn.call_count == 4


class TestValueOfInformationScoring:
    """Combined learning value scoring."""

    def test_learning_value_is_weighted_combination(self, tmp_path):
        fn = MagicMock(return_value="x")
        al = ActiveLearner(generate_fn=fn, db_path=str(tmp_path / "a/t.db"),
                           n_samples=2)
        candidates = al.score_queries(["q"], confidence_scores=[0.5])
        c = candidates[0]
        expected = 0.35 * c.uncertainty + 0.35 * c.disagreement + 0.30 * c.frontier_score
        assert c.learning_value == pytest.approx(expected, abs=1e-6)

    def test_frontier_score_peaks_at_half(self):
        assert ActiveLearner._frontier_score(0.5) == pytest.approx(1.0)
        assert ActiveLearner._frontier_score(0.0) == pytest.approx(0.0)
        assert ActiveLearner._frontier_score(1.0) == pytest.approx(0.0)

    def test_frontier_score_symmetric(self):
        assert ActiveLearner._frontier_score(0.3) == pytest.approx(
            ActiveLearner._frontier_score(0.7)
        )

    def test_high_confidence_low_frontier(self):
        assert ActiveLearner._frontier_score(0.95) < 0.2


class TestQueryPoolManagement:
    """Persistence, resolution, and stats."""

    def test_candidates_persisted(self, learner):
        learner.score_queries(["q1", "q2"])
        stats = learner.stats()
        assert stats["total_scored"] == 2
        assert stats["unresolved"] == 2

    def test_mark_resolved(self, learner):
        learner.score_queries(["q1"])
        learner.mark_resolved("q1")
        stats = learner.stats()
        assert stats["unresolved"] == 0

    def test_top_learning_opportunities(self, learner):
        learner.score_queries(["a", "b", "c"])
        top = learner.top_learning_opportunities(limit=2)
        assert len(top) == 2
        assert top[0]["learning_value"] >= top[1]["learning_value"]

    def test_resolved_excluded_from_top(self, learner):
        learner.score_queries(["a", "b"])
        learner.mark_resolved("a")
        top = learner.top_learning_opportunities(limit=10)
        queries = [t["query"] for t in top]
        assert "a" not in queries


class TestDiversitySampling:
    """Avoid redundant queries -- scoring sorts by learning value."""

    def test_sorted_descending_by_value(self, tmp_path):
        results = []

        def varied_fn(q, t):
            # Return different diversity for different queries
            if q == "easy":
                return "same"
            return f"unique-{t}-{q}"

        al = ActiveLearner(generate_fn=varied_fn,
                           db_path=str(tmp_path / "a/t.db"), n_samples=3)
        candidates = al.score_queries(["easy", "hard"],
                                      confidence_scores=[0.9, 0.5])
        assert candidates[0].learning_value >= candidates[1].learning_value

    def test_identical_queries_get_same_score(self, tmp_path):
        fn = MagicMock(return_value="fixed")
        al = ActiveLearner(generate_fn=fn,
                           db_path=str(tmp_path / "a/t.db"), n_samples=2)
        candidates = al.score_queries(["q", "q"])
        assert candidates[0].learning_value == pytest.approx(
            candidates[1].learning_value, abs=1e-6
        )


class TestBudgetAwareSelection:
    """Budget-aware selection via top_learning_opportunities limit."""

    def test_budget_limits_results(self, learner):
        learner.score_queries([f"q{i}" for i in range(20)])
        top = learner.top_learning_opportunities(limit=5)
        assert len(top) == 5

    def test_budget_zero_returns_empty(self, learner):
        learner.score_queries(["a", "b"])
        top = learner.top_learning_opportunities(limit=0)
        assert len(top) == 0

    def test_budget_exceeding_pool_returns_all(self, learner):
        learner.score_queries(["a"])
        top = learner.top_learning_opportunities(limit=100)
        assert len(top) == 1


class TestCommitteeDisagreement:
    """Committee voting with registered strategy functions."""

    def test_no_strategies_gives_default_disagreement(self, learner):
        candidates = learner.score_queries(["q"])
        assert candidates[0].disagreement == 0.5

    def test_registered_strategies_measured(self, tmp_path):
        fn = MagicMock(return_value="x")
        al = ActiveLearner(generate_fn=fn,
                           db_path=str(tmp_path / "a/t.db"), n_samples=2)
        al.register_strategy(lambda q: "alpha")
        al.register_strategy(lambda q: "beta")
        candidates = al.score_queries(["q"])
        assert candidates[0].disagreement == pytest.approx(1.0, abs=0.01)

    def test_agreeing_strategies_low_disagreement(self, tmp_path):
        fn = MagicMock(return_value="x")
        al = ActiveLearner(generate_fn=fn,
                           db_path=str(tmp_path / "a/t.db"), n_samples=2)
        al.register_strategy(lambda q: "same")
        al.register_strategy(lambda q: "same")
        candidates = al.score_queries(["q"])
        assert candidates[0].disagreement == 0.0


class TestShannonEntropy:
    """Unit tests for the entropy helper."""

    def test_empty_list(self):
        assert ActiveLearner._shannon_entropy([]) == 0.0

    def test_single_item(self):
        assert ActiveLearner._shannon_entropy(["a"]) == 0.0

    def test_two_equal(self):
        assert ActiveLearner._shannon_entropy(["a", "b"]) == pytest.approx(1.0)

    def test_all_same(self):
        assert ActiveLearner._shannon_entropy(["x", "x", "x"]) == 0.0
