"""Tests for core.procedural_memory -- state-action-outcome storage."""

import pytest

from core.procedural_memory import ProceduralMemory, ProceduralExperience


@pytest.fixture
def pm(tmp_path):
    mem = ProceduralMemory(str(tmp_path / "proc.db"))
    try:
        yield mem
    finally:
        if hasattr(mem, "close"):
            mem.close()


class TestStoreAndRecall:

    def test_store_and_recall(self, pm):
        exp = ProceduralExperience(
            state_hash="s1", action="run tests", outcome="all passed", success=True,
        )
        pm.store(exp)
        results = pm.recall("s1")
        assert len(results) == 1
        assert results[0].action == "run tests"
        assert results[0].success is True

    def test_recall_empty(self, pm):
        assert pm.recall("nonexistent") == []

    def test_multiple_actions_same_state(self, pm):
        pm.store(ProceduralExperience("s1", "action_a", "ok", True))
        pm.store(ProceduralExperience("s1", "action_b", "fail", False))
        results = pm.recall("s1", max_results=10)
        assert len(results) == 2
        actions = {r.action for r in results}
        assert actions == {"action_a", "action_b"}

    def test_replace_same_state_action(self, pm):
        pm.store(ProceduralExperience("s1", "act", "old", False))
        pm.store(ProceduralExperience("s1", "act", "new", True))
        results = pm.recall("s1")
        assert len(results) == 1
        assert results[0].outcome == "new"
        assert results[0].success is True

    def test_max_results_limit(self, pm):
        for i in range(10):
            pm.store(ProceduralExperience(
                "s1", f"action_{i}", f"outcome_{i}", True,
            ))
        results = pm.recall("s1", max_results=3)
        assert len(results) == 3

    def test_metadata_roundtrip(self, pm):
        exp = ProceduralExperience(
            "s1", "act", "out", True,
            metadata={"tool": "pytest", "count": 42},
        )
        pm.store(exp)
        results = pm.recall("s1")
        assert results[0].metadata["tool"] == "pytest"
        assert results[0].metadata["count"] == 42


class TestPruneOld:

    def test_prune_removes_oldest(self, pm):
        for i in range(20):
            pm.store(ProceduralExperience(
                f"state_{i:03d}", f"act_{i}", f"out_{i}", True,
            ))
        deleted = pm.prune_old(keep=10)
        assert deleted == 10

    def test_prune_noop_under_limit(self, pm):
        pm.store(ProceduralExperience("s1", "a1", "o1", True))
        assert pm.prune_old(keep=100) == 0


class TestDifferentStates:

    def test_isolation_between_states(self, pm):
        pm.store(ProceduralExperience("s1", "act", "out1", True))
        pm.store(ProceduralExperience("s2", "act", "out2", False))
        r1 = pm.recall("s1")
        r2 = pm.recall("s2")
        assert len(r1) == 1
        assert len(r2) == 1
        assert r1[0].outcome == "out1"
        assert r2[0].outcome == "out2"
