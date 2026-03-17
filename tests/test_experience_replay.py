"""
Tests for core.experience_replay -- P2Value priority scoring.
No external deps; pure SQLite + in-memory.
"""

from __future__ import annotations

import pytest

from core.experience_replay import Experience, ExperienceStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _store(tmp_path, **kwargs):
    db = str(tmp_path / "exp.db")
    return ExperienceStore(db_path=db, **kwargs)


def _code_answer(text=None):
    """Answer that passes the code-content check (>= 50 chars with code pattern)."""
    if text is not None:
        return text
    return "def foo():\n    return bar()\n\ndef process_data(items):\n    return [x for x in items]"


# ---------------------------------------------------------------------------
# Experience dataclass
# ---------------------------------------------------------------------------

class TestExperience:

    def test_defaults(self):
        e = Experience(
            exp_id="e1", query="q", answer="a",
            source_files=[], confidence=0.9, timestamp=1.0,
        )
        assert e.pass_count == 0
        assert e.fail_count == 0
        assert e.p2value == 0.0

    def test_all_fields(self):
        e = Experience(
            exp_id="e1", query="q", answer="a",
            source_files=["f.py"], confidence=0.8, timestamp=1.0,
            use_count=3, pass_count=9, fail_count=1, p2value=0.85,
        )
        assert e.pass_count == 9
        assert e.p2value == 0.85


# ---------------------------------------------------------------------------
# P2Value computation
# ---------------------------------------------------------------------------

class TestP2Value:

    def test_confidence_only_when_no_tests(self, tmp_path):
        s = _store(tmp_path)
        p2v = s.compute_p2value(confidence=0.8)
        assert p2v == 0.8

    def test_blends_confidence_and_pass_rate(self, tmp_path):
        s = _store(tmp_path, p2value_alpha=0.4)
        # alpha=0.4: 0.4*0.8 + 0.6*(9/10) = 0.32 + 0.54 = 0.86
        p2v = s.compute_p2value(confidence=0.8, pass_count=9, fail_count=1)
        # Near-miss boost: fail_count==1, so 0.86 * 1.3 = 1.118 -> capped at 1.0
        assert p2v == 1.0

    def test_near_miss_boost(self, tmp_path):
        s = _store(tmp_path, p2value_alpha=0.5, near_miss_boost=1.3)
        # 0.5*0.6 + 0.5*(4/5) = 0.3 + 0.4 = 0.7
        # Near-miss: 0.7 * 1.3 = 0.91
        p2v = s.compute_p2value(confidence=0.6, pass_count=4, fail_count=1)
        assert abs(p2v - 0.91) < 0.01

    def test_no_boost_for_multiple_failures(self, tmp_path):
        s = _store(tmp_path, p2value_alpha=0.5, near_miss_boost=1.3)
        # fail_count=3 -> no boost
        p2v_multi = s.compute_p2value(confidence=0.6, pass_count=7, fail_count=3)
        p2v_one = s.compute_p2value(confidence=0.6, pass_count=7, fail_count=1)
        # Near-miss (1 fail) should score higher
        assert p2v_one > p2v_multi

    def test_capped_at_one(self, tmp_path):
        s = _store(tmp_path, near_miss_boost=2.0)
        p2v = s.compute_p2value(confidence=0.99, pass_count=99, fail_count=1)
        assert p2v <= 1.0

    def test_zero_confidence(self, tmp_path):
        s = _store(tmp_path)
        p2v = s.compute_p2value(confidence=0.0, pass_count=10, fail_count=0)
        assert p2v >= 0.0


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

class TestStore:

    def test_stores_valid_experience(self, tmp_path):
        s = _store(tmp_path)
        ok = s.store("e1", "sort a list", _code_answer(), ["sort.py"], 0.9)
        assert ok is True

    def test_rejects_low_confidence(self, tmp_path):
        s = _store(tmp_path, min_confidence=0.6)
        ok = s.store("e1", "query", _code_answer(), [], 0.3)
        assert ok is False

    def test_rejects_short_answer(self, tmp_path):
        s = _store(tmp_path)
        ok = s.store("e1", "query", "short", [], 0.9)
        assert ok is False

    def test_rejects_non_code_answer(self, tmp_path):
        s = _store(tmp_path)
        ok = s.store("e1", "query", "a" * 60, [], 0.9)
        assert ok is False

    def test_stores_with_test_results(self, tmp_path):
        s = _store(tmp_path)
        ok = s.store("e1", "fix bug", _code_answer(), ["bug.py"],
                      confidence=0.8, pass_count=9, fail_count=1)
        assert ok is True
        stats = s.stats()
        assert stats["near_miss_count"] == 1

    def test_p2value_stored_in_db(self, tmp_path):
        s = _store(tmp_path, p2value_alpha=0.5)
        s.store("e1", "query", _code_answer(), [], 0.8, pass_count=8, fail_count=2)
        stats = s.stats()
        assert stats["avg_p2value"] > 0

    def test_eviction_by_p2value(self, tmp_path):
        """Low-p2value experiences evicted first when store is full."""
        s = _store(tmp_path)
        s.MAX_EXPERIENCES = 3
        # Store 3 experiences with varying p2value
        s.store("low", "query1", _code_answer(), [], 0.6, pass_count=1, fail_count=5)
        s.store("mid", "query2", _code_answer(), [], 0.7, pass_count=5, fail_count=2)
        s.store("high", "query3", _code_answer(), [], 0.95, pass_count=10, fail_count=0)
        # Store one more -- should evict lowest p2value
        s.store("new", "query4", _code_answer(), [], 0.8, pass_count=8, fail_count=1)
        stats = s.stats()
        assert stats["total"] == 3


# ---------------------------------------------------------------------------
# Retrieve
# ---------------------------------------------------------------------------

class TestRetrieve:

    def test_retrieves_by_keyword(self, tmp_path):
        s = _store(tmp_path)
        s.store("e1", "sort algorithm python", _code_answer(), ["sort.py"], 0.9)
        s.store("e2", "database connection pool", _code_answer(), ["db.py"], 0.9)
        results = s.retrieve("sort list python")
        assert len(results) >= 1
        assert results[0].exp_id == "e1"

    def test_p2value_affects_ranking(self, tmp_path):
        s = _store(tmp_path)
        # Same keywords, different p2values
        s.store("low", "implement sort", _code_answer(), [], 0.6, pass_count=2, fail_count=5)
        s.store("high", "implement sort",
                _code_answer("def sort_impl(data):\n    result = sorted(data)\n    return result  # optimized"),
                [], 0.9, pass_count=9, fail_count=1)
        results = s.retrieve("implement sort")
        assert len(results) == 2
        # Higher p2value should rank first
        assert results[0].p2value >= results[1].p2value

    def test_near_miss_prioritized(self, tmp_path):
        """Near-miss (1 failure) should rank above multi-failure with same confidence."""
        s = _store(tmp_path, p2value_alpha=0.5, near_miss_boost=1.5)
        s.store("multi_fail", "binary search tree", _code_answer(), [],
                0.7, pass_count=7, fail_count=3)
        s.store("near_miss", "binary search tree",
                _code_answer("def search(tree, key):\n    node = find_node(tree, key)\n    return node  # near miss"),
                [], 0.7, pass_count=9, fail_count=1)
        results = s.retrieve("binary search tree")
        assert len(results) == 2
        assert results[0].exp_id == "near_miss"

    def test_empty_query(self, tmp_path):
        s = _store(tmp_path)
        results = s.retrieve("")
        assert results == []

    def test_increments_use_count(self, tmp_path):
        s = _store(tmp_path)
        s.store("e1", "sort algorithm", _code_answer(), [], 0.9)
        s.retrieve("sort algorithm")
        s.retrieve("sort algorithm")
        # Retrieve a third time and check
        results = s.retrieve("sort algorithm")
        # use_count reflects the state BEFORE current increment
        assert results[0].use_count >= 2


# ---------------------------------------------------------------------------
# Replay blend (RLEP)
# ---------------------------------------------------------------------------

class TestReplayBlend:

    def test_blends_new_and_replay(self, tmp_path):
        s = _store(tmp_path)
        # Populate store with replays
        for i in range(5):
            s.store(f"old{i}", f"old query {i}", _code_answer(), [], 0.9)

        new_exps = [
            Experience(exp_id=f"new{i}", query=f"new {i}", answer=_code_answer(),
                       source_files=[], confidence=0.8, timestamp=float(i))
            for i in range(5)
        ]
        blended = s.replay_blend(new_exps, replay_ratio=0.3, max_total=10)
        new_ids = {e.exp_id for e in blended if e.exp_id.startswith("new")}
        old_ids = {e.exp_id for e in blended if e.exp_id.startswith("old")}
        assert len(new_ids) > 0
        assert len(old_ids) > 0
        assert len(blended) <= 10

    def test_empty_store_returns_new_only(self, tmp_path):
        s = _store(tmp_path)
        new_exps = [
            Experience(exp_id="n1", query="q", answer="a",
                       source_files=[], confidence=0.8, timestamp=1.0),
        ]
        blended = s.replay_blend(new_exps, replay_ratio=0.5, max_total=5)
        # Only new experiences available
        assert len(blended) >= 1

    def test_replay_ratio_controls_mix(self, tmp_path):
        s = _store(tmp_path)
        for i in range(20):
            s.store(f"r{i}", f"replay query {i}", _code_answer(), [], 0.9)

        new_exps = [
            Experience(exp_id=f"n{i}", query=f"new {i}", answer=_code_answer(),
                       source_files=[], confidence=0.8, timestamp=float(i))
            for i in range(20)
        ]
        blended = s.replay_blend(new_exps, replay_ratio=0.5, max_total=10)
        replay_count = sum(1 for e in blended if e.exp_id.startswith("r"))
        # ~50% should be replays (allow some variance from sampling)
        assert 3 <= replay_count <= 7


# ---------------------------------------------------------------------------
# Format and stats
# ---------------------------------------------------------------------------

class TestFormatAndStats:

    def test_format_as_examples(self, tmp_path):
        s = _store(tmp_path)
        exps = [
            Experience(exp_id="e1", query="sort list", answer=_code_answer(),
                       source_files=["sort.py"], confidence=0.9, timestamp=1.0),
        ]
        text = s.format_as_examples(exps)
        assert "sort list" in text
        assert "sort.py" in text

    def test_format_empty(self, tmp_path):
        s = _store(tmp_path)
        assert s.format_as_examples([]) == ""

    def test_stats_empty(self, tmp_path):
        s = _store(tmp_path)
        stats = s.stats()
        assert stats["total"] == 0

    def test_stats_with_data(self, tmp_path):
        s = _store(tmp_path)
        s.store("e1", "query1", _code_answer(), [], 0.9, pass_count=9, fail_count=1)
        s.store("e2", "query2", _code_answer(), [], 0.8, pass_count=8, fail_count=0)
        stats = s.stats()
        assert stats["total"] == 2
        assert stats["avg_p2value"] > 0
        assert stats["near_miss_count"] == 1


# ---------------------------------------------------------------------------
# GATE TEST: Near-miss prioritization
# ---------------------------------------------------------------------------

class TestGateNearMissPriority:
    """
    Gate test for Sprint 19: verify that near-miss experiences
    (failed exactly 1 test) are retrieved before experiences with
    more failures, even at the same confidence level.
    """

    def test_near_miss_beats_multi_failure(self, tmp_path):
        """
        Store two experiences with identical confidence and keywords.
        The near-miss (1 failure) should always rank above the
        multi-failure (3 failures).
        """
        s = _store(tmp_path, p2value_alpha=0.5, near_miss_boost=1.3)

        s.store(
            "multi", "implement hash table chaining",
            _code_answer("def hash_insert(table, key):\n    bucket = table.get_bucket(key)\n    return bucket.insert(key)"),
            ["hash.py"], confidence=0.75, pass_count=7, fail_count=3,
        )
        s.store(
            "near", "implement hash table chaining",
            _code_answer("def hash_lookup(table, key):\n    bucket = table.get_bucket(key)\n    return bucket.find(key)"),
            ["hash.py"], confidence=0.75, pass_count=9, fail_count=1,
        )

        results = s.retrieve("hash table chaining")
        assert len(results) == 2
        assert results[0].exp_id == "near", (
            f"Near-miss should rank first, got {results[0].exp_id}"
        )
        assert results[0].p2value > results[1].p2value
