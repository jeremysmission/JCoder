"""
Tests for Sprint 21 research integration features:
- MemRL Q-value updates (experience_replay.py)
- Self-certainty scoring (llm_backend.py, runtime.py)
- BestOfN self-certainty integration (best_of_n.py)
- Tool result caching (tools.py)
- Curriculum scheduling (learning_cycle.py)
- FTS5 OPTIMIZE scheduling (weekly_knowledge_update.py)
- Q-value feedback loop (bridge.py)
"""

from __future__ import annotations

import json
import math
import sqlite3
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# MemRL Q-value tests
# ---------------------------------------------------------------------------

from core.experience_replay import Experience, ExperienceStore


def _code_answer(text=None):
    if text is not None:
        return text
    return "def foo():\n    return bar()\n\ndef process_data(items):\n    return [x for x in items]"


class TestQValueUpdate:

    def test_q_value_default(self):
        e = Experience(
            exp_id="e1", query="q", answer="a",
            source_files=[], confidence=0.9, timestamp=1.0,
        )
        assert e.q_value == 0.0

    def test_q_value_field(self):
        e = Experience(
            exp_id="e1", query="q", answer="a",
            source_files=[], confidence=0.9, timestamp=1.0,
            q_value=0.75,
        )
        assert e.q_value == 0.75

    def test_bellman_update_success(self, tmp_path):
        s = ExperienceStore(str(tmp_path / "exp.db"))
        s.store("e1", "sort algorithm", _code_answer(), ["sort.py"], 0.8)
        s.update_q_value("e1", reward=1.0)
        results = s.retrieve("sort algorithm")
        assert results[0].q_value > 0

    def test_bellman_update_failure(self, tmp_path):
        s = ExperienceStore(str(tmp_path / "exp.db"))
        s.store("e1", "sort algorithm", _code_answer(), ["sort.py"], 0.8)
        # Get initial Q-value
        initial = s.retrieve("sort algorithm")[0].q_value
        s.update_q_value("e1", reward=0.0)
        updated = s.retrieve("sort algorithm")[0].q_value
        assert updated <= initial

    def test_bellman_convergence(self, tmp_path):
        s = ExperienceStore(str(tmp_path / "exp.db"), q_learning_rate=0.1)
        s.store("e1", "sort algorithm", _code_answer(), ["sort.py"], 0.8)
        # Repeated success updates should converge toward 1.0
        for _ in range(50):
            s.update_q_value("e1", reward=1.0)
        results = s.retrieve("sort algorithm")
        assert results[0].q_value > 0.9

    def test_bellman_update_nonexistent(self, tmp_path):
        s = ExperienceStore(str(tmp_path / "exp.db"))
        # Should not crash
        s.update_q_value("nonexistent", reward=1.0)

    def test_q_value_in_retrieval_scoring(self, tmp_path):
        s = ExperienceStore(
            str(tmp_path / "exp.db"),
            q_value_weight=0.3,
        )
        s.store("low_q", "implement parser", _code_answer(), [], 0.8)
        s.store("high_q", "implement parser",
                _code_answer("def parse(text):\n    tokens = tokenize(text)\n    return build_ast(tokens)"),
                [], 0.8)
        # Give high_q a much higher Q-value
        for _ in range(30):
            s.update_q_value("high_q", reward=1.0)
        results = s.retrieve("implement parser")
        assert results[0].exp_id == "high_q"

    def test_q_value_in_stats(self, tmp_path):
        s = ExperienceStore(str(tmp_path / "exp.db"))
        s.store("e1", "query", _code_answer(), [], 0.9)
        stats = s.stats()
        assert "avg_q_value" in stats

    def test_schema_migration(self, tmp_path):
        """Old databases without q_value column should auto-migrate."""
        db_path = tmp_path / "legacy.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE experiences (
                exp_id TEXT PRIMARY KEY,
                query TEXT NOT NULL,
                answer TEXT NOT NULL,
                source_files_json TEXT,
                confidence REAL,
                timestamp REAL,
                use_count INTEGER DEFAULT 0,
                keywords TEXT,
                pass_count INTEGER DEFAULT 0,
                fail_count INTEGER DEFAULT 0,
                p2value REAL DEFAULT 0.0
            )
        """)
        conn.commit()
        conn.close()
        # Opening with new code should add q_value column
        s = ExperienceStore(str(db_path))
        s.store("e1", "query", _code_answer(), [], 0.9)
        results = s.retrieve("query")
        assert hasattr(results[0], "q_value")

    def test_q_value_in_replay_blend(self, tmp_path):
        s = ExperienceStore(str(tmp_path / "exp.db"))
        for i in range(5):
            s.store(f"r{i}", f"replay query {i}", _code_answer(), [], 0.9)
        new_exps = [
            Experience(exp_id="n0", query="new", answer=_code_answer(),
                       source_files=[], confidence=0.8, timestamp=1.0),
        ]
        blended = s.replay_blend(new_exps, replay_ratio=0.5, max_total=4)
        for exp in blended:
            assert hasattr(exp, "q_value")


# ---------------------------------------------------------------------------
# Self-certainty scoring tests
# ---------------------------------------------------------------------------

from agent.llm_backend import ChatResponse, ToolCall


class TestSelfCertainty:

    def test_no_logprobs(self):
        r = ChatResponse(content="hello")
        assert r.self_certainty is None

    def test_empty_logprobs(self):
        r = ChatResponse(content="hello", logprobs=[])
        assert r.self_certainty is None

    def test_high_certainty(self):
        """Logprobs with high probability on chosen token = high certainty."""
        logprobs = [
            {
                "logprob": math.log(0.99),
                "top_logprobs": [
                    {"logprob": math.log(0.99)},
                    {"logprob": math.log(0.005)},
                ],
            }
            for _ in range(5)
        ]
        r = ChatResponse(content="hello", logprobs=logprobs)
        sc = r.self_certainty
        assert sc is not None
        assert sc > 0.9  # exp(log(0.99)) ~ 0.99

    def test_low_certainty(self):
        """Low-probability chosen tokens = low certainty."""
        logprobs = [
            {
                "logprob": math.log(0.1),
                "top_logprobs": [{"logprob": math.log(0.1)} for _ in range(5)],
            }
            for _ in range(5)
        ]
        r = ChatResponse(content="hello", logprobs=logprobs)
        sc = r.self_certainty
        assert sc is not None
        assert sc < 0.2  # exp(log(0.1)) = 0.1

    def test_logprobs_field_preserved(self):
        lp = [{"top_logprobs": [{"logprob": -0.5}]}]
        r = ChatResponse(content="test", logprobs=lp)
        assert r.logprobs == lp


# ---------------------------------------------------------------------------
# BestOfN self-certainty tests
# ---------------------------------------------------------------------------

from core.best_of_n import BestOfNGenerator, Candidate


class TestBestOfNSelfCertainty:

    def test_candidate_has_self_certainty(self):
        c = Candidate(text="code", index=0, self_certainty=0.85)
        assert c.self_certainty == 0.85

    def test_default_self_certainty(self):
        c = Candidate(text="code", index=0)
        assert c.self_certainty == 0.0

    def test_self_certainty_disabled_by_default(self):
        rt = MagicMock()
        rt.generate.return_value = "def f(): return 1"
        gen = BestOfNGenerator(runtime=rt, n=2)
        assert gen.use_self_certainty is False

    def test_self_certainty_enabled(self):
        rt = MagicMock()
        rt.generate.return_value = "def f(): return 1"
        gen = BestOfNGenerator(runtime=rt, n=1, use_self_certainty=True)
        assert gen.use_self_certainty is True

    def test_logprobs_used_when_available(self):
        from core.runtime import GenerationResult
        rt = MagicMock()
        rt.generate_with_logprobs.return_value = GenerationResult(
            text="def solve(): return 42",
            logprobs=[
                {"logprob": math.log(0.95), "top_logprobs": [{"logprob": math.log(0.95)}]}
                for _ in range(10)
            ],
        )
        gen = BestOfNGenerator(
            runtime=rt, n=1, temperature_spread=(0.1,),
            use_self_certainty=True,
        )
        result = gen.generate("test", [])
        assert rt.generate_with_logprobs.call_count == 1
        assert rt.generate.call_count == 0

    def test_fallback_without_logprobs(self):
        rt = MagicMock(spec=["generate"])
        rt.generate.return_value = "def f(): return 1"
        gen = BestOfNGenerator(
            runtime=rt, n=1, temperature_spread=(0.1,),
            use_self_certainty=True,
        )
        result = gen.generate("test", [])
        assert rt.generate.call_count == 1


# ---------------------------------------------------------------------------
# Tool result cache tests
# ---------------------------------------------------------------------------

from agent.tools import ToolResultCache, ToolResult, _cache_key


class TestToolResultCache:

    def test_put_and_get(self):
        cache = ToolResultCache()
        r = ToolResult(success=True, output="content")
        cache.put("key1", r)
        assert cache.get("key1", ttl=30.0) is r

    def test_ttl_expiry(self):
        cache = ToolResultCache()
        r = ToolResult(success=True, output="content")
        cache.put("key1", r)
        # Simulate expiry
        cache._store["key1"] = (time.monotonic() - 100, r)
        assert cache.get("key1", ttl=30.0) is None

    def test_hit_miss_tracking(self):
        cache = ToolResultCache()
        r = ToolResult(success=True, output="data")
        cache.put("k", r)
        cache.get("k", ttl=30.0)  # hit
        cache.get("miss", ttl=30.0)  # miss
        stats = cache.stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5

    def test_invalidate(self):
        cache = ToolResultCache()
        cache.put("k1", ToolResult(True, "a"))
        cache.put("k2", ToolResult(True, "b"))
        cache.invalidate()
        assert cache.get("k1", ttl=30.0) is None
        assert cache.get("k2", ttl=30.0) is None

    def test_lru_eviction(self):
        cache = ToolResultCache(max_entries=2)
        cache.put("k1", ToolResult(True, "a"))
        cache.put("k2", ToolResult(True, "b"))
        cache.put("k3", ToolResult(True, "c"))
        assert cache.stats()["entries"] == 2
        # k1 (oldest) should be evicted
        assert cache.get("k1", ttl=30.0) is None

    def test_cache_key_deterministic(self):
        k1 = _cache_key("read_file", {"path": "/a/b.py"})
        k2 = _cache_key("read_file", {"path": "/a/b.py"})
        assert k1 == k2

    def test_cache_key_different_args(self):
        k1 = _cache_key("read_file", {"path": "/a.py"})
        k2 = _cache_key("read_file", {"path": "/b.py"})
        assert k1 != k2


# ---------------------------------------------------------------------------
# Curriculum scheduling tests
# ---------------------------------------------------------------------------


class TestCurriculumScheduling:

    def test_easy_to_hard_ordering(self):
        from scripts.learning_cycle import generate_study_queries

        baseline_report = {
            "category_scores": {"python": 0.3, "go": 0.4, "security": 0.8},
            "results": [
                {"question_id": "q1", "score": 0.9, "category": "security"},
                {"question_id": "q2", "score": 0.2, "category": "python"},
                {"question_id": "q3", "score": 0.5, "category": "go"},
                {"question_id": "q4", "score": 0.1, "category": "python"},
                {"question_id": "q5", "score": 0.7, "category": "go"},
            ],
        }
        eval_questions = [
            {"id": "q1", "question": "SQL injection", "category": "security"},
            {"id": "q2", "question": "list comprehension", "category": "python"},
            {"id": "q3", "question": "goroutines", "category": "go"},
            {"id": "q4", "question": "decorators", "category": "python"},
            {"id": "q5", "question": "channels", "category": "go"},
        ]

        import tempfile
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8",
        ) as f:
            json.dump(eval_questions, f)
            eval_path = f.name

        study = generate_study_queries(
            baseline_report, eval_path, n_queries=10,
        )
        Path(eval_path).unlink()

        # Verify easy-to-hard: first query should have highest baseline_score
        scores = [q["baseline_score"] for q in study]
        assert scores == sorted(scores, reverse=True), (
            f"Study queries not sorted easy-to-hard: {scores}"
        )


# ---------------------------------------------------------------------------
# FTS5 OPTIMIZE tests
# ---------------------------------------------------------------------------


class TestFTS5Optimize:

    def test_optimize_empty_dir(self, tmp_path):
        from scripts.weekly_knowledge_update import optimize_fts5_indexes
        # Empty directory
        assert optimize_fts5_indexes(tmp_path) == 0

    def test_optimize_nonexistent_dir(self, tmp_path):
        from scripts.weekly_knowledge_update import optimize_fts5_indexes
        assert optimize_fts5_indexes(tmp_path / "nope") == 0

    def test_optimize_fts5_db(self, tmp_path):
        from scripts.weekly_knowledge_update import optimize_fts5_indexes
        # Create a real FTS5 database
        db_path = tmp_path / "test.fts5.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE VIRTUAL TABLE chunks USING fts5(content)")
        conn.execute("INSERT INTO chunks VALUES ('hello world')")
        conn.execute("INSERT INTO chunks VALUES ('foo bar baz')")
        conn.commit()
        conn.close()

        optimized = optimize_fts5_indexes(tmp_path, max_indexes=5)
        assert optimized == 1

    def test_optimize_limits_to_max(self, tmp_path):
        from scripts.weekly_knowledge_update import optimize_fts5_indexes
        # Create 3 FTS5 databases
        for i in range(3):
            db_path = tmp_path / f"idx{i}.fts5.db"
            conn = sqlite3.connect(str(db_path))
            conn.execute("CREATE VIRTUAL TABLE chunks USING fts5(content)")
            conn.execute("INSERT INTO chunks VALUES ('test data')")
            conn.commit()
            conn.close()

        optimized = optimize_fts5_indexes(tmp_path, max_indexes=2)
        assert optimized == 2


# ---------------------------------------------------------------------------
# GenerationResult self-certainty tests
# ---------------------------------------------------------------------------

from core.runtime import GenerationResult


class TestGenerationResult:

    def test_no_logprobs(self):
        r = GenerationResult(text="hello")
        assert r.self_certainty is None

    def test_with_logprobs(self):
        r = GenerationResult(
            text="code",
            logprobs=[
                {
                    "logprob": math.log(0.9),
                    "top_logprobs": [
                        {"logprob": math.log(0.9)},
                        {"logprob": math.log(0.1)},
                    ],
                }
                for _ in range(5)
            ],
        )
        sc = r.self_certainty
        assert sc is not None
        assert 0.8 <= sc <= 1.0  # exp(log(0.9)) = 0.9

    def test_certainty_bounded(self):
        r = GenerationResult(
            text="x",
            logprobs=[
                {"logprob": 0.0, "top_logprobs": [{"logprob": 0.0}]}  # p=1.0
                for _ in range(3)
            ],
        )
        sc = r.self_certainty
        assert sc is not None
        assert 0.0 <= sc <= 1.0
