"""
Tests for core.star_reasoner -- STaR Self-Taught Reasoner.
All LLM calls are mocked; no live runtime needed.
"""

from __future__ import annotations

import os
from typing import List
from unittest.mock import MagicMock, patch

import pytest

from core.star_reasoner import (
    ReasoningTrace,
    STaRReasoner,
    STaRResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_runtime(response: str = "Step 1: parse input\nStep 2: compute\nANSWER: 42"):
    rt = MagicMock()
    rt.generate.return_value = response
    return rt


def _make_reasoner(tmp_path, runtime=None, verify_fn=None):
    rt = runtime or _mock_runtime()
    return STaRReasoner(
        runtime=rt,
        verify_fn=verify_fn,
        db_path=str(tmp_path / "star.db"),
        seed=42,
    )


def _context_fn(query: str) -> List[str]:
    return ["def foo(): return 42"]


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:

    def test_creates_db(self, tmp_path):
        r = _make_reasoner(tmp_path)
        assert os.path.exists(tmp_path / "star.db")

    def test_custom_verify_fn(self, tmp_path):
        fn = lambda q, a: 0.9
        r = _make_reasoner(tmp_path, verify_fn=fn)
        assert r.verify_fn is fn


# ---------------------------------------------------------------------------
# answer_with_reasoning
# ---------------------------------------------------------------------------

class TestAnswerWithReasoning:

    def test_splits_reasoning_from_answer(self, tmp_path):
        rt = _mock_runtime("Step 1: check types\nStep 2: compute\nANSWER: The result is 42")
        r = _make_reasoner(tmp_path, runtime=rt)
        reasoning, answer = r.answer_with_reasoning("What is 6*7?", ["context"])
        assert "Step 1" in reasoning
        assert "42" in answer

    def test_no_answer_marker(self, tmp_path):
        rt = _mock_runtime("Just a plain response without markers")
        r = _make_reasoner(tmp_path, runtime=rt)
        reasoning, answer = r.answer_with_reasoning("test", ["ctx"])
        assert reasoning == answer  # both get the full text

    def test_uses_few_shot_examples(self, tmp_path):
        rt = _mock_runtime("ANSWER: result")
        r = _make_reasoner(tmp_path, runtime=rt)
        # Store a trace first so examples are available
        r._store_trace("prev q", "prev reasoning", "prev answer", True, 0.8, 0)
        r.answer_with_reasoning("new question", ["ctx"])
        # Runtime should have been called with the examples in the prompt
        call_args = rt.generate.call_args
        assert rt.generate.called

    def test_empty_context(self, tmp_path):
        rt = _mock_runtime("ANSWER: no context needed")
        r = _make_reasoner(tmp_path, runtime=rt)
        reasoning, answer = r.answer_with_reasoning("test", [])
        assert "no context needed" in answer


# ---------------------------------------------------------------------------
# run_iteration
# ---------------------------------------------------------------------------

class TestRunIteration:

    def test_basic_iteration(self, tmp_path):
        # Verify fn always returns 0.8 (correct)
        rt = _mock_runtime("Reasoning here\nANSWER: correct answer")
        r = _make_reasoner(tmp_path, runtime=rt, verify_fn=lambda q, a: 0.8)

        result = r.run_iteration(
            queries=["q1", "q2", "q3"],
            context_fn=_context_fn,
            iteration=0,
        )
        assert isinstance(result, STaRResult)
        assert result.queries_attempted == 3
        assert result.correct_first_try == 3
        assert result.accuracy == 1.0
        assert result.traces_stored == 3

    def test_failed_verification_no_trace(self, tmp_path):
        rt = _mock_runtime("ANSWER: wrong")
        r = _make_reasoner(tmp_path, runtime=rt, verify_fn=lambda q, a: 0.2)

        result = r.run_iteration(
            queries=["hard question"],
            context_fn=_context_fn,
            iteration=0,
        )
        assert result.correct_first_try == 0
        assert result.traces_stored == 0

    def test_rationalization_with_ground_truth(self, tmp_path):
        call_count = [0]

        def _verify(q, a):
            call_count[0] += 1
            return 0.2  # always fails

        rt = _mock_runtime("ANSWER: wrong answer")
        # Rationalization returns valid reasoning
        rt.generate.side_effect = [
            "ANSWER: wrong",  # first attempt
            "Step 1: because X\nStep 2: therefore Y",  # rationalization
        ]
        r = _make_reasoner(tmp_path, runtime=rt, verify_fn=_verify)

        result = r.run_iteration(
            queries=["q1"],
            context_fn=_context_fn,
            ground_truth=["correct answer"],
            iteration=1,
        )
        assert result.rationalized == 1
        assert result.traces_stored == 1

    def test_improvement_tracked(self, tmp_path):
        rt = _mock_runtime("ANSWER: ok")
        r = _make_reasoner(tmp_path, runtime=rt, verify_fn=lambda q, a: 0.8)

        # Iteration 0
        r.run_iteration(["q1"], _context_fn, iteration=0)
        # Iteration 1
        result = r.run_iteration(["q1", "q2"], _context_fn, iteration=1)
        assert isinstance(result.improvement_over_previous, float)


# ---------------------------------------------------------------------------
# _verify
# ---------------------------------------------------------------------------

class TestVerify:

    def test_custom_verify_fn(self, tmp_path):
        r = _make_reasoner(tmp_path, verify_fn=lambda q, a: 0.95)
        score = r._verify("test", "answer", None)
        assert score == 0.95

    def test_ground_truth_overlap(self, tmp_path):
        r = _make_reasoner(tmp_path)
        score = r._verify("test", "the answer is pathlib", "pathlib")
        assert score >= 0.8

    def test_llm_self_verify(self, tmp_path):
        rt = _mock_runtime("7")
        r = _make_reasoner(tmp_path, runtime=rt)
        score = r._verify("test", "some answer", None)
        assert score == 0.7

    def test_parse_failure_defaults(self, tmp_path):
        rt = _mock_runtime("I cannot rate this")
        r = _make_reasoner(tmp_path, runtime=rt)
        score = r._verify("test", "answer", None)
        assert score == 0.5


# ---------------------------------------------------------------------------
# Trace storage and retrieval
# ---------------------------------------------------------------------------

class TestTraceStorage:

    def test_store_and_retrieve(self, tmp_path):
        r = _make_reasoner(tmp_path)
        r._store_trace("q", "reasoning", "answer", True, 0.7, 0)
        examples = r._get_best_examples(k=5)
        assert len(examples) == 1
        assert examples[0].query == "q"
        assert examples[0].correct is True

    def test_only_correct_traces_returned(self, tmp_path):
        r = _make_reasoner(tmp_path)
        r._store_trace("good", "r1", "a1", True, 0.8, 0)
        r._store_trace("bad", "r2", "a2", False, 0.3, 0)
        examples = r._get_best_examples(k=10)
        assert len(examples) == 1
        assert examples[0].query == "good"

    def test_staleness_increments(self, tmp_path):
        r = _make_reasoner(tmp_path)
        r._store_trace("q", "r", "a", True, 0.5, 0)
        r._increment_staleness()
        r._increment_staleness()
        examples = r._get_best_examples(k=1)
        assert examples[0].staleness == 2


# ---------------------------------------------------------------------------
# format_examples
# ---------------------------------------------------------------------------

class TestFormatExamples:

    def test_empty_traces(self, tmp_path):
        r = _make_reasoner(tmp_path)
        assert r._format_examples([]) == ""

    def test_formats_correctly(self, tmp_path):
        traces = [
            ReasoningTrace("t1", "How?", "Because X", "42", True, 0.5),
        ]
        r = _make_reasoner(tmp_path)
        text = r._format_examples(traces)
        assert "Q: How?" in text
        assert "ANSWER: 42" in text


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------

class TestStats:

    def test_empty_stats(self, tmp_path):
        r = _make_reasoner(tmp_path)
        s = r.stats()
        assert s["total_traces"] == 0
        assert s["accuracy"] == 0.0

    def test_populated_stats(self, tmp_path):
        r = _make_reasoner(tmp_path)
        r._store_trace("q1", "r1", "a1", True, 0.8, 0)
        r._store_trace("q2", "r2", "a2", True, 0.6, 0)
        r._store_trace("q3", "r3", "a3", False, 0.3, 0)
        s = r.stats()
        assert s["total_traces"] == 3
        assert s["correct_traces"] == 2
        assert s["accuracy"] == pytest.approx(0.667, abs=0.01)
