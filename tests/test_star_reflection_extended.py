"""
Extended tests for STaR reasoning and Reflection engine (Sprint 9).
Covers rationalization, chain-of-thought verification, filtering,
bootstrap learning, self-critique, iterative refinement, and depth limits.
All LLM calls are mocked; no live runtime needed.
"""

from __future__ import annotations

from typing import Dict, List
from unittest.mock import MagicMock, call

import pytest

from core.star_reasoner import ReasoningTrace, STaRReasoner, STaRResult
from core.reflection import ReflectionEngine, _extract_score


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _mock_runtime(*responses):
    """Create a mock runtime returning responses in order (or cycling last)."""
    rt = MagicMock()
    if len(responses) == 1:
        rt.generate.return_value = responses[0]
    else:
        rt.generate.side_effect = list(responses)
    return rt


def _make_reasoner(tmp_path, runtime=None, verify_fn=None, seed=42):
    rt = runtime or _mock_runtime("Step 1: think\nANSWER: 42")
    return STaRReasoner(
        runtime=rt, verify_fn=verify_fn,
        db_path=str(tmp_path / "star.db"), seed=seed,
    )


def _context_fn(query: str) -> List[str]:
    return ["def example(): return 1"]


def _sample_chunks():
    return [
        {"content": "def foo(): return 42", "source_path": "src/foo.py"},
        {"content": "class Bar:\n    pass", "source_path": "src/bar.py"},
    ]


# ===========================================================================
# STaR Reasoner -- Rationalization Generation
# ===========================================================================

class TestRationalizationGeneration:
    """Rationalization: generate reasoning that leads to a known answer."""

    def test_rationalize_returns_reasoning(self, tmp_path):
        rt = _mock_runtime(
            "Step 1: We know the answer uses pathlib.\n"
            "Step 2: pathlib.Path handles cross-platform paths.\n"
            "Therefore, pathlib is the correct approach."
        )
        r = _make_reasoner(tmp_path, runtime=rt)
        result = r._rationalize("How to handle file paths?", "Use pathlib")
        assert result is not None
        assert "pathlib" in result

    def test_rationalize_none_when_too_short(self, tmp_path):
        rt = _mock_runtime("short")  # fewer than 30 chars
        r = _make_reasoner(tmp_path, runtime=rt)
        result = r._rationalize("question", "answer")
        assert result is None

    def test_rationalize_none_on_exception(self, tmp_path):
        rt = MagicMock()
        rt.generate.side_effect = RuntimeError("LLM down")
        r = _make_reasoner(tmp_path, runtime=rt)
        result = r._rationalize("q", "a")
        assert result is None

    def test_rationalize_hint_truncated_to_300(self, tmp_path):
        rt = _mock_runtime(
            "Step 1: analyze the long answer and extract key points for reasoning."
        )
        r = _make_reasoner(tmp_path, runtime=rt)
        long_answer = "x" * 500
        r._rationalize("q", long_answer)
        prompt_used = rt.generate.call_args[1]["question"]
        # The hint in the prompt should be truncated
        assert "x" * 300 in prompt_used
        assert "x" * 301 not in prompt_used


# ===========================================================================
# STaR Reasoner -- Chain-of-Thought Verification
# ===========================================================================

class TestChainOfThoughtVerification:
    """Verify that reasoning chains are scored correctly."""

    def test_custom_verifier_overrides_llm(self, tmp_path):
        r = _make_reasoner(tmp_path, verify_fn=lambda q, a: 0.95)
        score = r._verify("q", "a", "ground_truth")
        assert score == 0.95

    def test_ground_truth_substring_match(self, tmp_path):
        r = _make_reasoner(tmp_path)
        # answer contains ground truth
        score = r._verify("q", "the answer is pathlib module", "pathlib")
        assert score >= 0.8

    def test_ground_truth_reverse_match(self, tmp_path):
        r = _make_reasoner(tmp_path)
        # ground truth contains answer
        score = r._verify("q", "pathlib", "use pathlib for paths")
        assert score >= 0.8

    def test_llm_verify_parses_score(self, tmp_path):
        rt = _mock_runtime("9")
        r = _make_reasoner(tmp_path, runtime=rt)
        score = r._verify("q", "a", None)
        assert score == 0.9

    def test_llm_verify_unparseable_defaults(self, tmp_path):
        rt = _mock_runtime("cannot determine quality")
        r = _make_reasoner(tmp_path, runtime=rt)
        score = r._verify("q", "a", None)
        assert score == 0.5

    def test_llm_verify_exception_defaults(self, tmp_path):
        rt = MagicMock()
        rt.generate.side_effect = Exception("timeout")
        r = _make_reasoner(tmp_path, runtime=rt)
        score = r._verify("q", "a", None)
        assert score == 0.5


# ===========================================================================
# STaR Reasoner -- Correct vs Incorrect Rationalization Scoring
# ===========================================================================

class TestRationalizationScoring:
    """Correct rationalizations stored; incorrect ones discarded."""

    def test_correct_answer_stored_without_rationalization(self, tmp_path):
        rt = _mock_runtime("Reasoning\nANSWER: correct")
        r = _make_reasoner(tmp_path, runtime=rt, verify_fn=lambda q, a: 0.9)
        result = r.run_iteration(["q1"], _context_fn, iteration=0)
        assert result.correct_first_try == 1
        assert result.rationalized == 0
        assert result.traces_stored == 1

    def test_incorrect_without_gt_not_stored(self, tmp_path):
        rt = _mock_runtime("ANSWER: wrong")
        r = _make_reasoner(tmp_path, runtime=rt, verify_fn=lambda q, a: 0.1)
        result = r.run_iteration(
            queries=["q1"], context_fn=_context_fn,
            ground_truth=None, iteration=0,
        )
        assert result.correct_first_try == 0
        assert result.rationalized == 0
        assert result.traces_stored == 0

    def test_incorrect_with_gt_rationalizes(self, tmp_path):
        rt = MagicMock()
        rt.generate.side_effect = [
            "ANSWER: wrong",  # initial attempt
            "Step 1: The correct approach uses os.path because it handles "
            "separators. Step 2: os.path.join is cross-platform.",
        ]
        r = _make_reasoner(tmp_path, runtime=rt, verify_fn=lambda q, a: 0.2)
        result = r.run_iteration(
            ["q1"], _context_fn,
            ground_truth=["Use os.path.join"], iteration=0,
        )
        assert result.rationalized == 1
        assert result.traces_stored == 1


# ===========================================================================
# STaR Reasoner -- Filtering Low-Quality Rationalizations
# ===========================================================================

class TestRationalizationFiltering:
    """Low-quality rationalizations (too short) are filtered out."""

    def test_short_rationalization_filtered(self, tmp_path):
        rt = MagicMock()
        rt.generate.side_effect = [
            "ANSWER: wrong",  # initial attempt
            "ok",  # too-short rationalization (< 30 chars)
        ]
        r = _make_reasoner(tmp_path, runtime=rt, verify_fn=lambda q, a: 0.1)
        result = r.run_iteration(
            ["q1"], _context_fn,
            ground_truth=["correct answer"], iteration=0,
        )
        assert result.rationalized == 0
        assert result.traces_stored == 0

    def test_adequate_rationalization_kept(self, tmp_path):
        rt = MagicMock()
        rt.generate.side_effect = [
            "ANSWER: wrong",
            "Step 1: We analyze the input format carefully. "
            "Step 2: Then apply the transformation rule.",
        ]
        r = _make_reasoner(tmp_path, runtime=rt, verify_fn=lambda q, a: 0.1)
        result = r.run_iteration(
            ["q1"], _context_fn,
            ground_truth=["correct"], iteration=0,
        )
        assert result.rationalized == 1

    def test_only_correct_traces_in_examples(self, tmp_path):
        r = _make_reasoner(tmp_path)
        r._store_trace("good", "solid reasoning", "right", True, 0.9, 0)
        r._store_trace("bad", "flawed reasoning", "wrong", False, 0.2, 0)
        r._store_trace("ok", "decent reasoning", "close", True, 0.6, 0)
        examples = r._get_best_examples(k=10)
        assert all(e.correct for e in examples)
        assert len(examples) == 2


# ===========================================================================
# STaR Reasoner -- Bootstrap Learning from Filtered Examples
# ===========================================================================

class TestBootstrapLearning:
    """Multi-iteration bootstrap: traces from iteration N used in N+1."""

    def test_examples_injected_into_prompt(self, tmp_path):
        rt = _mock_runtime("ANSWER: result")
        r = _make_reasoner(tmp_path, runtime=rt, verify_fn=lambda q, a: 0.9)
        # Seed a trace
        r._store_trace("prev q", "prev reasoning chain", "prev answer", True, 0.8, 0)
        r.answer_with_reasoning("new q", ["ctx"])
        prompt = rt.generate.call_args[1]["question"]
        assert "prev q" in prompt

    def test_multi_iteration_accuracy_tracking(self, tmp_path):
        rt = _mock_runtime("ANSWER: ok")
        r = _make_reasoner(tmp_path, runtime=rt, verify_fn=lambda q, a: 0.8)
        r0 = r.run_iteration(["q1", "q2"], _context_fn, iteration=0)
        r1 = r.run_iteration(["q1", "q2"], _context_fn, iteration=1)
        assert r0.accuracy == 1.0
        assert r1.accuracy == 1.0
        assert isinstance(r1.improvement_over_previous, float)

    def test_priority_sampling_shuffles(self, tmp_path):
        r = _make_reasoner(tmp_path, seed=123)
        queries = [f"q{i}" for i in range(20)]
        shuffled = r._prioritize_queries(queries, 0)
        assert set(shuffled) == set(queries)
        # With 20 items and a fixed seed, very unlikely to stay sorted
        assert shuffled != queries

    def test_staleness_increases_over_iterations(self, tmp_path):
        r = _make_reasoner(tmp_path)
        r._store_trace("q", "r", "a", True, 0.5, 0)
        for _ in range(5):
            r._increment_staleness()
        examples = r._get_best_examples(k=1)
        assert examples[0].staleness == 5


# ===========================================================================
# Reflection -- Self-Critique Generation
# ===========================================================================

class TestSelfCritique:
    """ReflectionEngine generates critique scores for answers."""

    def test_low_relevance_detected(self):
        rt = _mock_runtime("2")
        engine = ReflectionEngine(rt)
        score = engine.score_relevance("How to sort a list?", [
            {"content": "def connect_db(): ...", "source_path": "db.py"}
        ])
        assert score == 0.2

    def test_high_support_detected(self):
        rt = _mock_runtime("9")
        engine = ReflectionEngine(rt)
        score = engine.score_support("sorted() returns a new list", [
            {"content": "sorted(iterable) -> list", "source_path": "builtins.py"}
        ])
        assert score == 0.9

    def test_full_critique_identifies_weakness(self):
        rt = MagicMock()
        # Low relevance, low support, decent usefulness
        rt.generate.side_effect = ["3", "2", "7"]
        engine = ReflectionEngine(rt)
        result = engine.full_reflection("How to sort?", _sample_chunks(), "Use sorted()")
        assert result["relevant"] == 0.3
        assert result["supported"] == 0.2
        assert result["useful"] == 0.7
        assert result["confidence"] < 0.5


# ===========================================================================
# Reflection -- Improvement Suggestion Extraction
# ===========================================================================

class TestImprovementExtraction:
    """Identify which signal is weakest to guide improvement."""

    def test_weakest_signal_identified(self):
        rt = MagicMock()
        rt.generate.side_effect = ["8", "3", "7"]
        engine = ReflectionEngine(rt)
        result = engine.full_reflection("q", _sample_chunks(), "a")
        weakest = min(result, key=lambda k: result[k] if k != "confidence" else 999)
        assert weakest == "supported"

    def test_all_signals_high_means_high_confidence(self):
        rt = MagicMock()
        rt.generate.side_effect = ["9", "9", "9"]
        engine = ReflectionEngine(rt)
        result = engine.full_reflection("q", _sample_chunks(), "a")
        assert result["confidence"] >= 0.8

    def test_all_signals_low_means_low_confidence(self):
        rt = MagicMock()
        rt.generate.side_effect = ["1", "1", "1"]
        engine = ReflectionEngine(rt)
        result = engine.full_reflection("q", _sample_chunks(), "a")
        assert result["confidence"] <= 0.2


# ===========================================================================
# Reflection -- Iterative Refinement (reflect -> improve -> reflect)
# ===========================================================================

class TestIterativeRefinement:
    """Simulate reflect -> improve -> reflect cycles."""

    def test_quality_improves_across_iterations(self):
        """Simulate: first reflection low, then after 'improvement' higher."""
        rt = MagicMock()
        # Iteration 1: low scores
        # Iteration 2: higher scores (after simulated improvement)
        rt.generate.side_effect = [
            "4", "3", "5",  # round 1: rel=0.4, sup=0.3, use=0.5
            "7", "8", "8",  # round 2: rel=0.7, sup=0.8, use=0.8
        ]
        engine = ReflectionEngine(rt)
        r1 = engine.full_reflection("q", _sample_chunks(), "initial answer")
        r2 = engine.full_reflection("q", _sample_chunks(), "improved answer")
        assert r2["confidence"] > r1["confidence"]

    def test_three_iteration_refinement(self):
        """Three rounds of reflection show monotonic improvement."""
        rt = MagicMock()
        rt.generate.side_effect = [
            "3", "3", "3",  # round 1
            "5", "5", "5",  # round 2
            "8", "8", "8",  # round 3
        ]
        engine = ReflectionEngine(rt)
        scores = []
        for _ in range(3):
            result = engine.full_reflection("q", _sample_chunks(), "a")
            scores.append(result["confidence"])
        assert scores[0] < scores[1] < scores[2]

    def test_no_improvement_detected(self):
        """When scores stay flat, no improvement is claimed."""
        rt = MagicMock()
        rt.generate.side_effect = ["5", "5", "5", "5", "5", "5"]
        engine = ReflectionEngine(rt)
        r1 = engine.full_reflection("q", _sample_chunks(), "a")
        r2 = engine.full_reflection("q", _sample_chunks(), "a")
        assert r1["confidence"] == pytest.approx(r2["confidence"], abs=0.01)


# ===========================================================================
# Reflection -- Depth Limit
# ===========================================================================

class TestReflectionDepthLimit:
    """Ensure reflection loops respect a maximum iteration count."""

    def test_depth_limit_stops_iteration(self):
        """Simulate a reflect-improve loop with a depth limit."""
        max_depth = 3
        rt = MagicMock()
        # All low scores -- would never converge
        rt.generate.return_value = "3"
        engine = ReflectionEngine(rt)

        iterations_run = 0
        threshold = 0.7
        for _ in range(max_depth):
            result = engine.full_reflection("q", _sample_chunks(), "a")
            iterations_run += 1
            if result["confidence"] >= threshold:
                break
        assert iterations_run == max_depth
        # 3 calls per full_reflection * 3 iterations = 9
        assert rt.generate.call_count == 9

    def test_early_exit_when_threshold_met(self):
        """Loop exits early when confidence exceeds threshold."""
        rt = MagicMock()
        # First round: high scores -- should exit immediately
        rt.generate.side_effect = ["9", "9", "9"] + ["5", "5", "5"] * 4
        engine = ReflectionEngine(rt)

        max_depth = 5
        iterations_run = 0
        for _ in range(max_depth):
            result = engine.full_reflection("q", _sample_chunks(), "a")
            iterations_run += 1
            if result["confidence"] >= 0.7:
                break
        assert iterations_run == 1
        assert rt.generate.call_count == 3


# ===========================================================================
# Reflection -- Quality Tracking Across Iterations
# ===========================================================================

class TestQualityTracking:
    """Track how scores evolve across multiple reflection rounds."""

    def test_confidence_history_recorded(self):
        rt = MagicMock()
        rt.generate.side_effect = [
            "4", "4", "4",
            "6", "6", "6",
            "8", "8", "8",
        ]
        engine = ReflectionEngine(rt)
        history = []
        for _ in range(3):
            result = engine.full_reflection("q", _sample_chunks(), "a")
            history.append(result["confidence"])
        assert len(history) == 3
        assert history[-1] > history[0]

    def test_individual_signal_tracking(self):
        rt = MagicMock()
        rt.generate.side_effect = [
            "3", "8", "5",  # round 1: rel low, sup high, use mid
            "7", "8", "5",  # round 2: rel improved, others same
        ]
        engine = ReflectionEngine(rt)
        r1 = engine.full_reflection("q", _sample_chunks(), "a")
        r2 = engine.full_reflection("q", _sample_chunks(), "a")
        assert r2["relevant"] > r1["relevant"]
        assert r2["supported"] == r1["supported"]
        assert r2["useful"] == r1["useful"]

    def test_weighted_confidence_formula(self):
        """Verify the exact formula: 0.3*rel + 0.4*sup + 0.3*use."""
        rt = MagicMock()
        rt.generate.side_effect = ["5", "10", "3"]
        engine = ReflectionEngine(rt)
        result = engine.full_reflection("q", _sample_chunks(), "a")
        expected = 0.3 * 0.5 + 0.4 * 1.0 + 0.3 * 0.3
        assert result["confidence"] == pytest.approx(expected, abs=0.01)

    def test_degradation_detected(self):
        """If scores get worse, confidence drops."""
        rt = MagicMock()
        rt.generate.side_effect = [
            "8", "8", "8",  # round 1: good
            "3", "3", "3",  # round 2: degraded
        ]
        engine = ReflectionEngine(rt)
        r1 = engine.full_reflection("q", _sample_chunks(), "a")
        r2 = engine.full_reflection("q", _sample_chunks(), "a")
        assert r2["confidence"] < r1["confidence"]
