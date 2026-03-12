"""
Tests for core.cascade -- Model Cascade (Cascadia Pattern).
All runtimes are mocked; no live LLM needed.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from core.cascade import (
    CascadeLevel,
    CascadeResult,
    ModelCascade,
    estimate_complexity,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_model_config(name="test-model"):
    mc = MagicMock()
    mc.model = name
    return mc


def _make_levels(n=3):
    """Create n cascade levels with increasing max_complexity."""
    return [
        CascadeLevel(
            name=f"tier_{i}",
            model_config=_mock_model_config(f"model_{i}"),
            max_complexity=0.3 * (i + 1),
            timeout_s=30,
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# estimate_complexity
# ---------------------------------------------------------------------------

class TestEstimateComplexity:

    def test_simple_query(self):
        score = estimate_complexity("What is a function?")
        assert score < 0.3

    def test_moderate_query(self):
        score = estimate_complexity(
            "How does the retrieval engine handle concurrent requests "
            "and what optimization strategies are used?"
        )
        assert score > 0.2

    def test_complex_query(self):
        score = estimate_complexity(
            "I need to refactor the database schema migration pipeline "
            "to support distributed async processing and optimize "
            "performance for concurrent parallel operations"
        )
        assert score > 0.5

    def test_very_short_query(self):
        score = estimate_complexity("find foo")
        assert score < 0.2

    def test_code_refs_increase_complexity(self):
        base = estimate_complexity("explain this function")
        with_refs = estimate_complexity(
            "explain core.retrieval.engine and utils.helper"
        )
        assert with_refs >= base

    def test_clamped_to_zero_one(self):
        low = estimate_complexity("hi")
        high = estimate_complexity(
            "refactor redesign architect optimize migrate security "
            "concurrent distributed async database schema and also "
            "integration performance vulnerability compare versus "
            "what would happen if we trade-off these three approaches, "
            "specifically for core.engine.main and utils.helper.run "
            "and db.schema.migrate?"
        )
        assert 0.0 <= low <= 1.0
        assert 0.0 <= high <= 1.0

    def test_multi_question(self):
        single = estimate_complexity("What does foo do?")
        multi = estimate_complexity("What does foo do? And where is bar defined?")
        assert multi >= single


# ---------------------------------------------------------------------------
# CascadeLevel / CascadeResult dataclasses
# ---------------------------------------------------------------------------

class TestDataclasses:

    def test_cascade_level(self):
        lv = CascadeLevel(
            name="fast", model_config=_mock_model_config(),
            max_complexity=0.3,
        )
        assert lv.timeout_s == 60

    def test_cascade_result(self):
        r = CascadeResult(
            answer="42", model_used="fast",
            level_index=0, complexity_score=0.1,
            escalated=False,
        )
        assert r.latency_ms == 0.0


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:

    @patch("core.cascade.NetworkGate")
    def test_sorts_levels(self, mock_gate):
        levels = [
            CascadeLevel("big", _mock_model_config(), 0.9),
            CascadeLevel("small", _mock_model_config(), 0.3),
            CascadeLevel("mid", _mock_model_config(), 0.6),
        ]
        cascade = ModelCascade(levels, gate=MagicMock())
        assert cascade.levels[0].name == "small"
        assert cascade.levels[1].name == "mid"
        assert cascade.levels[2].name == "big"

    @patch("core.cascade.NetworkGate")
    def test_custom_threshold(self, mock_gate):
        cascade = ModelCascade(
            _make_levels(2), gate=MagicMock(),
            confidence_threshold=0.7,
        )
        assert cascade.confidence_threshold == 0.7


# ---------------------------------------------------------------------------
# route
# ---------------------------------------------------------------------------

class TestRoute:

    @patch("core.cascade.Runtime")
    @patch("core.cascade.NetworkGate")
    def test_simple_query_uses_first_level(self, mock_gate, mock_rt_cls):
        mock_rt = MagicMock()
        mock_rt.generate.return_value = "simple answer"
        mock_rt_cls.return_value = mock_rt

        cascade = ModelCascade(_make_levels(3), gate=MagicMock())
        result = cascade.route("What is foo?", ["context"])
        assert isinstance(result, CascadeResult)
        assert result.answer == "simple answer"
        assert result.escalated is False

    @patch("core.cascade.Runtime")
    @patch("core.cascade.NetworkGate")
    def test_complex_query_skips_low_levels(self, mock_gate, mock_rt_cls):
        mock_rt = MagicMock()
        mock_rt.generate.return_value = "complex answer"
        mock_rt_cls.return_value = mock_rt

        levels = _make_levels(3)  # thresholds: 0.3, 0.6, 0.9
        cascade = ModelCascade(levels, gate=MagicMock())
        # Force high complexity
        result = cascade.route(
            "Refactor the distributed database schema migration "
            "to optimize concurrent async parallel processing",
            ["ctx"],
        )
        assert isinstance(result, CascadeResult)
        assert result.complexity_score > 0.3

    @patch("core.cascade.Runtime")
    @patch("core.cascade.NetworkGate")
    def test_escalation_on_low_confidence(self, mock_gate, mock_rt_cls):
        mock_rt = MagicMock()
        mock_rt.generate.return_value = "answer"
        mock_rt_cls.return_value = mock_rt

        levels = _make_levels(3)
        cascade = ModelCascade(levels, gate=MagicMock(), confidence_threshold=0.9)

        # Confidence fn returns low -> should escalate
        call_count = [0]
        def conf_fn(answer):
            call_count[0] += 1
            if call_count[0] < 3:
                return 0.1  # low confidence
            return 0.95  # high confidence at last level

        result = cascade.route("What is foo?", ["ctx"], confidence_fn=conf_fn)
        assert result.answer == "answer"

    @patch("core.cascade.Runtime")
    @patch("core.cascade.NetworkGate")
    def test_escalation_on_exception(self, mock_gate, mock_rt_cls):
        call_count = [0]
        def gen_side_effect(q, ctx, **kw):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("model crashed")
            return "fallback answer"

        mock_rt = MagicMock()
        mock_rt.generate.side_effect = gen_side_effect
        mock_rt_cls.return_value = mock_rt

        cascade = ModelCascade(_make_levels(2), gate=MagicMock())
        result = cascade.route("What is foo?", ["ctx"])
        assert result.escalated is True

    @patch("core.cascade.Runtime")
    @patch("core.cascade.NetworkGate")
    def test_all_levels_fail(self, mock_gate, mock_rt_cls):
        mock_rt = MagicMock()
        mock_rt.generate.side_effect = RuntimeError("crash")
        mock_rt_cls.return_value = mock_rt

        cascade = ModelCascade(_make_levels(2), gate=MagicMock())
        result = cascade.route("foo", ["ctx"])
        assert "failed" in result.answer.lower()
        assert result.escalated is True

    @patch("core.cascade.Runtime")
    @patch("core.cascade.NetworkGate")
    def test_latency_tracked(self, mock_gate, mock_rt_cls):
        mock_rt = MagicMock()
        mock_rt.generate.return_value = "answer"
        mock_rt_cls.return_value = mock_rt

        cascade = ModelCascade(_make_levels(1), gate=MagicMock())
        result = cascade.route("What is foo?", ["ctx"])
        assert result.latency_ms >= 0


# ---------------------------------------------------------------------------
# close
# ---------------------------------------------------------------------------

class TestClose:

    @patch("core.cascade.Runtime")
    @patch("core.cascade.NetworkGate")
    def test_close_releases_runtimes(self, mock_gate, mock_rt_cls):
        mock_rt = MagicMock()
        mock_rt.generate.return_value = "answer"
        mock_rt_cls.return_value = mock_rt

        cascade = ModelCascade(_make_levels(2), gate=MagicMock())
        cascade.route("test", ["ctx"])
        cascade.close()
        assert len(cascade._runtimes) == 0
