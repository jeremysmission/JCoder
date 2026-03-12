"""
Tests for core.reflection -- ReflectionEngine (Self-RAG).
All LLM calls are mocked; no live runtime needed.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from core.reflection import (
    ReflectionEngine,
    _extract_score,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_runtime(response: str = "7"):
    rt = MagicMock()
    rt.generate.return_value = response
    return rt


def _sample_chunks():
    return [
        {"content": "def foo(): return 42", "source_path": "src/foo.py"},
        {"content": "class Bar:\n    pass", "source_path": "src/bar.py"},
    ]


# ---------------------------------------------------------------------------
# _extract_score
# ---------------------------------------------------------------------------

class TestExtractScore:

    def test_single_digit(self):
        assert _extract_score("7") == 0.7

    def test_ten(self):
        assert _extract_score("10") == 1.0

    def test_zero(self):
        assert _extract_score("0") == 0.0

    def test_with_text(self):
        assert _extract_score("I would rate this a 8 out of 10") == 0.8

    def test_no_number(self):
        assert _extract_score("great answer") == 0.5

    def test_whitespace(self):
        assert _extract_score("  9  \n") == 0.9

    def test_multiple_numbers_takes_first(self):
        assert _extract_score("7 out of 10") == 0.7


# ---------------------------------------------------------------------------
# score_relevance (ISREL)
# ---------------------------------------------------------------------------

class TestScoreRelevance:

    def test_returns_score(self):
        rt = _mock_runtime("8")
        engine = ReflectionEngine(rt)
        score = engine.score_relevance("What does foo do?", _sample_chunks())
        assert score == 0.8

    def test_runtime_called(self):
        rt = _mock_runtime("6")
        engine = ReflectionEngine(rt)
        engine.score_relevance("test", _sample_chunks())
        rt.generate.assert_called_once()
        call_kwargs = rt.generate.call_args
        assert "relevant" in str(call_kwargs).lower() or "relevance" in str(call_kwargs).lower()


# ---------------------------------------------------------------------------
# score_support (ISSUP)
# ---------------------------------------------------------------------------

class TestScoreSupport:

    def test_returns_score(self):
        rt = _mock_runtime("9")
        engine = ReflectionEngine(rt)
        score = engine.score_support("foo returns 42", _sample_chunks())
        assert score == 0.9

    def test_runtime_called(self):
        rt = _mock_runtime("5")
        engine = ReflectionEngine(rt)
        engine.score_support("answer text", _sample_chunks())
        rt.generate.assert_called_once()


# ---------------------------------------------------------------------------
# score_usefulness (ISUSE)
# ---------------------------------------------------------------------------

class TestScoreUsefulness:

    def test_returns_score(self):
        rt = _mock_runtime("7")
        engine = ReflectionEngine(rt)
        score = engine.score_usefulness("What does foo do?", "foo returns 42")
        assert score == 0.7

    def test_runtime_called(self):
        rt = _mock_runtime("4")
        engine = ReflectionEngine(rt)
        engine.score_usefulness("q", "a")
        rt.generate.assert_called_once()


# ---------------------------------------------------------------------------
# full_reflection
# ---------------------------------------------------------------------------

class TestFullReflection:

    def test_returns_all_scores(self):
        rt = MagicMock()
        rt.generate.side_effect = ["8", "7", "9"]
        engine = ReflectionEngine(rt)
        result = engine.full_reflection("What is foo?", _sample_chunks(), "foo returns 42")
        assert result["relevant"] == 0.8
        assert result["supported"] == 0.7
        assert result["useful"] == 0.9
        assert "confidence" in result

    def test_confidence_is_weighted_aggregate(self):
        rt = MagicMock()
        # rel=8, sup=6, use=10
        rt.generate.side_effect = ["8", "6", "10"]
        engine = ReflectionEngine(rt)
        result = engine.full_reflection("q", _sample_chunks(), "a")
        # confidence = 0.3*0.8 + 0.4*0.6 + 0.3*1.0 = 0.24 + 0.24 + 0.30 = 0.78
        assert result["confidence"] == pytest.approx(0.78, abs=0.01)

    def test_three_llm_calls(self):
        rt = MagicMock()
        rt.generate.side_effect = ["5", "5", "5"]
        engine = ReflectionEngine(rt)
        engine.full_reflection("q", _sample_chunks(), "a")
        assert rt.generate.call_count == 3

    def test_parse_failure_defaults_to_half(self):
        rt = MagicMock()
        rt.generate.side_effect = ["no number", "also no number", "still no number"]
        engine = ReflectionEngine(rt)
        result = engine.full_reflection("q", _sample_chunks(), "a")
        assert result["relevant"] == 0.5
        assert result["supported"] == 0.5
        assert result["useful"] == 0.5


# ---------------------------------------------------------------------------
# _format_chunks
# ---------------------------------------------------------------------------

class TestFormatChunks:

    def test_formats_with_source(self):
        engine = ReflectionEngine(_mock_runtime())
        text = engine._format_chunks(_sample_chunks())
        assert "src/foo.py" in text
        assert "def foo" in text

    def test_empty_chunks(self):
        engine = ReflectionEngine(_mock_runtime())
        text = engine._format_chunks([])
        assert text == "(no chunks)"

    def test_truncation(self):
        huge_chunks = [
            {"content": "x" * 3000, "source_path": f"file_{i}.py"}
            for i in range(10)
        ]
        engine = ReflectionEngine(_mock_runtime())
        text = engine._format_chunks(huge_chunks, max_chars=500)
        assert len(text) <= 1500  # each chunk truncated to 500 chars max

    def test_missing_fields_handled(self):
        engine = ReflectionEngine(_mock_runtime())
        chunks = [{"content": "some code"}]  # no source_path
        text = engine._format_chunks(chunks)
        assert "some code" in text
        assert "?" in text  # default source
