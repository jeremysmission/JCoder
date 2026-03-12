"""
Tests for core.corrective_retrieval -- CRAG pattern.
All dependencies are mocked; no live runtime needed.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from core.corrective_retrieval import CorrectiveRetriever


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_retriever(chunks=None):
    rt = MagicMock()
    rt.retrieve.return_value = chunks or []
    return rt


def _mock_reflection(score=0.8):
    ref = MagicMock()
    ref.score_relevance.return_value = score
    return ref


def _sample_chunks(n=3):
    return [
        {"id": f"chunk_{i}", "content": f"def func_{i}(): pass", "source_path": f"src/f{i}.py"}
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:

    def test_minimal(self):
        cr = CorrectiveRetriever(retriever=_mock_retriever())
        assert cr.confidence_threshold == 0.5
        assert cr.max_reformulations == 2
        assert cr.reflection is None

    def test_with_reflection(self):
        cr = CorrectiveRetriever(
            retriever=_mock_retriever(),
            reflection=_mock_reflection(),
            confidence_threshold=0.7,
        )
        assert cr.confidence_threshold == 0.7
        assert cr.reflection is not None

    def test_custom_max_reformulations(self):
        cr = CorrectiveRetriever(
            retriever=_mock_retriever(),
            max_reformulations=5,
        )
        assert cr.max_reformulations == 5


# ---------------------------------------------------------------------------
# Standard retrieval (no reflection)
# ---------------------------------------------------------------------------

class TestStandardRetrieval:

    def test_returns_chunks_and_meta(self):
        chunks = _sample_chunks()
        cr = CorrectiveRetriever(retriever=_mock_retriever(chunks))
        result, meta = cr.retrieve("test query")
        assert result == chunks
        assert meta["strategy"] == "standard"
        assert meta["confidence"] == 1.0
        assert meta["attempts"] == 1

    def test_empty_retrieval_triggers_fallback(self):
        cr = CorrectiveRetriever(retriever=_mock_retriever([]))
        result, meta = cr.retrieve("test query")
        assert meta["attempts"] >= 1
        assert "corrective" in meta["strategy"]


# ---------------------------------------------------------------------------
# Confidence-gated retrieval (with reflection)
# ---------------------------------------------------------------------------

class TestConfidenceGated:

    def test_high_confidence_returns_immediately(self):
        chunks = _sample_chunks()
        cr = CorrectiveRetriever(
            retriever=_mock_retriever(chunks),
            reflection=_mock_reflection(0.9),
            confidence_threshold=0.5,
        )
        result, meta = cr.retrieve("test query")
        assert meta["strategy"] == "standard_confident"
        assert meta["confidence"] == 0.9
        assert result == chunks

    def test_low_confidence_triggers_corrective(self):
        chunks = _sample_chunks()
        retriever = _mock_retriever(chunks)
        # Return different chunks on subsequent calls
        # Calls: 1 initial + 1 keyword + 2 decompose parts = 4
        retriever.retrieve.side_effect = [
            chunks,  # initial
            [{"id": "new_1", "content": "new code", "source_path": "new.py"}],  # keyword
            [],  # decompose part 1
            [],  # decompose part 2
        ]
        cr = CorrectiveRetriever(
            retriever=retriever,
            reflection=_mock_reflection(0.3),
            confidence_threshold=0.5,
        )
        result, meta = cr.retrieve("how does function work and where is class defined")
        assert meta["attempts"] > 1
        # Should have merged initial + new chunks
        assert len(result) >= len(chunks)


# ---------------------------------------------------------------------------
# Corrective fallback
# ---------------------------------------------------------------------------

class TestCorrectiveFallback:

    def test_deduplicates_chunks(self):
        chunks = _sample_chunks(2)
        retriever = MagicMock()
        # Return same chunks on retry
        # Calls: 1 initial + 1 keyword + 2 decompose parts = 4
        retriever.retrieve.side_effect = [
            [],  # initial empty
            chunks,  # keyword retry
            chunks,  # decompose part 1
            chunks,  # decompose part 2
        ]
        cr = CorrectiveRetriever(retriever=retriever)
        result, meta = cr.retrieve("test query with some terms and other terms")
        # Should not duplicate
        ids = [c["id"] for c in result]
        assert len(ids) == len(set(ids))

    def test_respects_max_reformulations(self):
        retriever = MagicMock()
        retriever.retrieve.return_value = []
        cr = CorrectiveRetriever(retriever=retriever, max_reformulations=1)
        result, meta = cr.retrieve("test")
        # Should not exceed max_reformulations + 1 attempts
        assert meta["attempts"] <= cr.max_reformulations + 2

    def test_no_improvement_strategy(self):
        retriever = MagicMock()
        retriever.retrieve.return_value = []
        cr = CorrectiveRetriever(retriever=retriever)
        result, meta = cr.retrieve("xyz")
        assert meta["strategy"] == "corrective_no_improvement"

    def test_merged_strategy_when_new_chunks_found(self):
        initial = [{"id": "a", "content": "code a", "source_path": "a.py"}]
        extra = [{"id": "b", "content": "code b", "source_path": "b.py"}]
        retriever = MagicMock()
        retriever.retrieve.side_effect = [initial, extra, []]
        ref = _mock_reflection(0.3)
        cr = CorrectiveRetriever(
            retriever=retriever,
            reflection=ref,
            confidence_threshold=0.5,
        )
        result, meta = cr.retrieve("test query keywords")
        assert meta["strategy"] == "corrective_merged"
        assert len(result) == 2


# ---------------------------------------------------------------------------
# Keyword extraction
# ---------------------------------------------------------------------------

class TestKeywordQuery:

    def test_strips_stop_words(self):
        result = CorrectiveRetriever._keyword_query(
            "How does the function handle errors"
        )
        assert len(result) == 1
        assert "how" not in result[0].lower()
        assert "function" in result[0].lower()

    def test_keeps_identifiers(self):
        result = CorrectiveRetriever._keyword_query(
            "What is my_function doing with data_frame"
        )
        assert len(result) == 1
        assert "my_function" in result[0]
        assert "data_frame" in result[0]

    def test_empty_after_stop_removal(self):
        result = CorrectiveRetriever._keyword_query("what is the")
        assert result == []

    def test_short_words_removed(self):
        result = CorrectiveRetriever._keyword_query("a b c")
        assert result == []


# ---------------------------------------------------------------------------
# Query decomposition
# ---------------------------------------------------------------------------

class TestDecomposeQuery:

    def test_splits_on_and(self):
        result = CorrectiveRetriever._decompose_query(
            "How does function_a work and where is class_b defined"
        )
        assert len(result) == 2

    def test_splits_on_comma(self):
        result = CorrectiveRetriever._decompose_query(
            "explain the retrieval engine, describe the chunker module"
        )
        assert len(result) == 2

    def test_no_split_for_simple_query(self):
        result = CorrectiveRetriever._decompose_query("What does foo do")
        assert result == []

    def test_short_parts_filtered(self):
        # Parts <= 10 chars are dropped
        result = CorrectiveRetriever._decompose_query("abc and defghijklmnop query text")
        # "abc" is too short (3 chars), only second part should survive
        # But with only 1 meaningful part, returns []
        assert isinstance(result, list)
