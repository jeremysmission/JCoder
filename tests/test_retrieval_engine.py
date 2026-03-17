"""Tests for core/retrieval_engine.py -- adaptive retrieval + confidence gating."""

import sys, os
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.retrieval_engine import RetrievalEngine


@pytest.fixture
def mock_embedder():
    e = MagicMock()
    e.embed_single.return_value = [0.1] * 768
    return e


@pytest.fixture
def mock_index():
    idx = MagicMock()
    idx.hybrid_search.return_value = [
        (0.9, {"content": f"chunk_{i}", "source_path": f"src/{i}.py", "score": 0.9 - i * 0.1})
        for i in range(20)
    ]
    return idx


@pytest.fixture
def engine(mock_embedder, mock_index):
    return RetrievalEngine(
        embedder=mock_embedder,
        index=mock_index,
        top_k=50,
        rerank_top_n=10,
        adaptive_k=True,
        min_k=3,
        max_k=15,
    )


class TestComplexityEstimation:
    def test_simple_query(self, engine):
        score = engine._estimate_complexity("what is a list")
        assert score < 0.3

    def test_complex_query(self, engine):
        score = engine._estimate_complexity(
            "how does the authentication middleware interact with the "
            "database connection pool and what are the trade-offs between "
            "connection pooling strategies in a distributed architecture"
        )
        assert score > 0.3

    def test_api_lookup(self, engine):
        score = engine._estimate_complexity("what is the syntax for dict comprehension")
        assert score < 0.2

    def test_architecture_question(self, engine):
        score = engine._estimate_complexity("explain the design pattern used here")
        assert score > 0.1

    def test_clamped_0_to_1(self, engine):
        assert 0.0 <= engine._estimate_complexity("x") <= 1.0
        assert 0.0 <= engine._estimate_complexity("a " * 100) <= 1.0


class TestAdaptiveK:
    def test_simple_query_gets_fewer_chunks(self, engine):
        k = engine._adaptive_top_n("what is a list")
        assert k <= 5

    def test_complex_query_gets_more_chunks(self, engine):
        k = engine._adaptive_top_n(
            "how does the authentication middleware interact with the "
            "database connection pool in a distributed architecture"
        )
        assert k >= 5

    def test_respects_min_k(self, engine):
        k = engine._adaptive_top_n("x")
        assert k >= engine.min_k

    def test_respects_max_k(self, engine):
        k = engine._adaptive_top_n("a " * 100 + "explain architecture design pattern")
        assert k <= engine.max_k

    def test_disabled_returns_rerank_top_n(self, mock_embedder, mock_index):
        engine = RetrievalEngine(
            embedder=mock_embedder, index=mock_index,
            adaptive_k=False, rerank_top_n=7,
        )
        assert engine._adaptive_top_n("anything") == 7


class TestConfidenceGating:
    def test_filters_low_confidence(self, engine):
        results = [
            {"content": "good", "federated_score": 0.8},
            {"content": "ok", "federated_score": 0.3},
            {"content": "bad", "federated_score": 0.05},
        ]
        gated = engine._apply_confidence_gate(results, target_n=10)
        assert len(gated) == 2  # "bad" dropped (below 0.15 floor)

    def test_all_below_floor_returns_one(self, engine):
        results = [
            {"content": "low1", "federated_score": 0.05},
            {"content": "low2", "federated_score": 0.02},
        ]
        gated = engine._apply_confidence_gate(results, target_n=10)
        assert len(gated) == 1  # Best one only

    def test_empty_returns_empty(self, engine):
        assert engine._apply_confidence_gate([], target_n=5) == []

    def test_respects_target_n(self, engine):
        results = [{"content": f"r{i}", "federated_score": 0.9} for i in range(20)]
        gated = engine._apply_confidence_gate(results, target_n=5)
        assert len(gated) == 5


class TestRetrieve:
    def test_returns_results(self, engine):
        results = engine.retrieve("how do I sort a list")
        assert len(results) >= 1
        assert "content" in results[0]

    def test_adaptive_varies_result_count(self, engine):
        simple = engine.retrieve("what is a list")
        complex_q = engine.retrieve(
            "explain how the authentication architecture interacts "
            "with the database design pattern in this distributed system"
        )
        # Complex should get more results (or at least not fewer)
        assert len(complex_q) >= len(simple) or len(simple) <= engine.min_k

    def test_no_results_returns_empty(self, mock_embedder):
        idx = MagicMock()
        idx.hybrid_search.return_value = []
        engine = RetrievalEngine(embedder=mock_embedder, index=idx)
        assert engine.retrieve("nonexistent topic") == []

    def test_with_reranker(self, mock_embedder, mock_index):
        reranker = MagicMock()
        reranker.enabled = True
        reranker.rerank.return_value = [(0, 0.95), (2, 0.80)]
        engine = RetrievalEngine(
            embedder=mock_embedder, index=mock_index, reranker=reranker,
        )
        results = engine.retrieve("test query")
        assert len(results) >= 1
        reranker.rerank.assert_called_once()
