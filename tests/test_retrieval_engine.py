"""Tests for core/retrieval_engine.py -- retrieval pipeline."""

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
    )


class TestRetrieve:
    def test_returns_results(self, engine):
        results = engine.retrieve("how do I sort a list")
        assert len(results) >= 1
        assert "content" in results[0]

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

    def test_respects_rerank_top_n(self, mock_embedder, mock_index):
        engine = RetrievalEngine(
            embedder=mock_embedder, index=mock_index, rerank_top_n=5,
        )
        results = engine.retrieve("test query")
        assert len(results) <= 5
