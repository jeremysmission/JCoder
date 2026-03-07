"""Tests for RetrievalEngine federated search path."""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from core.federated_search import FederatedSearch, SearchResult
from core.retrieval_engine import RetrievalEngine


def _mock_embedder():
    emb = MagicMock()
    emb.embed_single = MagicMock(return_value=np.zeros(768))
    return emb


def _mock_index():
    idx = MagicMock()
    idx.index = None
    idx.metadata = []
    idx.hybrid_search = MagicMock(return_value=[])
    return idx


def _mock_federated(results: list[SearchResult]) -> FederatedSearch:
    fed = MagicMock(spec=FederatedSearch)
    fed.list_indexes.return_value = [{"name": "test", "count": 10, "weight": 1.0}]
    fed.search.return_value = results
    return fed


class TestFederatedRetrievalPath:

    def test_uses_federated_when_available(self):
        results = [
            SearchResult(
                content="hello world",
                source="a.py",
                index_name="code",
                score=0.9,
                metadata={"content": "hello world", "source_path": "a.py"},
            )
        ]
        fed = _mock_federated(results)
        engine = RetrievalEngine(
            embedder=_mock_embedder(),
            index=_mock_index(),
            federated=fed,
        )
        out = engine.retrieve("hello")
        assert len(out) == 1
        assert out[0]["content"] == "hello world"
        assert out[0]["federated_index"] == "code"
        assert out[0]["federated_score"] == 0.9
        fed.search.assert_called_once()

    def test_falls_back_to_single_when_no_federated(self):
        embedder = _mock_embedder()
        index = _mock_index()
        index.hybrid_search.return_value = [
            (0.8, {"content": "single result", "source_path": "b.py"})
        ]
        engine = RetrievalEngine(
            embedder=embedder,
            index=index,
            federated=None,
        )
        out = engine.retrieve("test")
        assert len(out) == 1
        assert out[0]["content"] == "single result"
        index.hybrid_search.assert_called_once()

    def test_falls_back_when_federated_empty(self):
        fed = MagicMock(spec=FederatedSearch)
        fed.list_indexes.return_value = []  # no indexes registered
        embedder = _mock_embedder()
        index = _mock_index()
        index.hybrid_search.return_value = [
            (0.7, {"content": "fallback", "source_path": "c.py"})
        ]
        engine = RetrievalEngine(
            embedder=embedder,
            index=index,
            federated=fed,
        )
        out = engine.retrieve("query")
        assert len(out) == 1
        assert out[0]["content"] == "fallback"

    def test_federated_empty_results(self):
        fed = _mock_federated([])
        engine = RetrievalEngine(
            embedder=_mock_embedder(),
            index=_mock_index(),
            federated=fed,
        )
        out = engine.retrieve("nothing")
        assert out == []

    def test_federated_with_reranker(self):
        results = [
            SearchResult(content=f"doc {i}", source=f"{i}.py",
                         index_name="code", score=0.9 - i * 0.1,
                         metadata={"content": f"doc {i}", "source_path": f"{i}.py"})
            for i in range(5)
        ]
        fed = _mock_federated(results)

        reranker = MagicMock()
        reranker.enabled = True
        # Reranker reverses the order
        reranker.rerank.return_value = [(4, 0.95), (3, 0.90), (2, 0.85)]

        engine = RetrievalEngine(
            embedder=_mock_embedder(),
            index=_mock_index(),
            reranker=reranker,
            federated=fed,
            rerank_top_n=3,
        )
        out = engine.retrieve("test")
        assert len(out) == 3
        assert out[0]["content"] == "doc 4"  # reranker put index 4 first
        assert out[1]["content"] == "doc 3"

    def test_federated_preserves_metadata(self):
        results = [
            SearchResult(
                content="code snippet",
                source="lib.py",
                index_name="csn_python",
                score=0.85,
                metadata={
                    "content": "code snippet",
                    "source_path": "lib.py",
                    "language": "python",
                    "repo": "example/repo",
                },
            )
        ]
        fed = _mock_federated(results)
        engine = RetrievalEngine(
            embedder=_mock_embedder(),
            index=_mock_index(),
            federated=fed,
        )
        out = engine.retrieve("python code")
        assert out[0]["language"] == "python"
        assert out[0]["repo"] == "example/repo"
        assert out[0]["federated_index"] == "csn_python"

    def test_federated_respects_top_k(self):
        results = [
            SearchResult(content=f"r{i}", source=f"{i}.py",
                         index_name="idx", score=0.5,
                         metadata={"content": f"r{i}"})
            for i in range(20)
        ]
        fed = _mock_federated(results)
        engine = RetrievalEngine(
            embedder=_mock_embedder(),
            index=_mock_index(),
            federated=fed,
            top_k=20,
            rerank_top_n=5,
        )
        out = engine.retrieve("test")
        # Without reranker, returns rerank_top_n items
        assert len(out) == 5
        fed.search.assert_called_with("test", top_k=20)

    def test_metadata_defaults_filled(self):
        """SearchResult with empty metadata still gets content/source."""
        results = [
            SearchResult(
                content="bare result",
                source="x.py",
                index_name="bare",
                score=0.5,
                metadata={},
            )
        ]
        fed = _mock_federated(results)
        engine = RetrievalEngine(
            embedder=_mock_embedder(),
            index=_mock_index(),
            federated=fed,
        )
        out = engine.retrieve("test")
        assert out[0]["content"] == "bare result"
        assert out[0]["source_path"] == "x.py"
