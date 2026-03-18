"""
Unit tests for core.retrieval_engine.RetrievalEngine.

All external dependencies (embedder, index, reranker, federated search)
are mocked -- no FAISS, no Ollama, no network calls needed.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Ensure project root is on sys.path (conftest also does this)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.retrieval_engine import RetrievalEngine


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------

def _make_embedder(vector: Optional[np.ndarray] = None) -> MagicMock:
    """Return a mock IEmbedder whose embed_single returns *vector*."""
    emb = MagicMock()
    if vector is None:
        vector = np.random.randn(768).astype(np.float32)
    emb.embed_single.return_value = vector
    return emb


def _make_index(candidates: Optional[List[Tuple[float, Dict]]] = None) -> MagicMock:
    """Return a mock IndexEngine whose hybrid_search returns *candidates*."""
    idx = MagicMock()
    idx.hybrid_search.return_value = candidates or []
    return idx


def _make_reranker(
    reranked: Optional[List[Tuple[int, float]]] = None,
    enabled: bool = True,
) -> MagicMock:
    """Return a mock IReranker."""
    rr = MagicMock()
    rr.enabled = enabled
    rr.rerank.return_value = reranked or []
    return rr


def _make_federated(
    indexes: Optional[List[str]] = None,
    results: Optional[list] = None,
) -> MagicMock:
    """Return a mock FederatedSearch."""
    fed = MagicMock()
    fed.list_indexes.return_value = indexes or []
    fed.search.return_value = results or []
    return fed


def _candidate(score: float, content: str, path: str = "a.py") -> Tuple[float, Dict]:
    """Helper to build a (score, metadata) candidate tuple."""
    return (score, {"content": content, "source_path": path})


# ---------------------------------------------------------------------------
# Constructor tests
# ---------------------------------------------------------------------------

class TestConstructor:
    def test_defaults(self):
        emb = _make_embedder()
        idx = _make_index()
        engine = RetrievalEngine(embedder=emb, index=idx)
        assert engine.top_k == 50
        assert engine.rerank_top_n == 10
        assert engine.reranker is None
        assert engine.federated is None

    def test_custom_params(self):
        emb = _make_embedder()
        idx = _make_index()
        rr = _make_reranker()
        engine = RetrievalEngine(
            embedder=emb, index=idx, reranker=rr,
            top_k=20, rerank_top_n=5,
        )
        assert engine.top_k == 20
        assert engine.rerank_top_n == 5
        assert engine.reranker is rr

    def test_with_federated(self):
        emb = _make_embedder()
        idx = _make_index()
        fed = _make_federated()
        engine = RetrievalEngine(embedder=emb, index=idx, federated=fed)
        assert engine.federated is fed


# ---------------------------------------------------------------------------
# Single-index retrieval (_retrieve_single)
# ---------------------------------------------------------------------------

class TestRetrieveSingle:
    def test_embeds_query(self):
        vec = np.ones(768, dtype=np.float32)
        emb = _make_embedder(vec)
        idx = _make_index([])
        engine = RetrievalEngine(embedder=emb, index=idx)

        engine.retrieve("hello world")

        emb.embed_single.assert_called_once_with("hello world")
        idx.hybrid_search.assert_called_once()
        call_args = idx.hybrid_search.call_args
        np.testing.assert_array_equal(call_args[0][0], vec)
        assert call_args[0][1] == "hello world"

    def test_empty_candidates_returns_empty(self):
        emb = _make_embedder()
        idx = _make_index([])
        engine = RetrievalEngine(embedder=emb, index=idx)

        result = engine.retrieve("anything")
        assert result == []

    def test_no_reranker_returns_top_n(self):
        candidates = [
            _candidate(0.9, "chunk A"),
            _candidate(0.8, "chunk B"),
            _candidate(0.7, "chunk C"),
        ]
        emb = _make_embedder()
        idx = _make_index(candidates)
        engine = RetrievalEngine(
            embedder=emb, index=idx, rerank_top_n=2,
        )

        result = engine.retrieve("query")
        assert len(result) == 2
        assert result[0]["content"] == "chunk A"
        assert result[1]["content"] == "chunk B"

    def test_reranker_reorders(self):
        candidates = [
            _candidate(0.9, "chunk A"),
            _candidate(0.8, "chunk B"),
            _candidate(0.7, "chunk C"),
        ]
        # Reranker says index 2 (chunk C) is best, then index 0 (chunk A)
        rr = _make_reranker(reranked=[(2, 0.95), (0, 0.80)])
        emb = _make_embedder()
        idx = _make_index(candidates)
        engine = RetrievalEngine(
            embedder=emb, index=idx, reranker=rr, rerank_top_n=2,
        )

        result = engine.retrieve("query")
        assert len(result) == 2
        assert result[0]["content"] == "chunk C"
        assert result[1]["content"] == "chunk A"
        rr.rerank.assert_called_once()

    def test_reranker_disabled_skips_rerank(self):
        candidates = [_candidate(0.9, "chunk A")]
        rr = _make_reranker(enabled=False)
        emb = _make_embedder()
        idx = _make_index(candidates)
        engine = RetrievalEngine(
            embedder=emb, index=idx, reranker=rr, rerank_top_n=5,
        )

        result = engine.retrieve("query")
        assert len(result) == 1
        assert result[0]["content"] == "chunk A"
        rr.rerank.assert_not_called()

    def test_top_k_forwarded_to_hybrid_search(self):
        emb = _make_embedder()
        idx = _make_index([])
        engine = RetrievalEngine(embedder=emb, index=idx, top_k=77)

        engine.retrieve("query")
        call_args = idx.hybrid_search.call_args
        assert call_args[0][2] == 77


# ---------------------------------------------------------------------------
# Federated retrieval (_retrieve_federated)
# ---------------------------------------------------------------------------

class TestRetrieveFederated:
    @staticmethod
    def _search_result(content: str, source: str, idx_name: str, score: float):
        """Build a mock SearchResult object."""
        sr = MagicMock()
        sr.content = content
        sr.source = source
        sr.index_name = idx_name
        sr.score = score
        sr.metadata = {"lang": "python"}
        return sr

    def test_federated_path_used_when_indexes_exist(self):
        sr1 = self._search_result("code A", "a.py", "idx1", 0.9)
        fed = _make_federated(indexes=["idx1"], results=[sr1])
        emb = _make_embedder()
        idx = _make_index()
        engine = RetrievalEngine(
            embedder=emb, index=idx, federated=fed, rerank_top_n=10,
        )

        result = engine.retrieve("query")
        assert len(result) == 1
        assert result[0]["content"] == "code A"
        assert result[0]["federated_index"] == "idx1"
        assert result[0]["federated_score"] == 0.9
        # Embedder should NOT be called in federated path
        emb.embed_single.assert_not_called()
        idx.hybrid_search.assert_not_called()

    def test_federated_empty_results(self):
        fed = _make_federated(indexes=["idx1"], results=[])
        emb = _make_embedder()
        idx = _make_index()
        engine = RetrievalEngine(
            embedder=emb, index=idx, federated=fed,
        )

        result = engine.retrieve("query")
        assert result == []

    def test_federated_falls_back_to_single_when_no_indexes(self):
        fed = _make_federated(indexes=[], results=[])
        candidates = [_candidate(0.9, "fallback chunk")]
        emb = _make_embedder()
        idx = _make_index(candidates)
        engine = RetrievalEngine(
            embedder=emb, index=idx, federated=fed, rerank_top_n=5,
        )

        result = engine.retrieve("query")
        assert len(result) == 1
        assert result[0]["content"] == "fallback chunk"
        emb.embed_single.assert_called_once()

    def test_federated_with_reranker(self):
        sr1 = self._search_result("code A", "a.py", "idx1", 0.5)
        sr2 = self._search_result("code B", "b.py", "idx1", 0.9)
        fed = _make_federated(indexes=["idx1"], results=[sr1, sr2])
        # Reranker picks index 1 (code B) first
        rr = _make_reranker(reranked=[(1, 0.99), (0, 0.50)])
        emb = _make_embedder()
        idx = _make_index()
        engine = RetrievalEngine(
            embedder=emb, index=idx, federated=fed,
            reranker=rr, rerank_top_n=2,
        )

        result = engine.retrieve("query")
        assert len(result) == 2
        assert result[0]["content"] == "code B"
        assert result[1]["content"] == "code A"

    def test_federated_metadata_defaults(self):
        """SearchResult with None metadata still produces valid dicts."""
        sr = MagicMock()
        sr.content = "the code"
        sr.source = "file.py"
        sr.index_name = "main"
        sr.score = 0.7
        sr.metadata = None
        fed = _make_federated(indexes=["main"], results=[sr])
        emb = _make_embedder()
        idx = _make_index()
        engine = RetrievalEngine(
            embedder=emb, index=idx, federated=fed, rerank_top_n=5,
        )

        result = engine.retrieve("query")
        assert len(result) == 1
        assert result[0]["content"] == "the code"
        assert result[0]["source_path"] == "file.py"

    def test_federated_top_n_capping(self):
        results = [
            self._search_result(f"code {i}", f"{i}.py", "idx1", 0.9 - i * 0.1)
            for i in range(10)
        ]
        fed = _make_federated(indexes=["idx1"], results=results)
        emb = _make_embedder()
        idx = _make_index()
        engine = RetrievalEngine(
            embedder=emb, index=idx, federated=fed, rerank_top_n=3,
        )

        result = engine.retrieve("query")
        assert len(result) == 3
