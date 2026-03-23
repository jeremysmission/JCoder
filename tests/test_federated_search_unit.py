"""Unit tests for core/federated_search.py (R18).

Tests RRF fusion, deduplication, per-index weighting, caching,
and quality filtering without needing real FTS5 indexes.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from core.federated_search import FederatedSearch, SearchResult, _content_hash, _SearchCache


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_index(results=None, count=100):
    """Create a mock IndexEngine."""
    idx = MagicMock()
    idx.count = count
    idx.index = None  # No FAISS -- forces FTS5
    idx.metadata = []
    idx.search_fts5_direct = MagicMock(return_value=results or [])
    return idx


# ---------------------------------------------------------------------------
# Content hash
# ---------------------------------------------------------------------------

class TestContentHash:

    def test_same_content_same_hash(self):
        assert _content_hash("hello") == _content_hash("hello")

    def test_different_content_different_hash(self):
        assert _content_hash("hello") != _content_hash("world")


# ---------------------------------------------------------------------------
# SearchCache
# ---------------------------------------------------------------------------

class TestSearchCache:

    def test_put_and_get(self):
        cache = _SearchCache(maxsize=10, ttl_s=60)
        cache.put("key1", ["result1"])
        assert cache.get("key1") == ["result1"]

    def test_miss_returns_none(self):
        cache = _SearchCache(maxsize=10, ttl_s=60)
        assert cache.get("missing") is None

    def test_invalidate_clears_all(self):
        cache = _SearchCache(maxsize=10, ttl_s=60)
        cache.put("k1", ["r1"])
        cache.put("k2", ["r2"])
        cache.invalidate()
        assert cache.get("k1") is None
        assert cache.get("k2") is None


# ---------------------------------------------------------------------------
# Index management
# ---------------------------------------------------------------------------

class TestIndexManagement:

    def test_add_and_list_indexes(self):
        fs = FederatedSearch()
        fs.add_index("test", _mock_index(), weight=1.5)
        indexes = fs.list_indexes()
        assert len(indexes) == 1
        assert indexes[0]["name"] == "test"
        assert indexes[0]["weight"] == 1.5

    def test_remove_index(self):
        fs = FederatedSearch()
        fs.add_index("test", _mock_index())
        fs.remove_index("test")
        assert fs.list_indexes() == []

    def test_add_index_invalid_weight_raises(self):
        fs = FederatedSearch()
        with pytest.raises(ValueError):
            fs.add_index("test", _mock_index(), weight=0)

    def test_stats(self):
        fs = FederatedSearch()
        fs.add_index("a", _mock_index(count=50))
        fs.add_index("b", _mock_index(count=100))
        s = fs.stats()
        assert s["total_chunks"] == 150
        assert s["index_count"] == 2


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

class TestSearch:

    def test_empty_search(self):
        fs = FederatedSearch()
        results = fs.search("test query")
        assert results == []

    def test_single_index_search(self):
        idx = _mock_index([
            (0.9, {"content": "hello world", "source_path": "x.py"}),
        ])
        fs = FederatedSearch()
        fs.add_index("test", idx)
        results = fs.search("hello", top_k=5)
        assert len(results) == 1
        assert results[0].content == "hello world"
        assert results[0].index_name == "test"

    def test_rrf_merges_multiple_indexes(self):
        idx1 = _mock_index([
            (0.9, {"content": "from index1", "source_path": "a.py"}),
        ])
        idx2 = _mock_index([
            (0.8, {"content": "from index2", "source_path": "b.py"}),
        ])
        fs = FederatedSearch()
        fs.add_index("idx1", idx1, weight=1.0)
        fs.add_index("idx2", idx2, weight=1.0)
        results = fs.search("test", top_k=5)
        assert len(results) == 2

    def test_deduplication(self):
        """Same content from two indexes should merge."""
        same_content = {"content": "duplicate content", "source_path": "x.py"}
        idx1 = _mock_index([(0.9, dict(same_content))])
        idx2 = _mock_index([(0.8, dict(same_content))])
        fs = FederatedSearch()
        fs.add_index("idx1", idx1)
        fs.add_index("idx2", idx2)
        results = fs.search("test", top_k=5)
        # Should merge into 1 result with boosted RRF score
        assert len(results) == 1

    def test_weight_affects_ranking(self):
        """Higher-weighted index should rank first."""
        idx_low = _mock_index([
            (0.9, {"content": "low weight result", "source_path": "a.py"}),
        ])
        idx_high = _mock_index([
            (0.9, {"content": "high weight result", "source_path": "b.py"}),
        ])
        fs = FederatedSearch()
        fs.add_index("low", idx_low, weight=0.5)
        fs.add_index("high", idx_high, weight=2.0)
        results = fs.search("test", top_k=5)
        assert len(results) == 2
        assert results[0].content == "high weight result"

    def test_search_specific_indexes(self):
        idx1 = _mock_index([(0.9, {"content": "a", "source_path": "a.py"})])
        idx2 = _mock_index([(0.8, {"content": "b", "source_path": "b.py"})])
        fs = FederatedSearch()
        fs.add_index("idx1", idx1)
        fs.add_index("idx2", idx2)
        results = fs.search("test", top_k=5, indexes=["idx1"])
        assert len(results) == 1
        assert results[0].index_name == "idx1"


# ---------------------------------------------------------------------------
# Quality filtering
# ---------------------------------------------------------------------------

class TestQualityFiltering:

    def test_min_quality_passed_to_search(self):
        idx = _mock_index([])
        fs = FederatedSearch(min_quality=3)
        fs.add_index("test", idx)
        fs.search("test query", top_k=5)
        # Verify min_quality was passed through
        call_args = idx.search_fts5_direct.call_args
        assert call_args.kwargs.get("min_quality") == 3

    def test_default_no_quality_filter(self):
        idx = _mock_index([])
        fs = FederatedSearch()
        fs.add_index("test", idx)
        fs.search("test", top_k=5)
        call_args = idx.search_fts5_direct.call_args
        assert call_args.kwargs.get("min_quality") == 0


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------

class TestCaching:

    def test_cached_results_returned(self):
        idx = _mock_index([
            (0.9, {"content": "cached", "source_path": "x.py"}),
        ])
        fs = FederatedSearch()
        fs.add_index("test", idx)
        # First call
        r1 = fs.search("test query", top_k=5)
        # Second call should hit cache
        r2 = fs.search("test query", top_k=5)
        assert len(r1) == len(r2)
        # FTS5 should only be called once (cached on second)
        assert idx.search_fts5_direct.call_count == 1

    def test_add_index_invalidates_cache(self):
        idx = _mock_index([(0.9, {"content": "a", "source_path": "a.py"})])
        fs = FederatedSearch()
        fs.add_index("test", idx)
        fs.search("query", top_k=5)
        # Adding new index should invalidate
        fs.add_index("test2", _mock_index())
        fs.search("query", top_k=5)
        assert idx.search_fts5_direct.call_count == 2


# ---------------------------------------------------------------------------
# SearchResult
# ---------------------------------------------------------------------------

class TestSearchResult:

    def test_search_result_fields(self):
        r = SearchResult(content="code", source="x.py", index_name="test", score=0.9)
        assert r.content == "code"
        assert r.metadata == {}
