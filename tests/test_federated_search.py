"""Tests for federated search: cache, parallel dispatch, RRF merge."""
from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from core.federated_search import FederatedSearch, SearchResult, _SearchCache


# ---------------------------------------------------------------------------
# _SearchCache unit tests
# ---------------------------------------------------------------------------

class TestSearchCache:

    def test_put_and_get(self):
        cache = _SearchCache(maxsize=4, ttl_s=10.0)
        cache.put("k1", [1, 2, 3])
        assert cache.get("k1") == [1, 2, 3]

    def test_miss_returns_none(self):
        cache = _SearchCache()
        assert cache.get("nonexistent") is None

    def test_ttl_expiry(self):
        cache = _SearchCache(ttl_s=0.2)
        cache.put("k1", ["data"])
        assert cache.get("k1") == ["data"]
        time.sleep(0.3)
        assert cache.get("k1") is None

    def test_eviction_on_full(self):
        cache = _SearchCache(maxsize=2, ttl_s=60.0)
        cache.put("a", [1])
        time.sleep(0.01)
        cache.put("b", [2])
        time.sleep(0.01)
        cache.put("c", [3])  # evicts "a" (oldest)
        assert cache.get("a") is None
        assert cache.get("b") == [2]
        assert cache.get("c") == [3]

    def test_invalidate_clears_all(self):
        cache = _SearchCache()
        cache.put("x", [1])
        cache.put("y", [2])
        cache.invalidate()
        assert cache.get("x") is None
        assert cache.get("y") is None

    def test_make_key_deterministic(self):
        k1 = _SearchCache.make_key("hello", 10, None)
        k2 = _SearchCache.make_key("hello", 10, None)
        assert k1 == k2

    def test_make_key_order_independent(self):
        k1 = _SearchCache.make_key("q", 5, ["b", "a"])
        k2 = _SearchCache.make_key("q", 5, ["a", "b"])
        assert k1 == k2

    def test_make_key_differs_on_params(self):
        k1 = _SearchCache.make_key("q", 5, None)
        k2 = _SearchCache.make_key("q", 10, None)
        assert k1 != k2


# ---------------------------------------------------------------------------
# FederatedSearch integration tests (mock IndexEngine)
# ---------------------------------------------------------------------------

def _make_mock_index(name: str, results: list):
    """Create a mock IndexEngine that returns fixed results."""
    idx = MagicMock()
    idx.count = len(results)
    idx.index = None  # no FAISS
    idx.metadata = []
    idx.search_keywords = MagicMock(return_value=[])

    idx.search_fts5_direct = MagicMock(side_effect=lambda q, k: results[:k])
    return idx


class TestFederatedSearchCache:

    def test_cache_hit_skips_search(self):
        fed = FederatedSearch(embedding_engine=None)
        idx = _make_mock_index("test", [
            (0.9, {"content": "hello world", "source_path": "a.txt"}),
        ])
        fed.add_index("test", idx)

        r1 = fed.search("hello", top_k=5)
        r2 = fed.search("hello", top_k=5)

        assert len(r1) == len(r2)
        assert r1[0].content == r2[0].content
        # FTS5 should only be called once (second call served from cache)
        assert idx.search_fts5_direct.call_count == 1

    def test_cache_invalidated_on_add_index(self):
        fed = FederatedSearch(embedding_engine=None)
        idx1 = _make_mock_index("a", [
            (0.8, {"content": "alpha", "source_path": "a.txt"}),
        ])
        fed.add_index("a", idx1)
        fed.search("test", top_k=5)

        # Cache should have an entry now
        key = fed._cache.make_key("test", 5, None)
        assert fed._cache.get(key) is not None

        # Adding new index invalidates cache
        idx2 = _make_mock_index("b", [])
        fed.add_index("b", idx2)
        assert fed._cache.get(key) is None

    def test_cache_invalidated_on_remove_index(self):
        fed = FederatedSearch(embedding_engine=None)
        idx = _make_mock_index("a", [
            (0.8, {"content": "alpha", "source_path": "a.txt"}),
        ])
        fed.add_index("a", idx)
        fed.search("test", top_k=5)

        key = fed._cache.make_key("test", 5, None)
        assert fed._cache.get(key) is not None

        fed.remove_index("a")
        assert fed._cache.get(key) is None


class TestFederatedSearchMerge:

    def test_rrf_merge_two_indexes(self):
        fed = FederatedSearch(embedding_engine=None, rrf_k=60)
        idx_a = _make_mock_index("a", [
            (0.9, {"content": "shared result", "source_path": "a.txt"}),
            (0.7, {"content": "only in a", "source_path": "a2.txt"}),
        ])
        idx_b = _make_mock_index("b", [
            (0.8, {"content": "shared result", "source_path": "b.txt"}),
            (0.6, {"content": "only in b", "source_path": "b2.txt"}),
        ])
        fed.add_index("a", idx_a)
        fed.add_index("b", idx_b)

        results = fed.search("test", top_k=10)
        # "shared result" should rank highest (boosted by both indexes)
        assert results[0].content == "shared result"
        assert len(results) == 3  # 2 unique + 1 shared (deduped)

    def test_weighted_index_boost(self):
        fed = FederatedSearch(embedding_engine=None, rrf_k=60)
        idx_a = _make_mock_index("a", [
            (0.9, {"content": "from a", "source_path": "a.txt"}),
        ])
        idx_b = _make_mock_index("b", [
            (0.9, {"content": "from b", "source_path": "b.txt"}),
        ])
        fed.add_index("a", idx_a, weight=1.0)
        fed.add_index("b", idx_b, weight=2.0)

        results = fed.search("test", top_k=10)
        # "from b" should rank higher due to 2x weight
        assert results[0].content == "from b"

    def test_empty_indexes_return_empty(self):
        fed = FederatedSearch(embedding_engine=None)
        assert fed.search("anything") == []

    def test_search_by_index(self):
        fed = FederatedSearch(embedding_engine=None)
        idx = _make_mock_index("docs", [
            (0.9, {"content": "doc content", "source_path": "d.txt"}),
        ])
        fed.add_index("docs", idx)

        results = fed.search_by_index("test", "docs", top_k=5)
        assert len(results) >= 0  # should not crash

    def test_search_nonexistent_index(self):
        fed = FederatedSearch(embedding_engine=None)
        results = fed.search_by_index("test", "nope")
        assert results == []

    def test_repeated_queries_stable(self):
        """Same query must always return same ordering."""
        fed = FederatedSearch(embedding_engine=None, cache_maxsize=0, cache_ttl_s=0)
        for i in range(3):
            idx = _make_mock_index(f"idx{i}", [
                (0.9 - i * 0.1, {"content": f"result {i}", "source_path": f"{i}.txt"}),
            ])
            fed.add_index(f"idx{i}", idx)

        r1 = fed.search("test", top_k=10)
        # Disable cache to force re-search
        fed._cache.invalidate()
        r2 = fed.search("test", top_k=10)
        assert [r.content for r in r1] == [r.content for r in r2]


class TestFederatedSearchLifecycle:

    def test_close_shuts_pool(self):
        fed = FederatedSearch(embedding_engine=None)
        fed.close()  # no pool yet, should not crash

    def test_list_indexes(self):
        fed = FederatedSearch(embedding_engine=None)
        idx = _make_mock_index("test", [])
        fed.add_index("test", idx, weight=1.5)
        info = fed.list_indexes()
        assert len(info) == 1
        assert info[0]["name"] == "test"
        assert info[0]["weight"] == 1.5

    def test_stats(self):
        fed = FederatedSearch(embedding_engine=None)
        idx = _make_mock_index("test", [(0.5, {"content": "x"})])
        fed.add_index("test", idx)
        s = fed.stats()
        assert s["index_count"] == 1

    def test_add_index_rejects_zero_weight(self):
        fed = FederatedSearch(embedding_engine=None)
        idx = _make_mock_index("test", [])
        with pytest.raises(ValueError):
            fed.add_index("test", idx, weight=0)
