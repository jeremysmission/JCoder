"""Extended tests for FederatedSearch -- discovery, parallel search, merging,
dedup, score normalization, error handling, config-driven selection, timeouts.

All external dependencies are mocked.
"""
from __future__ import annotations

import threading
import time
from concurrent.futures import Future
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from core.federated_search import (
    FederatedSearch,
    SearchResult,
    _SearchCache,
    _content_hash,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_index(results=None, count=0, metadata=None):
    """Return a MagicMock IndexEngine with sensible defaults.

    By default metadata is empty list (falsy) and search_fts5_direct exists,
    so _search_single takes the lazy FTS5 path and returns our results directly.
    """
    idx = MagicMock()
    idx.index = None  # no FAISS index by default
    idx.metadata = metadata if metadata is not None else []
    idx.count = count
    idx.hybrid_search = MagicMock(return_value=results or [])
    idx.search_keywords = MagicMock(return_value=[])
    # Provide search_fts5_direct so _search_single uses the lazy FTS5 path
    idx.search_fts5_direct = MagicMock(return_value=results or [])
    return idx


def _result_row(content, source="src.py"):
    """Helper to build a (score, meta) tuple the way IndexEngine returns."""
    return (0.9, {"content": content, "source_path": source})


def _make_embedder():
    emb = MagicMock()
    emb.embed_single = MagicMock(return_value=np.zeros(384))
    return emb


# ---------------------------------------------------------------------------
# Discovery of multiple indexes
# ---------------------------------------------------------------------------

class TestIndexDiscovery:

    def test_add_multiple_indexes(self):
        fs = FederatedSearch()
        fs.add_index("code", _make_index(count=100))
        fs.add_index("docs", _make_index(count=50))
        fs.add_index("memory", _make_index(count=10))
        info = fs.list_indexes()
        names = {i["name"] for i in info}
        assert names == {"code", "docs", "memory"}

    def test_list_indexes_reports_weight_and_count(self):
        fs = FederatedSearch()
        fs.add_index("a", _make_index(count=7), weight=1.5)
        info = fs.list_indexes()
        assert info[0]["weight"] == 1.5
        assert info[0]["count"] == 7

    def test_remove_index(self):
        fs = FederatedSearch()
        fs.add_index("tmp", _make_index())
        fs.remove_index("tmp")
        assert fs.list_indexes() == []

    def test_remove_nonexistent_is_noop(self):
        fs = FederatedSearch()
        fs.remove_index("ghost")  # should not raise

    def test_reject_zero_or_negative_weight(self):
        fs = FederatedSearch()
        with pytest.raises(ValueError):
            fs.add_index("bad", _make_index(), weight=0)
        with pytest.raises(ValueError):
            fs.add_index("bad", _make_index(), weight=-1.0)

    def test_stats(self):
        fs = FederatedSearch()
        fs.add_index("a", _make_index(count=10))
        fs.add_index("b", _make_index(count=20))
        s = fs.stats()
        assert s["total_chunks"] == 30
        assert s["index_count"] == 2


# ---------------------------------------------------------------------------
# Parallel search across indexes
# ---------------------------------------------------------------------------

class TestParallelSearch:

    def test_single_index_skips_threadpool(self):
        idx = _make_index(results=[_result_row("hit1")])
        fs = FederatedSearch()
        fs.add_index("only", idx)
        results = fs.search("query", top_k=5)
        assert len(results) == 1
        assert fs._pool is None  # no pool created for 1 index

    def test_multiple_indexes_use_threadpool(self):
        fs = FederatedSearch(max_workers=2)
        fs.add_index("a", _make_index(results=[_result_row("from a")]))
        fs.add_index("b", _make_index(results=[_result_row("from b")]))
        results = fs.search("q", top_k=10)
        assert len(results) == 2
        assert fs._pool is not None
        fs.close()

    def test_parallel_collects_results_from_all(self):
        fs = FederatedSearch(max_workers=4)
        for name in ["x", "y", "z"]:
            fs.add_index(name, _make_index(results=[_result_row(f"doc-{name}")]))
        results = fs.search("test", top_k=10)
        contents = {r.content for r in results}
        assert contents == {"doc-x", "doc-y", "doc-z"}
        fs.close()


# ---------------------------------------------------------------------------
# Result merging and deduplication
# ---------------------------------------------------------------------------

class TestMergingDedup:

    def test_duplicate_content_merged(self):
        """Same content from two indexes should collapse to one result."""
        fs = FederatedSearch()
        fs.add_index("a", _make_index(results=[_result_row("shared doc")]))
        fs.add_index("b", _make_index(results=[_result_row("shared doc")]))
        results = fs.search("q", top_k=10)
        assert len(results) == 1
        # RRF score should be higher than either single contribution
        assert results[0].score > 1.0 / (60 + 1)

    def test_distinct_content_kept(self):
        fs = FederatedSearch()
        fs.add_index("a", _make_index(results=[_result_row("alpha")]))
        fs.add_index("b", _make_index(results=[_result_row("beta")]))
        results = fs.search("q", top_k=10)
        assert len(results) == 2

    def test_top_k_limits_output(self):
        fs = FederatedSearch()
        rows = [_result_row(f"doc{i}") for i in range(20)]
        fs.add_index("big", _make_index(results=rows))
        results = fs.search("q", top_k=5)
        assert len(results) == 5

    def test_content_hash_deterministic(self):
        assert _content_hash("hello") == _content_hash("hello")
        assert _content_hash("a") != _content_hash("b")


# ---------------------------------------------------------------------------
# Score normalization across indexes
# ---------------------------------------------------------------------------

class TestScoreNormalization:

    def test_weight_boosts_rrf_score(self):
        """Higher-weight index should produce higher RRF contribution."""
        fs = FederatedSearch()
        fs.add_index("lo", _make_index(results=[_result_row("lo-doc")]), weight=1.0)
        fs.add_index("hi", _make_index(results=[_result_row("hi-doc")]), weight=3.0)
        results = fs.search("q", top_k=10)
        scores = {r.content: r.score for r in results}
        assert scores["hi-doc"] > scores["lo-doc"]

    def test_rrf_rank_ordering(self):
        """First result in an index gets higher RRF than second."""
        rows = [_result_row("first"), _result_row("second")]
        fs = FederatedSearch()
        fs.add_index("idx", _make_index(results=rows))
        results = fs.search("q", top_k=10)
        assert results[0].content == "first"
        assert results[0].score > results[1].score

    def test_rrf_k_parameter_affects_scores(self):
        rows = [_result_row("doc")]
        fs_lo = FederatedSearch(rrf_k=10)
        fs_lo.add_index("i", _make_index(results=rows))
        fs_hi = FederatedSearch(rrf_k=100)
        fs_hi.add_index("i", _make_index(results=rows))
        r_lo = fs_lo.search("q", top_k=1)
        r_hi = fs_hi.search("q", top_k=1)
        assert r_lo[0].score > r_hi[0].score  # smaller k -> bigger score


# ---------------------------------------------------------------------------
# Index unavailable handling
# ---------------------------------------------------------------------------

class TestIndexUnavailable:

    def test_search_with_missing_index_name_skips(self):
        fs = FederatedSearch()
        fs.add_index("real", _make_index(results=[_result_row("ok")]))
        results = fs.search("q", top_k=10, indexes=["real", "ghost"])
        assert len(results) == 1

    def test_search_all_missing_returns_empty(self):
        fs = FederatedSearch()
        fs.add_index("real", _make_index())
        results = fs.search("q", top_k=10, indexes=["nope"])
        assert results == []

    def test_index_raising_exception_is_handled(self):
        """If one index throws during search, the rest still return."""
        good = _make_index(results=[_result_row("good")])
        bad = _make_index()
        bad.search_keywords.side_effect = RuntimeError("db locked")
        bad.metadata = [{"content": "x"}]  # force keyword path

        fs = FederatedSearch(max_workers=2)
        fs.add_index("good", good)
        fs.add_index("bad", bad)
        results = fs.search("q", top_k=10)
        # Should get at least the good result; bad silently dropped
        contents = {r.content for r in results}
        assert "good" in contents
        fs.close()

    def test_search_by_index_unknown_returns_empty(self):
        fs = FederatedSearch()
        assert fs.search_by_index("q", "nonexistent") == []


# ---------------------------------------------------------------------------
# Empty index handling
# ---------------------------------------------------------------------------

class TestEmptyIndex:

    def test_no_indexes_returns_empty(self):
        fs = FederatedSearch()
        assert fs.search("hello") == []

    def test_index_returning_no_results(self):
        fs = FederatedSearch()
        fs.add_index("empty", _make_index(results=[]))
        assert fs.search("q") == []

    def test_stats_with_empty_indexes(self):
        fs = FederatedSearch()
        s = fs.stats()
        assert s["total_chunks"] == 0
        assert s["index_count"] == 0


# ---------------------------------------------------------------------------
# Config-driven index selection
# ---------------------------------------------------------------------------

class TestConfigDrivenSelection:

    def test_search_only_specified_indexes(self):
        fs = FederatedSearch()
        fs.add_index("code", _make_index(results=[_result_row("code-hit")]))
        fs.add_index("docs", _make_index(results=[_result_row("docs-hit")]))
        fs.add_index("mem", _make_index(results=[_result_row("mem-hit")]))
        results = fs.search("q", top_k=10, indexes=["code", "mem"])
        contents = {r.content for r in results}
        assert "code-hit" in contents
        assert "mem-hit" in contents
        assert "docs-hit" not in contents

    def test_resolve_targets_all(self):
        fs = FederatedSearch()
        fs.add_index("a", _make_index())
        fs.add_index("b", _make_index())
        assert set(fs._resolve_targets(None)) == {"a", "b"}

    def test_resolve_targets_subset(self):
        fs = FederatedSearch()
        fs.add_index("a", _make_index())
        fs.add_index("b", _make_index())
        assert fs._resolve_targets(["b"]) == ["b"]

    def test_search_by_index_delegates(self):
        fs = FederatedSearch()
        fs.add_index("solo", _make_index(results=[_result_row("solo-doc")]))
        results = fs.search_by_index("q", "solo", top_k=3)
        assert len(results) == 1
        assert results[0].index_name == "solo"


# ---------------------------------------------------------------------------
# Search timeout / cache
# ---------------------------------------------------------------------------

class TestSearchCache:

    def test_cache_returns_same_results(self):
        idx = _make_index(results=[_result_row("cached")])
        fs = FederatedSearch(cache_maxsize=16, cache_ttl_s=60)
        fs.add_index("c", idx)
        r1 = fs.search("q", top_k=5)
        r2 = fs.search("q", top_k=5)
        assert r1 is r2  # exact same list object from cache

    def test_cache_expires(self):
        cache = _SearchCache(maxsize=10, ttl_s=0.01)
        cache.put("k", ["data"])
        time.sleep(0.02)
        assert cache.get("k") is None

    def test_cache_evicts_oldest(self):
        cache = _SearchCache(maxsize=2, ttl_s=60)
        cache.put("a", [1])
        cache.put("b", [2])
        cache.put("c", [3])  # should evict "a"
        assert cache.get("a") is None
        assert cache.get("b") == [2]

    def test_cache_invalidate(self):
        cache = _SearchCache(maxsize=10, ttl_s=60)
        cache.put("x", [42])
        cache.invalidate()
        assert cache.get("x") is None

    def test_cache_key_varies_by_top_k(self):
        k1 = _SearchCache.make_key("q", 5, None)
        k2 = _SearchCache.make_key("q", 10, None)
        assert k1 != k2

    def test_cache_key_varies_by_indexes(self):
        k1 = _SearchCache.make_key("q", 5, ["a"])
        k2 = _SearchCache.make_key("q", 5, ["b"])
        assert k1 != k2

    def test_close_shuts_pool(self):
        fs = FederatedSearch(max_workers=2)
        fs.add_index("a", _make_index(results=[_result_row("x")]))
        fs.add_index("b", _make_index(results=[_result_row("y")]))
        fs.search("q")
        assert fs._pool is not None
        fs.close()
        assert fs._pool is None
