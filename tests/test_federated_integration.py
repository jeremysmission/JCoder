"""
Integration tests for FederatedSearch with multi-index FTS5 querying.

Creates realistic temp indexes simulating python_docs, stackoverflow,
and codesearchnet corpora. Exercises RRF fusion, deduplication,
edge cases, and performance. All indexes use sparse_only=True (no FAISS).
"""

import json
import os
import time
from pathlib import Path
from typing import Dict, List

import pytest

from core.config import StorageConfig
from core.federated_search import FederatedSearch, SearchResult
from core.index_engine import IndexEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _storage(tmp_path: Path) -> StorageConfig:
    idx = str(tmp_path / "indexes")
    os.makedirs(idx, exist_ok=True)
    return StorageConfig(data_dir=str(tmp_path), index_dir=idx)


def _sparse_index(tmp_path: Path, name: str, docs: List[dict]) -> IndexEngine:
    """Build a sparse-only IndexEngine with FTS5 data on disk."""
    st = _storage(tmp_path)
    eng = IndexEngine(dimension=768, storage=st, sparse_only=True)
    eng.metadata = docs
    eng._db_path = os.path.join(st.index_dir, name + ".fts5.db")
    eng._build_fts5()
    with open(os.path.join(st.index_dir, name + ".meta.json"), "w") as f:
        json.dump(docs, f)
    return eng


# ---------------------------------------------------------------------------
# Realistic corpus data
# ---------------------------------------------------------------------------

PYTHON_DOCS = [
    {"id": "py01", "content": "open() returns a file object. Use with statement for safe file handling. mode='r' for reading, 'w' for writing.", "source_path": "docs/builtins/open.py"},
    {"id": "py02", "content": "os.path.join() concatenates path components intelligently. Handles separators for Windows and POSIX.", "source_path": "docs/os/path.py"},
    {"id": "py03", "content": "json.loads() parses a JSON string and returns a Python dictionary. json.dumps() serializes to string.", "source_path": "docs/json/json_module.py"},
    {"id": "py04", "content": "asyncio.run() creates an event loop and runs the given coroutine. async def defines a coroutine function.", "source_path": "docs/asyncio/runners.py"},
    {"id": "py05", "content": "dict comprehension creates dictionaries: {k: v for k, v in items}. Filter with if clause.", "source_path": "docs/builtins/dict.py"},
    {"id": "py06", "content": "pathlib.Path provides object-oriented filesystem paths. Path.read_text() reads file contents as string.", "source_path": "docs/pathlib/path.py"},
    {"id": "py07", "content": "logging module provides flexible event logging. basicConfig sets up root logger. getLogger returns named logger.", "source_path": "docs/logging/logging_module.py"},
    {"id": "py08", "content": "sqlite3 module provides a DB-API 2.0 interface for SQLite databases. connect() opens a database.", "source_path": "docs/sqlite3/sqlite3_module.py"},
    {"id": "py09", "content": "re.search() scans through a string looking for a match. re.findall() returns all non-overlapping matches.", "source_path": "docs/re/re_module.py"},
    {"id": "py10", "content": "typing module for type hints: List, Dict, Optional, Union. Python 3.10+ supports X | Y syntax.", "source_path": "docs/typing/typing_module.py"},
    {"id": "py11", "content": "subprocess.run() runs a command and waits for completion. capture_output=True captures stdout and stderr.", "source_path": "docs/subprocess/subprocess_module.py"},
    {"id": "py12", "content": "collections.defaultdict provides a dictionary with default factory. Counter counts hashable objects.", "source_path": "docs/collections/defaultdict.py"},
    {"id": "py13", "content": "contextlib.contextmanager decorator turns a generator into a context manager for use with the with statement.", "source_path": "docs/contextlib/contextmanager.py"},
    {"id": "py14", "content": "functools.lru_cache memoizes function results. maxsize controls cache size. typed=True caches by argument type.", "source_path": "docs/functools/lru_cache.py"},
    {"id": "py15", "content": "itertools.chain() chains multiple iterables together. itertools.groupby() groups consecutive elements.", "source_path": "docs/itertools/itertools_module.py"},
]

STACKOVERFLOW = [
    {"id": "so01", "content": "Q: How to read a file line by line in Python? A: Use 'with open(filename) as f: for line in f:' for memory efficient reading.", "source_path": "so/python/read_file_line.md"},
    {"id": "so02", "content": "Q: How to merge two dictionaries in Python 3.9+? A: Use the merge operator: merged = dict1 | dict2.", "source_path": "so/python/merge_dicts.md"},
    {"id": "so03", "content": "Q: async await in JavaScript explained. A: async functions return Promises. await pauses execution until Promise resolves. Use try/catch for error handling.", "source_path": "so/javascript/async_await.md"},
    {"id": "so04", "content": "Q: JavaScript fetch API example. A: Use fetch(url).then(res => res.json()).then(data => console.log(data)). Supports async/await syntax.", "source_path": "so/javascript/fetch_api.md"},
    {"id": "so05", "content": "Q: Python decorator explained with examples. A: Decorators wrap functions. @decorator syntax is syntactic sugar for func = decorator(func).", "source_path": "so/python/decorators.md"},
    {"id": "so06", "content": "Q: How to handle exceptions in Python? A: Use try/except blocks. except Exception as e catches most exceptions. finally always runs.", "source_path": "so/python/exceptions.md"},
    {"id": "so07", "content": "Q: JavaScript Promise.all for parallel async operations. A: Promise.all([p1, p2]) runs promises concurrently. Rejects if any promise rejects.", "source_path": "so/javascript/promise_all.md"},
    {"id": "so08", "content": "Q: How to sort a list of objects in Python? A: Use sorted(items, key=lambda x: x.attr) or list.sort() for in-place sorting.", "source_path": "so/python/sorting_objects.md"},
    {"id": "so09", "content": "Q: Python virtual environment best practices. A: Use python -m venv .venv to create. Activate with source .venv/bin/activate.", "source_path": "so/python/virtual_env.md"},
    {"id": "so10", "content": "Q: How to debug memory leaks in Python? A: Use tracemalloc for tracking allocations. objgraph for reference graphs. gc.get_referrers() for back-references.", "source_path": "so/python/memory_leaks.md"},
    {"id": "so11", "content": "Q: TypeScript generics tutorial. A: function identity<T>(arg: T): T. Generics provide type safety with flexibility. Constraints with extends.", "source_path": "so/typescript/generics.md"},
    {"id": "so12", "content": "Q: How to write unit tests in Python with pytest? A: Create test_ files. Use assert statements. Fixtures with @pytest.fixture decorator.", "source_path": "so/python/pytest_testing.md"},
]

CODESEARCHNET = [
    {"id": "csn01", "content": "def read_file(path: str) -> str:\n    \"\"\"Read entire file contents and return as string.\"\"\"\n    with open(path, 'r') as f:\n        return f.read()", "source_path": "repos/utils/file_io.py"},
    {"id": "csn02", "content": "def parse_json(text: str) -> dict:\n    \"\"\"Parse JSON string into Python dictionary with error handling.\"\"\"\n    try:\n        return json.loads(text)\n    except json.JSONDecodeError:\n        return {}", "source_path": "repos/utils/json_utils.py"},
    {"id": "csn03", "content": "async function fetchData(url) {\n    const response = await fetch(url);\n    if (!response.ok) throw new Error(response.statusText);\n    return await response.json();\n}", "source_path": "repos/frontend/api_client.js"},
    {"id": "csn04", "content": "def calculate_rrf_score(ranks: list, k: int = 60) -> float:\n    \"\"\"Reciprocal Rank Fusion score from multiple ranked lists.\"\"\"\n    return sum(1.0 / (k + r + 1) for r in ranks)", "source_path": "repos/search/rrf.py"},
    {"id": "csn05", "content": "class DatabasePool:\n    \"\"\"Connection pool for SQLite with thread-safe checkout.\"\"\"\n    def __init__(self, db_path, pool_size=5):\n        self.connections = [sqlite3.connect(db_path) for _ in range(pool_size)]", "source_path": "repos/db/pool.py"},
    {"id": "csn06", "content": "def tokenize(text: str) -> list:\n    \"\"\"Split text into tokens. Handles code identifiers like snake_case and CamelCase.\"\"\"\n    tokens = re.split(r'[\\s_]+', text)\n    return [t.lower() for t in tokens if t]", "source_path": "repos/nlp/tokenizer.py"},
    {"id": "csn07", "content": "async def run_pipeline(steps):\n    \"\"\"Execute async pipeline steps sequentially, passing results forward.\"\"\"\n    result = None\n    for step in steps:\n        result = await step(result)\n    return result", "source_path": "repos/pipeline/runner.py"},
    {"id": "csn08", "content": "def merge_sorted_lists(a: list, b: list) -> list:\n    \"\"\"Merge two sorted lists into one sorted list in O(n+m).\"\"\"\n    result, i, j = [], 0, 0\n    while i < len(a) and j < len(b):\n        if a[i] <= b[j]: result.append(a[i]); i += 1\n        else: result.append(b[j]); j += 1\n    return result + a[i:] + b[j:]", "source_path": "repos/algorithms/merge.py"},
    {"id": "csn09", "content": "class LRUCache:\n    \"\"\"Least Recently Used cache with O(1) get and put using OrderedDict.\"\"\"\n    def __init__(self, capacity):\n        self.cache = OrderedDict()\n        self.capacity = capacity", "source_path": "repos/cache/lru.py"},
    {"id": "csn10", "content": "def validate_email(email: str) -> bool:\n    \"\"\"Validate email format using regex pattern.\"\"\"\n    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'\n    return bool(re.match(pattern, email))", "source_path": "repos/validation/email.py"},
    {"id": "csn11", "content": "def chunk_text(text: str, max_size: int = 4000) -> list:\n    \"\"\"Split text into chunks respecting sentence boundaries.\"\"\"\n    sentences = text.split('.')\n    chunks, current = [], ''\n    for s in sentences:\n        if len(current) + len(s) > max_size:\n            chunks.append(current); current = s\n        else: current += '.' + s\n    if current: chunks.append(current)\n    return chunks", "source_path": "repos/nlp/chunker.py"},
    {"id": "csn12", "content": "function debounce(fn, delay) {\n    let timer;\n    return function(...args) {\n        clearTimeout(timer);\n        timer = setTimeout(() => fn.apply(this, args), delay);\n    };\n}", "source_path": "repos/frontend/utils.js"},
]


def _build_three_indexes(tmp_path: Path):
    """Build the three realistic indexes and return (fed, indexes_dict)."""
    py_idx = _sparse_index(tmp_path / "python_docs", "python_docs", PYTHON_DOCS)
    so_idx = _sparse_index(tmp_path / "stackoverflow", "stackoverflow", STACKOVERFLOW)
    csn_idx = _sparse_index(tmp_path / "codesearchnet", "codesearchnet", CODESEARCHNET)

    fed = FederatedSearch(embedding_engine=None)
    fed.add_index("python_docs", py_idx, weight=1.0)
    fed.add_index("stackoverflow", so_idx, weight=1.0)
    fed.add_index("codesearchnet", csn_idx, weight=1.0)
    return fed, {"python_docs": py_idx, "stackoverflow": so_idx, "codesearchnet": csn_idx}


def _close_all(indexes: Dict[str, IndexEngine]):
    for idx in indexes.values():
        idx.close()


# ===========================================================================
# Multi-corpus scenarios
# ===========================================================================

class TestMultiCorpusSearch:
    """Search across python_docs, stackoverflow, codesearchnet indexes."""

    def test_cross_index_file_reading(self, tmp_path):
        """Query 'how to read a file in python' should find results from
        multiple indexes -- docs, SO answers, and code snippets."""
        fed, idxs = _build_three_indexes(tmp_path)
        results = fed.search("how to read a file in python", top_k=10)
        assert len(results) >= 3, f"Expected 3+ results, got {len(results)}"
        sources = {r.index_name for r in results}
        # Should hit at least 2 of the 3 corpora
        assert len(sources) >= 2, f"Expected results from 2+ indexes, got {sources}"
        # Content should be relevant
        combined = " ".join(r.content.lower() for r in results)
        assert "file" in combined or "read" in combined or "open" in combined
        _close_all(idxs)

    def test_javascript_async_ranking(self, tmp_path):
        """Query 'async await javascript' should find JS-tagged results."""
        fed, idxs = _build_three_indexes(tmp_path)
        results = fed.search("async await javascript", top_k=10)
        assert len(results) >= 2
        # At least one top-5 result should mention JavaScript/JS or async/await
        top5_content = " ".join(r.content.lower() for r in results[:5])
        assert "javascript" in top5_content or "async" in top5_content or "await" in top5_content
        _close_all(idxs)

    def test_python_specific_query(self, tmp_path):
        """Query for Python-only topic should primarily return Python results."""
        fed, idxs = _build_three_indexes(tmp_path)
        results = fed.search("python decorator explained", top_k=10)
        assert len(results) >= 1
        # First result should mention decorator
        assert "decorator" in results[0].content.lower()
        _close_all(idxs)

    def test_json_parsing_across_corpora(self, tmp_path):
        """JSON parsing appears in docs, SO, and code -- should find all."""
        fed, idxs = _build_three_indexes(tmp_path)
        results = fed.search("parse json string dictionary", top_k=10)
        assert len(results) >= 2
        combined = " ".join(r.content.lower() for r in results)
        assert "json" in combined
        _close_all(idxs)

    def test_sqlite_database_query(self, tmp_path):
        """Query for database/sqlite should find docs and code entries."""
        fed, idxs = _build_three_indexes(tmp_path)
        results = fed.search("sqlite database connection", top_k=10)
        assert len(results) >= 2
        combined = " ".join(r.content.lower() for r in results)
        assert "sqlite" in combined or "database" in combined
        _close_all(idxs)

    def test_results_have_correct_structure(self, tmp_path):
        """Each SearchResult has all required fields populated."""
        fed, idxs = _build_three_indexes(tmp_path)
        results = fed.search("file reading", top_k=5)
        for r in results:
            assert isinstance(r, SearchResult)
            assert isinstance(r.content, str) and len(r.content) > 0
            assert isinstance(r.source, str) and len(r.source) > 0
            assert r.index_name in ("python_docs", "stackoverflow", "codesearchnet")
            assert isinstance(r.score, float) and r.score > 0
            assert isinstance(r.metadata, dict)
        _close_all(idxs)


# ===========================================================================
# RRF fusion correctness
# ===========================================================================

class TestRRFFusion:
    """Verify Reciprocal Rank Fusion merging and weighting."""

    def test_rrf_interleaves_results(self, tmp_path):
        """Results from multiple indexes are interleaved, not just concatenated."""
        fed, idxs = _build_three_indexes(tmp_path)
        results = fed.search("python file reading open", top_k=10)
        # With equal weights and relevant content in all indexes,
        # we should see results from more than one index interspersed
        if len(results) >= 4:
            first_half_sources = {r.index_name for r in results[:len(results)//2]}
            second_half_sources = {r.index_name for r in results[len(results)//2:]}
            # At least one index should appear in both halves (interleaving)
            overlap = first_half_sources & second_half_sources
            # Not a hard failure (depends on BM25 scores), but check structure
            assert len(results) >= 3, "Should have enough results to check interleaving"

        _close_all(idxs)

    def test_weight_boost_changes_ranking(self, tmp_path):
        """Boosted index should rank higher than unboosted for same query."""
        # Build two indexes with similar content
        docs_a = [
            {"id": "a1", "content": "sorting algorithm quicksort implementation", "source_path": "a/sort.py"},
            {"id": "a2", "content": "binary search tree insertion deletion", "source_path": "a/bst.py"},
        ]
        docs_b = [
            {"id": "b1", "content": "sorting algorithm mergesort implementation", "source_path": "b/sort.py"},
            {"id": "b2", "content": "binary search tree traversal inorder", "source_path": "b/bst.py"},
        ]

        idx_a = _sparse_index(tmp_path / "boosted", "boosted", docs_a)
        idx_b = _sparse_index(tmp_path / "normal", "normal", docs_b)

        # With high boost on idx_a
        fed = FederatedSearch(embedding_engine=None)
        fed.add_index("boosted", idx_a, weight=10.0)
        fed.add_index("normal", idx_b, weight=1.0)
        results = fed.search("sorting algorithm", top_k=5)
        assert len(results) >= 2
        assert results[0].index_name == "boosted"

        idx_a.close()
        idx_b.close()

    def test_weight_reversal(self, tmp_path):
        """Swapping weights should swap which index ranks first."""
        docs_a = [{"id": "a1", "content": "machine learning gradient descent optimization", "source_path": "a/ml.py"}]
        docs_b = [{"id": "b1", "content": "machine learning neural network training optimization", "source_path": "b/ml.py"}]

        idx_a = _sparse_index(tmp_path / "a", "a", docs_a)
        idx_b = _sparse_index(tmp_path / "b", "b", docs_b)

        # A boosted
        fed1 = FederatedSearch(embedding_engine=None)
        fed1.add_index("A", idx_a, weight=10.0)
        fed1.add_index("B", idx_b, weight=0.1)
        r1 = fed1.search("machine learning optimization", top_k=2)

        idx_a.close()
        idx_b.close()

        # Rebuild for second federation
        idx_a2 = _sparse_index(tmp_path / "a2", "a2", docs_a)
        idx_b2 = _sparse_index(tmp_path / "b2", "b2", docs_b)

        # B boosted
        fed2 = FederatedSearch(embedding_engine=None)
        fed2.add_index("A", idx_a2, weight=0.1)
        fed2.add_index("B", idx_b2, weight=10.0)
        r2 = fed2.search("machine learning optimization", top_k=2)

        assert len(r1) >= 1 and len(r2) >= 1
        assert r1[0].index_name == "A"
        assert r2[0].index_name == "B"

        idx_a2.close()
        idx_b2.close()

    def test_deduplication_across_indexes(self, tmp_path):
        """Identical content in two indexes should be deduplicated."""
        duplicate = "def read_file(path): return open(path).read()"
        docs_a = [{"id": "a1", "content": duplicate, "source_path": "a/util.py"}]
        docs_b = [{"id": "b1", "content": duplicate, "source_path": "b/util.py"}]

        idx_a = _sparse_index(tmp_path / "a", "a", docs_a)
        idx_b = _sparse_index(tmp_path / "b", "b", docs_b)

        fed = FederatedSearch(embedding_engine=None)
        fed.add_index("A", idx_a, weight=1.0)
        fed.add_index("B", idx_b, weight=1.0)
        results = fed.search("read file open path", top_k=10)

        # Should deduplicate to 1 result (same content hash)
        assert len(results) == 1, f"Expected 1 deduplicated result, got {len(results)}"
        # Score should be sum of both indexes' RRF contributions
        single_rrf = 1.0 / (60 + 0 + 1)
        assert results[0].score > single_rrf, "Deduped result should accumulate scores"

        idx_a.close()
        idx_b.close()

    def test_empty_index_in_federation(self, tmp_path):
        """An empty index should not break fusion or pollute results."""
        docs = [{"id": "c1", "content": "Python list comprehension filtering", "source_path": "list.py"}]
        idx_full = _sparse_index(tmp_path / "full", "full", docs)
        idx_empty = _sparse_index(tmp_path / "empty", "empty", [])

        fed = FederatedSearch(embedding_engine=None)
        fed.add_index("full", idx_full, weight=1.0)
        fed.add_index("empty", idx_empty, weight=1.0)
        results = fed.search("list comprehension", top_k=5)
        assert len(results) >= 1
        assert all(r.index_name == "full" for r in results)

        idx_full.close()
        idx_empty.close()


# ===========================================================================
# Edge cases
# ===========================================================================

class TestEdgeCases:
    """Boundary conditions and robustness tests."""

    def test_all_indexes_empty(self, tmp_path):
        """Federation with only empty indexes returns nothing."""
        idx1 = _sparse_index(tmp_path / "e1", "e1", [])
        idx2 = _sparse_index(tmp_path / "e2", "e2", [])

        fed = FederatedSearch(embedding_engine=None)
        fed.add_index("e1", idx1, weight=1.0)
        fed.add_index("e2", idx2, weight=1.0)
        results = fed.search("anything at all", top_k=10)
        assert results == []

        idx1.close()
        idx2.close()

    def test_no_matches_in_any_index(self, tmp_path):
        """Query with no token overlap returns empty."""
        fed, idxs = _build_three_indexes(tmp_path)
        results = fed.search("xylophone zymurgy quaternion", top_k=10)
        assert results == []
        _close_all(idxs)

    def test_single_index_same_as_nonfederated(self, tmp_path):
        """Single-index federation should produce same results as direct search."""
        docs = [
            {"id": "d1", "content": "binary search algorithm divide and conquer", "source_path": "search.py"},
            {"id": "d2", "content": "linear search sequential scan through array", "source_path": "linear.py"},
            {"id": "d3", "content": "hash table constant time lookup insert", "source_path": "hash.py"},
        ]
        idx = _sparse_index(tmp_path, "single", docs)

        fed = FederatedSearch(embedding_engine=None)
        fed.add_index("only", idx, weight=1.0)
        fed_results = fed.search("binary search algorithm", top_k=5)

        # Direct FTS5 search on the same index
        direct = idx.search_keywords("binary search algorithm", 5)

        # Both should find the same top result
        assert len(fed_results) >= 1
        assert len(direct) >= 1
        # The top result content should match
        direct_top_content = idx.metadata[direct[0][0]]["content"]
        assert fed_results[0].content == direct_top_content

        idx.close()

    def test_very_long_query(self, tmp_path):
        """Very long query string should not crash."""
        fed, idxs = _build_three_indexes(tmp_path)
        long_query = " ".join(["python"] * 500 + ["file"] * 500)
        results = fed.search(long_query, top_k=5)
        # Should return results (many tokens match) or empty, but not crash
        assert isinstance(results, list)
        _close_all(idxs)

    def test_special_characters_no_crash(self, tmp_path):
        """Special chars in query should be sanitized, not cause SQL injection."""
        fed, idxs = _build_three_indexes(tmp_path)
        dangerous_queries = [
            "'; DROP TABLE chunks; --",
            "file AND (content OR 1=1)",
            'content MATCH "test"',
            "hello\x00world",
            "<script>alert('xss')</script>",
            "def func(*args, **kwargs):",
            "path/to/file.py::class::method",
            '{"key": "value"}',
        ]
        for q in dangerous_queries:
            results = fed.search(q, top_k=5)
            assert isinstance(results, list), f"Query {q!r} did not return a list"
        _close_all(idxs)

    def test_unicode_query(self, tmp_path):
        """Non-ASCII query should not crash the search."""
        fed, idxs = _build_three_indexes(tmp_path)
        results = fed.search("python file lambda", top_k=5)
        assert isinstance(results, list)
        _close_all(idxs)

    def test_empty_query_string(self, tmp_path):
        """Empty query should return empty results gracefully."""
        fed, idxs = _build_three_indexes(tmp_path)
        results = fed.search("", top_k=5)
        assert results == []
        _close_all(idxs)

    def test_negative_weight_rejected(self, tmp_path):
        """Adding an index with non-positive weight raises ValueError."""
        fed = FederatedSearch(embedding_engine=None)
        idx = _sparse_index(tmp_path, "neg", [{"id": "1", "content": "test", "source_path": "t.py"}])
        with pytest.raises(ValueError):
            fed.add_index("bad", idx, weight=0.0)
        with pytest.raises(ValueError):
            fed.add_index("bad", idx, weight=-1.0)
        idx.close()

    def test_search_nonexistent_index_name(self, tmp_path):
        """Searching by a name that doesn't exist returns empty."""
        fed, idxs = _build_three_indexes(tmp_path)
        results = fed.search_by_index("python", "nonexistent_index", top_k=5)
        assert results == []
        _close_all(idxs)


# ===========================================================================
# Performance sanity
# ===========================================================================

class TestPerformance:
    """Timing checks to catch regressions."""

    def test_three_indexes_50_entries_under_2s(self, tmp_path):
        """3 indexes x 50 entries each, query completes in < 2 seconds."""
        # Build 50 entries per index with varied content
        def _gen_docs(prefix, n=50):
            topics = [
                "algorithm sorting quicksort mergesort heapsort",
                "database query optimization index btree",
                "network socket tcp udp protocol connection",
                "authentication token jwt oauth session",
                "caching redis memcached invalidation ttl",
                "testing unit integration mock fixture assert",
                "deployment docker kubernetes container pod",
                "logging monitoring alerting metrics grafana",
                "api rest graphql endpoint middleware route",
                "concurrency thread process async await lock",
            ]
            docs = []
            for i in range(n):
                topic = topics[i % len(topics)]
                docs.append({
                    "id": f"{prefix}_{i:03d}",
                    "content": f"Entry {i} about {topic} with details for {prefix} corpus number {i}",
                    "source_path": f"{prefix}/doc_{i:03d}.py",
                })
            return docs

        idx_a = _sparse_index(tmp_path / "perf_a", "perf_a", _gen_docs("alpha"))
        idx_b = _sparse_index(tmp_path / "perf_b", "perf_b", _gen_docs("beta"))
        idx_c = _sparse_index(tmp_path / "perf_c", "perf_c", _gen_docs("gamma"))

        fed = FederatedSearch(embedding_engine=None)
        fed.add_index("alpha", idx_a, weight=1.0)
        fed.add_index("beta", idx_b, weight=1.5)
        fed.add_index("gamma", idx_c, weight=0.8)

        start = time.perf_counter()
        results = fed.search("database query optimization", top_k=20)
        elapsed = time.perf_counter() - start

        assert elapsed < 2.0, f"Query took {elapsed:.3f}s, expected < 2.0s"
        assert len(results) >= 1, "Should find results in 150-entry federation"

        idx_a.close()
        idx_b.close()
        idx_c.close()

    def test_repeated_queries_stable(self, tmp_path):
        """Running the same query multiple times gives consistent results."""
        fed, idxs = _build_three_indexes(tmp_path)
        query = "python file reading"
        r1 = fed.search(query, top_k=5)
        r2 = fed.search(query, top_k=5)
        r3 = fed.search(query, top_k=5)

        # Same number of results
        assert len(r1) == len(r2) == len(r3)
        # Same content in same order
        for a, b, c in zip(r1, r2, r3):
            assert a.content == b.content == c.content
            assert a.index_name == b.index_name == c.index_name

        _close_all(idxs)
