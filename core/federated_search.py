"""
Federated Search
----------------
Search across multiple IndexEngine instances and merge results with RRF.

Non-programmer explanation:
JCoder has several knowledge bases (Stack Overflow, code docs, personal
memory, etc.). This module runs one query across ALL of them at once,
then merges the rankings so the single best answer floats to the top
regardless of which index it came from.
"""

import hashlib
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .embedding_engine import EmbeddingEngine
from .index_engine import IndexEngine


@dataclass
class SearchResult:
    """Single result from a federated search."""
    content: str
    source: str
    index_name: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


def _content_hash(text: str) -> str:
    """Fast 64-char hex digest for deduplication."""
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


class FederatedSearch:
    """Search across multiple IndexEngine instances with result fusion."""

    def __init__(
        self,
        embedding_engine: Optional[EmbeddingEngine] = None,
        rrf_k: int = 60,
        max_workers: int = 8,
    ):
        self._indexes: Dict[str, IndexEngine] = {}
        self._weights: Dict[str, float] = {}
        self._embedder = embedding_engine
        self._rrf_k = rrf_k
        self._max_workers = max_workers
        self._pool: Optional[ThreadPoolExecutor] = None

    # -- Index management --------------------------------------------------

    def add_index(self, name: str, index: IndexEngine, weight: float = 1.0):
        """Register an index for federated search.

        Weight controls how much this index's results are boosted in RRF.
        Example: agent_memory weight=1.5 prefers personal experience.
        """
        if weight <= 0:
            raise ValueError(f"Weight must be positive, got {weight}")
        self._indexes[name] = index
        self._weights[name] = weight

    def remove_index(self, name: str):
        """Unregister an index."""
        self._indexes.pop(name, None)
        self._weights.pop(name, None)

    # -- Search ------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 10,
        indexes: Optional[List[str]] = None,
    ) -> List[SearchResult]:
        """Search across all (or specified) indexes and merge results.

        1. Embed query once (if embedder available).
        2. Search each index independently via hybrid or FTS5-only.
        3. Apply per-index weight multipliers in RRF.
        4. Deduplicate by content hash.
        5. Return top_k ranked results.
        """
        targets = self._resolve_targets(indexes)
        if not targets:
            return []

        query_vector = None
        if self._embedder is not None:
            query_vector = self._embedder.embed_single(query)

        # Parallel search across all targeted indexes
        per_index = self._search_all(targets, query, query_vector, top_k)

        # RRF merge with dedup
        merged: Dict[str, Dict[str, Any]] = {}
        for name, results in per_index:
            weight = self._weights[name]
            for rank, (score, meta) in enumerate(results):
                rrf_score = weight * (1.0 / (self._rrf_k + rank + 1))
                content = meta.get("content", "")
                h = _content_hash(content)

                if h in merged:
                    merged[h]["rrf"] += rrf_score
                    if rrf_score > merged[h]["best_rrf"]:
                        merged[h]["result"] = self._make_result(
                            meta, name, rrf_score,
                        )
                        merged[h]["best_rrf"] = rrf_score
                else:
                    merged[h] = {
                        "rrf": rrf_score,
                        "best_rrf": rrf_score,
                        "result": self._make_result(meta, name, rrf_score),
                    }

        # Sort by total RRF score, patch final score, return top_k
        ranked = sorted(merged.values(), key=lambda x: x["rrf"], reverse=True)
        output: List[SearchResult] = []
        for entry in ranked[:top_k]:
            r = entry["result"]
            r.score = entry["rrf"]
            output.append(r)
        return output

    def search_by_index(
        self, query: str, index_name: str, top_k: int = 10,
    ) -> List[SearchResult]:
        """Search a specific index only."""
        if index_name not in self._indexes:
            return []
        return self.search(query, top_k=top_k, indexes=[index_name])

    # -- Info --------------------------------------------------------------

    def list_indexes(self) -> List[Dict[str, Any]]:
        """Return info about registered indexes."""
        return [
            {
                "name": name,
                "count": idx.count,
                "weight": self._weights[name],
            }
            for name, idx in self._indexes.items()
        ]

    def stats(self) -> Dict[str, Any]:
        """Total chunks across all indexes, per-index breakdown."""
        per_index = {
            name: idx.count for name, idx in self._indexes.items()
        }
        return {
            "total_chunks": sum(per_index.values()),
            "index_count": len(per_index),
            "per_index": per_index,
        }

    def close(self):
        """Shut down the thread pool (if active)."""
        if self._pool is not None:
            self._pool.shutdown(wait=False)
            self._pool = None

    # -- Internals ---------------------------------------------------------

    def _resolve_targets(self, indexes: Optional[List[str]]) -> List[str]:
        """Return list of valid index names to search."""
        if indexes is None:
            return list(self._indexes.keys())
        return [n for n in indexes if n in self._indexes]

    def _search_all(
        self,
        targets: List[str],
        query: str,
        query_vector: Optional[np.ndarray],
        top_k: int,
    ) -> List[Tuple[str, List]]:
        """Search all target indexes, parallel when >1 index."""
        if len(targets) <= 1:
            # Single index -- no thread overhead
            results = []
            for name in targets:
                r = self._search_single(
                    self._indexes[name], query, query_vector, top_k,
                )
                results.append((name, r))
            return results

        # Lazy-init thread pool (reused across searches)
        if self._pool is None:
            workers = min(self._max_workers, len(targets))
            self._pool = ThreadPoolExecutor(max_workers=workers)

        def _do_search(name: str) -> Tuple[str, List]:
            r = self._search_single(
                self._indexes[name], query, query_vector, top_k,
            )
            return (name, r)

        futures = {
            self._pool.submit(_do_search, name): name
            for name in targets
        }
        # Collect results, preserving deterministic order by index name
        result_map: Dict[str, List] = {}
        for future in as_completed(futures):
            try:
                name, hits = future.result()
                result_map[name] = hits
            except Exception:
                pass  # Index search failure is non-fatal
        # Return in original target order for deterministic RRF
        return [(name, result_map[name]) for name in targets
                if name in result_map]

    @staticmethod
    def _search_single(
        index: IndexEngine,
        query_text: str,
        query_vector: Optional[np.ndarray],
        k: int,
    ) -> List:
        """Search one IndexEngine, using hybrid or FTS5-only."""
        if query_vector is not None and index.index is not None:
            return index.hybrid_search(query_vector, query_text, k)
        # Lazy FTS5: no metadata preloaded, query DB directly
        if not index.metadata and hasattr(index, "search_fts5_direct"):
            return index.search_fts5_direct(query_text, k)
        # FTS5 with preloaded metadata (small indexes, .meta.json)
        kw_results = index.search_keywords(query_text, k)
        return [
            (score, index.metadata[idx])
            for idx, score in kw_results
            if idx < len(index.metadata)
        ]

    @staticmethod
    def _make_result(
        meta: Dict[str, Any], index_name: str, score: float,
    ) -> SearchResult:
        """Build a SearchResult from index metadata."""
        return SearchResult(
            content=meta.get("content", ""),
            source=meta.get("source_path", meta.get("source", "")),
            index_name=index_name,
            score=score,
            metadata=meta,
        )
