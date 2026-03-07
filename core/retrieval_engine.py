"""
Retrieval Engine
----------------
Combines embedding, hybrid search, and reranking into one pipeline.
Optionally uses FederatedSearch to query multiple FTS5 indexes at once.

Non-programmer explanation:
This is the part that:
1) Converts your question into a fingerprint (embedding).
2) Runs both vector and keyword search (hybrid).
   -- Or, if federated search is wired in, searches ALL indexes at once.
3) Re-scores the top results with a smarter model (reranker).
4) Returns the best code chunks for the LLM to read.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, List, Optional

from .index_engine import IndexEngine
from .interfaces import IEmbedder, IReranker

if TYPE_CHECKING:
    from .federated_search import FederatedSearch

log = logging.getLogger(__name__)


class RetrievalEngine:
    """
    Orchestrates: embed query -> hybrid search -> rerank -> return top chunks.

    When a FederatedSearch instance is provided, retrieve() fans out across
    all registered FTS5 indexes instead of hitting a single IndexEngine.
    Single-index mode is preserved as the default.
    """

    def __init__(
        self,
        embedder: IEmbedder,
        index: IndexEngine,
        reranker: Optional[IReranker] = None,
        top_k: int = 50,
        rerank_top_n: int = 10,
        federated: Optional[FederatedSearch] = None,
    ):
        self.embedder = embedder
        self.index = index
        self.reranker = reranker
        self.top_k = top_k
        self.rerank_top_n = rerank_top_n
        self.federated = federated

    def retrieve(self, query: str) -> List[Dict]:
        """
        Full retrieval pipeline: embed -> search -> rerank.

        If a FederatedSearch is wired in and has indexes registered,
        it is used instead of the single-index path. Results are converted
        to the same List[Dict] format callers expect.
        """
        if self.federated and self.federated.list_indexes():
            return self._retrieve_federated(query)
        return self._retrieve_single(query)

    def _retrieve_single(self, query: str) -> List[Dict]:
        """Original single-index path: embed -> hybrid search -> rerank."""
        # Step 1: Embed the query
        query_vector = self.embedder.embed_single(query)

        # Step 2: Hybrid search (FAISS vectors + FTS5 keywords)
        candidates = self.index.hybrid_search(query_vector, query, self.top_k)

        if not candidates:
            return []

        # Step 3: Rerank if available
        if self.reranker and self.reranker.enabled:
            docs = [c[1].get("content", "") for c in candidates]
            reranked = self.reranker.rerank(query, docs, self.rerank_top_n)
            return [candidates[idx][1] for idx, _score in reranked]

        # No reranker -- return top_n from hybrid search directly
        return [meta for _score, meta in candidates[: self.rerank_top_n]]

    def _retrieve_federated(self, query: str) -> List[Dict]:
        """Federated path: fan out across all registered FTS5 indexes."""
        results = self.federated.search(query, top_k=self.top_k)
        if not results:
            return []

        # Convert SearchResult objects to the metadata dicts callers expect
        candidates = []
        for sr in results:
            meta = dict(sr.metadata) if sr.metadata else {}
            meta.setdefault("content", sr.content)
            meta.setdefault("source_path", sr.source)
            meta["federated_index"] = sr.index_name
            meta["federated_score"] = sr.score
            candidates.append(meta)

        # Rerank across the merged results if available
        if self.reranker and self.reranker.enabled:
            docs = [c.get("content", "") for c in candidates]
            reranked = self.reranker.rerank(query, docs, self.rerank_top_n)
            return [candidates[idx] for idx, _score in reranked]

        return candidates[: self.rerank_top_n]
