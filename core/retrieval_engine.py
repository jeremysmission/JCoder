"""
Retrieval Engine
----------------
Combines embedding, hybrid search, and reranking into one pipeline.

Non-programmer explanation:
This is the part that:
1) Converts your question into a fingerprint (embedding).
2) Runs both vector and keyword search (hybrid).
3) Re-scores the top results with a smarter model (reranker).
4) Returns the best code chunks for the LLM to read.
"""

from typing import Dict, List, Optional

from .index_engine import IndexEngine
from .interfaces import IEmbedder, IReranker


class RetrievalEngine:
    """
    Orchestrates: embed query -> hybrid search -> rerank -> return top chunks.
    """

    def __init__(
        self,
        embedder: IEmbedder,
        index: IndexEngine,
        reranker: Optional[IReranker] = None,
        top_k: int = 50,
        rerank_top_n: int = 10,
    ):
        self.embedder = embedder
        self.index = index
        self.reranker = reranker
        self.top_k = top_k
        self.rerank_top_n = rerank_top_n

    def retrieve(self, query: str) -> List[Dict]:
        """
        Full retrieval pipeline: embed -> hybrid search -> rerank.
        Returns list of metadata dicts for the top results.
        """
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
