"""
Reranker
--------
Cross-encoder rescoring via vLLM reranker server.

Non-programmer explanation:
The first search (FAISS + keyword) casts a wide net and returns ~50 results.
The reranker is a smarter, slower model that looks at each result paired
with your question and re-scores them. It narrows 50 rough matches
down to the 10 best ones before the LLM sees them.
"""

from typing import List, Tuple

import httpx

from .config import ModelConfig
from .http_factory import make_client
from .network_gate import NetworkGate


class Reranker:
    """
    Calls vLLM's cross-encoder scoring endpoint.
    Takes (query, document) pairs and returns relevance scores.
    """

    def __init__(self, config: ModelConfig, timeout: int = 120,
                 gate: NetworkGate = None):
        self.endpoint = config.endpoint.rstrip("/")
        self.model_name = config.name
        self.enabled = config.enabled
        self._client = make_client(timeout_s=timeout)
        self._gate = gate

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_n: int = 10,
    ) -> List[Tuple[int, float]]:
        """
        Score each document against the query and return top_n results.
        Returns list of (original_index, score) sorted by score descending.
        """
        if not self.enabled or not documents:
            # Pass-through: return original order with dummy scores
            return [(i, 1.0) for i in range(min(top_n, len(documents)))]

        url = f"{self.endpoint}/score"
        if self._gate:
            self._gate.guard(url)
        response = self._client.post(
            url,
            json={
                "model": self.model_name,
                "text_1": query,
                "text_2": documents,
            },
        )
        response.raise_for_status()

        scores = response.json()
        # vLLM /score returns list of {"index": i, "score": float}
        scored = [(item["index"], item["score"]) for item in scores["data"]]
        scored.sort(key=lambda x: x[1], reverse=True)

        return scored[:top_n]

    def close(self):
        """Release HTTP connection pool."""
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
