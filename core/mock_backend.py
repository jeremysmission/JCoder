"""Mock backends for CPU-only testing without vLLM.

Implements all Protocol interfaces with deterministic behavior
so the full pipeline (chunk -> embed -> index -> search -> rerank -> answer)
can be validated without any GPU or model server running.
"""

import hashlib
from typing import Dict, List, Tuple

import numpy as np


class MockEmbedder:
    """Hash-based deterministic embeddings. Same text always gets same vector."""

    def __init__(self, dimension: int = 768):
        self.dimension = dimension

    def embed(self, texts: List[str]) -> np.ndarray:
        vectors = np.array([self._hash_to_vector(t) for t in texts], dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms

    def embed_single(self, text: str) -> np.ndarray:
        return self.embed([text])[0]

    def _hash_to_vector(self, text: str) -> np.ndarray:
        """SHA-256 -> deterministic float vector of target dimension."""
        h = hashlib.sha256(text.replace("\r\n", "\n").encode("utf-8")).digest()
        # Repeat hash bytes to fill dimension, then map to [-1, 1]
        repeated = (h * ((self.dimension * 4 // len(h)) + 1))[:self.dimension * 4]
        arr = np.frombuffer(bytearray(repeated), dtype=np.uint8)[:self.dimension]
        return (arr.astype(np.float32) / 128.0) - 1.0

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class MockReranker:
    """Scores by keyword overlap between query and document."""

    def __init__(self):
        self.enabled = True

    def rerank(
        self, query: str, documents: List[str], top_n: int = 10,
    ) -> List[Tuple[int, float]]:
        query_words = set(query.lower().split())
        scored = []
        for i, doc in enumerate(documents):
            doc_words = set(doc.lower().split())
            overlap = len(query_words & doc_words)
            scored.append((i, float(overlap)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_n]

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class MockLLM:
    """Returns top sources + extracted snippets. No reasoning, just citation plumbing test."""

    def generate(self, question: str, context_chunks: List[str]) -> str:
        if not context_chunks:
            return "No context provided."

        lines = [f"Question: {question}", "", "Relevant excerpts:"]
        for i, chunk in enumerate(context_chunks[:5]):
            preview = chunk[:200].replace("\n", " ")
            lines.append(f"  [{i+1}] {preview}")

        return "\n".join(lines)

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass
