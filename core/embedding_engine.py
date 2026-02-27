"""
Embedding Engine
----------------
Converts text into numerical vectors by calling the vLLM embedding server.

Non-programmer explanation:
Think of this like turning text into a fingerprint.
Similar code snippets get similar fingerprints.
We send text to a local AI server (vLLM) that returns
a list of numbers representing the meaning of that text.
"""

from typing import List

import httpx
import numpy as np

from .config import ModelConfig


class EmbeddingEngine:
    """
    Responsible ONLY for converting text into vectors.
    Calls vLLM's OpenAI-compatible /v1/embeddings endpoint.

    No indexing logic. No retrieval logic. Just embedding.
    """

    def __init__(self, config: ModelConfig, timeout: int = 120):
        self.endpoint = config.endpoint.rstrip("/")
        self.model_name = config.name
        self.dimension = config.dimension or 768
        self._client = httpx.Client(timeout=httpx.Timeout(timeout))

    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Convert multiple text chunks into vector form.
        Returns an (N, dimension) numpy array of normalized embeddings.
        """
        response = self._client.post(
            f"{self.endpoint}/embeddings",
            json={
                "model": self.model_name,
                "input": texts,
            },
        )
        response.raise_for_status()

        data = response.json()["data"]
        # vLLM returns embeddings sorted by index
        vectors = [item["embedding"] for item in sorted(data, key=lambda x: x["index"])]
        result = np.array(vectors, dtype=np.float32)

        # Normalize for cosine similarity via inner product
        norms = np.linalg.norm(result, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        result = result / norms

        return result

    def embed_single(self, text: str) -> np.ndarray:
        """Convert a single query into a vector."""
        return self.embed([text])[0]

    def close(self):
        """Release HTTP connection pool."""
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
