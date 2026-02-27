"""Protocol interfaces -- stable contracts that backends must implement.

Any backend (vLLM, Ollama, ONNX, mock) can be swapped in
as long as it satisfies these protocols. Core pipeline code
only depends on these, never on concrete implementations.
"""

from typing import Dict, List, Protocol, Tuple, runtime_checkable

import numpy as np


@runtime_checkable
class IEmbedder(Protocol):
    """Converts text into numerical vectors."""

    def embed(self, texts: List[str]) -> np.ndarray:
        """Embed multiple texts. Returns (N, dimension) array."""
        ...

    def embed_single(self, text: str) -> np.ndarray:
        """Embed a single text. Returns (dimension,) array."""
        ...


@runtime_checkable
class IRetriever(Protocol):
    """Searches an index and returns candidate chunks."""

    def retrieve(self, query: str) -> List[Dict]:
        """Return top matching chunks as metadata dicts."""
        ...


@runtime_checkable
class IReranker(Protocol):
    """Re-scores candidates with a cross-encoder."""

    def rerank(
        self, query: str, documents: List[str], top_n: int = 10,
    ) -> List[Tuple[int, float]]:
        """Return (original_index, score) sorted by relevance."""
        ...


@runtime_checkable
class ILLM(Protocol):
    """Generates text from a prompt + context."""

    def generate(
        self, question: str, context_chunks: List[str],
    ) -> str:
        """Return generated answer text."""
        ...
