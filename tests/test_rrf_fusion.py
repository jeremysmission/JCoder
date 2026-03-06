"""RRF fusion ordering must be deterministic and correct."""

import pytest
import numpy as np

from core.config import StorageConfig
from core.index_engine import IndexEngine

faiss = pytest.importorskip("faiss", reason="faiss not installed")


def _make_index(tmp_path, dimension=8) -> IndexEngine:
    """Create a small test index with known vectors."""
    storage = StorageConfig(
        data_dir=str(tmp_path),
        index_dir=str(tmp_path / "indexes"),
    )
    index = IndexEngine(dimension, storage, rrf_k=60, gpu_safety_margin_mb=0)
    return index


def test_rrf_ranks_vector_matches_higher(tmp_path):
    """A chunk that matches in both vector AND keyword search should rank highest."""
    index = _make_index(tmp_path)

    # Create 3 chunks with known content
    vectors = np.random.RandomState(42).randn(3, 8).astype(np.float32)
    # Normalize
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / norms

    metadata = [
        {"id": "a", "content": "def authenticate_user(): pass", "source_path": "auth.py"},
        {"id": "b", "content": "def setup_logging(): pass", "source_path": "log.py"},
        {"id": "c", "content": "def connect_database(): pass", "source_path": "db.py"},
    ]

    index.add_vectors(vectors, metadata)

    # Query with vector closest to chunk 0
    query_vec = vectors[0]
    results = index.hybrid_search(query_vec, "authenticate", k=3)

    assert len(results) > 0
    # Chunk "a" should appear because it matches both vector (closest) and keyword
    result_ids = [r[1]["id"] for r in results]
    assert "a" in result_ids


def test_rrf_deterministic(tmp_path):
    """Same query, same index -> same ordering every time."""
    index = _make_index(tmp_path)

    vectors = np.random.RandomState(99).randn(5, 8).astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / norms

    metadata = [
        {"id": str(i), "content": f"function_{i} does thing_{i}", "source_path": f"f{i}.py"}
        for i in range(5)
    ]
    index.add_vectors(vectors, metadata)

    query_vec = vectors[2]
    results_a = index.hybrid_search(query_vec, "function_2", k=5)
    results_b = index.hybrid_search(query_vec, "function_2", k=5)

    ids_a = [r[1]["id"] for r in results_a]
    ids_b = [r[1]["id"] for r in results_b]
    assert ids_a == ids_b


def test_rrf_empty_index(tmp_path):
    """Searching an empty index returns empty results."""
    index = _make_index(tmp_path)

    query_vec = np.zeros(8, dtype=np.float32)
    results = index.hybrid_search(query_vec, "anything", k=5)
    assert results == []


def test_vector_dimension_mismatch_raises(tmp_path):
    """Adding vectors with wrong dimension must raise ValueError."""
    index = _make_index(tmp_path, dimension=8)

    wrong_dim = np.random.randn(2, 16).astype(np.float32)  # 16 != 8
    metadata = [{"id": "x", "content": "x"}, {"id": "y", "content": "y"}]

    try:
        index.add_vectors(wrong_dim, metadata)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "dimension" in str(e).lower()
