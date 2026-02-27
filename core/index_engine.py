"""
Index Engine
------------
Hybrid search: FAISS vectors + SQLite FTS5 keywords, fused with RRF.

Non-programmer explanation:
Finding relevant code needs two strategies:
1. FAISS (vector search) -- finds code with similar *meaning*
2. SQLite FTS5 (keyword search) -- finds code with exact *words*

Neither alone is good enough for code. You need "authentication" to match
semantically AND "def authenticate" to match literally. This module runs
both searches and merges the rankings using Reciprocal Rank Fusion (RRF).
"""

import json
import os
import sqlite3
from typing import Dict, List, Tuple

import faiss
import numpy as np

from .config import StorageConfig


def _gpu_min_free_mb() -> int:
    """Query free VRAM per GPU, return the MINIMUM across all GPUs.

    Per-GPU check because TP splits load across GPUs.
    If any single GPU is below the safety margin, FAISS GPU is unsafe.
    Returns 0 if query fails.
    """
    try:
        import pynvml
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        if count == 0:
            pynvml.nvmlShutdown()
            return 0
        per_gpu = []
        for i in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            per_gpu.append(mem.free // (1024 * 1024))
        pynvml.nvmlShutdown()
        return min(per_gpu)
    except Exception:
        return 0


class IndexEngine:
    """
    Hybrid vector + keyword index with RRF fusion.

    Uses FAISS GPU when available and VRAM exceeds safety margin.
    Falls back to CPU otherwise.
    Uses SQLite FTS5 for keyword search (built into Python's sqlite3).
    """

    def __init__(
        self,
        dimension: int,
        storage: StorageConfig,
        rrf_k: int = 60,
        gpu_safety_margin_mb: int = 2048,
        sparse_only: bool = False,
    ):
        self.dimension = dimension
        self.storage = storage
        self.rrf_k = rrf_k
        self.metadata: List[Dict] = []
        self._fts5_error_logged = False
        self._sparse_only = sparse_only

        # FAISS index -- try GPU if enough VRAM, fall back to CPU
        self._flat_index = faiss.IndexFlatIP(dimension)
        self._gpu_available = False

        if hasattr(faiss, "index_cpu_to_all_gpus"):
            free_mb = _gpu_min_free_mb()
            if free_mb > gpu_safety_margin_mb:
                try:
                    self.index = faiss.index_cpu_to_all_gpus(self._flat_index)
                    self._gpu_available = True
                except Exception:
                    self.index = self._flat_index
            else:
                reason = (f"free VRAM {free_mb} MB < safety margin {gpu_safety_margin_mb} MB"
                          if free_mb > 0 else "VRAM query failed")
                print(f"[WARN] FAISS GPU skipped: {reason}. Using CPU.")
                self.index = self._flat_index
        else:
            self.index = self._flat_index

        # SQLite FTS5 for keyword search
        self._db_path = os.path.join(storage.index_dir, "fts5.db")
        os.makedirs(storage.index_dir, exist_ok=True)
        self._init_fts5()

    def _init_fts5(self):
        """Create the FTS5 virtual table if it doesn't exist."""
        conn = sqlite3.connect(self._db_path)
        conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS chunks "
            "USING fts5(content, source_path, chunk_id)"
        )
        conn.commit()
        conn.close()

    def add_vectors(self, vectors: np.ndarray, metadata: List[Dict]):
        """
        Add vectors to FAISS and text to FTS5.
        Validates dimension match before insertion.
        """
        if vectors.shape[1] != self.dimension:
            raise ValueError(
                f"Vector dimension {vectors.shape[1]} does not match "
                f"index dimension {self.dimension}"
            )

        self.index.add(vectors)
        self.metadata.extend(metadata)

        # Also insert into FTS5 for keyword search
        conn = sqlite3.connect(self._db_path)
        rows = [
            (m.get("content", ""), m.get("source_path", ""), m.get("id", ""))
            for m in metadata
        ]
        conn.executemany(
            "INSERT INTO chunks(content, source_path, chunk_id) VALUES (?, ?, ?)",
            rows,
        )
        conn.commit()
        conn.close()

    def search_vectors(self, query_vector: np.ndarray, k: int) -> List[Tuple[int, float]]:
        """
        Dense vector search via FAISS.
        Returns list of (metadata_index, score).
        """
        if len(self.metadata) == 0:
            return []
        query_vector = np.array([query_vector], dtype=np.float32)
        scores, indices = self.index.search(query_vector, min(k, len(self.metadata)))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:  # FAISS returns -1 for missing entries
                results.append((int(idx), float(score)))
        return results

    @staticmethod
    def _sanitize_fts5_query(query: str) -> str:
        """Strip characters that are invalid in FTS5 MATCH expressions."""
        import re
        # Keep only alphanumeric, spaces, and underscores
        cleaned = re.sub(r"[^\w\s]", " ", query)
        # Collapse whitespace and strip
        tokens = cleaned.split()
        if not tokens:
            return '""'
        # Quote each token so FTS5 treats them as literals
        return " ".join(f'"{t}"' for t in tokens)

    def search_keywords(self, query: str, k: int) -> List[Tuple[int, float]]:
        """
        Keyword search via SQLite FTS5 BM25.
        Returns list of (metadata_index, bm25_score).

        If the quoted-token query returns 0 results, falls back to
        OR-joined normalization search (broader match, lower precision).
        """
        fts_query = self._sanitize_fts5_query(query)
        conn = sqlite3.connect(self._db_path)
        rows = []

        # Primary: quoted literal tokens (high precision)
        try:
            cursor = conn.execute(
                "SELECT chunk_id, rank FROM chunks WHERE chunks MATCH ? "
                "ORDER BY rank LIMIT ?",
                (fts_query, k),
            )
            rows = cursor.fetchall()
        except sqlite3.OperationalError as e:
            if not self._fts5_error_logged:
                print(f"[WARN] FTS5 query error (logged once): {e}")
                self._fts5_error_logged = True

        # Fallback: OR-joined tokens (broader recall)
        if not rows:
            import re
            tokens = re.sub(r"[^\w\s]", " ", query).split()
            if tokens:
                or_query = " OR ".join(f'"{t}"' for t in tokens)
                try:
                    cursor = conn.execute(
                        "SELECT chunk_id, rank FROM chunks WHERE chunks MATCH ? "
                        "ORDER BY rank LIMIT ?",
                        (or_query, k),
                    )
                    rows = cursor.fetchall()
                except sqlite3.OperationalError:
                    pass

        conn.close()

        # Map chunk_id back to metadata index
        id_to_idx = {m.get("id"): i for i, m in enumerate(self.metadata)}
        results = []
        for chunk_id, rank in rows:
            idx = id_to_idx.get(chunk_id)
            if idx is not None:
                # FTS5 rank is negative (lower = better), invert for consistency
                results.append((idx, -float(rank)))
        return results

    @staticmethod
    def _is_identifier_heavy(query: str) -> bool:
        """Detect queries containing code identifiers (snake_case, CamelCase, dots, parens)."""
        import re
        patterns = [
            r'[a-z]+_[a-z]+',      # snake_case
            r'[a-z][a-zA-Z]*[A-Z]', # camelCase/CamelCase
            r'\w+\.\w+',            # foo.bar
            r'\w+\(',               # func(
        ]
        return any(re.search(p, query) for p in patterns)

    def hybrid_search(self, query_vector: np.ndarray, query_text: str, k: int) -> List[Tuple[float, Dict]]:
        """
        Run both vector and keyword search, fuse with RRF.
        In sparse_only mode, skip dense retrieval (mock embeddings are noise).
        Returns list of (fused_score, metadata_dict).
        """
        # Widen sparse pool for identifier-heavy queries
        k_sparse = k * 3 if self._is_identifier_heavy(query_text) else k

        # Sparse-only mode: skip dense retrieval entirely
        if self._sparse_only:
            keyword_results = self.search_keywords(query_text, k_sparse)
            results = []
            for idx, score in keyword_results[:k]:
                results.append((score, self.metadata[idx]))
            return results

        vector_results = self.search_vectors(query_vector, k)
        keyword_results = self.search_keywords(query_text, k_sparse)

        # Reciprocal Rank Fusion
        rrf_scores: Dict[int, float] = {}

        for rank, (idx, _score) in enumerate(vector_results):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (self.rrf_k + rank + 1)

        for rank, (idx, _score) in enumerate(keyword_results):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (self.rrf_k + rank + 1)

        # Sort by fused score descending
        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in ranked[:k]:
            results.append((score, self.metadata[idx]))
        return results

    def save(self, name: str):
        """Save FAISS index and metadata to disk."""
        path = os.path.join(self.storage.index_dir, name)
        os.makedirs(self.storage.index_dir, exist_ok=True)

        # If GPU index, convert back to CPU for saving
        if self._gpu_available:
            cpu_index = faiss.index_gpu_to_cpu(self.index)
            faiss.write_index(cpu_index, path + ".faiss")
        else:
            faiss.write_index(self.index, path + ".faiss")

        with open(path + ".meta.json", "w", encoding="utf-8") as f:
            json.dump(self.metadata, f)

    def load(self, name: str):
        """Load FAISS index and metadata from disk, rebuild FTS5."""
        path = os.path.join(self.storage.index_dir, name)

        cpu_index = faiss.read_index(path + ".faiss")
        if self._gpu_available:
            try:
                self.index = faiss.index_cpu_to_all_gpus(cpu_index)
            except Exception:
                self.index = cpu_index
        else:
            self.index = cpu_index

        with open(path + ".meta.json", "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        # Rebuild FTS5 from loaded metadata so keyword search stays in sync
        conn = sqlite3.connect(self._db_path)
        conn.execute("DROP TABLE IF EXISTS chunks")
        conn.execute(
            "CREATE VIRTUAL TABLE chunks USING fts5(content, source_path, chunk_id)"
        )
        rows = [
            (m.get("content", ""), m.get("source_path", ""), m.get("id", ""))
            for m in self.metadata
        ]
        conn.executemany(
            "INSERT INTO chunks(content, source_path, chunk_id) VALUES (?, ?, ?)",
            rows,
        )
        conn.commit()
        conn.close()

    @property
    def count(self) -> int:
        """Number of vectors in the index."""
        return len(self.metadata)
