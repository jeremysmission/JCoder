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
import re
import sqlite3
from typing import Dict, List, Optional, Tuple

try:
    import faiss
except ImportError:
    faiss = None  # type: ignore[assignment]
import numpy as np

from .config import StorageConfig


def _normalize_for_search(text: str) -> str:
    """Normalize code text for FTS5 keyword matching.

    Splits underscore/dot/path identifiers into separate tokens
    and breaks CamelCase so FTS5 can match partial identifier words.
    """
    # Replace _ - . / \ : with spaces
    out = re.sub(r"[_\-./\\:]", " ", text)
    # Split CamelCase: "MockReranker" -> "Mock Reranker"
    out = re.sub(r"([a-z])([A-Z])", r"\1 \2", out)
    out = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", out)
    return out.lower()


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
    except (ImportError, OSError):
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
        self._fts_conn: sqlite3.Connection = None
        self._db_path: str = ""
        self._fts5_query_spec: Optional[Dict[str, str]] = None
        os.makedirs(storage.index_dir, exist_ok=True)

        # FAISS index -- try GPU if enough VRAM, fall back to CPU
        if faiss is None:
            self._flat_index = None
            self._gpu_available = False
            self.index = None
            self._sparse_only = True  # auto-degrade to FTS5-only
            print("[WARN] faiss not installed. Dense vector search disabled, using FTS5-only.")
            return
        self._flat_index = faiss.IndexFlatIP(dimension)
        self._gpu_available = False

        if hasattr(faiss, "index_cpu_to_all_gpus"):
            free_mb = _gpu_min_free_mb()
            if free_mb > gpu_safety_margin_mb:
                try:
                    self.index = faiss.index_cpu_to_all_gpus(self._flat_index)
                    self._gpu_available = True
                except RuntimeError as e:
                    print(f"[WARN] FAISS GPU init failed: {e}. Using CPU.")
                    self.index = self._flat_index
            else:
                reason = (f"free VRAM {free_mb} MB < safety margin {gpu_safety_margin_mb} MB"
                          if free_mb > 0 else "VRAM query failed")
                print(f"[WARN] FAISS GPU skipped: {reason}. Using CPU.")
                self.index = self._flat_index
        else:
            self.index = self._flat_index

        # FTS5 DB path is set per-index in save()/load()

    def _get_fts_conn(self) -> sqlite3.Connection:
        """Return persistent FTS5 connection, opening if needed.

        check_same_thread=False allows reuse from ThreadPoolExecutor workers
        in FederatedSearch parallel queries. Safe because FTS5 queries are
        read-only and each IndexEngine owns a single connection.
        """
        if self._fts_conn is None and self._db_path:
            self._fts_conn = sqlite3.connect(
                self._db_path, check_same_thread=False,
            )
            self._configure_fts_conn(self._fts_conn)
        return self._fts_conn

    @staticmethod
    def _configure_fts_conn(conn: sqlite3.Connection) -> None:
        """Apply conservative SQLite settings for read-heavy FTS5 use."""
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")

    def _metadata_at(self, idx: int) -> Optional[Dict]:
        if 0 <= idx < len(self.metadata):
            return self.metadata[idx]
        return None

    def _close_fts_conn(self):
        """Close persistent FTS5 connection if open."""
        if self._fts_conn is not None:
            self._fts_conn.close()
            self._fts_conn = None

    def _resolve_fts5_query_spec(self) -> Dict[str, str]:
        """Resolve column names for modern and legacy FTS5 schemas."""
        if self._fts5_query_spec is not None:
            return self._fts5_query_spec

        conn = self._get_fts_conn()
        if conn is None:
            raise sqlite3.OperationalError("FTS5 database path is not configured")

        columns = {
            row[1]
            for row in conn.execute("PRAGMA table_info(chunks)").fetchall()
        }
        content_col = "search_content" if "search_content" in columns else "content"
        if content_col not in columns:
            raise sqlite3.OperationalError(
                "chunks table missing search_content/content column"
            )

        source_col = ""
        if "source_path" in columns:
            source_col = "source_path"
        elif "source" in columns:
            source_col = "source"

        if "chunk_id" in columns:
            id_expr = "chunk_id"
        elif source_col:
            id_expr = f"{source_col} || ':' || CAST(rowid AS TEXT)"
        else:
            id_expr = "CAST(rowid AS TEXT)"

        self._fts5_query_spec = {
            "match_col": content_col,
            "content_col": content_col,
            "source_expr": source_col or "''",
            "id_expr": id_expr,
        }
        return self._fts5_query_spec

    def close(self):
        """Release FTS5 connection."""
        self._close_fts_conn()

    def add_vectors(self, vectors: np.ndarray, metadata: List[Dict]):
        """
        Add vectors to FAISS and accumulate metadata.
        FTS5 is built on save() for per-index isolation.
        Validates dimension match before insertion.
        """
        if vectors.shape[1] != self.dimension:
            raise ValueError(
                f"Vector dimension {vectors.shape[1]} does not match "
                f"index dimension {self.dimension}"
            )

        if self.index is not None:
            self.index.add(vectors)
        self.metadata.extend(metadata)

    def search_vectors(self, query_vector: np.ndarray, k: int) -> List[Tuple[int, float]]:
        """
        Dense vector search via FAISS.
        Returns list of (metadata_index, score).
        Returns [] when FAISS is unavailable.
        """
        if self.index is None or len(self.metadata) == 0:
            return []
        query_vector = np.array([query_vector], dtype=np.float32)
        scores, indices = self.index.search(query_vector, min(k, len(self.metadata)))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:  # FAISS returns -1 for missing entries
                results.append((int(idx), float(score)))
        return results

    @staticmethod
    def _sanitize_fts5_query(raw: str) -> str:
        """Sanitize arbitrary text into a safe FTS5 MATCH expression.

        Always returns a non-empty string. At minimum returns '""'.
        Normalizes identifiers, strips special chars, quotes each token.
        """
        normalized = _normalize_for_search(raw)
        tokens = normalized.split()
        if not tokens:
            return '""'
        return " OR ".join(f'"{t}"' for t in tokens)

    def search_keywords(self, query: str, k: int) -> List[Tuple[int, float]]:
        """
        Keyword search via SQLite FTS5 BM25 against normalized search_content.
        Returns list of (metadata_index, bm25_score).

        Query is normalized the same way as indexed content so identifiers
        like memory_safety_margin_mb match query words "safety margin".
        """
        or_query = self._sanitize_fts5_query(query)
        if or_query == '""':
            return []
        conn = self._get_fts_conn()
        if conn is None:
            return []
        rows = []

        try:
            spec = self._resolve_fts5_query_spec()
            cursor = conn.execute(
                f"SELECT {spec['id_expr']} AS hit_id, rank FROM chunks "
                f"WHERE {spec['match_col']} MATCH ? "
                "ORDER BY rank LIMIT ?",
                (or_query, k),
            )
            rows = cursor.fetchall()
        except sqlite3.OperationalError as e:
            if not self._fts5_error_logged:
                print(f"[WARN] FTS5 query error (logged once): {e}")
                self._fts5_error_logged = True

        # Map chunk_id back to metadata index
        id_to_idx = {m.get("id"): i for i, m in enumerate(self.metadata)}
        results = []
        for chunk_id, rank in rows:
            idx = id_to_idx.get(chunk_id)
            if idx is not None:
                # FTS5 rank is negative (lower = better), invert for consistency
                results.append((idx, -float(rank)))
        return results

    def search_fts5_direct(self, query: str, k: int) -> List[Tuple[float, Dict]]:
        """Search FTS5 and return content directly without metadata array.

        Returns list of (score, {id, content, source_path}) -- same shape
        as what FederatedSearch._search_single expects, but fetched in a
        single SQL query. No preloaded metadata needed.

        Use this for standalone FTS5 indexes where loading all rows into
        RAM is impractical (e.g. 500 MB+ corpus indexes).
        """
        or_query = self._sanitize_fts5_query(query)
        if or_query == '""':
            return []
        conn = self._get_fts_conn()
        if conn is None:
            return []

        try:
            spec = self._resolve_fts5_query_spec()
            cursor = conn.execute(
                f"SELECT {spec['id_expr']} AS hit_id, "
                f"{spec['content_col']} AS hit_content, "
                f"{spec['source_expr']} AS hit_source, rank "
                "FROM chunks "
                f"WHERE {spec['match_col']} MATCH ? "
                "ORDER BY rank LIMIT ?",
                (or_query, k),
            )
            return [
                (-float(rank), {"id": cid, "content": content, "source_path": src})
                for cid, content, src, rank in cursor.fetchall()
            ]
        except sqlite3.OperationalError as e:
            if not self._fts5_error_logged:
                print(f"[WARN] FTS5 direct query error (logged once): {e}")
                self._fts5_error_logged = True
            return []

    # Intent token sets for path-prior boosting
    _INTENT_CONFIG = {"config", "yaml", "yml", "policy", "policies", "ports",
                      "models", "timeout", "cap", "budget"}
    _INTENT_INTERFACE = {"protocol", "interface", "iembedder", "iretriever", "illm"}
    _INTENT_DOCTOR = {"doctor", "health", "gpu", "nvidia", "endpoint"}
    _INTENT_MOCK = {"mock", "mockreranker", "mock_backend", "deterministic", "fake"}

    @staticmethod
    def _path_prior_boost(query_text: str, source_path: str) -> float:
        """Deterministic additive boost for short files that match query intent."""
        norm_q = set(_normalize_for_search(query_text).split())
        norm_p = source_path.replace("\\", "/").lower()

        boost = 0.0

        if norm_q & IndexEngine._INTENT_CONFIG:
            if "/config/" in norm_p:
                boost = max(boost, 0.30)
            for name in ("policies.yaml", "ports.yaml", "models.yaml", "default.yaml"):
                if norm_p.endswith(name):
                    boost = max(boost, 0.25)
            if norm_p.endswith(".yaml") or norm_p.endswith(".yml"):
                boost = max(boost, 0.20)

        if norm_q & IndexEngine._INTENT_INTERFACE:
            if norm_p.endswith("interfaces.py"):
                boost = max(boost, 0.25)

        if norm_q & IndexEngine._INTENT_DOCTOR:
            if norm_p.endswith("doctor.py"):
                boost = max(boost, 0.25)

        if norm_q & IndexEngine._INTENT_MOCK:
            if norm_p.endswith("mock_backend.py"):
                boost = max(boost, 0.25)

        return boost

    @staticmethod
    def _is_identifier_heavy(query: str) -> bool:
        """Detect queries containing code identifiers (snake_case, CamelCase, dots, parens)."""
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
            # Apply path-prior boost before final sort
            boosted = []
            for idx, score in keyword_results:
                meta = self._metadata_at(idx)
                if meta is None:
                    continue
                path = meta.get("source_path", "")
                boosted.append(
                    (score + self._path_prior_boost(query_text, path), meta)
                )
            boosted.sort(key=lambda x: x[0], reverse=True)
            results = []
            for score, meta in boosted[:k]:
                results.append((score, meta))
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
            meta = self._metadata_at(idx)
            if meta is not None:
                results.append((score, meta))
        return results

    def _build_fts5(self):
        """Build per-index FTS5 DB from current metadata."""
        self._close_fts_conn()
        self._fts5_query_spec = None
        conn = sqlite3.connect(self._db_path)
        self._configure_fts_conn(conn)
        try:
            conn.execute("BEGIN")
            conn.execute("DROP TABLE IF EXISTS chunks")
            conn.execute(
                "CREATE VIRTUAL TABLE chunks "
                "USING fts5(search_content, source_path, chunk_id)"
            )
            rows = [
                (
                    _normalize_for_search(m.get("content", "")),
                    m.get("source_path", ""),
                    m.get("id", ""),
                )
                for m in self.metadata
            ]
            conn.executemany(
                "INSERT INTO chunks(search_content, source_path, chunk_id) "
                "VALUES (?, ?, ?)",
                rows,
            )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
        # Re-open as persistent connection (thread-safe for parallel search)
        self._fts_conn = sqlite3.connect(
            self._db_path, check_same_thread=False,
        )
        self._configure_fts_conn(self._fts_conn)

    def save(self, name: str):
        """Save FAISS index, metadata, and per-index FTS5 DB to disk."""
        path = os.path.join(self.storage.index_dir, name)
        os.makedirs(self.storage.index_dir, exist_ok=True)

        # Save FAISS index if available
        if self.index is not None and faiss is not None:
            if self._gpu_available:
                cpu_index = faiss.index_gpu_to_cpu(self.index)
                faiss.write_index(cpu_index, path + ".faiss")
            else:
                faiss.write_index(self.index, path + ".faiss")

        with open(path + ".meta.json", "w", encoding="utf-8") as f:
            json.dump(self.metadata, f)

        # Build per-index FTS5 DB
        self._db_path = path + ".fts5.db"
        self._fts5_query_spec = None
        self._build_fts5()

    def load(self, name: str):
        """Load FAISS index, metadata, and per-index FTS5 DB from disk."""
        self._close_fts_conn()
        path = os.path.join(self.storage.index_dir, name)

        # Load FAISS index if available
        faiss_path = path + ".faiss"
        if faiss is not None and os.path.exists(faiss_path):
            cpu_index = faiss.read_index(faiss_path)
            if self._gpu_available:
                try:
                    self.index = faiss.index_cpu_to_all_gpus(cpu_index)
                except RuntimeError as e:
                    print(f"[WARN] FAISS GPU reload failed: {e}. Using CPU.")
                    self.index = cpu_index
            else:
                self.index = cpu_index

        with open(path + ".meta.json", "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        # Connect to per-index FTS5 DB (built during save)
        self._db_path = path + ".fts5.db"
        self._fts5_query_spec = None
        if not os.path.exists(self._db_path):
            # Legacy index without per-index FTS5 -- rebuild
            self._build_fts5()

    @property
    def count(self) -> int:
        """Number of vectors in the index."""
        return len(self.metadata)
