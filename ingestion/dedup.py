"""MinHash-based near-duplicate detection with LSH for fast candidate lookup.

Filters duplicate/near-duplicate content before embedding to avoid wasting
compute on the 953K file corpus. Pure Python + numpy, no external MinHash libs.

Flow: SHA-256 exact check (O(1)) -> MinHash + LSH candidates -> Jaccard verify.
"""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np

# Large Mersenne prime for hash universe
_MERSENNE_PRIME = (1 << 61) - 1
_MAX_HASH = (1 << 32) - 1
_SEED = 42


@dataclass
class DedupStats:
    total_seen: int = 0
    unique: int = 0
    exact_dupes: int = 0
    near_dupes: int = 0


class MinHashDedup:
    """Near-duplicate detection using MinHash + Locality-Sensitive Hashing."""

    def __init__(
        self,
        num_perm: int = 128,
        threshold: float = 0.8,
        num_bands: int = 0,
        persist_path: str = "",
    ):
        if not 0.0 < threshold < 1.0:
            raise ValueError("threshold must be in (0, 1)")
        if num_perm < 2:
            raise ValueError("num_perm must be >= 2")

        self.num_perm = num_perm
        self.threshold = threshold
        self.persist_path = persist_path

        # --- LSH band/row calculation ---
        if num_bands > 0:
            self.num_bands = num_bands
            self.rows_per_band = num_perm // num_bands
        else:
            self.num_bands, self.rows_per_band = self._optimal_bands(
                num_perm, threshold
            )

        # --- Deterministic hash coefficients: h_i(x) = (a_i * x + b_i) % p ---
        rng = np.random.RandomState(_SEED)
        self._coeff_a = rng.randint(1, _MERSENNE_PRIME, size=num_perm, dtype=np.int64)
        self._coeff_b = rng.randint(0, _MERSENNE_PRIME, size=num_perm, dtype=np.int64)

        # --- State ---
        self._content_hashes: Set[str] = set()
        # doc_id -> signature
        self._signatures: Dict[str, np.ndarray] = {}
        # bucket_hash -> set of doc_ids
        self._lsh_buckets: Dict[int, Set[str]] = {}
        self._stats = DedupStats()
        self._next_id = 0

        # Resume from disk if available
        if persist_path:
            self.load()

    # ------------------------------------------------------------------
    # LSH band calculation
    # ------------------------------------------------------------------

    @staticmethod
    def _optimal_bands(num_perm: int, threshold: float) -> Tuple[int, int]:
        """Find (bands, rows) so that P(candidate) ~ 0.5 at the threshold."""
        best_bands = 1
        best_err = float("inf")
        for b in range(1, num_perm + 1):
            r = num_perm // b
            if r < 1 or b * r != num_perm:
                continue
            # P = 1 - (1 - t^r)^b  -- want this near 0.5
            prob = 1.0 - (1.0 - threshold ** r) ** b
            err = abs(prob - 0.5)
            if err < best_err:
                best_err = err
                best_bands = b
        best_rows = num_perm // best_bands
        return best_bands, best_rows

    # ------------------------------------------------------------------
    # Shingling
    # ------------------------------------------------------------------

    @staticmethod
    def _shingle(text: str, k: int = 5) -> Set[str]:
        """Convert text to k-character shingles (overlapping substrings)."""
        text = text.lower().strip()
        if len(text) < k:
            return {text} if text else set()
        return {text[i : i + k] for i in range(len(text) - k + 1)}

    # ------------------------------------------------------------------
    # MinHash
    # ------------------------------------------------------------------

    def _minhash(self, shingles: Set[str]) -> np.ndarray:
        """Compute MinHash signature from shingles using vectorized ops."""
        if not shingles:
            return np.full(self.num_perm, _MAX_HASH, dtype=np.uint32)

        # Hash each shingle to a 32-bit integer
        hashes = np.array(
            [struct.unpack("<I", hashlib.md5(s.encode()).digest()[:4])[0] for s in shingles],
            dtype=np.int64,
        )

        # Vectorized: for each permutation, compute min over all shingles
        # shape: (num_perm, num_shingles)
        a = self._coeff_a[:, np.newaxis]  # (num_perm, 1)
        b = self._coeff_b[:, np.newaxis]  # (num_perm, 1)
        h = hashes[np.newaxis, :]         # (1, num_shingles)

        permuted = (a * h + b) % _MERSENNE_PRIME  # (num_perm, num_shingles)
        signature = permuted.min(axis=1).astype(np.uint32)
        return signature

    # ------------------------------------------------------------------
    # LSH buckets
    # ------------------------------------------------------------------

    def _lsh_bucket_keys(self, signature: np.ndarray) -> List[int]:
        """Compute LSH bucket hashes from signature bands."""
        keys = []
        for band_idx in range(self.num_bands):
            start = band_idx * self.rows_per_band
            end = start + self.rows_per_band
            band = signature[start:end]
            # Hash the band slice to a single bucket key
            h = hashlib.sha1(band.tobytes()).digest()
            bucket_key = struct.unpack("<q", h[:8])[0] ^ band_idx
            keys.append(bucket_key)
        return keys

    # ------------------------------------------------------------------
    # Jaccard similarity from signatures
    # ------------------------------------------------------------------

    @staticmethod
    def jaccard_similarity(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
        """Estimate Jaccard similarity from two MinHash signatures."""
        return float(np.sum(sig_a == sig_b)) / len(sig_a)

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def is_duplicate(self, text: str, doc_id: str = "") -> bool:
        """Check if text is duplicate of something already seen.

        Registers the text if it is not a duplicate.
        Returns True if duplicate, False if unique.
        """
        self._stats.total_seen += 1

        if not doc_id:
            doc_id = f"_auto_{self._next_id}"
            self._next_id += 1

        # --- Exact duplicate: O(1) ---
        content_hash = hashlib.sha256(text.encode()).hexdigest()
        if content_hash in self._content_hashes:
            self._stats.exact_dupes += 1
            return True

        # --- MinHash + LSH ---
        shingles = self._shingle(text)
        signature = self._minhash(shingles)
        bucket_keys = self._lsh_bucket_keys(signature)

        # Gather candidate doc_ids from buckets
        candidates: Set[str] = set()
        for key in bucket_keys:
            if key in self._lsh_buckets:
                candidates.update(self._lsh_buckets[key])

        # Verify candidates with full Jaccard similarity
        for cand_id in candidates:
            sim = self.jaccard_similarity(signature, self._signatures[cand_id])
            if sim >= self.threshold:
                self._stats.near_dupes += 1
                return True

        # --- Not a duplicate: register it ---
        self._content_hashes.add(content_hash)
        self._signatures[doc_id] = signature
        for key in bucket_keys:
            if key not in self._lsh_buckets:
                self._lsh_buckets[key] = set()
            self._lsh_buckets[key].add(doc_id)
        self._stats.unique += 1
        return False

    def add(self, text: str, doc_id: str = "") -> bool:
        """Add text to the index. Returns True if it was unique (not a dupe)."""
        return not self.is_duplicate(text, doc_id)

    def stats(self) -> DedupStats:
        """Return deduplication statistics."""
        return DedupStats(
            total_seen=self._stats.total_seen,
            unique=self._stats.unique,
            exact_dupes=self._stats.exact_dupes,
            near_dupes=self._stats.near_dupes,
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Persist state to disk for resume across runs."""
        if not self.persist_path:
            return
        path = Path(self.persist_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Signatures as {doc_id: list_of_ints}
        sigs = {k: v.tolist() for k, v in self._signatures.items()}
        # Buckets as {str(key): list_of_doc_ids}
        buckets = {str(k): list(v) for k, v in self._lsh_buckets.items()}

        state = {
            "content_hashes": list(self._content_hashes),
            "signatures": sigs,
            "lsh_buckets": buckets,
            "stats": {
                "total_seen": self._stats.total_seen,
                "unique": self._stats.unique,
                "exact_dupes": self._stats.exact_dupes,
                "near_dupes": self._stats.near_dupes,
            },
            "next_id": self._next_id,
            "num_perm": self.num_perm,
            "threshold": self.threshold,
            "num_bands": self.num_bands,
        }
        path.write_text(json.dumps(state), encoding="utf-8")

    def load(self) -> None:
        """Load persisted state. Silently no-ops if file missing."""
        if not self.persist_path:
            return
        path = Path(self.persist_path)
        if not path.exists():
            return

        state = json.loads(path.read_text(encoding="utf-8"))

        # Validate compatibility
        if state.get("num_perm") != self.num_perm:
            return  # Incompatible, start fresh

        self._content_hashes = set(state["content_hashes"])
        self._signatures = {
            k: np.array(v, dtype=np.uint32)
            for k, v in state["signatures"].items()
        }
        self._lsh_buckets = {
            int(k): set(v) for k, v in state["lsh_buckets"].items()
        }
        s = state["stats"]
        self._stats = DedupStats(
            total_seen=s["total_seen"],
            unique=s["unique"],
            exact_dupes=s["exact_dupes"],
            near_dupes=s["near_dupes"],
        )
        self._next_id = state.get("next_id", 0)

    def reset(self) -> None:
        """Clear all state."""
        self._content_hashes.clear()
        self._signatures.clear()
        self._lsh_buckets.clear()
        self._stats = DedupStats()
        self._next_id = 0
