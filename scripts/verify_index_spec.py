"""Verify rare_format_documents FAISS index matches HybridRAG spec.

Checks: 768 dimensions, L2 normalized vectors, proper metadata structure.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

INDEX_NAME = "rare_format_documents"
INDEX_DIR = "data/indexes"
EXPECTED_DIM = 768


def main():
    faiss_path = os.path.join(INDEX_DIR, f"{INDEX_NAME}.faiss")
    meta_path = os.path.join(INDEX_DIR, f"{INDEX_NAME}.meta.json")
    fts5_path = os.path.join(INDEX_DIR, f"{INDEX_NAME}.fts5.db")

    errors = []
    warnings = []

    # Check files exist
    for path, label in [(faiss_path, "FAISS"), (meta_path, "metadata"), (fts5_path, "FTS5")]:
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / 1e6
            print(f"  [{label}] {path} ({size_mb:.1f} MB)")
        else:
            errors.append(f"{label} file missing: {path}")
            print(f"  [{label}] MISSING: {path}")

    if errors:
        print(f"\nFAILED: {len(errors)} errors")
        for e in errors:
            print(f"  - {e}")
        return False

    # Check FAISS index
    try:
        import faiss
        index = faiss.read_index(faiss_path)
        dim = index.d
        n_vectors = index.ntotal
        print(f"\n  FAISS: {n_vectors:,} vectors, {dim} dimensions")

        if dim != EXPECTED_DIM:
            errors.append(f"Dimension mismatch: got {dim}, expected {EXPECTED_DIM}")
        else:
            print(f"  Dimension: OK ({dim})")

        # Check L2 normalization on sample vectors
        if n_vectors > 0:
            sample_size = min(100, n_vectors)
            sample = index.reconstruct_batch(list(range(sample_size)))
            norms = np.linalg.norm(sample, axis=1)
            avg_norm = np.mean(norms)
            max_dev = np.max(np.abs(norms - 1.0))
            print(f"  Norm check (sample={sample_size}): avg={avg_norm:.6f}, max_dev={max_dev:.6f}")
            if max_dev > 0.01:
                warnings.append(f"L2 norm deviation > 0.01: max_dev={max_dev:.6f}")
            else:
                print(f"  L2 normalized: OK")

    except ImportError:
        errors.append("faiss not installed")
    except Exception as e:
        errors.append(f"FAISS read error: {e}")

    # Check metadata
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        print(f"\n  Metadata: {len(metadata):,} entries")

        if n_vectors != len(metadata):
            warnings.append(f"Vector/metadata count mismatch: {n_vectors} vs {len(metadata)}")
        else:
            print(f"  Vector/metadata match: OK ({n_vectors:,})")

        # Check chunk structure
        if metadata:
            sample = metadata[0]
            required = {"content", "source"}
            found = set(sample.keys())
            missing = required - found
            if missing:
                warnings.append(f"Metadata missing fields: {missing}")
            else:
                print(f"  Metadata fields: OK ({', '.join(sorted(found)[:6])}...)")

            # Check chunk sizes (HybridRAG compat: ~1200 chars)
            sample_chunks = metadata[:1000]
            avg_len = np.mean([len(m.get("content", "")) for m in sample_chunks])
            print(f"  Avg chunk length: {avg_len:.0f} chars (target: ~1200)")

    except Exception as e:
        errors.append(f"Metadata read error: {e}")

    # Check FTS5
    try:
        import sqlite3
        conn = sqlite3.connect(fts5_path)
        cursor = conn.execute("SELECT COUNT(*) FROM chunks")
        fts5_count = cursor.fetchone()[0]
        conn.close()
        print(f"\n  FTS5: {fts5_count:,} entries")
    except Exception as e:
        warnings.append(f"FTS5 check: {e}")

    # Summary
    print(f"\n{'='*50}")
    if errors:
        print(f"FAILED: {len(errors)} errors")
        for e in errors:
            print(f"  ERROR: {e}")
    elif warnings:
        print(f"PASSED with {len(warnings)} warnings")
        for w in warnings:
            print(f"  WARN: {w}")
    else:
        print("PASSED: All checks OK - HybridRAG compatible")

    return len(errors) == 0


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
