"""
Build FAISS vector indexes from FTS5 metadata via Ollama embedding.

Handles CPU-only Ollama by embedding one-at-a-time with adaptive fallback.
Processes indexes smallest-first for quick wins.

Usage:
    python -u tools/build_faiss_indexes.py [--max-chunks N] [--batch-size N]
    python -u tools/build_faiss_indexes.py --index-name NAME
"""

import argparse
import io
import json
import os
import sqlite3
import sys
import hashlib
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

# Unbuffered stdout for real-time progress
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import load_config


EMBED_ENDPOINT = "http://localhost:11434/v1/embeddings"
EMBED_MODEL = "nomic-embed-text"
DIM = 768


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.replace("\r\n", "\n").encode("utf-8")).hexdigest()


def _ensure_meta_json(index_name: str, index_dir: str) -> Optional[str]:
    """Generate .meta.json from FTS5 if it does not exist. Returns path or None."""
    base = os.path.join(index_dir, index_name)
    meta_path = base + ".meta.json"
    fts5_path = base + ".fts5.db"

    if os.path.exists(meta_path):
        return meta_path

    if not os.path.exists(fts5_path):
        return None

    conn = sqlite3.connect(fts5_path)
    try:
        cur = conn.cursor()
        cur.execute("SELECT count(*) FROM chunks")
        total = cur.fetchone()[0]
        if total == 0:
            return None

        print(f"  [OK] Generating meta.json from FTS5 ({total:,} chunks)...")
        cur.execute("SELECT search_content, source_path, chunk_id FROM chunks")
        metadata = []
        for row in cur:
            metadata.append({
                "id": row[2],
                "content": row[0],
                "source_path": row[1] or "",
                "source_type": "fts5_extract",
                "ingestion_date": datetime.now().isoformat(),
                "content_hash": _content_hash(row[0]),
            })
    finally:
        conn.close()

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f)
    print(f"  [OK] Wrote {meta_path}")
    return meta_path


def _embed_single(client, text: str) -> List[float]:
    """Embed a single text, truncating if needed."""
    resp = client.post(
        EMBED_ENDPOINT,
        json={"model": EMBED_MODEL, "input": [text]},
    )
    if resp.status_code == 200:
        return resp.json()["data"][0]["embedding"]
    # Truncate on context error
    if resp.status_code == 400:
        resp2 = client.post(
            EMBED_ENDPOINT,
            json={"model": EMBED_MODEL, "input": [text[:2000]]},
        )
        resp2.raise_for_status()
        return resp2.json()["data"][0]["embedding"]
    resp.raise_for_status()


def _embed_batch(client, texts: List[str]) -> "np.ndarray":
    """Embed a batch; fall back to one-at-a-time on context error."""
    import numpy as np
    resp = client.post(
        EMBED_ENDPOINT,
        json={"model": EMBED_MODEL, "input": texts},
    )
    if resp.status_code == 200:
        data = resp.json()["data"]
        vecs = [item["embedding"] for item in sorted(data, key=lambda x: x["index"])]
        return np.array(vecs, dtype=np.float32)

    if resp.status_code == 400:
        # Fall back to individual embedding
        vecs = [_embed_single(client, t) for t in texts]
        return np.array(vecs, dtype=np.float32)

    resp.raise_for_status()


def build_faiss(index_name: str, index_dir: str, batch_size: int = 4) -> bool:
    """Build FAISS index for one index. Returns True on success."""
    import httpx
    import numpy as np
    import faiss

    base = os.path.join(index_dir, index_name)
    faiss_path = base + ".faiss"

    # Skip if already has FAISS with non-zero vectors
    if os.path.exists(faiss_path):
        existing = faiss.read_index(faiss_path)
        if existing.ntotal > 0:
            # Check if it is a placeholder (all zeros)
            sample = existing.reconstruct(0)
            if any(v != 0.0 for v in sample):
                print(f"  [WARN] {index_name}.faiss already has real vectors, skipping")
                return True

    meta_path = _ensure_meta_json(index_name, index_dir)
    if not meta_path:
        print(f"  [FAIL] No FTS5 or meta.json for {index_name}")
        return False

    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    total = len(metadata)
    if total == 0:
        print(f"  [WARN] {index_name}: 0 chunks, skipping")
        return False

    print(f"  [OK] {total:,} chunks to embed")
    all_vectors = np.zeros((total, DIM), dtype=np.float32)

    client = httpx.Client(
        timeout=httpx.Timeout(120.0),
        transport=httpx.HTTPTransport(retries=2),
    )

    try:
        t0 = time.time()
        done = 0
        for i in range(0, total, batch_size):
            batch = [m["content"] for m in metadata[i:i + batch_size]]
            arr = _embed_batch(client, batch)
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            arr = arr / norms
            end = min(i + batch_size, total)
            all_vectors[i:end] = arr
            done = end

            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0
            eta = (total - done) / rate if rate > 0 else 0
            if done % 20 == 0 or done == total:
                print(f"  {done:,}/{total:,} ({rate:.1f} chunks/s, ETA {eta:.0f}s)")

        elapsed = time.time() - t0
        print(f"  [OK] Embedded in {elapsed:.0f}s ({total / elapsed:.1f} chunks/s)")
    finally:
        client.close()

    # Build and save FAISS
    index = faiss.IndexFlatIP(DIM)
    index.add(all_vectors)
    faiss.write_index(index, faiss_path)
    size_mb = os.path.getsize(faiss_path) / (1024 * 1024)
    print(f"  [OK] Saved {faiss_path} ({total:,} vectors, {size_mb:.1f} MB)")
    return True


def discover_targets(index_dir: str, max_chunks: int) -> List[tuple]:
    """Find FTS5 indexes that need FAISS, sorted by chunk count (smallest first)."""
    targets = []
    for f in os.listdir(index_dir):
        if not f.endswith(".fts5.db"):
            continue
        name = f[:-8]  # strip .fts5.db
        base = os.path.join(index_dir, name)
        faiss_path = base + ".faiss"

        # Check chunk count
        meta_path = base + ".meta.json"
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as mf:
                chunks = len(json.load(mf))
        else:
            conn = sqlite3.connect(os.path.join(index_dir, f))
            try:
                cur = conn.cursor()
                cur.execute("SELECT count(*) FROM chunks")
                chunks = cur.fetchone()[0]
            finally:
                conn.close()

        if chunks == 0:
            continue
        if chunks > max_chunks:
            continue

        # Skip if already has real FAISS vectors
        needs_embed = True
        if os.path.exists(faiss_path):
            try:
                import faiss
                idx = faiss.read_index(faiss_path)
                if idx.ntotal > 0:
                    sample = idx.reconstruct(0)
                    if any(v != 0.0 for v in sample):
                        needs_embed = False
            except Exception:
                pass

        if needs_embed:
            fts5_size = os.path.getsize(os.path.join(index_dir, f))
            targets.append((name, chunks, fts5_size))

    targets.sort(key=lambda t: t[1])
    return targets


def main():
    parser = argparse.ArgumentParser(description="Build FAISS indexes from FTS5 via Ollama")
    parser.add_argument("--index-name", help="Embed a specific index")
    parser.add_argument("--max-chunks", type=int, default=50000,
                        help="Skip indexes with more than N chunks (default: 50000)")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Embedding batch size (default: 4, safe for CPU Ollama)")
    args = parser.parse_args()

    cfg = load_config(None)
    index_dir = cfg.storage.index_dir

    if args.index_name:
        print(f"[OK] Building FAISS for: {args.index_name}")
        ok = build_faiss(args.index_name, index_dir, args.batch_size)
        return 0 if ok else 1

    # Auto-discover targets
    targets = discover_targets(index_dir, args.max_chunks)
    if not targets:
        print("[OK] No indexes need FAISS embedding")
        return 0

    print(f"[OK] Found {len(targets)} indexes to embed:")
    total_chunks = 0
    for name, chunks, fts5_size in targets:
        sz = fts5_size / (1024 * 1024)
        print(f"  {chunks:>6,} chunks  {sz:>6.1f} MB  {name}")
        total_chunks += chunks
    print(f"  Total: {total_chunks:,} chunks")
    print()

    succeeded = 0
    failed = 0
    for name, chunks, _ in targets:
        print(f"[OK] === {name} ({chunks:,} chunks) ===")
        try:
            ok = build_faiss(name, index_dir, args.batch_size)
            if ok:
                succeeded += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  [FAIL] {e}")
            failed += 1
        print()

    print(f"[OK] Done: {succeeded} succeeded, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
