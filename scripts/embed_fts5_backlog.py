"""Build FAISS indexes for FTS5 databases that lack vector indexes.

Extracts text from FTS5, embeds via direct CUDA, saves FAISS + meta.json.
Skips databases larger than --max-size-mb (default 100MB).
"""
import json
import os
import sqlite3
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import ModelConfig, StorageConfig

INDEX_DIR = "data/indexes"
MIN_CHUNK_CHARS = 50
BATCH_SIZE = 512


def get_fts5_texts(db_path):
    """Extract text content from FTS5 database."""
    conn = sqlite3.connect(db_path)
    try:
        # Try common column names
        for col in ("search_content", "content", "text", "body"):
            try:
                cursor = conn.execute(f"SELECT {col} FROM chunks")
                rows = cursor.fetchall()
                if rows:
                    return [r[0] for r in rows if r[0] and len(r[0]) >= MIN_CHUNK_CHARS]
            except sqlite3.OperationalError:
                continue

        # Fallback: get all columns and use the first text-like one
        cursor = conn.execute("PRAGMA table_info(chunks)")
        cols = [r[1] for r in cursor.fetchall()]
        for col in cols:
            try:
                cursor = conn.execute(f"SELECT {col} FROM chunks LIMIT 1")
                val = cursor.fetchone()
                if val and isinstance(val[0], str) and len(val[0]) > 20:
                    cursor = conn.execute(f"SELECT {col} FROM chunks")
                    return [r[0] for r in cursor.fetchall()
                            if r[0] and len(r[0]) >= MIN_CHUNK_CHARS]
            except sqlite3.OperationalError:
                continue
    finally:
        conn.close()
    return []


def main():
    max_size_mb = 100  # Skip databases larger than this
    if "--max-size-mb" in sys.argv:
        idx = sys.argv.index("--max-size-mb")
        max_size_mb = int(sys.argv[idx + 1])

    # Find FTS5 DBs without FAISS
    missing = []
    for f in sorted(os.listdir(INDEX_DIR)):
        if f.endswith(".fts5.db"):
            base = f.replace(".fts5.db", "")
            faiss_path = os.path.join(INDEX_DIR, f"{base}.faiss")
            if not os.path.exists(faiss_path):
                size_mb = os.path.getsize(os.path.join(INDEX_DIR, f)) / 1e6
                if size_mb > max_size_mb:
                    print(f"  SKIP {base}: {size_mb:.0f}MB > {max_size_mb}MB limit",
                          flush=True)
                    continue
                if size_mb == 0:
                    print(f"  SKIP {base}: empty", flush=True)
                    continue
                missing.append((base, size_mb))

    if not missing:
        print("All FTS5 databases have FAISS indexes!", flush=True)
        return

    print(f"Found {len(missing)} FTS5 databases without FAISS:", flush=True)
    for name, size in missing:
        print(f"  {name}: {size:.1f}MB", flush=True)

    # Initialize embedder
    os.environ.setdefault("JCODER_EMBED_DEVICE", "cuda:1")
    embed_config = ModelConfig(name="nomic-embed-text-v2-moe", dimension=768)

    from core.embedding_engine import EmbeddingEngine
    embedder = EmbeddingEngine(config=embed_config)
    print(f"\nDirect CUDA: {embedder._use_direct_cuda}", flush=True)

    import faiss

    for name, size_mb in missing:
        db_path = os.path.join(INDEX_DIR, f"{name}.fts5.db")
        faiss_path = os.path.join(INDEX_DIR, f"{name}.faiss")
        meta_path = os.path.join(INDEX_DIR, f"{name}.meta.json")

        print(f"\n{'='*50}", flush=True)
        print(f"Processing: {name} ({size_mb:.1f}MB)", flush=True)

        texts = get_fts5_texts(db_path)
        if not texts:
            print(f"  No text found in {name}, skipping", flush=True)
            continue

        print(f"  Extracted {len(texts):,} chunks (>={MIN_CHUNK_CHARS} chars)",
              flush=True)

        index = faiss.IndexFlatIP(768)
        metadata = []
        embedded = 0
        t0 = time.time()

        for batch_start in range(0, len(texts), BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, len(texts))
            batch_texts = texts[batch_start:batch_end]

            try:
                vectors = embedder.embed(batch_texts)
                index.add(vectors)
                for t in batch_texts:
                    metadata.append({"content": t, "source": name})
                embedded += len(batch_texts)
            except Exception as e:
                print(f"  [ERROR] {batch_start}-{batch_end}: {e}", flush=True)

        elapsed = time.time() - t0
        rate = embedded / elapsed if elapsed > 0 else 0

        # Save
        faiss.write_index(index, faiss_path)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f)

        print(f"  Done: {embedded:,} vectors in {elapsed:.1f}s ({rate:.0f}/s)",
              flush=True)
        print(f"  Saved: {faiss_path} ({os.path.getsize(faiss_path)/1e6:.1f}MB)",
              flush=True)

    embedder.close()
    print(f"\nAll done!", flush=True)


if __name__ == "__main__":
    main()
