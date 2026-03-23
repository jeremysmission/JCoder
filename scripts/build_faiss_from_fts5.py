"""
Build FAISS vector indexes from existing FTS5 databases.
Uses Ollama nomic-embed-text for 768-dim embeddings.

Processes FTS5 databases that don't yet have a corresponding .faiss file.
Resumable — saves progress after every batch.

Usage:
    python scripts/build_faiss_from_fts5.py [--db rare_formats] [--batch 64] [--max-chunks 50000]
"""

import json
import os
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def embed_batch(texts: list, model: str = "nomic-embed-text") -> np.ndarray:
    """Embed a batch of texts using Ollama."""
    import urllib.request
    vectors = []
    for text in texts:
        payload = json.dumps({"model": model, "prompt": text[:2000]}).encode()
        req = urllib.request.Request(
            "http://localhost:11434/api/embeddings",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
            vectors.append(data["embedding"])
    return np.array(vectors, dtype=np.float32)


def load_fts5_chunks(db_path: str, max_chunks: int = 50000) -> list:
    """Load chunks from FTS5 database (handles different table schemas)."""
    conn = sqlite3.connect(db_path)
    # Discover table name
    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    if not tables:
        conn.close()
        return []
    table = tables[0]
    # Discover columns
    cols = [r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()]
    # Find content column (first text-like column)
    content_col = cols[0]
    source_col = cols[1] if len(cols) > 1 else "rowid"
    id_col = cols[2] if len(cols) > 2 else "rowid"
    cursor = conn.execute(
        f"SELECT {content_col}, {source_col}, {id_col} FROM {table} LIMIT ?",
        (max_chunks,),
    )
    chunks = [{"content": str(row[0]), "source": str(row[1]), "id": str(row[2])}
              for row in cursor.fetchall() if row[0] and len(str(row[0]).strip()) > 20]
    conn.close()
    return chunks


def build_faiss_index(db_name: str, index_dir: str, batch_size: int = 32,
                      max_chunks: int = 50000):
    """Build FAISS index from an FTS5 database."""
    import faiss

    db_path = os.path.join(index_dir, f"{db_name}.fts5.db")
    faiss_path = os.path.join(index_dir, f"{db_name}.faiss")
    meta_path = os.path.join(index_dir, f"{db_name}.faiss.meta.json")
    progress_path = os.path.join(index_dir, f"{db_name}.faiss.progress.json")

    if not os.path.exists(db_path):
        print(f"  [SKIP] {db_name}: FTS5 database not found")
        return

    # Load chunks
    print(f"  Loading {db_name}...")
    chunks = load_fts5_chunks(db_path, max_chunks)
    if not chunks:
        print(f"  [SKIP] {db_name}: No chunks found")
        return
    print(f"  {len(chunks)} chunks loaded")

    # Resume from progress
    start_idx = 0
    all_vectors = []
    if os.path.exists(progress_path):
        try:
            with open(progress_path) as f:
                progress = json.load(f)
            start_idx = progress.get("embedded_count", 0)
            if start_idx >= len(chunks):
                print(f"  [SKIP] {db_name}: Already complete ({start_idx} embedded)")
                return
            # Load partial vectors
            if os.path.exists(faiss_path + ".partial.npy"):
                all_vectors = list(np.load(faiss_path + ".partial.npy"))
            print(f"  Resuming from chunk {start_idx}")
        except Exception:
            start_idx = 0

    # Embed in batches
    t0 = time.monotonic()
    for i in range(start_idx, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        texts = [c["content"] for c in batch]
        try:
            vectors = embed_batch(texts)
            all_vectors.extend(vectors)
        except Exception as e:
            print(f"  [ERROR] Batch {i}: {e}")
            break

        embedded = i + len(batch)
        elapsed = time.monotonic() - t0
        rate = (embedded - start_idx) / elapsed if elapsed > 0 else 0

        if embedded % (batch_size * 10) == 0 or embedded >= len(chunks):
            print(f"  [{embedded}/{len(chunks)}] {rate:.1f} chunks/s")
            # Save progress
            np.save(faiss_path + ".partial.npy", np.array(all_vectors, dtype=np.float32))
            with open(progress_path, "w") as f:
                json.dump({"embedded_count": embedded, "total": len(chunks)}, f)

    if not all_vectors:
        print(f"  [SKIP] {db_name}: No vectors produced")
        return

    # Build FAISS index
    matrix = np.array(all_vectors, dtype=np.float32)
    # L2 normalize for cosine similarity
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    matrix = matrix / norms

    dimension = matrix.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner product = cosine after normalization
    index.add(matrix)
    faiss.write_index(index, faiss_path)

    # Save metadata
    meta = [{"source": c["source"], "id": c["id"]} for c in chunks[:len(all_vectors)]]
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f)

    # Cleanup progress files
    for tmp in [faiss_path + ".partial.npy", progress_path]:
        if os.path.exists(tmp):
            os.remove(tmp)

    elapsed = time.monotonic() - t0
    print(f"  [OK] {db_name}: {index.ntotal} vectors, dim={dimension}, "
          f"{elapsed:.0f}s, saved to {faiss_path}")


def main():
    os.chdir(os.environ.get("JCODER_ROOT", str(Path(__file__).resolve().parent.parent)))
    index_dir = "data/indexes"
    batch_size = 32
    max_chunks = 50000
    target_db = None

    args = sys.argv[1:]
    for i, arg in enumerate(args):
        if arg == "--db" and i + 1 < len(args):
            target_db = args[i + 1]
        elif arg == "--batch" and i + 1 < len(args):
            batch_size = int(args[i + 1])
        elif arg == "--max-chunks" and i + 1 < len(args):
            max_chunks = int(args[i + 1])

    print(f"=== FAISS Index Builder ===")
    print(f"  Index dir: {index_dir}")
    print(f"  Batch size: {batch_size}")
    print(f"  Max chunks per DB: {max_chunks}")

    if target_db:
        build_faiss_index(target_db, index_dir, batch_size, max_chunks)
    else:
        # Find FTS5 databases without FAISS counterparts
        fts5_dbs = sorted(Path(index_dir).glob("*.fts5.db"))
        missing = []
        for db in fts5_dbs:
            faiss_path = db.with_suffix("").with_suffix(".faiss")
            if not faiss_path.exists():
                missing.append(db.stem.replace(".fts5", ""))

        print(f"  FTS5 databases: {len(fts5_dbs)}")
        print(f"  Missing FAISS: {len(missing)}")

        # Prioritize new databases from tonight
        priority = ["rare_formats", "research_corpus", "wayback_research",
                     "fresh_knowledge", "research_feed"]
        ordered = [m for m in priority if m in missing] + [m for m in missing if m not in priority]

        for db_name in ordered:
            print(f"\n--- {db_name} ---")
            build_faiss_index(db_name, index_dir, batch_size, max_chunks)


if __name__ == "__main__":
    main()
