"""
High-throughput SE archive ingestion for JCoder.

Optimized for 300K+ sanitized markdown files.
Uses concurrent I/O, streaming FTS5 insertion, and batched processing.

Phase 1: Builds FTS5 keyword index (no GPU needed).
Phase 2: Embeds chunks when vLLM/Ollama is available.

Usage:
    python scripts/bulk_ingest_se.py --source-dir PATH --index-name NAME [--workers N]
    python scripts/bulk_ingest_se.py --phase2-embed --index-name NAME [--embed-endpoint URL]
"""

import argparse
import hashlib
import json
import os
import sqlite3
import sys
import time
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import load_config, StorageConfig


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_CHARS = 4000
BATCH_SIZE = 5000
IO_WORKERS = 16
EMBED_BATCH = 64
DEFAULT_SOURCE = (
    Path(os.environ.get("JCODER_DATA_DIR", "data"))
    / "clean_source" / "_ingest_runs" / "parallel_20260303_213210"
)


# ---------------------------------------------------------------------------
# Text normalization (must match index_engine._normalize_for_search)
# ---------------------------------------------------------------------------

def _normalize_for_search(text: str) -> str:
    out = re.sub(r"[_\-./\\:]", " ", text)
    out = re.sub(r"([a-z])([A-Z])", r"\1 \2", out)
    out = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", out)
    return out.lower()


# ---------------------------------------------------------------------------
# Chunking (character-split only, optimized for markdown)
# ---------------------------------------------------------------------------

def _content_hash(text: str) -> str:
    normalized = text.replace("\r\n", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _chunk_id(source_path: str, start_pos: int, chunk_text: str) -> str:
    raw = f"{source_path}:{start_pos}:{chunk_text.replace(chr(13) + chr(10), chr(10))}"
    return hashlib.sha256(raw.encode("utf-8", errors="replace")).hexdigest()


def _chunk_text(content: str, source_path: str, max_chars: int) -> List[Dict]:
    chunks = []
    pos = 0
    while pos < len(content):
        end = min(pos + max_chars, len(content))
        if end < len(content):
            nl = content.rfind("\n", pos, end)
            if nl > pos:
                end = nl + 1
        chunk_text = content[pos:end]
        if chunk_text.strip():
            content_hash = _content_hash(chunk_text)
            cid = _chunk_id(source_path, pos, chunk_text)
            chunks.append({
                "id": cid,
                "content": chunk_text,
                "source_path": source_path,
                "source_type": ".md",
                "ingestion_date": datetime.now(tz=None).isoformat(),
                "content_hash": content_hash,
                "chunker_version": "bulk_ingest_v1",
                "node_type": "markdown_split",
            })
        pos = end
    return chunks


# ---------------------------------------------------------------------------
# File discovery and reading
# ---------------------------------------------------------------------------

def _discover_md_files(source_dir: Path) -> List[Path]:
    files = []
    for dirpath, _, filenames in os.walk(source_dir):
        for fn in filenames:
            if fn.endswith(".md"):
                files.append(Path(dirpath) / fn)
    return files


def _read_and_chunk(path: Path) -> Tuple[List[Dict], bool]:
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
        if not content.strip():
            return ([], False)
        return (_chunk_text(content, str(path), MAX_CHARS), False)
    except Exception:
        return ([], True)


# ---------------------------------------------------------------------------
# FTS5 builder
# ---------------------------------------------------------------------------

def _create_fts5_db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("DROP TABLE IF EXISTS chunks")
    conn.execute(
        "CREATE VIRTUAL TABLE chunks "
        "USING fts5(search_content, source_path, chunk_id)"
    )
    return conn


def _insert_fts5_batch(conn: sqlite3.Connection, chunks: List[Dict]):
    rows = [
        (
            _normalize_for_search(c.get("content", "")),
            c.get("source_path", ""),
            c.get("id", ""),
        )
        for c in chunks
    ]
    conn.executemany(
        "INSERT INTO chunks(search_content, source_path, chunk_id) "
        "VALUES (?, ?, ?)",
        rows,
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Phase 1: Build FTS5 index + metadata
# ---------------------------------------------------------------------------

@dataclass
class IngestStats:
    files_discovered: int = 0
    files_read: int = 0
    files_empty: int = 0
    files_error: int = 0
    chunks_produced: int = 0
    fts5_rows: int = 0
    elapsed_seconds: float = 0.0


def phase1_build_fts5(
    source_dir: Path,
    index_name: str,
    index_dir: str,
    io_workers: int = IO_WORKERS,
) -> IngestStats:
    stats = IngestStats()
    t0 = time.time()

    # Discover files
    print(f"[OK] Scanning {source_dir} for .md files...")
    files = _discover_md_files(source_dir)
    stats.files_discovered = len(files)
    print(f"[OK] Found {len(files):,} .md files")

    if not files:
        print("[WARN] No .md files found")
        return stats

    # Prepare output paths
    os.makedirs(index_dir, exist_ok=True)
    base_path = os.path.join(index_dir, index_name)
    fts5_path = base_path + ".fts5.db"
    meta_path = base_path + ".meta.json"

    # Create FTS5 DB
    fts_conn = _create_fts5_db(fts5_path)

    # Process files in parallel batches
    all_metadata = []
    batch_chunks = []

    print(f"[OK] Processing with {io_workers} I/O workers...")
    processed = 0

    with ThreadPoolExecutor(max_workers=io_workers) as executor:
        futures = {executor.submit(_read_and_chunk, f): f for f in files}

        for future in as_completed(futures):
            chunks, had_error = future.result()
            processed += 1

            if had_error:
                stats.files_error += 1
            elif chunks:
                stats.files_read += 1
                stats.chunks_produced += len(chunks)
                batch_chunks.extend(chunks)
            else:
                stats.files_empty += 1

            # Flush batch to FTS5
            if len(batch_chunks) >= BATCH_SIZE:
                _insert_fts5_batch(fts_conn, batch_chunks)
                stats.fts5_rows += len(batch_chunks)
                all_metadata.extend(batch_chunks)
                batch_chunks = []

            if processed % 10000 == 0:
                elapsed = time.time() - t0
                rate = processed / elapsed if elapsed > 0 else 0
                print(
                    f"  {processed:,}/{len(files):,} files "
                    f"({stats.chunks_produced:,} chunks, "
                    f"{rate:.0f} files/s)"
                )

    # Flush remaining
    if batch_chunks:
        _insert_fts5_batch(fts_conn, batch_chunks)
        stats.fts5_rows += len(batch_chunks)
        all_metadata.extend(batch_chunks)

    fts_conn.close()

    # Save metadata
    print(f"[OK] Writing metadata ({len(all_metadata):,} chunks)...")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(all_metadata, f)

    # Create minimal FAISS index (placeholder for phase 2)
    try:
        import faiss
        import numpy as np
        dim = 768
        index = faiss.IndexFlatIP(dim)
        # Add zero vectors as placeholders
        zeros = np.zeros((len(all_metadata), dim), dtype=np.float32)
        index.add(zeros)
        faiss.write_index(index, base_path + ".faiss")
        print(f"[OK] Placeholder FAISS index saved ({dim}d, {len(all_metadata):,} vectors)")
    except ImportError:
        print("[WARN] faiss not available, skipping FAISS placeholder")

    stats.elapsed_seconds = time.time() - t0
    return stats


# ---------------------------------------------------------------------------
# Phase 2: Embed chunks and rebuild FAISS
# ---------------------------------------------------------------------------

def phase2_embed(
    index_name: str,
    index_dir: str,
    embed_endpoint: str,
    embed_model: str,
    embed_batch_size: int = EMBED_BATCH,
) -> None:
    import httpx
    import numpy as np
    import faiss

    base_path = os.path.join(index_dir, index_name)
    meta_path = base_path + ".meta.json"

    print(f"[OK] Loading metadata from {meta_path}...")
    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    total = len(metadata)
    print(f"[OK] {total:,} chunks to embed")

    dim = 768
    all_vectors = np.zeros((total, dim), dtype=np.float32)

    endpoint = embed_endpoint.rstrip("/") + "/embeddings"
    client = httpx.Client(timeout=httpx.Timeout(120.0), transport=httpx.HTTPTransport(retries=2))

    def _embed_batch(texts):
        """Embed a batch; on context-length error, fall back to one-at-a-time."""
        resp = client.post(
            endpoint, json={"model": embed_model, "input": texts},
        )
        if resp.status_code == 200:
            data = resp.json()["data"]
            vecs = [item["embedding"] for item in sorted(data, key=lambda x: x["index"])]
            return np.array(vecs, dtype=np.float32)
        # Context-length exceeded -- embed individually
        if resp.status_code == 400 and len(texts) > 1:
            vecs = []
            for t in texts:
                r = client.post(
                    endpoint, json={"model": embed_model, "input": [t]},
                )
                if r.status_code == 200:
                    vecs.append(r.json()["data"][0]["embedding"])
                else:
                    # Truncate to ~2000 chars as last resort
                    r2 = client.post(
                        endpoint,
                        json={"model": embed_model, "input": [t[:2000]]},
                    )
                    r2.raise_for_status()
                    vecs.append(r2.json()["data"][0]["embedding"])
            return np.array(vecs, dtype=np.float32)
        resp.raise_for_status()

    try:
        t0 = time.time()
        done = 0
        for i in range(0, total, embed_batch_size):
            batch = [m["content"] for m in metadata[i:i + embed_batch_size]]
            arr = _embed_batch(batch)
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            arr = arr / norms
            end = min(i + embed_batch_size, total)
            all_vectors[i:end] = arr
            done = end

            if (i // embed_batch_size) % 10 == 0:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                print(f"  {done:,}/{total:,} embedded ({rate:.0f} chunks/s)")

        elapsed = time.time() - t0
        print(f"[OK] Embedding complete in {elapsed:.0f}s ({total / elapsed:.0f} chunks/s)")
    finally:
        client.close()

    # Rebuild FAISS index
    print("[OK] Building FAISS index...")
    index = faiss.IndexFlatIP(dim)
    index.add(all_vectors)
    faiss.write_index(index, base_path + ".faiss")
    print(f"[OK] FAISS index saved ({dim}d, {total:,} vectors)")


# ---------------------------------------------------------------------------
# Generate meta.json from FTS5 (for indexes that only have .fts5.db)
# ---------------------------------------------------------------------------

def generate_meta_from_fts5(index_name: str, index_dir: str) -> int:
    """Extract chunk content from FTS5 and write .meta.json for Phase 2."""
    base_path = os.path.join(index_dir, index_name)
    fts5_path = base_path + ".fts5.db"
    meta_path = base_path + ".meta.json"

    if not os.path.exists(fts5_path):
        print(f"[FAIL] FTS5 database not found: {fts5_path}")
        return 1

    if os.path.exists(meta_path):
        print(f"[WARN] {meta_path} already exists, skipping (delete to regenerate)")
        return 0

    conn = sqlite3.connect(fts5_path)
    cur = conn.cursor()
    cur.execute("SELECT count(*) FROM chunks")
    total = cur.fetchone()[0]

    if total == 0:
        print(f"[WARN] {index_name}: 0 chunks in FTS5, skipping")
        conn.close()
        return 0

    print(f"[OK] Extracting {total:,} chunks from {index_name}.fts5.db...")
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

    conn.close()

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f)

    size_mb = os.path.getsize(meta_path) / (1024 * 1024)
    print(f"[OK] Wrote {meta_path} ({len(metadata):,} chunks, {size_mb:.1f} MB)")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="High-throughput SE ingestion for JCoder"
    )
    parser.add_argument(
        "--source-dir", type=Path, default=DEFAULT_SOURCE,
        help="Directory containing sanitized .md files",
    )
    parser.add_argument(
        "--index-name", default="stackoverflow",
        help="Index name (default: stackoverflow)",
    )
    parser.add_argument(
        "--workers", type=int, default=IO_WORKERS,
        help=f"I/O worker threads (default: {IO_WORKERS})",
    )
    parser.add_argument(
        "--phase2-embed", action="store_true",
        help="Run phase 2: embed chunks with real model",
    )
    parser.add_argument(
        "--embed-endpoint", default="http://localhost:11434/v1",
        help="Embedding server endpoint (default: Ollama)",
    )
    parser.add_argument(
        "--embed-model", default="nomic-embed-text-v2-moe",
        help="Embedding model name",
    )
    parser.add_argument(
        "--embed-batch-size", type=int, default=EMBED_BATCH,
        help=f"Embedding batch size (default: {EMBED_BATCH}; use 8 for CPU-only Ollama)",
    )
    parser.add_argument(
        "--generate-meta", action="store_true",
        help="Generate .meta.json from FTS5 database (needed before Phase 2)",
    )
    args = parser.parse_args()

    cfg = load_config(None)
    index_dir = cfg.storage.index_dir

    if args.generate_meta:
        return generate_meta_from_fts5(args.index_name, index_dir)

    if args.phase2_embed:
        print(f"[OK] Phase 2: Embedding chunks for index '{args.index_name}'")
        phase2_embed(
            args.index_name, index_dir,
            args.embed_endpoint, args.embed_model,
            embed_batch_size=args.embed_batch_size,
        )
        return 0

    print(f"[OK] Phase 1: Building FTS5 index '{args.index_name}'")
    print(f"  Source: {args.source_dir}")
    print(f"  Index dir: {index_dir}")
    print(f"  Workers: {args.workers}")

    stats = phase1_build_fts5(
        args.source_dir, args.index_name, index_dir, args.workers,
    )

    print(f"\nResults:")
    print(f"  Files discovered: {stats.files_discovered:,}")
    print(f"  Files read:       {stats.files_read:,}")
    print(f"  Files empty:      {stats.files_empty:,}")
    print(f"  Files error:      {stats.files_error:,}")
    print(f"  Chunks produced:  {stats.chunks_produced:,}")
    print(f"  FTS5 rows:        {stats.fts5_rows:,}")
    print(f"  Time:             {stats.elapsed_seconds:.0f}s ({stats.elapsed_seconds / 60:.1f}m)")
    if stats.elapsed_seconds > 0:
        print(f"  Throughput:       {stats.files_discovered / stats.elapsed_seconds:.0f} files/s")

    # Write log
    log_dir = Path(index_dir).parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"bulk_ingest_{ts}.json"
    log_data = {
        "generated_at": datetime.now().isoformat(),
        "source_dir": str(args.source_dir),
        "index_name": args.index_name,
        "index_dir": index_dir,
        "stats": {
            "files_discovered": stats.files_discovered,
            "files_read": stats.files_read,
            "files_empty": stats.files_empty,
            "files_error": stats.files_error,
            "chunks_produced": stats.chunks_produced,
            "fts5_rows": stats.fts5_rows,
            "elapsed_seconds": round(stats.elapsed_seconds, 1),
        },
    }
    log_path.write_text(json.dumps(log_data, indent=2), encoding="utf-8")
    print(f"  Log:              {log_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
