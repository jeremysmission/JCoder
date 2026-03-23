"""
FTS5 Index Builder Demo
------------------------
Builds an FTS5 full-text search index from any source code repository.
Then runs interactive search queries against it.

Usage:
    python tools/fts5_demo.py D:\HybridRAG3_Educational
    python tools/fts5_demo.py D:\some\folder --output my_index.fts5.db
"""

import io
import os
import sqlite3
import sys
import time
from pathlib import Path

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace"
    )

# Import central extension registry from core
try:
    from ingestion.chunker import LANGUAGE_MAP
    _HAS_LANGUAGE_MAP = True
except ImportError:
    _HAS_LANGUAGE_MAP = False
    LANGUAGE_MAP = {}

# File types to index: base set + all code extensions from LANGUAGE_MAP
_BASE_INDEX_EXTENSIONS = {
    ".md", ".txt", ".yaml", ".yml", ".json", ".toml",
    ".ps1", ".sh", ".bat", ".cmd", ".html", ".htm", ".css",
    ".cfg", ".ini", ".xml", ".csv",
}
INDEX_EXTENSIONS = _BASE_INDEX_EXTENSIONS.copy()
if _HAS_LANGUAGE_MAP:
    INDEX_EXTENSIONS.update(LANGUAGE_MAP.keys())

SKIP_DIRS = {
    "__pycache__", ".git", ".venv", "node_modules", ".pytest_cache",
    ".mypy_cache", "dist", "build", "egg-info",
}

MAX_FILE_SIZE = 256 * 1024  # 256 KB


# ---------------------------------------------------------------------------
# Step 1: Crawl and chunk
# ---------------------------------------------------------------------------

def crawl_and_chunk(source_dir, chunk_size=800, overlap=100):
    """Walk a directory, read text files, split into chunks.

    Yields dicts: {content, source_path, chunk_id}
    """
    root = Path(source_dir).resolve()

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]

        for filename in sorted(filenames):
            filepath = Path(dirpath) / filename
            ext = filepath.suffix.lower()

            if ext not in INDEX_EXTENSIONS:
                continue
            try:
                if filepath.stat().st_size > MAX_FILE_SIZE:
                    continue
            except OSError:
                continue

            try:
                text = filepath.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue

            if len(text.strip()) < 20:
                continue

            rel_path = str(filepath.relative_to(root))
            lines = text.splitlines()
            buf = []
            buf_len = 0
            chunk_idx = 0

            for line in lines:
                buf.append(line)
                buf_len += len(line)

                if buf_len >= chunk_size:
                    content = "\n".join(buf)
                    yield {
                        "content": content,
                        "source_path": rel_path,
                        "chunk_id": f"{rel_path}::chunk_{chunk_idx}",
                    }
                    chunk_idx += 1

                    # Keep tail as overlap
                    overlap_lines = []
                    overlap_len = 0
                    for ol in reversed(buf):
                        if overlap_len + len(ol) > overlap:
                            break
                        overlap_lines.insert(0, ol)
                        overlap_len += len(ol)
                    buf = overlap_lines
                    buf_len = overlap_len

            if buf and len("\n".join(buf).strip()) > 20:
                yield {
                    "content": "\n".join(buf),
                    "source_path": rel_path,
                    "chunk_id": f"{rel_path}::chunk_{chunk_idx}",
                }


# ---------------------------------------------------------------------------
# Step 2: Build FTS5 index
# ---------------------------------------------------------------------------

def build_index(source_dir, db_path, batch_size=500):
    """Build an FTS5 index from a source directory."""
    print(f"[1/3] Crawling: {source_dir}")

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("DROP TABLE IF EXISTS chunks")
    conn.execute(
        "CREATE VIRTUAL TABLE chunks "
        "USING fts5(search_content, source_path, chunk_id)"
    )

    batch = []
    total_chunks = 0
    total_files = set()
    t0 = time.monotonic()

    print("[2/3] Indexing...")

    for chunk in crawl_and_chunk(source_dir):
        total_files.add(chunk["source_path"])
        batch.append((
            chunk["content"],
            chunk["source_path"],
            chunk["chunk_id"],
        ))

        if len(batch) >= batch_size:
            conn.executemany(
                "INSERT INTO chunks(search_content, source_path, chunk_id) "
                "VALUES (?, ?, ?)",
                batch,
            )
            conn.commit()
            total_chunks += len(batch)
            print(f"     {total_chunks} chunks indexed...", end="\r")
            batch = []

    if batch:
        conn.executemany(
            "INSERT INTO chunks(search_content, source_path, chunk_id) "
            "VALUES (?, ?, ?)",
            batch,
        )
        conn.commit()
        total_chunks += len(batch)

    elapsed = time.monotonic() - t0
    db_size = os.path.getsize(db_path)

    print(f"[3/3] Done!                          ")
    print(f"     Files indexed:  {len(total_files)}")
    print(f"     Chunks created: {total_chunks}")
    print(f"     Index size:     {db_size / 1024:.0f} KB")
    print(f"     Time:           {elapsed:.1f}s")
    print(f"     Output:         {db_path}")
    print()

    conn.close()
    return total_chunks


# ---------------------------------------------------------------------------
# Step 3: Search
# ---------------------------------------------------------------------------

def search_index(db_path, query, top_k=10):
    """Search the FTS5 index and return results."""
    conn = sqlite3.connect(db_path)

    terms = query.strip().split()
    fts_query = " OR ".join(terms[:10])

    try:
        rows = conn.execute(
            "SELECT source_path, chunk_id, "
            "snippet(chunks, 0, '>>>', '<<<', '...', 25) "
            "FROM chunks WHERE chunks MATCH ? "
            "ORDER BY rank LIMIT ?",
            (fts_query, top_k),
        ).fetchall()
    except Exception as exc:
        print(f"[FAIL] Search error: {exc}")
        return []
    finally:
        conn.close()

    return rows


def run_demo_queries(db_path):
    """Run a set of demo queries to show the index working."""
    demos = [
        "embedding engine ollama",
        "config yaml default model",
        "query pipeline retrieval",
        "test mock fixture",
        "security credential encryption",
        "GUI panel settings",
        "chunking overlap split",
        "FastAPI server endpoint",
    ]

    print("=" * 60)
    print("  FTS5 Search Demo -- sample queries")
    print("=" * 60)

    for query in demos:
        t0 = time.monotonic()
        results = search_index(db_path, query, top_k=3)
        elapsed = (time.monotonic() - t0) * 1000

        print(f"\n  Query: \"{query}\"  ({elapsed:.0f}ms, {len(results)} hits)")
        if not results:
            print("    (no results)")
            continue
        for i, (source, chunk_id, snippet) in enumerate(results, 1):
            # Clean up snippet for display
            snippet_clean = snippet.replace("\n", " ")[:120]
            print(f"    [{i}] {source}")
            print(f"        {snippet_clean}")

    print(f"\n{'=' * 60}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Build and search an FTS5 index from a code repo"
    )
    parser.add_argument("source_dir", help="Directory to index")
    parser.add_argument(
        "--output", default="",
        help="Output .fts5.db path (default: <name>.fts5.db in current dir)"
    )
    parser.add_argument(
        "--search-only", action="store_true",
        help="Skip building, go straight to search"
    )
    args = parser.parse_args()

    source = Path(args.source_dir).resolve()
    if not source.is_dir():
        print(f"[FAIL] Not a directory: {source}")
        sys.exit(1)

    db_name = source.name + ".fts5.db"
    db_path = args.output or db_name

    if not args.search_only:
        build_index(str(source), db_path)

    if not os.path.exists(db_path):
        print(f"[FAIL] Index not found: {db_path}")
        sys.exit(1)

    run_demo_queries(db_path)


if __name__ == "__main__":
    main()
