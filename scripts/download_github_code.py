"""
Download code corpus from bigcode/the-stack-smol-xl and build FTS5 indexes.

Downloads Parquet files directly via HTTP (no streaming dependency),
filters by language, and builds per-language FTS5 indexes.

This dataset is ungated and contains ~1.7 GB of permissively-licensed code
across many programming languages.

Usage:
    cd D:\\JCoder
    .venv\\Scripts\\python scripts\\download_github_code.py

Safe to interrupt and resume -- Parquet files are cached, progress tracked.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
import sqlite3
import sys
import time
from pathlib import Path

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace"
    )

from core.download_manager import DownloadManager, fetch_huggingface_parquet_urls

DATA_ROOT = Path(os.environ.get("JCODER_DATA", "data"))
INDEX_DIR = DATA_ROOT / "indexes"
DOWNLOAD_DIR = DATA_ROOT / "downloads" / "stack_smol_xl"
INDEX_DIR.mkdir(parents=True, exist_ok=True)
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

DATASET_ID = "bigcode/the-stack-smol-xl"

# Languages we want (lowercase for matching)
TARGET_LANGUAGES = {
    "python", "javascript", "typescript", "java", "go",
    "c", "c++", "rust", "shell", "c#", "ruby", "php",
}

BATCH_SIZE = 5000
MAX_CHARS = 4000

_NORMALIZE_RE = re.compile(r"[_\-./\\:]")
_CAMEL_RE1 = re.compile(r"([a-z])([A-Z])")
_CAMEL_RE2 = re.compile(r"([A-Z]+)([A-Z][a-z])")


def _normalize(text: str) -> str:
    out = _NORMALIZE_RE.sub(" ", text)
    out = _CAMEL_RE1.sub(r"\1 \2", out)
    out = _CAMEL_RE2.sub(r"\1 \2", out)
    return out.lower()


def _lang_key(lang: str) -> str:
    return lang.lower().replace("+", "p").replace("#", "sharp")


def _download_parquet_files() -> list:
    """Download all Parquet files to local cache. Returns list of local paths."""
    with DownloadManager(DOWNLOAD_DIR, read_timeout_s=300.0) as downloader:
        print(f"Fetching Parquet file list from {DATASET_ID}...")
        urls = fetch_huggingface_parquet_urls(downloader, DATASET_ID)
        print(f"  {len(urls)} Parquet files found")

        local_files = []
        for i, url in enumerate(urls):
            local_path = DOWNLOAD_DIR / f"train_{i:02d}.parquet"
            result = downloader.download_file(
                url,
                local_path.relative_to(DOWNLOAD_DIR),
                min_existing_bytes=1000,
                chunk_size=256 * 1024,
            )
            if not result.ok:
                raise RuntimeError(f"Download failed for {url}: {result.error}")

            size_mb = result.path.stat().st_size / 1e6
            if result.status == "cached":
                print(f"  [{i+1}/{len(urls)}] Already cached ({size_mb:.0f} MB)")
            else:
                print(f"  [{i+1}/{len(urls)}] Downloaded {size_mb:.0f} MB")
            local_files.append(result.path)

        return local_files


def _process_parquet_files(parquet_files: list):
    """Read Parquet files, filter by language, build per-language FTS5 indexes."""
    import pyarrow.parquet as pq

    # Track per-language DB connections and counters
    connections = {}
    counters = {}

    print(f"\nProcessing {len(parquet_files)} Parquet files...")

    total_rows = 0
    total_indexed = 0

    for file_idx, pq_path in enumerate(parquet_files):
        print(f"\n--- Parquet file {file_idx + 1}/{len(parquet_files)} ---")
        t0 = time.time()

        table = pq.read_table(str(pq_path))
        columns = table.column_names
        print(f"  Columns: {columns}")
        print(f"  Rows: {len(table):,}")
        total_rows += len(table)

        # Determine column names (varies by dataset)
        content_col = "content" if "content" in columns else "code"
        lang_col = "lang" if "lang" in columns else "language"

        for row_idx in range(len(table)):
            content = str(table[content_col][row_idx])
            lang = str(table[lang_col][row_idx]).lower() if lang_col in columns else "unknown"

            if lang not in TARGET_LANGUAGES:
                continue

            if len(content) < 50 or len(content) > 100_000:
                continue
            if content.count("\n") < 3:
                continue

            lk = _lang_key(lang)

            # Lazily create DB connection
            if lk not in connections:
                db_path = INDEX_DIR / f"stack_{lk}.fts5.db"
                conn = sqlite3.connect(str(db_path))
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=NORMAL")
                conn.execute(
                    "CREATE VIRTUAL TABLE IF NOT EXISTS chunks "
                    "USING fts5(search_content, source_path, chunk_id)"
                )
                connections[lk] = {"conn": conn, "batch": [], "db_path": db_path}
                counters[lk] = 0

            entry = connections[lk]
            counters[lk] += 1
            count = counters[lk]

            # Chunk and add to batch
            source_id = f"stack_{lk}_{count:07d}"
            pos = 0
            cidx = 0
            while pos < len(content):
                end = min(pos + MAX_CHARS, len(content))
                if end < len(content):
                    nl = content.rfind("\n", pos, end)
                    if nl > pos:
                        end = nl + 1
                chunk = content[pos:end]
                if chunk.strip():
                    cid = hashlib.sha256(f"{source_id}:{cidx}".encode()).hexdigest()
                    entry["batch"].append((_normalize(chunk), source_id, cid))
                    cidx += 1
                pos = end

            # Flush batch
            if len(entry["batch"]) >= BATCH_SIZE:
                entry["conn"].executemany(
                    "INSERT INTO chunks(search_content, source_path, chunk_id) "
                    "VALUES (?, ?, ?)",
                    entry["batch"],
                )
                entry["conn"].commit()
                total_indexed += len(entry["batch"])
                entry["batch"] = []

        elapsed = time.time() - t0
        print(f"  Processed in {elapsed:.0f}s. "
              f"Running totals: {total_rows:,} rows, {total_indexed:,} chunks")

    # Flush remaining batches and close
    print("\nFlushing remaining batches...")
    for lk, entry in connections.items():
        if entry["batch"]:
            entry["conn"].executemany(
                "INSERT INTO chunks(search_content, source_path, chunk_id) "
                "VALUES (?, ?, ?)",
                entry["batch"],
            )
            entry["conn"].commit()
            total_indexed += len(entry["batch"])
        entry["conn"].close()

    return counters, total_indexed


def main():
    print("=" * 60)
    print("Code Corpus Download: bigcode/the-stack-smol-xl")
    print(f"Download dir: {DOWNLOAD_DIR}")
    print(f"Index dir:    {INDEX_DIR}")
    print(f"Languages:    {', '.join(sorted(TARGET_LANGUAGES))}")
    print("=" * 60)

    t0 = time.time()

    # Step 1: Download Parquet files
    try:
        parquet_files = _download_parquet_files()
    except Exception as exc:
        print(f"[FAIL] Download failed: {exc}")
        return

    if not parquet_files:
        print("[FAIL] No Parquet files downloaded")
        return

    # Step 2: Check pyarrow availability
    try:
        import pyarrow.parquet  # noqa: F401
    except ImportError:
        print("[FAIL] pyarrow not installed. Run: pip install pyarrow")
        return

    # Step 3: Process and index
    counters, total_chunks = _process_parquet_files(parquet_files)

    elapsed = time.time() - t0
    print("\n" + "=" * 60)
    print("Summary")
    for lk in sorted(counters.keys()):
        count = counters[lk]
        db_path = INDEX_DIR / f"stack_{lk}.fts5.db"
        size_mb = db_path.stat().st_size / 1e6 if db_path.exists() else 0
        print(f"  {lk:12s}: {count:>8,} entries ({size_mb:.0f} MB)")
    print(f"\n  Total entries: {sum(counters.values()):,}")
    print(f"  Total chunks:  {total_chunks:,}")
    print(f"  Elapsed: {elapsed:.0f}s ({elapsed / 60:.1f} min)")
    print("=" * 60)


if __name__ == "__main__":
    main()
