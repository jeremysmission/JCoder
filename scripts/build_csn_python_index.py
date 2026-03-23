"""
Build csn_python.fts5.db directly from python.jsonl.

Reads the HuggingFace-downloaded JSONL (412K entries) and builds an FTS5
full-text search index. Skips the slow .md extraction step entirely.

Usage:
    cd D:\JCoder
    .venv\Scripts\python scripts\build_csn_python_index.py
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

DATA_ROOT = Path(os.environ.get("JCODER_DATA", "data"))
JSONL_DIR = DATA_ROOT / "raw_downloads" / "codesearchnet"
INDEX_DIR = DATA_ROOT / "indexes"
INDEX_DIR.mkdir(parents=True, exist_ok=True)

# All 6 CSN languages -- builds any missing index
LANGUAGES = ["python", "javascript", "java", "go", "ruby", "php"]

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


def _chunk_text(text: str, source_id: str):
    """Yield (search_content, source_path, chunk_id) tuples."""
    pos = 0
    chunk_idx = 0
    while pos < len(text):
        end = min(pos + MAX_CHARS, len(text))
        if end < len(text):
            nl = text.rfind("\n", pos, end)
            if nl > pos:
                end = nl + 1
        chunk = text[pos:end]
        if chunk.strip():
            cid = hashlib.sha256(f"{source_id}:{chunk_idx}".encode()).hexdigest()
            yield (_normalize(chunk), source_id, cid)
            chunk_idx += 1
        pos = end


def build_index_from_jsonl(language: str) -> int:
    """Build FTS5 index for one CSN language from its JSONL file.

    Returns number of chunks indexed.
    """
    jsonl_path = JSONL_DIR / f"{language}.jsonl"
    db_path = INDEX_DIR / f"csn_{language}.fts5.db"

    if not jsonl_path.exists():
        print(f"[WARN] {jsonl_path} not found, skipping {language}")
        return 0

    # Skip if index already exists and has data
    if db_path.exists():
        try:
            conn = sqlite3.connect(str(db_path))
            row = conn.execute(
                "SELECT 1 FROM chunks LIMIT 1"
            ).fetchone()
            conn.close()
            if row:
                size_mb = db_path.stat().st_size / (1024 * 1024)
                print(f"[OK] csn_{language}.fts5.db already exists "
                      f"({size_mb:.0f} MB), skipping")
                return -1  # signal: already exists
        except Exception:
            pass

    print(f"Building csn_{language}.fts5.db from {jsonl_path.name}...")
    t0 = time.time()

    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("DROP TABLE IF EXISTS chunks")
    conn.execute(
        "CREATE VIRTUAL TABLE chunks "
        "USING fts5(search_content, source_path, chunk_id)"
    )

    batch = []
    total_chunks = 0
    entries = 0
    skipped = 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue

            code = obj.get("whole_func_string",
                          obj.get("code",
                          obj.get("original_string",
                          obj.get("function", ""))))
            docstring = obj.get("func_documentation_string",
                               obj.get("docstring", ""))
            func_name = obj.get("func_name",
                               obj.get("identifier", ""))
            repo = obj.get("repository_name",
                          obj.get("repo", ""))

            if not code or len(code.strip()) < 10:
                skipped += 1
                continue

            # Build searchable text combining all fields
            parts = []
            if func_name:
                parts.append(func_name)
            if repo:
                parts.append(f"repo: {repo}")
            if docstring:
                parts.append(docstring.strip())
            parts.append(code.strip())
            text = "\n".join(parts)

            source_id = f"csn_{language}_{entries:06d}"
            for chunk_tuple in _chunk_text(text, source_id):
                batch.append(chunk_tuple)

            entries += 1

            if len(batch) >= BATCH_SIZE:
                conn.executemany(
                    "INSERT INTO chunks(search_content, source_path, chunk_id) "
                    "VALUES (?, ?, ?)",
                    batch,
                )
                conn.commit()
                total_chunks += len(batch)
                batch = []

            if entries % 50000 == 0:
                elapsed = time.time() - t0
                rate = entries / elapsed if elapsed > 0 else 0
                print(f"  {language}: {entries:,} entries -> "
                      f"{total_chunks:,} chunks ({rate:.0f} entries/s)")

    if batch:
        conn.executemany(
            "INSERT INTO chunks(search_content, source_path, chunk_id) "
            "VALUES (?, ?, ?)",
            batch,
        )
        conn.commit()
        total_chunks += len(batch)

    conn.close()
    elapsed = time.time() - t0
    size_mb = db_path.stat().st_size / (1024 * 1024)
    print(f"[OK] csn_{language}.fts5.db: {total_chunks:,} chunks from "
          f"{entries:,} entries ({skipped:,} skipped) in {elapsed:.0f}s "
          f"({size_mb:.0f} MB)")
    return total_chunks


def main():
    print("=" * 60)
    print("Build CSN FTS5 Indexes from JSONL")
    print(f"JSONL dir: {JSONL_DIR}")
    print(f"Index dir: {INDEX_DIR}")
    print("=" * 60)

    t0 = time.time()
    results = {}

    for lang in LANGUAGES:
        count = build_index_from_jsonl(lang)
        results[lang] = count

    elapsed = time.time() - t0
    print("\n" + "=" * 60)
    print("Summary")
    for lang, count in results.items():
        if count == -1:
            print(f"  {lang:12s}: already existed")
        elif count == 0:
            print(f"  {lang:12s}: MISSING JSONL")
        else:
            print(f"  {lang:12s}: {count:,} chunks (new)")
    print(f"  Elapsed: {elapsed:.0f}s ({elapsed / 60:.1f} min)")
    print("=" * 60)


if __name__ == "__main__":
    main()
