"""
Download multiple code corpora from HuggingFace and build FTS5 indexes.

Downloads Parquet files directly via HTTP, processes per-language,
builds FTS5 indexes. All datasets are ungated and freely available.

Datasets:
  1. codeparrot/codeparrot-clean    -- 1.8 GB, cleaned GitHub Python code
  2. bigcode/commitpackft           -- 0.3 GB, commit message + diff pairs (11 languages)
  3. jinaai/code_exercises          -- 0.5 GB, code exercise/solution pairs
  4. deepmind/code_contests         -- 0.1 GB, competitive programming problems+solutions
  5. nickrosh/Evol-Instruct-Code-80k-v1 -- 0.1 GB, evolved code instructions
  6. m-a-p/CodeFeedback-Filtered-Instruction -- 0.2 GB, code Q&A pairs
  7. teknium/OpenHermes-2.5         -- 0.8 GB, mixed instruction set (filter for code)

Usage:
    cd D:\\JCoder
    .venv\\Scripts\\python scripts\\download_code_corpora.py [--only DATASET]

Safe to interrupt and resume -- Parquet files cached, progress tracked.
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

import httpx
import pyarrow.parquet as pq

DATA_ROOT = Path(os.environ.get("JCODER_DATA", r"D:\JCoder_Data"))
INDEX_DIR = DATA_ROOT / "indexes"
DOWNLOAD_DIR = DATA_ROOT / "downloads"
INDEX_DIR.mkdir(parents=True, exist_ok=True)

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


def _download_parquet(url: str, local_path: Path) -> bool:
    """Download a single Parquet file. Returns True on success."""
    if local_path.exists() and local_path.stat().st_size > 1000:
        return True
    local_path.parent.mkdir(parents=True, exist_ok=True)
    partial = local_path.with_suffix(".partial")
    try:
        with httpx.stream("GET", url, follow_redirects=True,
                          timeout=httpx.Timeout(30.0, read=600.0)) as resp:
            resp.raise_for_status()
            with open(partial, "wb") as f:
                for chunk in resp.iter_bytes(chunk_size=256 * 1024):
                    f.write(chunk)
        if local_path.exists():
            local_path.unlink()
        partial.rename(local_path)
        return True
    except Exception as exc:
        print(f"    [WARN] Download failed: {exc}")
        return False


def _get_parquet_urls(dataset_id: str, config: str = "default",
                      split: str = "train") -> list:
    """Get Parquet file URLs from HuggingFace API."""
    url = f"https://huggingface.co/api/datasets/{dataset_id}/parquet"
    r = httpx.get(url, timeout=15)
    r.raise_for_status()
    data = r.json()
    if config in data:
        splits = data[config]
        if isinstance(splits, dict) and split in splits:
            return splits[split]
        if isinstance(splits, list):
            return splits
    return []


class FTS5Builder:
    """Incrementally builds an FTS5 index."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.conn = sqlite3.connect(str(db_path))
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS chunks "
            "USING fts5(search_content, source_path, chunk_id)"
        )
        self._batch = []
        self.total_chunks = 0
        self.total_entries = 0

    def add(self, text: str, source_id: str):
        """Add text, chunking if needed."""
        if not text or len(text.strip()) < 20:
            return
        self.total_entries += 1
        pos = 0
        cidx = 0
        while pos < len(text):
            end = min(pos + MAX_CHARS, len(text))
            if end < len(text):
                nl = text.rfind("\n", pos, end)
                if nl > pos:
                    end = nl + 1
            chunk = text[pos:end]
            if chunk.strip():
                cid = hashlib.sha256(f"{source_id}:{cidx}".encode()).hexdigest()
                self._batch.append((_normalize(chunk), source_id, cid))
                cidx += 1
            pos = end
        if len(self._batch) >= BATCH_SIZE:
            self._flush()

    def _flush(self):
        if self._batch:
            self.conn.executemany(
                "INSERT INTO chunks(search_content, source_path, chunk_id) "
                "VALUES (?, ?, ?)", self._batch)
            self.conn.commit()
            self.total_chunks += len(self._batch)
            self._batch = []

    def finish(self):
        self._flush()
        self.conn.close()
        size_mb = self.db_path.stat().st_size / 1e6
        return self.total_entries, self.total_chunks, size_mb


# -----------------------------------------------------------------------
# Dataset processors
# -----------------------------------------------------------------------

def process_codeparrot_clean():
    """codeparrot/codeparrot-clean -- Cleaned GitHub Python code."""
    ds_id = "codeparrot/codeparrot-clean"
    db_path = INDEX_DIR / "codeparrot_python.fts5.db"
    cache_dir = DOWNLOAD_DIR / "codeparrot_clean"

    if db_path.exists() and db_path.stat().st_size > 1_000_000:
        print(f"  [OK] {db_path.name} already exists ({db_path.stat().st_size/1e6:.0f} MB)")
        return

    print(f"  Downloading {ds_id}...")
    urls = _get_parquet_urls(ds_id)
    print(f"  {len(urls)} Parquet files")

    local_files = []
    for i, url in enumerate(urls):
        local = cache_dir / f"train_{i:02d}.parquet"
        print(f"    [{i+1}/{len(urls)}] ", end="", flush=True)
        if _download_parquet(url, local):
            print(f"{local.stat().st_size/1e6:.0f} MB", flush=True)
            local_files.append(local)
        else:
            print("FAILED", flush=True)

    print(f"  Building FTS5 index...")
    builder = FTS5Builder(db_path)
    for f in local_files:
        table = pq.read_table(str(f), columns=["content"])
        for i in range(len(table)):
            content = str(table["content"][i])
            if len(content) < 50 or len(content) > 100_000:
                continue
            builder.add(content, f"codeparrot_{builder.total_entries:07d}")

        print(f"    {f.name}: {builder.total_entries:,} entries, "
              f"{builder.total_chunks:,} chunks")

    entries, chunks, size = builder.finish()
    print(f"  [OK] {db_path.name}: {entries:,} entries, {chunks:,} chunks ({size:.0f} MB)")


def process_commitpackft():
    """bigcode/commitpackft -- Commit message + diff pairs, per language."""
    ds_id = "bigcode/commitpackft"
    code_langs = ["python", "javascript", "typescript", "java", "go",
                  "c", "c++", "rust", "shell", "ruby", "php"]

    db_path = INDEX_DIR / "commitpack.fts5.db"
    if db_path.exists() and db_path.stat().st_size > 1_000_000:
        print(f"  [OK] {db_path.name} already exists ({db_path.stat().st_size/1e6:.0f} MB)")
        return

    print(f"  Downloading {ds_id} (code languages)...")

    # Get all Parquet URLs
    url = f"https://huggingface.co/api/datasets/{ds_id}/parquet"
    r = httpx.get(url, timeout=15)
    data = r.json()

    builder = FTS5Builder(db_path)
    cache_dir = DOWNLOAD_DIR / "commitpackft"

    for lang in code_langs:
        if lang not in data:
            continue
        splits = data[lang]
        if not isinstance(splits, dict) or "train" not in splits:
            continue
        files = splits["train"]
        if not files:
            continue

        for file_url in files:
            local = cache_dir / f"{lang}.parquet"
            print(f"    {lang}: ", end="", flush=True)
            if not _download_parquet(file_url, local):
                print("FAILED")
                continue

            table = pq.read_table(str(local))
            cols = table.column_names
            msg_col = "message" if "message" in cols else "subject"
            diff_col = "old_contents" if "old_contents" in cols else None

            before = builder.total_entries
            for i in range(len(table)):
                msg = str(table[msg_col][i]) if msg_col in cols else ""
                new_content = str(table["new_contents"][i]) if "new_contents" in cols else ""
                old_content = str(table[diff_col][i]) if diff_col and diff_col in cols else ""

                text = f"Commit: {msg}\n\n{new_content}"
                if len(text.strip()) < 30:
                    continue
                builder.add(text, f"commit_{lang}_{builder.total_entries:07d}")

            added = builder.total_entries - before
            print(f"{added:,} entries")

    entries, chunks, size = builder.finish()
    print(f"  [OK] {db_path.name}: {entries:,} entries, {chunks:,} chunks ({size:.0f} MB)")


def process_code_exercises():
    """jinaai/code_exercises -- Code exercise + solution pairs."""
    ds_id = "jinaai/code_exercises"
    db_path = INDEX_DIR / "code_exercises.fts5.db"
    cache_dir = DOWNLOAD_DIR / "code_exercises"

    if db_path.exists() and db_path.stat().st_size > 1_000_000:
        print(f"  [OK] {db_path.name} already exists ({db_path.stat().st_size/1e6:.0f} MB)")
        return

    print(f"  Downloading {ds_id}...")
    urls = _get_parquet_urls(ds_id)
    print(f"  {len(urls)} Parquet files")

    local_files = []
    for i, url in enumerate(urls):
        local = cache_dir / f"train_{i:02d}.parquet"
        print(f"    [{i+1}/{len(urls)}] ", end="", flush=True)
        if _download_parquet(url, local):
            print(f"{local.stat().st_size/1e6:.0f} MB", flush=True)
            local_files.append(local)
        else:
            print("FAILED", flush=True)

    print(f"  Building FTS5 index...")
    builder = FTS5Builder(db_path)
    for f in local_files:
        table = pq.read_table(str(f))
        cols = table.column_names
        for i in range(len(table)):
            # Try common column names
            parts = []
            for col in cols:
                val = str(table[col][i])
                if val and val != "None" and len(val) > 10:
                    parts.append(val)
            text = "\n\n".join(parts)
            if len(text) < 50:
                continue
            builder.add(text, f"exercise_{builder.total_entries:07d}")

        print(f"    {f.name}: {builder.total_entries:,} entries")

    entries, chunks, size = builder.finish()
    print(f"  [OK] {db_path.name}: {entries:,} entries, {chunks:,} chunks ({size:.0f} MB)")


def process_code_contests():
    """deepmind/code_contests -- Competitive programming problems + solutions."""
    ds_id = "deepmind/code_contests"
    db_path = INDEX_DIR / "code_contests.fts5.db"
    cache_dir = DOWNLOAD_DIR / "code_contests"

    if db_path.exists() and db_path.stat().st_size > 1_000_000:
        print(f"  [OK] {db_path.name} already exists ({db_path.stat().st_size/1e6:.0f} MB)")
        return

    print(f"  Downloading {ds_id}...")
    urls = _get_parquet_urls(ds_id)
    print(f"  {len(urls)} Parquet files")

    local_files = []
    for i, url in enumerate(urls):
        local = cache_dir / f"train_{i:02d}.parquet"
        print(f"    [{i+1}/{len(urls)}] ", end="", flush=True)
        if _download_parquet(url, local):
            print(f"{local.stat().st_size/1e6:.0f} MB", flush=True)
            local_files.append(local)
        else:
            print("FAILED", flush=True)

    print(f"  Building FTS5 index...")
    builder = FTS5Builder(db_path)
    for f in local_files:
        table = pq.read_table(str(f))
        cols = table.column_names
        for i in range(len(table)):
            desc = str(table["description"][i]) if "description" in cols else ""
            solutions = ""
            if "solutions" in cols:
                sol_data = table["solutions"][i]
                if hasattr(sol_data, "as_py"):
                    sol_data = sol_data.as_py()
                if isinstance(sol_data, dict):
                    sol_list = sol_data.get("solution", [])
                    if isinstance(sol_list, list):
                        solutions = "\n\n---\n\n".join(str(s) for s in sol_list[:5])
                elif isinstance(sol_data, list):
                    solutions = "\n\n---\n\n".join(str(s) for s in sol_data[:5])

            text = f"{desc}\n\n{solutions}".strip()
            if len(text) < 50:
                continue
            builder.add(text, f"contest_{builder.total_entries:07d}")

        print(f"    {f.name}: {builder.total_entries:,} entries")

    entries, chunks, size = builder.finish()
    print(f"  [OK] {db_path.name}: {entries:,} entries, {chunks:,} chunks ({size:.0f} MB)")


def process_evol_instruct():
    """nickrosh/Evol-Instruct-Code-80k-v1 -- Evolved code instructions."""
    ds_id = "nickrosh/Evol-Instruct-Code-80k-v1"
    db_path = INDEX_DIR / "evol_instruct_code.fts5.db"
    cache_dir = DOWNLOAD_DIR / "evol_instruct"

    if db_path.exists() and db_path.stat().st_size > 1_000_000:
        print(f"  [OK] {db_path.name} already exists ({db_path.stat().st_size/1e6:.0f} MB)")
        return

    print(f"  Downloading {ds_id}...")
    urls = _get_parquet_urls(ds_id)
    print(f"  {len(urls)} Parquet files")

    local_files = []
    for i, url in enumerate(urls):
        local = cache_dir / f"train_{i:02d}.parquet"
        print(f"    [{i+1}/{len(urls)}] ", end="", flush=True)
        if _download_parquet(url, local):
            print(f"{local.stat().st_size/1e6:.0f} MB", flush=True)
            local_files.append(local)
        else:
            print("FAILED", flush=True)

    print(f"  Building FTS5 index...")
    builder = FTS5Builder(db_path)
    for f in local_files:
        table = pq.read_table(str(f))
        cols = table.column_names
        for i in range(len(table)):
            instruction = str(table["instruction"][i]) if "instruction" in cols else ""
            output = str(table["output"][i]) if "output" in cols else ""
            text = f"Q: {instruction}\n\nA: {output}".strip()
            if len(text) < 50:
                continue
            builder.add(text, f"evol_{builder.total_entries:07d}")

        print(f"    {f.name}: {builder.total_entries:,} entries")

    entries, chunks, size = builder.finish()
    print(f"  [OK] {db_path.name}: {entries:,} entries, {chunks:,} chunks ({size:.0f} MB)")


def process_code_feedback():
    """m-a-p/CodeFeedback-Filtered-Instruction -- Code Q&A pairs."""
    ds_id = "m-a-p/CodeFeedback-Filtered-Instruction"
    db_path = INDEX_DIR / "code_feedback.fts5.db"
    cache_dir = DOWNLOAD_DIR / "code_feedback"

    if db_path.exists() and db_path.stat().st_size > 1_000_000:
        print(f"  [OK] {db_path.name} already exists ({db_path.stat().st_size/1e6:.0f} MB)")
        return

    print(f"  Downloading {ds_id}...")
    urls = _get_parquet_urls(ds_id)
    print(f"  {len(urls)} Parquet files")

    local_files = []
    for i, url in enumerate(urls):
        local = cache_dir / f"train_{i:02d}.parquet"
        print(f"    [{i+1}/{len(urls)}] ", end="", flush=True)
        if _download_parquet(url, local):
            print(f"{local.stat().st_size/1e6:.0f} MB", flush=True)
            local_files.append(local)
        else:
            print("FAILED", flush=True)

    print(f"  Building FTS5 index...")
    builder = FTS5Builder(db_path)
    for f in local_files:
        table = pq.read_table(str(f))
        cols = table.column_names
        for i in range(len(table)):
            query = str(table["query"][i]) if "query" in cols else ""
            answer = str(table["answer"][i]) if "answer" in cols else ""
            text = f"Q: {query}\n\nA: {answer}".strip()
            if len(text) < 50:
                continue
            builder.add(text, f"cfb_{builder.total_entries:07d}")

        print(f"    {f.name}: {builder.total_entries:,} entries")

    entries, chunks, size = builder.finish()
    print(f"  [OK] {db_path.name}: {entries:,} entries, {chunks:,} chunks ({size:.0f} MB)")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

ALL_PROCESSORS = {
    "codeparrot": process_codeparrot_clean,
    "commitpack": process_commitpackft,
    "exercises": process_code_exercises,
    "contests": process_code_contests,
    "evol": process_evol_instruct,
    "feedback": process_code_feedback,
}


def main():
    only = None
    if len(sys.argv) > 2 and sys.argv[1] == "--only":
        only = sys.argv[2]

    print("=" * 60)
    print("JCoder Code Corpora Download + Indexing")
    print(f"Index dir: {INDEX_DIR}")
    if only:
        print(f"Running only: {only}")
    print("=" * 60)

    t0 = time.time()

    for name, processor in ALL_PROCESSORS.items():
        if only and name != only:
            continue
        print(f"\n--- {name.upper()} ---")
        try:
            processor()
        except Exception as exc:
            print(f"  [FAIL] {name}: {exc}")

    elapsed = time.time() - t0
    print("\n" + "=" * 60)
    print(f"Total time: {elapsed:.0f}s ({elapsed / 60:.1f} min)")

    # Show all indexes
    print("\nIndexes in " + str(INDEX_DIR) + ":")
    for db in sorted(INDEX_DIR.glob("*.fts5.db")):
        size_mb = db.stat().st_size / 1e6
        if size_mb > 0.1:
            print(f"  {db.name:35s} {size_mb:>8.0f} MB")
    print("=" * 60)


if __name__ == "__main__":
    main()
