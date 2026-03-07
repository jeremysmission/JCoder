"""
Download instruction-tuning and best-practices code datasets.

Focus: syntax correction, code quality, best practices, beginner-friendly.
These datasets contain instruction+response pairs that teach good coding habits.

Datasets:
  1. iamtarun/python_code_instructions_18k_alpaca -- Python instruction pairs
  2. sahil2801/CodeAlpaca-20k -- Code instruction pairs (multi-lang)
  3. TokenBender/code_instructions_122k_alpaca_style -- 122K code instructions
  4. nampdn-ai/tiny-codes -- Simple beginner code examples
  5. bigcode/self-oss-instruct-sc2-exec-filter-50k -- High-quality code instruct
  6. flytech/python-scripts-validation -- Python scripts with validation
  7. TIGER-Lab/MathInstruct -- Math+code reasoning pairs

Usage:
    cd D:\\JCoder
    .venv\\Scripts\\python scripts\\download_instruction_corpora.py [--only DATASET]
"""
from __future__ import annotations

import hashlib
import io
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
    url = f"https://huggingface.co/api/datasets/{dataset_id}/parquet"
    r = httpx.get(url, timeout=15, follow_redirects=True)
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


def _generic_qa_processor(
    dataset_id: str,
    db_name: str,
    cache_subdir: str,
    prefix: str,
    q_cols: list,
    a_cols: list,
    config: str = "default",
):
    """Generic processor for instruction Q&A datasets."""
    db_path = INDEX_DIR / db_name
    cache_dir = DOWNLOAD_DIR / cache_subdir

    if db_path.exists() and db_path.stat().st_size > 100_000:
        print(f"  [OK] {db_name} already exists ({db_path.stat().st_size/1e6:.0f} MB)")
        return

    print(f"  Downloading {dataset_id}...")
    try:
        urls = _get_parquet_urls(dataset_id, config=config)
    except Exception as exc:
        print(f"  [FAIL] Could not get Parquet URLs: {exc}")
        return

    if not urls:
        print(f"  [WARN] No Parquet files found for {dataset_id}")
        return

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

    if not local_files:
        print(f"  [FAIL] No files downloaded")
        return

    print(f"  Building FTS5 index...")
    builder = FTS5Builder(db_path)

    for f in local_files:
        table = pq.read_table(str(f))
        cols = table.column_names

        for i in range(len(table)):
            # Build question from q_cols
            q_parts = []
            for qc in q_cols:
                if qc in cols:
                    val = str(table[qc][i])
                    if val and val != "None":
                        q_parts.append(val)

            # Build answer from a_cols
            a_parts = []
            for ac in a_cols:
                if ac in cols:
                    val = str(table[ac][i])
                    if val and val != "None":
                        a_parts.append(val)

            question = " ".join(q_parts).strip()
            answer = "\n".join(a_parts).strip()

            if not question and not answer:
                continue

            text = f"Q: {question}\n\nA: {answer}" if question else answer
            if len(text) < 50:
                continue

            builder.add(text, f"{prefix}_{builder.total_entries:07d}")

        print(f"    {f.name}: {builder.total_entries:,} entries")

    entries, chunks, size = builder.finish()
    print(f"  [OK] {db_name}: {entries:,} entries, {chunks:,} chunks ({size:.0f} MB)")


# -----------------------------------------------------------------------
# Dataset processors
# -----------------------------------------------------------------------

def process_python_instructions():
    _generic_qa_processor(
        dataset_id="iamtarun/python_code_instructions_18k_alpaca",
        db_name="python_instructions.fts5.db",
        cache_subdir="python_instructions",
        prefix="pyinst",
        q_cols=["prompt", "instruction"],
        a_cols=["output", "response"],
    )


def process_code_alpaca():
    _generic_qa_processor(
        dataset_id="sahil2801/CodeAlpaca-20k",
        db_name="code_alpaca.fts5.db",
        cache_subdir="code_alpaca",
        prefix="alpaca",
        q_cols=["prompt", "instruction"],
        a_cols=["completion", "output", "response"],
    )


def process_code_instructions_122k():
    _generic_qa_processor(
        dataset_id="TokenBender/code_instructions_122k_alpaca_style",
        db_name="code_instructions_122k.fts5.db",
        cache_subdir="code_instructions_122k",
        prefix="ci122k",
        q_cols=["instruction", "input"],
        a_cols=["output"],
    )


def process_tiny_codes():
    _generic_qa_processor(
        dataset_id="nampdn-ai/tiny-codes",
        db_name="tiny_codes.fts5.db",
        cache_subdir="tiny_codes",
        prefix="tiny",
        q_cols=["prompt"],
        a_cols=["response"],
    )


def process_self_oss_instruct():
    _generic_qa_processor(
        dataset_id="bigcode/self-oss-instruct-sc2-exec-filter-50k",
        db_name="self_oss_instruct.fts5.db",
        cache_subdir="self_oss_instruct",
        prefix="ossinst",
        q_cols=["instruction", "prompt"],
        a_cols=["response", "output"],
    )


def process_python_scripts():
    _generic_qa_processor(
        dataset_id="flytech/python-scripts-validation",
        db_name="python_scripts_val.fts5.db",
        cache_subdir="python_scripts_val",
        prefix="pyval",
        q_cols=["instruction", "prompt", "input"],
        a_cols=["output", "response", "code"],
    )


def process_math_instruct():
    _generic_qa_processor(
        dataset_id="TIGER-Lab/MathInstruct",
        db_name="math_instruct.fts5.db",
        cache_subdir="math_instruct",
        prefix="math",
        q_cols=["instruction", "source"],
        a_cols=["output"],
    )


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

ALL_PROCESSORS = {
    "python_inst": process_python_instructions,
    "code_alpaca": process_code_alpaca,
    "code_122k": process_code_instructions_122k,
    "tiny_codes": process_tiny_codes,
    "self_oss": process_self_oss_instruct,
    "python_scripts": process_python_scripts,
    "math": process_math_instruct,
}


def main():
    only = None
    if len(sys.argv) > 2 and sys.argv[1] == "--only":
        only = sys.argv[2]

    print("=" * 60)
    print("JCoder Instruction Corpora Download + Indexing")
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

    print("\nIndexes in " + str(INDEX_DIR) + ":")
    for db in sorted(INDEX_DIR.glob("*.fts5.db")):
        size_mb = db.stat().st_size / 1e6
        if size_mb > 0.1:
            print(f"  {db.name:35s} {size_mb:>8.0f} MB")
    print("=" * 60)


if __name__ == "__main__":
    main()
