"""
Expansion Tier 1: Download + FTS5 index new code/AI/Rust/agentic datasets.

All datasets are ungated HuggingFace Parquet. Downloads cached, safe to resume.

Datasets:
  1. ajibawa-2023/Code-290k-ShareGPT      -- 290K code conversations (multi-lang)
  2. ajibawa-2023/Python-Code-23k-ShareGPT -- 23K Python-only conversations
  3. ajibawa-2023/Code-74k-ShareGPT        -- 74K code conversations
  4. glaiveai/glaive-code-assistant-v3      -- Code Q&A (bigger than v1)
  5. theblackcat102/evol-codealpaca-v1      -- WizardCoder Evol-Instruct 110K
  6. Fortytwo-Network/Strandset-Rust-v1    -- 191K Rust tasks (Apache 2.0)
  7. gaianet/learn-rust                     -- Rust learning dataset
  8. CShorten/ML-ArXiv-Papers              -- 100K+ ML papers from arXiv
  9. flytech/python-codes-25k              -- 25K Python codes
 10. nampdn-ai/tiny-codes                  -- Beginner code examples (large)
 11. LDJnr/Capybara                        -- Multi-turn reasoning conversations

Usage:
    cd D:\\JCoder
    .venv\\Scripts\\python scripts\\download_expansion_tier1.py [--only NAME]
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
    r = httpx.get(url, timeout=30, follow_redirects=True)
    r.raise_for_status()
    data = r.json()
    # Try exact config match
    if config in data:
        splits = data[config]
        if isinstance(splits, dict) and split in splits:
            return splits[split]
        if isinstance(splits, list):
            return splits
    # Fallback: try first config
    for cfg_name, cfg_data in data.items():
        if isinstance(cfg_data, dict) and split in cfg_data:
            return cfg_data[split]
        if isinstance(cfg_data, list):
            return cfg_data
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
                cid = hashlib.sha256(
                    f"{source_id}:{cidx}".encode()
                ).hexdigest()
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


# -------------------------------------------------------------------
# ShareGPT / Conversation format handler
# -------------------------------------------------------------------

def _extract_sharegpt_text(conversations) -> str:
    """Extract text from ShareGPT conversation format."""
    if conversations is None:
        return ""
    if hasattr(conversations, "as_py"):
        conversations = conversations.as_py()
    if isinstance(conversations, str):
        try:
            conversations = json.loads(conversations)
        except (json.JSONDecodeError, ValueError):
            return conversations
    if isinstance(conversations, list):
        parts = []
        for turn in conversations:
            if isinstance(turn, dict):
                role = turn.get("from", turn.get("role", ""))
                value = turn.get("value", turn.get("content", ""))
                if value:
                    label = "Human" if role in ("human", "user") else "Assistant"
                    parts.append(f"{label}: {value}")
            elif isinstance(turn, str):
                parts.append(turn)
        return "\n\n".join(parts)
    return str(conversations)


# -------------------------------------------------------------------
# Generic processors
# -------------------------------------------------------------------

def _generic_qa(dataset_id, db_name, cache_subdir, prefix,
                q_cols, a_cols, config="default"):
    """Generic Q&A processor for instruction datasets."""
    db_path = INDEX_DIR / db_name
    cache_dir = DOWNLOAD_DIR / cache_subdir

    if db_path.exists() and db_path.stat().st_size > 100_000:
        print(f"  [OK] {db_name} exists ({db_path.stat().st_size/1e6:.0f} MB)")
        return

    print(f"  Downloading {dataset_id}...")
    try:
        urls = _get_parquet_urls(dataset_id, config=config)
    except Exception as exc:
        print(f"  [FAIL] Could not get Parquet URLs: {exc}")
        return

    if not urls:
        print(f"  [WARN] No Parquet files found")
        return
    print(f"  {len(urls)} Parquet files")

    local_files = []
    for i, url in enumerate(urls):
        local = cache_dir / f"train_{i:04d}.parquet"
        print(f"    [{i+1}/{len(urls)}] ", end="", flush=True)
        if _download_parquet(url, local):
            sz = local.stat().st_size / 1e6
            print(f"{sz:.0f} MB", flush=True)
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
            q_parts = []
            for qc in q_cols:
                if qc in cols:
                    val = str(table[qc][i])
                    if val and val != "None":
                        q_parts.append(val)
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
    print(f"  [OK] {db_name}: {entries:,} entries, {chunks:,} chunks "
          f"({size:.0f} MB)")


def _generic_sharegpt(dataset_id, db_name, cache_subdir, prefix,
                      conv_col="conversations", config="default"):
    """Generic processor for ShareGPT-format conversation datasets."""
    db_path = INDEX_DIR / db_name
    cache_dir = DOWNLOAD_DIR / cache_subdir

    if db_path.exists() and db_path.stat().st_size > 100_000:
        print(f"  [OK] {db_name} exists ({db_path.stat().st_size/1e6:.0f} MB)")
        return

    print(f"  Downloading {dataset_id}...")
    try:
        urls = _get_parquet_urls(dataset_id, config=config)
    except Exception as exc:
        print(f"  [FAIL] Could not get Parquet URLs: {exc}")
        return

    if not urls:
        print(f"  [WARN] No Parquet files found")
        return
    print(f"  {len(urls)} Parquet files")

    local_files = []
    for i, url in enumerate(urls):
        local = cache_dir / f"train_{i:04d}.parquet"
        print(f"    [{i+1}/{len(urls)}] ", end="", flush=True)
        if _download_parquet(url, local):
            sz = local.stat().st_size / 1e6
            print(f"{sz:.0f} MB", flush=True)
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

        # Find the conversation column
        target_col = None
        for candidate in [conv_col, "conversations", "conversation",
                          "messages", "text", "content"]:
            if candidate in cols:
                target_col = candidate
                break

        if target_col is None:
            # Fallback: concatenate all string columns
            for i in range(len(table)):
                parts = []
                for c in cols:
                    val = str(table[c][i])
                    if val and val != "None" and len(val) > 10:
                        parts.append(val)
                text = "\n\n".join(parts)
                if len(text) < 50:
                    continue
                builder.add(text, f"{prefix}_{builder.total_entries:07d}")
        else:
            for i in range(len(table)):
                raw = table[target_col][i]
                text = _extract_sharegpt_text(raw)
                if len(text) < 50:
                    continue
                builder.add(text, f"{prefix}_{builder.total_entries:07d}")

        print(f"    {f.name}: {builder.total_entries:,} entries")

    entries, chunks, size = builder.finish()
    print(f"  [OK] {db_name}: {entries:,} entries, {chunks:,} chunks "
          f"({size:.0f} MB)")


# -------------------------------------------------------------------
# Dataset processors
# -------------------------------------------------------------------

def process_code_290k():
    """ajibawa-2023/Code-290k-ShareGPT -- 290K code conversations."""
    _generic_sharegpt(
        dataset_id="ajibawa-2023/Code-290k-ShareGPT",
        db_name="code_290k_sharegpt.fts5.db",
        cache_subdir="code_290k_sharegpt",
        prefix="c290k",
    )


def process_python_23k():
    """ajibawa-2023/Python-Code-23k-ShareGPT -- Python-only conversations."""
    _generic_sharegpt(
        dataset_id="ajibawa-2023/Python-Code-23k-ShareGPT",
        db_name="python_23k_sharegpt.fts5.db",
        cache_subdir="python_23k_sharegpt",
        prefix="py23k",
    )


def process_code_74k():
    """ajibawa-2023/Code-74k-ShareGPT -- 74K code conversations."""
    _generic_sharegpt(
        dataset_id="ajibawa-2023/Code-74k-ShareGPT",
        db_name="code_74k_sharegpt.fts5.db",
        cache_subdir="code_74k_sharegpt",
        prefix="c74k",
    )


def process_glaive_v3():
    """glaiveai/glaive-code-assistant-v3 -- Code Q&A set."""
    _generic_qa(
        dataset_id="glaiveai/glaive-code-assistant-v3",
        db_name="glaive_code_v3.fts5.db",
        cache_subdir="glaive_code_v3",
        prefix="glaive3",
        q_cols=["question", "instruction", "input"],
        a_cols=["answer", "output", "response"],
    )


def process_evol_codealpaca():
    """theblackcat102/evol-codealpaca-v1 -- WizardCoder Evol-Instruct 110K."""
    _generic_qa(
        dataset_id="theblackcat102/evol-codealpaca-v1",
        db_name="evol_codealpaca.fts5.db",
        cache_subdir="evol_codealpaca",
        prefix="evolca",
        q_cols=["instruction", "input"],
        a_cols=["output"],
    )


def process_strandset_rust():
    """Fortytwo-Network/Strandset-Rust-v1 -- 191K Rust tasks."""
    _generic_qa(
        dataset_id="Fortytwo-Network/Strandset-Rust-v1",
        db_name="strandset_rust.fts5.db",
        cache_subdir="strandset_rust",
        prefix="rust",
        q_cols=["instruction", "input", "prompt", "question"],
        a_cols=["output", "response", "answer", "completion"],
    )


def process_learn_rust():
    """gaianet/learn-rust -- Rust learning dataset."""
    _generic_sharegpt(
        dataset_id="gaianet/learn-rust",
        db_name="learn_rust.fts5.db",
        cache_subdir="learn_rust",
        prefix="lrust",
    )


def process_ml_arxiv():
    """CShorten/ML-ArXiv-Papers -- 100K+ ML papers from arXiv."""
    _generic_qa(
        dataset_id="CShorten/ML-ArXiv-Papers",
        db_name="ml_arxiv_papers.fts5.db",
        cache_subdir="ml_arxiv_papers",
        prefix="arxiv",
        q_cols=["title"],
        a_cols=["abstract"],
    )


def process_python_codes_25k():
    """flytech/python-codes-25k -- 25K Python codes."""
    _generic_qa(
        dataset_id="flytech/python-codes-25k",
        db_name="python_codes_25k.fts5.db",
        cache_subdir="python_codes_25k",
        prefix="pyc25",
        q_cols=["instruction", "input"],
        a_cols=["output"],
    )


def process_tiny_codes():
    """nampdn-ai/tiny-codes -- Beginner code examples (large)."""
    _generic_qa(
        dataset_id="nampdn-ai/tiny-codes",
        db_name="tiny_codes.fts5.db",
        cache_subdir="tiny_codes",
        prefix="tiny",
        q_cols=["prompt"],
        a_cols=["response"],
    )


def process_capybara():
    """LDJnr/Capybara -- Multi-turn reasoning conversations."""
    _generic_sharegpt(
        dataset_id="LDJnr/Capybara",
        db_name="capybara.fts5.db",
        cache_subdir="capybara",
        prefix="capy",
        conv_col="conversation",
    )


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

ALL_PROCESSORS = {
    "code_290k":       process_code_290k,
    "python_23k":      process_python_23k,
    "code_74k":        process_code_74k,
    "glaive_v3":       process_glaive_v3,
    "evol_codealpaca": process_evol_codealpaca,
    "strandset_rust":  process_strandset_rust,
    "learn_rust":      process_learn_rust,
    "ml_arxiv":        process_ml_arxiv,
    "python_codes_25k": process_python_codes_25k,
    "tiny_codes":      process_tiny_codes,
    "capybara":        process_capybara,
}


def main():
    only = None
    if len(sys.argv) > 2 and sys.argv[1] == "--only":
        only = sys.argv[2]

    print("=" * 60)
    print("JCoder Expansion Tier 1: Download + Index")
    print(f"Index dir: {INDEX_DIR}")
    print(f"Download dir: {DOWNLOAD_DIR}")
    if only:
        print(f"Running only: {only}")
    print("=" * 60)

    t0 = time.time()
    results = {}

    for name, processor in ALL_PROCESSORS.items():
        if only and name != only:
            continue
        print(f"\n--- {name.upper()} ---")
        try:
            processor()
            results[name] = "OK"
        except Exception as exc:
            print(f"  [FAIL] {name}: {exc}")
            results[name] = f"FAIL: {exc}"

    elapsed = time.time() - t0
    print("\n" + "=" * 60)
    print(f"Total time: {elapsed:.0f}s ({elapsed / 60:.1f} min)")

    # Results summary
    print("\nResults:")
    for name, status in results.items():
        print(f"  {name:25s} [{status}]")

    # Show all indexes
    print(f"\nAll indexes in {INDEX_DIR}:")
    total_size = 0
    for db in sorted(INDEX_DIR.glob("*.fts5.db")):
        size_mb = db.stat().st_size / 1e6
        if size_mb > 0.1:
            total_size += size_mb
            print(f"  {db.name:40s} {size_mb:>8.0f} MB")
    print(f"  {'TOTAL':40s} {total_size:>8.0f} MB")
    print("=" * 60)


if __name__ == "__main__":
    main()
