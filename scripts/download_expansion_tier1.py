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

import argparse
import hashlib
import io
import json
import os
import re
import sqlite3
import sys
import time
from pathlib import Path

import httpx

from core.download_manager import DownloadManager, fetch_huggingface_parquet_urls

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DATA_ROOT_ENV = os.environ.get("JCODER_DATA") or os.environ.get("JCODER_DATA_DIR")
DATA_ROOT = Path(_DATA_ROOT_ENV) if _DATA_ROOT_ENV else _PROJECT_ROOT / "data"
INDEX_DIR = DATA_ROOT / "indexes"
DOWNLOAD_DIR = DATA_ROOT / "downloads"

BATCH_SIZE = 5000
MAX_CHARS = 4000

_NORMALIZE_RE = re.compile(r"[_\-./\\:]")
_CAMEL_RE1 = re.compile(r"([a-z])([A-Z])")
_CAMEL_RE2 = re.compile(r"([A-Z]+)([A-Z][a-z])")
_DOWNLOADER: DownloadManager | None = None
_LEARN_RUST_RAW_FILES = [
    (
        "rust-books.txt",
        "https://huggingface.co/datasets/gaianet/learn-rust/raw/main/rust-books.txt",
    ),
    (
        "rust-qa.txt",
        "https://huggingface.co/datasets/gaianet/learn-rust/raw/main/rust-qa.txt",
    ),
]


def _configure_stdout() -> None:
    if sys.platform != "win32":
        return
    if getattr(sys.stdout, "buffer", None) is None:
        return
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace"
    )


def _normalize(text: str) -> str:
    out = _NORMALIZE_RE.sub(" ", text)
    out = _CAMEL_RE1.sub(r"\1 \2", out)
    out = _CAMEL_RE2.sub(r"\1 \2", out)
    return out.lower()


def _get_downloader() -> DownloadManager:
    global _DOWNLOADER
    if _DOWNLOADER is None:
        _DOWNLOADER = DownloadManager(DOWNLOAD_DIR, read_timeout_s=600.0)
    return _DOWNLOADER


def _close_downloader() -> None:
    global _DOWNLOADER
    if _DOWNLOADER is not None:
        _DOWNLOADER.close()
        _DOWNLOADER = None


def _download_parquet(url: str, local_path: Path) -> bool:
    relative_path = local_path.relative_to(DOWNLOAD_DIR)
    result = _get_downloader().download_file(
        url,
        relative_path,
        min_existing_bytes=1000,
        chunk_size=256 * 1024,
    )
    if result.ok:
        return True
    print(f"    [WARN] Download failed: {result.error}")
    return False


def _download_text_asset(url: str, local_path: Path) -> bool:
    relative_path = local_path.relative_to(DOWNLOAD_DIR)
    result = _get_downloader().download_file(
        url,
        relative_path,
        min_existing_bytes=10,
        chunk_size=256 * 1024,
    )
    if result.ok:
        return True
    print(f"    [WARN] Download failed: {result.error}")
    return False


def _get_parquet_urls(dataset_id: str, config: str = "default",
                      split: str = "train") -> list:
    return fetch_huggingface_parquet_urls(
        _get_downloader(),
        dataset_id,
        config=config,
        split=split,
        fallback_to_first_config=True,
    )


def _split_text_blocks(text: str) -> list[str]:
    normalized = text.replace("\r\n", "\n")
    blocks = [block.strip() for block in normalized.split("\n\n") if len(block.strip()) >= 50]
    if len(blocks) > 1:
        return blocks
    return [line.strip() for line in normalized.splitlines() if len(line.strip()) >= 50]


def _read_parquet_table(path: Path):
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow not installed. Run: pip install pyarrow") from exc

    return pq.read_table(str(path))


class FTS5Builder:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
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
                q_cols, a_cols, config="default") -> bool:
    """Generic Q&A processor for instruction datasets."""
    db_path = INDEX_DIR / db_name
    cache_dir = DOWNLOAD_DIR / cache_subdir

    if db_path.exists() and db_path.stat().st_size > 100_000:
        print(f"  [OK] {db_name} exists ({db_path.stat().st_size/1e6:.0f} MB)")
        return True

    print(f"  Downloading {dataset_id}...")
    try:
        urls = _get_parquet_urls(dataset_id, config=config)
    except Exception as exc:
        print(f"  [FAIL] Could not get Parquet URLs: {exc}")
        return False

    if not urls:
        print(f"  [WARN] No Parquet files found")
        return False
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
        return False

    print(f"  Building FTS5 index...")
    builder = FTS5Builder(db_path)

    for f in local_files:
        table = _read_parquet_table(f)
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
    return True


def _generic_sharegpt(dataset_id, db_name, cache_subdir, prefix,
                      conv_col="conversations", config="default") -> bool:
    """Generic processor for ShareGPT-format conversation datasets."""
    db_path = INDEX_DIR / db_name
    cache_dir = DOWNLOAD_DIR / cache_subdir

    if db_path.exists() and db_path.stat().st_size > 100_000:
        print(f"  [OK] {db_name} exists ({db_path.stat().st_size/1e6:.0f} MB)")
        return True

    print(f"  Downloading {dataset_id}...")
    try:
        urls = _get_parquet_urls(dataset_id, config=config)
    except Exception as exc:
        print(f"  [FAIL] Could not get Parquet URLs: {exc}")
        return False

    if not urls:
        print(f"  [WARN] No Parquet files found")
        return False
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
        return False

    print(f"  Building FTS5 index...")
    builder = FTS5Builder(db_path)

    for f in local_files:
        table = _read_parquet_table(f)
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
    return True


# -------------------------------------------------------------------
# Dataset processors
# -------------------------------------------------------------------

def process_code_290k() -> bool:
    """ajibawa-2023/Code-290k-ShareGPT -- 290K code conversations."""
    return _generic_sharegpt(
        dataset_id="ajibawa-2023/Code-290k-ShareGPT",
        db_name="code_290k_sharegpt.fts5.db",
        cache_subdir="code_290k_sharegpt",
        prefix="c290k",
    )


def process_python_23k() -> bool:
    """ajibawa-2023/Python-Code-23k-ShareGPT -- Python-only conversations."""
    return _generic_sharegpt(
        dataset_id="ajibawa-2023/Python-Code-23k-ShareGPT",
        db_name="python_23k_sharegpt.fts5.db",
        cache_subdir="python_23k_sharegpt",
        prefix="py23k",
    )


def process_code_74k() -> bool:
    """ajibawa-2023/Code-74k-ShareGPT -- 74K code conversations."""
    return _generic_sharegpt(
        dataset_id="ajibawa-2023/Code-74k-ShareGPT",
        db_name="code_74k_sharegpt.fts5.db",
        cache_subdir="code_74k_sharegpt",
        prefix="c74k",
    )


def process_glaive_v3() -> bool:
    """glaiveai/glaive-code-assistant-v3 -- Code Q&A set."""
    return _generic_qa(
        dataset_id="glaiveai/glaive-code-assistant-v3",
        db_name="glaive_code_v3.fts5.db",
        cache_subdir="glaive_code_v3",
        prefix="glaive3",
        q_cols=["question", "instruction", "input"],
        a_cols=["answer", "output", "response"],
    )


def process_evol_codealpaca() -> bool:
    """theblackcat102/evol-codealpaca-v1 -- WizardCoder Evol-Instruct 110K."""
    return _generic_qa(
        dataset_id="theblackcat102/evol-codealpaca-v1",
        db_name="evol_codealpaca.fts5.db",
        cache_subdir="evol_codealpaca",
        prefix="evolca",
        q_cols=["instruction", "input"],
        a_cols=["output"],
    )


def process_strandset_rust() -> bool:
    """Fortytwo-Network/Strandset-Rust-v1 -- 191K Rust tasks."""
    return _generic_qa(
        dataset_id="Fortytwo-Network/Strandset-Rust-v1",
        db_name="strandset_rust.fts5.db",
        cache_subdir="strandset_rust",
        prefix="rust",
        q_cols=["instruction", "input", "prompt", "question"],
        a_cols=["output", "response", "answer", "completion"],
    )


def process_learn_rust() -> bool:
    """gaianet/learn-rust -- Rust learning dataset."""
    db_path = INDEX_DIR / "learn_rust.fts5.db"
    cache_dir = DOWNLOAD_DIR / "learn_rust"

    if db_path.exists() and db_path.stat().st_size > 100_000:
        print(f"  [OK] {db_path.name} exists ({db_path.stat().st_size/1e6:.0f} MB)")
        return True

    print("  Downloading gaianet/learn-rust raw text assets...")
    local_files: list[Path] = []
    for index, (filename, url) in enumerate(_LEARN_RUST_RAW_FILES, start=1):
        local_path = cache_dir / filename
        print(f"    [{index}/{len(_LEARN_RUST_RAW_FILES)}] {filename}: ", end="", flush=True)
        if _download_text_asset(url, local_path):
            print(f"{local_path.stat().st_size / 1e6:.1f} MB", flush=True)
            local_files.append(local_path)
        else:
            print("FAILED", flush=True)

    if not local_files:
        print("  [FAIL] No learn-rust source files downloaded")
        return False

    print("  Building FTS5 index...")
    builder = FTS5Builder(db_path)
    total_sections = 0

    for path in local_files:
        text = path.read_text(encoding="utf-8", errors="replace")
        sections = _split_text_blocks(text)
        if not sections:
            continue
        for section in sections:
            builder.add(section, f"lrust_{path.stem}_{total_sections:07d}")
            total_sections += 1
        print(f"    {path.name}: {len(sections):,} sections")

    entries, chunks, size = builder.finish()
    if entries == 0:
        print("  [FAIL] learn_rust produced no indexable text")
        return False

    print(f"  [OK] {db_path.name}: {entries:,} entries, {chunks:,} chunks ({size:.0f} MB)")
    return True


def process_ml_arxiv() -> bool:
    """CShorten/ML-ArXiv-Papers -- 100K+ ML papers from arXiv."""
    return _generic_qa(
        dataset_id="CShorten/ML-ArXiv-Papers",
        db_name="ml_arxiv_papers.fts5.db",
        cache_subdir="ml_arxiv_papers",
        prefix="arxiv",
        q_cols=["title"],
        a_cols=["abstract"],
    )


def process_python_codes_25k() -> bool:
    """flytech/python-codes-25k -- 25K Python codes."""
    return _generic_qa(
        dataset_id="flytech/python-codes-25k",
        db_name="python_codes_25k.fts5.db",
        cache_subdir="python_codes_25k",
        prefix="pyc25",
        q_cols=["instruction", "input"],
        a_cols=["output"],
    )


def process_tiny_codes() -> bool:
    """nampdn-ai/tiny-codes -- Beginner code examples (large)."""
    return _generic_qa(
        dataset_id="nampdn-ai/tiny-codes",
        db_name="tiny_codes.fts5.db",
        cache_subdir="tiny_codes",
        prefix="tiny",
        q_cols=["prompt"],
        a_cols=["response"],
    )


def process_capybara() -> bool:
    """LDJnr/Capybara -- Multi-turn reasoning conversations."""
    return _generic_sharegpt(
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download expansion datasets and build FTS5 indexes")
    parser.add_argument("--only", help="Run only the named dataset processor")
    args = parser.parse_args(argv)
    only = args.only

    print("=" * 60)
    print("JCoder Expansion Tier 1: Download + Index")
    print(f"Index dir: {INDEX_DIR}")
    print(f"Download dir: {DOWNLOAD_DIR}")
    if only:
        print(f"Running only: {only}")
    print("=" * 60)

    t0 = time.time()
    results = {}
    had_failure = False

    selected = [
        (name, processor)
        for name, processor in ALL_PROCESSORS.items()
        if not only or name == only
    ]
    if only and not selected:
        print(f"[FAIL] Unknown dataset: {only}")
        return 1

    for name, processor in selected:
        print(f"\n--- {name.upper()} ---")
        try:
            ok = processor()
            if ok:
                results[name] = "OK"
            else:
                results[name] = "FAIL"
                had_failure = True
        except Exception as exc:
            print(f"  [FAIL] {name}: {exc}")
            results[name] = f"FAIL: {exc}"
            had_failure = True

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
    return 1 if had_failure else 0


if __name__ == "__main__":
    try:
        _configure_stdout()
        raise SystemExit(main())
    finally:
        _close_downloader()
