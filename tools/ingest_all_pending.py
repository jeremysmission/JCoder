"""
Batch Ingest All Pending Downloads into FTS5 Indexes
-----------------------------------------------------
Scans D:\JCoder_Data\downloads for datasets not yet indexed
and builds FTS5 indexes for each. Handles Parquet, JSONL, and
plain text/markdown formats.

Usage:
    cd D:\JCoder
    python tools/ingest_all_pending.py [--only DATASET] [--max-per-dataset N]
"""

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

DOWNLOADS_DIR = Path(os.environ.get("JCODER_DOWNLOADS", r"D:\JCoder_Data\downloads"))
INDEX_DIR = Path(os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "indexes",
))
INDEX_DIR.mkdir(parents=True, exist_ok=True)

MAX_CHUNK_CHARS = 4000
BATCH_SIZE = 5000

_NORMALIZE_RE = re.compile(r"[_\-./\\:]")
_CAMEL_RE1 = re.compile(r"([a-z])([A-Z])")
_CAMEL_RE2 = re.compile(r"([A-Z]+)([A-Z][a-z])")


def _normalize(text: str) -> str:
    out = _NORMALIZE_RE.sub(" ", text)
    out = _CAMEL_RE1.sub(r"\1 \2", out)
    out = _CAMEL_RE2.sub(r"\1 \2", out)
    return out.lower()


class FTS5Builder:
    """Incrementally builds an FTS5 index."""

    def __init__(self, name: str):
        self.name = name
        self.db_path = INDEX_DIR / f"{name}.fts5.db"
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("DROP TABLE IF EXISTS chunks")
        self.conn.execute(
            "CREATE VIRTUAL TABLE chunks "
            "USING fts5(search_content, source_path, chunk_id)"
        )
        self._batch = []
        self.total_chunks = 0

    def add(self, text: str, source_id: str):
        if not text or len(text.strip()) < 20:
            return
        pos = 0
        cidx = 0
        while pos < len(text):
            end = min(pos + MAX_CHUNK_CHARS, len(text))
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
        count = self.conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
        self.conn.close()
        size_mb = self.db_path.stat().st_size / (1024 * 1024)
        return count, size_mb


def _read_parquet_texts(parquet_dir: Path, text_fields, max_records=0):
    """Yield (source_id, text) from Parquet files."""
    try:
        import pyarrow.parquet as pq
    except ImportError:
        print("    [WARN] pyarrow not installed, skipping Parquet")
        return
    count = 0
    for pf in sorted(parquet_dir.glob("**/*.parquet")):
        try:
            table = pq.read_table(str(pf))
            for row in range(table.num_rows):
                if max_records and count >= max_records:
                    return
                parts = []
                for field in text_fields:
                    if field in table.column_names:
                        val = table.column(field)[row].as_py()
                        if val:
                            parts.append(str(val))
                if parts:
                    text = "\n\n".join(parts)
                    yield f"{parquet_dir.name}/{pf.stem}:{row}", text
                    count += 1
        except Exception as exc:
            print(f"    [WARN] Error reading {pf.name}: {exc}")


def _read_jsonl_texts(jsonl_path: Path, text_fields, max_records=0):
    """Yield (source_id, text) from a JSONL file."""
    count = 0
    with open(jsonl_path, "r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if max_records and count >= max_records:
                return
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            parts = []
            for field in text_fields:
                val = rec.get(field)
                if val:
                    parts.append(str(val))
            if parts:
                yield f"{jsonl_path.stem}:{i}", "\n\n".join(parts)
                count += 1


def _read_markdown_dir(md_dir: Path, max_records=0):
    """Yield (source_id, text) from markdown/text files."""
    count = 0
    for ext in ("*.md", "*.txt", "*.rst"):
        for fp in sorted(md_dir.glob(f"**/{ext}")):
            if max_records and count >= max_records:
                return
            try:
                text = fp.read_text(encoding="utf-8", errors="replace")
                if text.strip():
                    yield f"{md_dir.name}/{fp.name}", text
                    count += 1
            except Exception:
                pass


# Dataset-specific configurations: (index_name, text_fields, format)
DATASET_CONFIG = {
    "code_alpaca": {
        "index": "code_alpaca",
        "format": "jsonl",
        "fields": ["instruction", "input", "output"],
        "file_pattern": "*.json",
    },
    "code_contests": {
        "index": "code_contests",
        "format": "parquet",
        "fields": ["description", "solutions"],
    },
    "code_exercises": {
        "index": "code_exercises",
        "format": "parquet",
        "fields": ["problem", "solution", "explanation"],
    },
    "code_feedback": {
        "index": "code_feedback",
        "format": "parquet",
        "fields": ["query", "answer"],
    },
    "code_instructions_122k": {
        "index": "code_instructions_122k",
        "format": "parquet",
        "fields": ["instruction", "output"],
    },
    "codeparrot_clean": {
        "index": "codeparrot_clean",
        "format": "parquet",
        "fields": ["content"],
    },
    "evol_instruct": {
        "index": "evol_instruct",
        "format": "parquet",
        "fields": ["instruction", "output"],
    },
    "math_instruct": {
        "index": "math_instruct",
        "format": "parquet",
        "fields": ["instruction", "output"],
    },
    "self_oss_instruct": {
        "index": "self_oss_instruct",
        "format": "parquet",
        "fields": ["instruction", "response"],
    },
    "python_instructions": {
        "index": "python_instructions",
        "format": "parquet",
        "fields": ["instruction", "output"],
    },
    "python_docs": {
        "index": "python_docs",
        "format": "markdown",
        "fields": [],
    },
    "rfc": {
        "index": "rfc_docs",
        "format": "markdown",
        "fields": [],
    },
    "stack_smol_xl": {
        "index": "stack_smol_xl",
        "format": "parquet",
        "fields": ["content"],
    },
    "codesearchnet_python_instruct": {
        "index": "codesearchnet_python_instruct",
        "format": "parquet",
        "fields": ["INSTRUCTION", "RESPONSE"],
    },
    "code_instructions_120k": {
        "index": "code_instructions_120k",
        "format": "parquet",
        "fields": ["instruction", "input", "output"],
    },
    "python_code_18k": {
        "index": "python_code_18k",
        "format": "parquet",
        "fields": ["instruction", "input", "output"],
    },
    "codesearchnet_java_instruct": {
        "index": "codesearchnet_java_instruct",
        "format": "parquet",
        "fields": ["INSTRUCTION", "RESPONSE"],
    },
    "codesearchnet_ruby_instruct": {
        "index": "codesearchnet_ruby_instruct",
        "format": "parquet",
        "fields": ["INSTRUCTION", "RESPONSE"],
    },
    "codesearchnet_php_instruct": {
        "index": "codesearchnet_php_instruct",
        "format": "parquet",
        "fields": ["INSTRUCTION", "RESPONSE"],
    },
    "cot_code_instruct": {
        "index": "cot_code_instruct",
        "format": "parquet",
        "fields": ["instruction", "output"],
    },
    "openhermes_2_5": {
        "index": "openhermes_2_5",
        "format": "parquet",
        "fields": ["conversations"],
    },
    "magicoder_oss_instruct": {
        "index": "magicoder_oss_instruct",
        "format": "parquet",
        "fields": ["problem", "solution"],
    },
    "magicoder_evol_instruct": {
        "index": "magicoder_evol_instruct",
        "format": "parquet",
        "fields": ["instruction", "response"],
    },
    "glaive_code_assistant": {
        "index": "glaive_code_assistant",
        "format": "parquet",
        "fields": ["question", "answer"],
    },
}


def ingest_dataset(name: str, max_records: int = 0) -> dict:
    """Ingest a single dataset into FTS5. Returns stats dict."""
    if name not in DATASET_CONFIG:
        return {"status": "SKIP", "reason": f"No config for {name}"}

    cfg = DATASET_CONFIG[name]
    dl_dir = DOWNLOADS_DIR / name
    if not dl_dir.exists():
        return {"status": "SKIP", "reason": f"Dir not found: {dl_dir}"}

    t0 = time.monotonic()
    builder = FTS5Builder(cfg["index"])

    if cfg["format"] == "parquet":
        for src_id, text in _read_parquet_texts(dl_dir, cfg["fields"], max_records):
            builder.add(text, src_id)
    elif cfg["format"] == "jsonl":
        # Find JSONL/JSON files
        for pattern in ("*.jsonl", "*.json"):
            for fp in sorted(dl_dir.glob(f"**/{pattern}")):
                for src_id, text in _read_jsonl_texts(fp, cfg["fields"], max_records):
                    builder.add(text, src_id)
    elif cfg["format"] == "markdown":
        for src_id, text in _read_markdown_dir(dl_dir, max_records):
            builder.add(text, src_id)

    count, size_mb = builder.finish()
    elapsed = time.monotonic() - t0

    return {
        "status": "OK",
        "dataset": name,
        "index": cfg["index"],
        "chunks": count,
        "size_mb": round(size_mb, 2),
        "elapsed_s": round(elapsed, 1),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Ingest all pending downloads")
    parser.add_argument("--only", default="", help="Process only this dataset")
    parser.add_argument("--max-per-dataset", type=int, default=0,
                        help="Max records per dataset (0=unlimited)")
    args = parser.parse_args()

    # Find what's already indexed
    existing = set()
    for f in INDEX_DIR.iterdir():
        if f.name.endswith(".fts5.db"):
            existing.add(f.name.replace(".fts5.db", ""))

    results = {}
    targets = [args.only] if args.only else sorted(DATASET_CONFIG.keys())

    for name in targets:
        cfg = DATASET_CONFIG.get(name, {})
        idx_name = cfg.get("index", name)

        if idx_name in existing and not args.only:
            print(f"[SKIP] {name} -- already indexed as {idx_name}")
            continue

        print(f"[INFO] Ingesting {name}...")
        result = ingest_dataset(name, max_records=args.max_per_dataset)
        results[name] = result
        status = result.get("status", "?")
        chunks = result.get("chunks", 0)
        size = result.get("size_mb", 0)
        elapsed = result.get("elapsed_s", 0)
        print(f"  [{status}] {chunks:,} chunks, {size:.1f} MB, {elapsed:.1f}s")

    # Summary
    print("\n" + "=" * 60)
    print("BATCH INGESTION SUMMARY")
    print("=" * 60)
    total_chunks = 0
    total_mb = 0
    for name, res in results.items():
        c = res.get("chunks", 0)
        m = res.get("size_mb", 0)
        total_chunks += c
        total_mb += m
        print(f"  {name:30s} {c:>8,} chunks  {m:>7.1f} MB  [{res.get('status')}]")
    print(f"  {'TOTAL':30s} {total_chunks:>8,} chunks  {total_mb:>7.1f} MB")
    print("=" * 60)


if __name__ == "__main__":
    main()
