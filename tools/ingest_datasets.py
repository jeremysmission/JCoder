"""
Dataset Ingestion Tool
----------------------
Converts downloaded JSONL datasets (DS-1000, CommitPackFT, HumanEvalPack)
into FTS5 indexes compatible with JCoder's federated search.

Usage:
    python -m tools.ingest_datasets [--dataset ds1000|commitpackft|humanevalpack|all]
                                    [--index-dir data/indexes]
                                    [--max-per-lang N]
"""

import hashlib
import json
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional


DOWNLOADS_DIR = os.environ.get(
    "JCODER_DOWNLOADS", r"D:\JCoder_Data\downloads"
)
DEFAULT_INDEX_DIR = os.path.join("data", "indexes")
MAX_CHUNK_CHARS = 4000
CHUNKER_VERSION = "1.0-dataset"


def _hash(text: str) -> str:
    return hashlib.sha256(text.replace("\r\n", "\n").encode("utf-8")).hexdigest()


def _make_chunk(content: str, source_path: str, **extra) -> Dict:
    return {
        "id": _hash(content),
        "content": content,
        "source_path": source_path,
        "source_type": os.path.splitext(source_path)[1],
        "ingestion_date": datetime.now(timezone.utc).isoformat(),
        "content_hash": _hash(content),
        "chunker_version": CHUNKER_VERSION,
        **extra,
    }


def _sub_chunk(text: str, max_chars: int = MAX_CHUNK_CHARS) -> List[str]:
    """Split oversized text at line boundaries."""
    if len(text) <= max_chars:
        return [text]
    parts = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            nl = text.rfind("\n", start, end)
            if nl > start:
                end = nl + 1
        chunk = text[start:end].strip()
        if chunk:
            parts.append(chunk)
        start = end
    return parts


def _build_fts5(db_path: str, chunks: List[Dict]) -> int:
    """Build an FTS5 database from chunk dicts. Returns row count."""
    # Import normalize function from index_engine
    try:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from core.index_engine import _normalize_for_search
    except ImportError:
        def _normalize_for_search(t):
            return t.lower()

    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("DROP TABLE IF EXISTS chunks")
    conn.execute(
        "CREATE VIRTUAL TABLE chunks "
        "USING fts5(search_content, source_path, chunk_id)"
    )
    rows = [
        (
            _normalize_for_search(m.get("content", "")),
            m.get("source_path", ""),
            m.get("id", ""),
        )
        for m in chunks
    ]
    conn.executemany(
        "INSERT INTO chunks(search_content, source_path, chunk_id) VALUES (?, ?, ?)",
        rows,
    )
    conn.commit()
    count = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
    conn.close()
    return count


def _save_meta(meta_path: str, chunks: List[Dict]):
    """Save metadata JSON alongside FTS5 database."""
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f)


# ---------------------------------------------------------------------------
# DS-1000: Python data science problems
# ---------------------------------------------------------------------------

def ingest_ds1000(index_dir: str, max_records: int = 0) -> Dict:
    """Ingest DS-1000 JSONL into FTS5 index."""
    jsonl_path = os.path.join(DOWNLOADS_DIR, "ds1000", "ds1000.jsonl")
    if not os.path.exists(jsonl_path):
        return {"status": "SKIP", "reason": f"File not found: {jsonl_path}"}

    t0 = time.monotonic()
    chunks = []
    lib_counts = {}

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_records and i >= max_records:
                break
            rec = json.loads(line)
            meta = rec.get("metadata", {})
            lib = meta.get("library", "unknown")
            pid = meta.get("problem_id", i)

            # Build rich content: prompt + reference code + context
            parts = []
            parts.append(f"# DS-1000 Problem {pid} ({lib})")
            if rec.get("prompt"):
                parts.append(f"## Problem\n{rec['prompt']}")
            if rec.get("reference_code"):
                parts.append(f"## Reference Solution\n```python\n{rec['reference_code']}\n```")
            if rec.get("code_context"):
                parts.append(f"## Code Context\n```python\n{rec['code_context']}\n```")

            content = "\n\n".join(parts)
            vpath = f"ds1000/{lib.lower()}/problem_{pid}.py"

            for text in _sub_chunk(content):
                chunks.append(_make_chunk(
                    text, vpath,
                    language="python", library=lib,
                    source_kind="ds1000", problem_id=str(pid),
                ))

            lib_counts[lib] = lib_counts.get(lib, 0) + 1

    # Build FTS5 index
    db_path = os.path.join(index_dir, "ds1000.fts5.db")
    meta_path = os.path.join(index_dir, "ds1000.meta.json")
    row_count = _build_fts5(db_path, chunks)
    _save_meta(meta_path, chunks)

    elapsed = time.monotonic() - t0
    return {
        "status": "OK",
        "dataset": "ds1000",
        "records": sum(lib_counts.values()),
        "chunks": len(chunks),
        "fts5_rows": row_count,
        "db_path": db_path,
        "db_size_mb": round(os.path.getsize(db_path) / (1024 * 1024), 2),
        "elapsed_s": round(elapsed, 1),
        "by_library": lib_counts,
    }


# ---------------------------------------------------------------------------
# CommitPackFT: git commit diffs (instruction-like)
# ---------------------------------------------------------------------------

def ingest_commitpackft(
    index_dir: str, languages: Optional[List[str]] = None,
    max_per_lang: int = 0,
) -> Dict:
    """Ingest CommitPackFT JSONL files into per-language FTS5 indexes."""
    base = os.path.join(DOWNLOADS_DIR, "commitpackft", "data")
    if not os.path.exists(base):
        return {"status": "SKIP", "reason": f"Directory not found: {base}"}

    if languages is None:
        languages = ["python", "shell", "rust", "typescript", "javascript"]

    t0 = time.monotonic()
    results = {}

    for lang in languages:
        jsonl_path = os.path.join(base, lang, "data.jsonl")
        if not os.path.exists(jsonl_path):
            results[lang] = {"status": "SKIP", "reason": "file not found"}
            continue

        chunks = []
        skipped = 0
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if max_per_lang and i >= max_per_lang:
                    break
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    skipped += 1
                    continue

                # Build content from commit data
                subject = rec.get("subject", "")
                message = rec.get("message", "")
                new_file = rec.get("new_file", "")
                new_contents = rec.get("new_contents", "")

                # Skip empty or tiny commits
                if len(new_contents) < 20:
                    skipped += 1
                    continue

                parts = []
                if subject:
                    parts.append(f"# {subject}")
                if message and message != subject:
                    parts.append(f"## Commit Message\n{message}")
                parts.append(f"## File: {new_file}")

                # Determine extension for code fence
                ext = os.path.splitext(new_file)[1].lstrip(".")
                fence_lang = ext or lang
                parts.append(f"```{fence_lang}\n{new_contents}\n```")

                content = "\n\n".join(parts)
                vpath = f"commitpackft/{lang}/{new_file}"

                for text in _sub_chunk(content):
                    chunks.append(_make_chunk(
                        text, vpath,
                        language=lang, source_kind="commitpackft",
                        license=rec.get("license", ""),
                    ))

        # Build FTS5 index
        idx_name = f"commitpackft_{lang}"
        db_path = os.path.join(index_dir, f"{idx_name}.fts5.db")
        meta_path = os.path.join(index_dir, f"{idx_name}.meta.json")
        row_count = _build_fts5(db_path, chunks)
        _save_meta(meta_path, chunks)

        results[lang] = {
            "status": "OK",
            "chunks": len(chunks),
            "fts5_rows": row_count,
            "skipped": skipped,
            "db_size_mb": round(os.path.getsize(db_path) / (1024 * 1024), 2),
        }

    elapsed = time.monotonic() - t0
    return {
        "status": "OK",
        "dataset": "commitpackft",
        "elapsed_s": round(elapsed, 1),
        "languages": results,
    }


# ---------------------------------------------------------------------------
# HumanEvalPack: coding problems across 5 languages
# ---------------------------------------------------------------------------

def ingest_humanevalpack(index_dir: str) -> Dict:
    """Ingest HumanEvalPack JSONL files into a single FTS5 index."""
    base = os.path.join(DOWNLOADS_DIR, "humanevalpack")
    if not os.path.exists(base):
        return {"status": "SKIP", "reason": f"Directory not found: {base}"}

    t0 = time.monotonic()
    chunks = []
    lang_counts = {}

    for fname in sorted(os.listdir(base)):
        if not fname.endswith(".jsonl"):
            continue
        lang = fname.replace("humanevalpack_", "").replace(".jsonl", "")
        count = 0

        with open(os.path.join(base, fname), "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                task_id = rec.get("task_id", "")

                parts = []
                parts.append(f"# {task_id}")
                if rec.get("docstring"):
                    parts.append(f"## Description\n{rec['docstring']}")
                if rec.get("declaration"):
                    parts.append(f"## Declaration\n```{lang}\n{rec['declaration']}\n```")
                if rec.get("canonical_solution"):
                    parts.append(f"## Solution\n```{lang}\n{rec['canonical_solution']}\n```")
                if rec.get("buggy_solution"):
                    parts.append(f"## Buggy Version ({rec.get('bug_type', '')})\n"
                                 f"```{lang}\n{rec['buggy_solution']}\n```")
                if rec.get("test"):
                    parts.append(f"## Tests\n```{lang}\n{rec['test']}\n```")

                content = "\n\n".join(parts)
                vpath = f"humanevalpack/{lang}/{task_id}.{lang}"

                for text in _sub_chunk(content):
                    chunks.append(_make_chunk(
                        text, vpath,
                        language=lang, source_kind="humanevalpack",
                        task_id=task_id,
                    ))
                count += 1

        lang_counts[lang] = count

    # Build single FTS5 index
    db_path = os.path.join(index_dir, "humanevalpack.fts5.db")
    meta_path = os.path.join(index_dir, "humanevalpack.meta.json")
    row_count = _build_fts5(db_path, chunks)
    _save_meta(meta_path, chunks)

    elapsed = time.monotonic() - t0
    return {
        "status": "OK",
        "dataset": "humanevalpack",
        "records": sum(lang_counts.values()),
        "chunks": len(chunks),
        "fts5_rows": row_count,
        "db_path": db_path,
        "db_size_mb": round(os.path.getsize(db_path) / (1024 * 1024), 2),
        "elapsed_s": round(elapsed, 1),
        "by_language": lang_counts,
    }


# ---------------------------------------------------------------------------
# Research papers, curated lists, and JCoder meta-docs
# ---------------------------------------------------------------------------

RESEARCH_DIRS = [
    os.path.join(DOWNLOADS_DIR, "research_papers", "markdown"),
    os.path.join(DOWNLOADS_DIR, "self_learning_docs"),
]
JCODER_DOC_PATTERNS = [
    os.path.join("docs", "11_RESEARCH_AGENTIC_SELF_LEARNING.md"),
    os.path.join("docs", "11a_CITATIONS_RANKED.md"),
    os.path.join("docs", "11b_DEEP_DIVE_SUBAGENT_FINDINGS.md"),
]


def _collect_md_files(dirs: List[str], extra_files: List[str]) -> List[str]:
    """Collect all .md files from directories plus explicit file paths."""
    paths = []
    for d in dirs:
        if not os.path.isdir(d):
            continue
        for fname in sorted(os.listdir(d)):
            if fname.endswith(".md"):
                paths.append(os.path.join(d, fname))
    for f in extra_files:
        if os.path.isfile(f):
            paths.append(f)
    return paths


def ingest_research(index_dir: str, jcoder_root: str = "") -> Dict:
    """Ingest self-learning research papers, curated lists, and JCoder docs."""
    if not jcoder_root:
        jcoder_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    extra = [os.path.join(jcoder_root, p) for p in JCODER_DOC_PATTERNS]
    md_files = _collect_md_files(RESEARCH_DIRS, extra)

    if not md_files:
        return {"status": "SKIP", "reason": "No markdown files found"}

    t0 = time.monotonic()
    chunks = []
    file_stats = {}

    for fpath in md_files:
        fname = os.path.basename(fpath)
        with open(fpath, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()

        if not text.strip():
            file_stats[fname] = {"chunks": 0, "chars": 0}
            continue

        # Determine source kind from path
        if "research_papers" in fpath:
            source_kind = "research_paper"
        elif "self_learning_docs" in fpath:
            source_kind = "curated_list"
        else:
            source_kind = "jcoder_meta"

        vpath = f"self_learning/{source_kind}/{fname}"

        for text_chunk in _sub_chunk(text):
            chunks.append(_make_chunk(
                text_chunk, vpath,
                language="markdown", source_kind=source_kind,
                original_file=fname,
            ))

        file_stats[fname] = {
            "chunks": sum(1 for c in chunks if c.get("original_file") == fname),
            "chars": len(text),
        }

    db_path = os.path.join(index_dir, "self_learning.fts5.db")
    meta_path = os.path.join(index_dir, "self_learning.meta.json")
    row_count = _build_fts5(db_path, chunks)
    _save_meta(meta_path, chunks)

    elapsed = time.monotonic() - t0
    return {
        "status": "OK",
        "dataset": "self_learning_research",
        "files": len(md_files),
        "chunks": len(chunks),
        "fts5_rows": row_count,
        "db_path": db_path,
        "db_size_mb": round(os.path.getsize(db_path) / (1024 * 1024), 2),
        "elapsed_s": round(elapsed, 1),
        "file_details": file_stats,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Ingest JSONL datasets into FTS5")
    parser.add_argument("--dataset", default="all",
                        choices=["ds1000", "commitpackft", "humanevalpack", "research", "all"])
    parser.add_argument("--index-dir", default=DEFAULT_INDEX_DIR)
    parser.add_argument("--max-per-lang", type=int, default=0,
                        help="Limit records per language (0 = unlimited)")
    args = parser.parse_args()

    os.makedirs(args.index_dir, exist_ok=True)
    results = {}

    if args.dataset in ("ds1000", "all"):
        print("[INFO] Ingesting DS-1000...")
        results["ds1000"] = ingest_ds1000(args.index_dir)
        print(f"  -> {json.dumps(results['ds1000'], indent=2)}")

    if args.dataset in ("commitpackft", "all"):
        print("[INFO] Ingesting CommitPackFT...")
        results["commitpackft"] = ingest_commitpackft(
            args.index_dir, max_per_lang=args.max_per_lang)
        print(f"  -> {json.dumps(results['commitpackft'], indent=2)}")

    if args.dataset in ("humanevalpack", "all"):
        print("[INFO] Ingesting HumanEvalPack...")
        results["humanevalpack"] = ingest_humanevalpack(args.index_dir)
        print(f"  -> {json.dumps(results['humanevalpack'], indent=2)}")

    if args.dataset in ("research", "all"):
        print("[INFO] Ingesting self-learning research...")
        results["research"] = ingest_research(args.index_dir)
        print(f"  -> {json.dumps(results['research'], indent=2)}")

    # Summary
    print("\n" + "=" * 60)
    print("INGESTION SUMMARY")
    print("=" * 60)
    total_chunks = 0
    for name, res in results.items():
        chunks = res.get("chunks", 0)
        if not chunks:
            langs = res.get("languages", {})
            chunks = sum(v.get("chunks", 0) for v in langs.values() if isinstance(v, dict))
        total_chunks += chunks
        status = res.get("status", "?")
        elapsed = res.get("elapsed_s", 0)
        print(f"  {name}: {status} | {chunks} chunks | {elapsed}s")
    print(f"  TOTAL: {total_chunks} chunks")
    print("=" * 60)

    return results


if __name__ == "__main__":
    main()
