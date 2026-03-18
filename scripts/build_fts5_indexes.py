"""
Build FTS5 keyword indexes for all available clean_source data.

No embedding server needed -- pure SQLite FTS5 full-text search.
Indexes enable keyword-based code retrieval immediately while
waiting for the BEAST hardware for vector embeddings.

Usage:
    cd D:\\JCoder
    python scripts/build_fts5_indexes.py [--source NAME] [--max-files N]
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import sqlite3
import sys
import time
from pathlib import Path

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

if sys.platform == "win32" and __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from ingestion.chunker import Chunker, LANGUAGE_MAP

DATA_ROOT = Path(os.environ.get("JCODER_DATA", r"D:\JCoder_Data"))
CLEAN_DIR = DATA_ROOT / "clean_source"
INDEX_DIR = DATA_ROOT / "indexes"

# Extensions that should use AST-aware chunking
AST_EXTENSIONS = {ext for ext, lang in LANGUAGE_MAP.items() if lang is not None}

# Map source dirs to index names and chunking strategies
# type: "qa" = Q&A markdown, "docs" = heading-split docs, "code" = AST-aware code
SOURCE_CONFIG = {
    "codesearchnet/python":     {"index": "csn_python",     "type": "qa"},
    "codesearchnet/javascript": {"index": "csn_javascript", "type": "qa"},
    "codesearchnet/java":       {"index": "csn_java",       "type": "qa"},
    "codesearchnet/go":         {"index": "csn_go",         "type": "qa"},
    "codesearchnet/php":        {"index": "csn_php",        "type": "qa"},
    "codesearchnet/ruby":       {"index": "csn_ruby",       "type": "qa"},
    "python_docs":              {"index": "python_docs",    "type": "docs"},
    "rfc":                      {"index": "rfc",            "type": "docs"},
    "repos":                    {"index": "repos",          "type": "code"},
}

# Normalize text for FTS5 (same as index_engine._normalize_for_search)
def _normalize(text: str) -> str:
    out = re.sub(r"[_\-./\\:]", " ", text)
    out = re.sub(r"([a-z])([A-Z])", r"\1 \2", out)
    out = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", out)
    return out.lower()


def _chunk_qa_file(fpath: Path, max_chars: int = 4000) -> list:
    """Parse Q&A markdown (SO, CSN) into chunks."""
    text = fpath.read_text(encoding="utf-8", errors="replace")
    chunks = []
    for block in re.split(r"^\s*---\s*$", text, flags=re.MULTILINE):
        block = block.strip()
        if not block or len(block) < 20:
            continue
        # Extract title
        title_m = re.search(r"^#\s+(.+)$", block, re.MULTILINE)
        title = title_m.group(1).strip() if title_m else ""
        # Sub-chunk if too large
        if len(block) <= max_chars:
            chunks.append({"content": block, "title": title, "source": str(fpath)})
        else:
            start = 0
            while start < len(block):
                end = min(start + max_chars, len(block))
                if end < len(block):
                    nl = block.rfind("\n", start, end)
                    if nl > start:
                        end = nl + 1
                chunk = block[start:end].strip()
                if chunk:
                    chunks.append({"content": chunk, "title": title, "source": str(fpath)})
                start = end
    return chunks


def _chunk_docs_file(fpath: Path, max_chars: int = 4000) -> list:
    """Split documentation markdown at heading boundaries."""
    text = fpath.read_text(encoding="utf-8", errors="replace")
    chunks = []
    for sec in re.split(r"(?=^##\s)", text, flags=re.MULTILINE):
        sec = sec.strip()
        if not sec or len(sec) < 20:
            continue
        if len(sec) <= max_chars:
            chunks.append({"content": sec, "title": fpath.stem, "source": str(fpath)})
        else:
            start = 0
            while start < len(sec):
                end = min(start + max_chars, len(sec))
                if end < len(sec):
                    nl = sec.rfind("\n", start, end)
                    if nl > start:
                        end = nl + 1
                chunk = sec[start:end].strip()
                if chunk:
                    chunks.append({"content": chunk, "title": fpath.stem, "source": str(fpath)})
                start = end
    return chunks


# Shared Chunker instance for AST-aware code chunking
_code_chunker = Chunker(max_chars=4000)


def _chunk_code_file(fpath: Path, max_chars: int = 4000) -> list:
    """Chunk a code file using AST-aware splitting.

    Python/JS/TS/Java/Go/Rust files get tree-sitter AST chunking at
    function/class boundaries.  Unknown extensions fall back to character
    splitting.  Syntax errors also fall back gracefully.
    """
    try:
        raw_chunks = _code_chunker.chunk_file(str(fpath))
    except Exception:
        # Any read/parse failure -> empty
        return []

    results = []
    for c in raw_chunks:
        content = c.get("content", "")
        if content and content.strip():
            results.append({
                "content": content,
                "title": fpath.stem,
                "source": str(fpath),
            })
    return results


def build_fts5_index(source_name: str, config: dict, max_files: int = 0) -> dict:
    """Build one FTS5 index for a data source. Returns stats dict."""
    source_dir = CLEAN_DIR / source_name
    if not source_dir.exists():
        print(f"[WARN] {source_name}: directory not found, skipping")
        return {"name": source_name, "files": 0, "chunks": 0, "skipped": True}

    index_name = config["index"]
    chunk_type = config["type"]
    db_path = INDEX_DIR / f"{index_name}.fts5.db"

    # Check if index exists and is non-trivial
    if db_path.exists() and db_path.stat().st_size > 100_000:
        size_mb = db_path.stat().st_size / (1024 * 1024)
        print(f"[OK] {index_name}: index exists ({size_mb:.1f} MB), skipping (delete to rebuild)")
        return {"name": source_name, "index": index_name, "files": 0, "chunks": 0,
                "skipped": True, "size_mb": size_mb}

    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    # Collect files (recursive for sources with subdirectories like python_docs)
    # Use islice when max_files is set to avoid sorting 300K+ filenames
    import itertools

    if chunk_type == "code":
        # Collect code files with AST-supported extensions + text fallback
        all_exts = set(LANGUAGE_MAP.keys()) | {".txt", ".cfg", ".ini", ".toml"}
        collected: list[Path] = []
        for ext in sorted(all_exts):
            pattern = f"*{ext}"
            if max_files and len(collected) >= max_files:
                break
            found = list(source_dir.rglob(pattern))
            collected.extend(found)
        if max_files:
            collected = collected[:max_files]
        md_files = collected
    else:
        if max_files:
            md_files = list(itertools.islice(source_dir.glob("*.md"), max_files))
            if not md_files:
                md_files = list(itertools.islice(source_dir.rglob("*.md"), max_files))
        else:
            md_files = sorted(source_dir.glob("*.md"))
            if not md_files:
                md_files = sorted(source_dir.rglob("*.md"))

    if not md_files:
        print(f"[WARN] {source_name}: no files found")
        return {"name": source_name, "files": 0, "chunks": 0, "skipped": True}

    print(f"[OK] {source_name}: indexing {len(md_files)} files -> {index_name}")

    if chunk_type == "code":
        chunk_fn = _chunk_code_file
    elif chunk_type == "qa":
        chunk_fn = _chunk_qa_file
    else:
        chunk_fn = _chunk_docs_file

    t0 = time.monotonic()
    stats = {"name": source_name, "index": index_name, "files": 0,
             "chunks": 0, "errors": 0, "skipped": False}
    error_lines: list[str] = []

    # Build in batches to SQLite
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("DROP TABLE IF EXISTS chunks")
        conn.execute(
            "CREATE VIRTUAL TABLE chunks "
            "USING fts5(search_content, source_path, chunk_id, title)"
        )

        batch = []
        chunk_id = 0

        for i, fpath in enumerate(md_files):
            try:
                chunks = chunk_fn(fpath)
                for c in chunks:
                    chunk_id += 1
                    batch.append((
                        _normalize(c["content"]),
                        c["source"],
                        f"{index_name}_{chunk_id:08d}",
                        c.get("title", ""),
                    ))
                    stats["chunks"] += 1

                    if len(batch) >= 5000:
                        conn.executemany(
                            "INSERT INTO chunks(search_content, source_path, chunk_id, title) "
                            "VALUES (?, ?, ?, ?)", batch
                        )
                        conn.commit()
                        batch.clear()

                stats["files"] += 1
            except Exception as exc:
                stats["errors"] += 1
                error_lines.append(f"{fpath.name}: {type(exc).__name__}: {exc}")
                if stats["errors"] <= 5:
                    print(f"[WARN] {fpath.name}: {exc}")

            if (i + 1) % 5000 == 0:
                elapsed = time.monotonic() - t0
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                print(f"     {source_name}: {i+1}/{len(md_files)} files, "
                      f"{stats['chunks']:,} chunks ({rate:.0f} files/s)")

        # Flush remaining
        if batch:
            conn.executemany(
                "INSERT INTO chunks(search_content, source_path, chunk_id, title) "
                "VALUES (?, ?, ?, ?)", batch
            )
            conn.commit()
    finally:
        conn.close()

    # Write error log if any errors occurred
    if error_lines:
        error_log_path = INDEX_DIR / f"{index_name}.errors.log"
        error_log_path.write_text("\n".join(error_lines) + "\n", encoding="utf-8")

    elapsed = time.monotonic() - t0
    size_mb = db_path.stat().st_size / (1024 * 1024)
    stats["size_mb"] = size_mb
    stats["elapsed_s"] = elapsed
    print(f"[OK] {index_name}: {stats['chunks']:,} chunks, {size_mb:.1f} MB, "
          f"{elapsed:.1f}s ({stats['errors']} errors)")
    return stats


def main():
    parser = argparse.ArgumentParser(description="Build FTS5 indexes for JCoder")
    parser.add_argument("--source", default="", help="Process only this source")
    parser.add_argument("--max-files", type=int, default=0, help="Limit files per source")
    args = parser.parse_args()

    print("\nJCoder FTS5 Index Builder")
    print("=" * 50)

    all_stats = []
    t_total = time.monotonic()

    for source_name, config in SOURCE_CONFIG.items():
        if args.source and args.source not in source_name:
            continue
        stats = build_fts5_index(source_name, config, max_files=args.max_files)
        all_stats.append(stats)

    elapsed = time.monotonic() - t_total
    total_chunks = sum(s["chunks"] for s in all_stats)
    total_size = sum(s.get("size_mb", 0) for s in all_stats)

    print("=" * 50)
    print(f"[OK] Total: {total_chunks:,} chunks, {total_size:.1f} MB, {elapsed:.1f}s")
    for s in all_stats:
        status = "SKIP" if s.get("skipped") else "OK"
        print(f"  [{status}] {s['name']}: {s['chunks']:,} chunks"
              + (f", {s.get('size_mb', 0):.1f} MB" if s.get("size_mb") else ""))


if __name__ == "__main__":
    main()
