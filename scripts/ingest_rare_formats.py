"""
Ingest rare format files into HybridRAG-compatible FTS5 + FAISS indexes.

Processes all downloaded rare format files (.drawio, .rst, .svg, .dia, .epub,
.xlsx) using the parser registry and DocumentChunker (1200/200 overlap).

Usage:
    python scripts/ingest_rare_formats.py [--source data/raw_downloads/rare_formats]
"""

import os
import sys
import time
from pathlib import Path

# Fix encoding for Windows
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingestion.parser_registry import DOCUMENT_EXTENSIONS, parse_file
from ingestion.chunker import DocumentChunker, Chunker


def collect_files(source_dir: Path) -> list:
    """Walk source directory and collect all parseable document files."""
    files = []
    code_exts = {".py", ".js", ".ts", ".java", ".go", ".rs", ".c", ".cpp",
                 ".rb", ".php", ".kt", ".cs"}
    all_exts = set(DOCUMENT_EXTENSIONS) | code_exts | {".md", ".txt", ".json",
                                                         ".yaml", ".yml"}
    for dirpath, _dirs, filenames in os.walk(source_dir):
        # Skip .git dirs
        if ".git" in dirpath.split(os.sep):
            continue
        for fn in filenames:
            ext = os.path.splitext(fn)[1].lower()
            if ext in all_exts:
                files.append(os.path.join(dirpath, fn))
    files.sort()
    return files


def main():
    source = Path("data/raw_downloads/rare_formats")
    for i, arg in enumerate(sys.argv[1:]):
        if arg == "--source" and i + 1 < len(sys.argv) - 1:
            source = Path(sys.argv[i + 2])

    if not source.exists():
        print(f"Source not found: {source}")
        return

    print(f"=== Ingesting rare format files from {source} ===\n")

    files = collect_files(source)
    print(f"Found {len(files)} files to process")

    doc_chunker = DocumentChunker(chunk_size=1200, overlap=200)
    code_chunker = Chunker(max_chars=1200)  # Also use 1200 for HybridRAG compat

    all_chunks = []
    stats = {"processed": 0, "skipped": 0, "errors": 0, "by_ext": {}}
    t0 = time.monotonic()

    for fpath in files:
        ext = os.path.splitext(fpath)[1].lower()
        try:
            if ext in DOCUMENT_EXTENSIONS:
                text, details = parse_file(fpath)
                if text.strip():
                    chunks = doc_chunker.chunk_text(text, fpath)
                    all_chunks.extend(chunks)
                    stats["processed"] += 1
                    stats["by_ext"][ext] = stats["by_ext"].get(ext, 0) + 1
                else:
                    stats["skipped"] += 1
            else:
                # Code/text files — use code chunker at 1200 chars
                chunks = code_chunker.chunk_file(fpath)
                all_chunks.extend(chunks)
                stats["processed"] += 1
                stats["by_ext"][ext] = stats["by_ext"].get(ext, 0) + 1
        except Exception as e:
            stats["errors"] += 1
            if stats["errors"] <= 10:
                print(f"  [ERROR] {fpath}: {e}")

        if stats["processed"] % 100 == 0 and stats["processed"] > 0:
            elapsed = time.monotonic() - t0
            rate = stats["processed"] / elapsed
            print(f"  [{stats['processed']}/{len(files)}] "
                  f"{len(all_chunks)} chunks, {rate:.0f} files/s")

    elapsed = time.monotonic() - t0
    print(f"\n=== Processing Complete ===")
    print(f"  Files processed: {stats['processed']}")
    print(f"  Files skipped: {stats['skipped']}")
    print(f"  Errors: {stats['errors']}")
    print(f"  Total chunks: {len(all_chunks)}")
    print(f"  Elapsed: {elapsed:.1f}s")
    print(f"\n  By extension:")
    for ext, count in sorted(stats["by_ext"].items()):
        print(f"    {ext}: {count} files")

    # Save chunks to JSON for indexing
    import json
    out_path = Path("data") / "rare_format_chunks.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Save as JSONL for streaming
    with open(out_path, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    print(f"\n  Chunks saved to: {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")

    # Build FTS5 index
    import sqlite3
    db_path = Path("data/indexes/rare_formats.fts5.db")
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("BEGIN")
        conn.execute("DROP TABLE IF EXISTS chunks")
        conn.execute(
            "CREATE VIRTUAL TABLE chunks "
            "USING fts5(search_content, source_path, chunk_id)"
        )
        rows = [
            (chunk.get("content", ""), chunk.get("source_path", ""), chunk.get("id", ""))
            for chunk in all_chunks
        ]
        conn.executemany(
            "INSERT INTO chunks(search_content, source_path, chunk_id) VALUES (?, ?, ?)",
            rows,
        )
        conn.commit()
        print(f"  FTS5 index: {db_path} ({len(rows)} rows)")
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    print(f"\n=== DONE ===")


if __name__ == "__main__":
    os.chdir(os.environ.get("JCODER_ROOT", str(Path(__file__).resolve().parent.parent)))
    main()
