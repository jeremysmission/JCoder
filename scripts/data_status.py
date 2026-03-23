"""
Show JCoder data download and indexing status at a glance.

Usage:
    cd D:\\JCoder
    .venv\\Scripts\\python scripts\\data_status.py
"""
from __future__ import annotations

import io
import os
import sqlite3
import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace"
    )

DATA_ROOT = Path(os.environ.get("JCODER_DATA", "data"))
INDEX_DIR_DATA = DATA_ROOT / "indexes"
INDEX_DIR_PROJ = Path("data") / "indexes"
RAW_DIR = DATA_ROOT / "raw_downloads"
CLEAN_DIR = DATA_ROOT / "clean_source"
PROGRESS_DIR = DATA_ROOT / "prep_stage" / "github_code"


def _size_str(path: Path) -> str:
    if not path.exists():
        return "MISSING"
    size = path.stat().st_size
    if size > 1e9:
        return f"{size / 1e9:.1f} GB"
    if size > 1e6:
        return f"{size / 1e6:.0f} MB"
    return f"{size / 1e3:.0f} KB"


def _fts5_info(db_path: Path) -> str:
    """Get quick info about an FTS5 database."""
    if not db_path.exists():
        return "MISSING"
    size = _size_str(db_path)
    try:
        conn = sqlite3.connect(str(db_path))
        row = conn.execute(
            "SELECT 1 FROM chunks LIMIT 1"
        ).fetchone()
        conn.close()
        return f"{size}" + (" (has data)" if row else " (EMPTY)")
    except Exception:
        return f"{size} (error)"


def _count_files(d: Path, ext: str = ".md") -> int:
    if not d.exists():
        return 0
    return sum(1 for _ in d.rglob(f"*{ext}"))


def _read_counter(path: Path) -> int:
    if path.exists():
        try:
            return int(path.read_text().strip())
        except (ValueError, OSError):
            pass
    return 0


def main():
    import shutil
    data_drive = os.environ.get("JCODER_DATA_DRIVE", "C:\\")
    usage = shutil.disk_usage(data_drive)
    free_gb = usage.free / 1e9

    print("=" * 70)
    print("JCoder Data Status")
    print(f"{data_drive} -- {free_gb:.0f} GB free / {usage.total / 1e9:.0f} GB total")
    print("=" * 70)

    # Raw downloads
    print("\n--- RAW DOWNLOADS ---")
    csn_dir = RAW_DIR / "codesearchnet"
    csn_langs = ["python", "javascript", "java", "go", "ruby", "php"]
    for lang in csn_langs:
        jsonl = csn_dir / f"{lang}.jsonl"
        done = csn_dir / f"{lang}.done"
        status = "DONE" if done.exists() else "partial"
        print(f"  CSN {lang:12s}: {_size_str(jsonl):>8s}  [{status}]")

    # FTS5 indexes -- JCoder_Data/indexes
    print("\n--- FTS5 INDEXES (JCoder_Data/indexes) ---")
    if INDEX_DIR_DATA.exists():
        for db in sorted(INDEX_DIR_DATA.glob("*.fts5.db")):
            print(f"  {db.stem:30s}: {_fts5_info(db)}")
    else:
        print("  (directory missing)")

    # FTS5 indexes -- JCoder/data/indexes
    print("\n--- FTS5 INDEXES (JCoder/data/indexes) ---")
    if INDEX_DIR_PROJ.exists():
        for db in sorted(INDEX_DIR_PROJ.glob("*.fts5.db")):
            print(f"  {db.stem:30s}: {_fts5_info(db)}")
    else:
        print("  (directory missing)")

    # GitHub code progress
    print("\n--- GITHUB CODE DOWNLOAD PROGRESS ---")
    gh_langs = {
        "Python": "python", "JavaScript": "javascript",
        "TypeScript": "typescript", "Java": "java", "Go": "go",
        "C": "c", "C++": "cpp", "Rust": "rust", "Shell": "shell",
        "C#": "csharp",
    }
    for display, key in gh_langs.items():
        count = _read_counter(PROGRESS_DIR / f"{key}.count")
        db = INDEX_DIR_DATA / f"ghcode_{key}.fts5.db"
        idx_info = _fts5_info(db) if db.exists() else "no index"
        if count > 0:
            print(f"  {display:12s}: {count:>8,} entries  [{idx_info}]")
        else:
            print(f"  {display:12s}: not started")

    # Clean source file counts
    print("\n--- EXTRACTED FILES (clean_source) ---")
    if CLEAN_DIR.exists():
        for subdir in sorted(CLEAN_DIR.iterdir()):
            if subdir.is_dir() and not subdir.name.startswith("_"):
                count = _count_files(subdir)
                if count > 0:
                    print(f"  {subdir.name:25s}: {count:>8,} .md files")

    print("=" * 70)


if __name__ == "__main__":
    main()
