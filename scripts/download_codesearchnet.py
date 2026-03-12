"""
Download CodeSearchNet dataset from GitHub releases (S3 mirrors).

Provides function + docstring pairs across 6 languages.
Downloads zip archives, extracts JSONL.GZ files, and converts
entries to markdown format compatible with corpus_pipeline._chunks_from_qa.

Output: D:\\JCoder_Data\\clean_source\\codesearchnet\\{language}\\

Usage:
    cd D:\\JCoder
    .venv\\Scripts\\python scripts\\download_codesearchnet.py
"""
from __future__ import annotations

import gzip
import io
import json
import os
import sys
import time
import zipfile
from pathlib import Path

from core.download_manager import DownloadManager

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Wrap stdout for Windows Unicode safety
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LANGUAGES = ["python", "javascript", "java", "go", "php", "ruby"]

BASE_URL = "https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/{language}.zip"

DATA_ROOT = Path(os.environ.get("JCODER_DATA", r"D:\JCoder_Data"))
DOWNLOAD_DIR = DATA_ROOT / "downloads" / "codesearchnet"
OUTPUT_DIR = DATA_ROOT / "clean_source" / "codesearchnet"

ENTRIES_PER_FILE = 50
MAX_RETRIES = 3
CHUNK_SIZE = 1024 * 256  # 256 KB read chunks for streaming


def _ensure_dirs():
    """Create all required directories."""
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _download_zip(language: str, downloader: DownloadManager) -> Path:
    """Download a language zip from S3. Returns path to zip file.

    Skips if already downloaded. Retries up to MAX_RETRIES times.
    """
    url = BASE_URL.format(language=language)
    zip_path = DOWNLOAD_DIR / f"{language}.zip"

    if zip_path.exists() and zip_path.stat().st_size > 1000:
        size_mb = zip_path.stat().st_size / (1024 * 1024)
        print(f"[OK] {language}.zip already exists ({size_mb:.1f} MB), skipping download")
        return zip_path

    print(f"     Downloading {language}.zip...")
    result = downloader.download_file(
        url,
        f"{language}.zip",
        min_existing_bytes=1000,
        chunk_size=CHUNK_SIZE,
        progress_label=f"{language}.zip",
        progress_every_bytes=5 * 1024 * 1024,
    )
    if result.ok:
        size_mb = result.path.stat().st_size / (1024 * 1024)
        print(f"[OK] Downloaded {language}.zip ({size_mb:.1f} MB)")
        return result.path

    print(f"[FAIL] Could not download {language}.zip: {result.error}")
    return zip_path  # Caller checks existence


def _extract_jsonl_entries(zip_path: Path, language: str):
    """Extract JSONL.GZ files from zip and yield parsed entries.

    CodeSearchNet zips contain: {language}/final/jsonl/{split}/*.jsonl.gz
    Each JSONL line has: func_name, original_string, docstring, language, url, repo
    """
    if not zip_path.exists():
        return

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            jsonl_gz_files = [
                name for name in zf.namelist()
                if name.endswith(".jsonl.gz") and "__MACOSX" not in name
            ]

            if not jsonl_gz_files:
                print(f"[WARN] No .jsonl.gz files found in {zip_path.name}")
                return

            print(f"     Found {len(jsonl_gz_files)} JSONL.GZ files in {zip_path.name}")

            for gz_name in sorted(jsonl_gz_files):
                try:
                    raw = zf.read(gz_name)
                    decompressed = gzip.decompress(raw)
                    for line in decompressed.decode("utf-8", errors="replace").splitlines():
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            entry = json.loads(line)
                            yield entry
                        except json.JSONDecodeError:
                            continue
                except Exception as exc:
                    print(f"[WARN] Error reading {gz_name}: {exc}")
                    continue
    except zipfile.BadZipFile:
        print(f"[FAIL] Corrupt zip file: {zip_path}")
    except Exception as exc:
        print(f"[FAIL] Error extracting {zip_path}: {exc}")


def _entry_to_markdown(entry: dict) -> str:
    """Convert a single JSONL entry to markdown matching _chunks_from_qa format.

    Expected format:
        # {func_name}
        - language: {language}
        - repo: {repo}
        ## Technical Explanation
        {docstring}
        ## Code
        {original_string}
        ---
    """
    func_name = entry.get("func_name", "").strip()
    language = entry.get("language", "").strip()
    repo = entry.get("repo", "").strip()
    docstring = entry.get("docstring", "").strip()
    code = entry.get("original_string", "").strip()

    # Skip entries with no useful content
    if not code or len(code) < 10:
        return ""

    # Use func_name or generate a placeholder
    title = func_name if func_name else "anonymous_function"

    parts = [f"# {title}"]
    if language:
        parts.append(f"- language: {language}")
    if repo:
        parts.append(f"- repo: {repo}")

    if docstring:
        parts.append(f"## Technical Explanation\n{docstring}")

    parts.append(f"## Code\n{code}")
    parts.append("---")

    return "\n".join(parts)


def _convert_language(language: str, zip_path: Path) -> dict:
    """Convert all entries for one language to grouped markdown files.

    Groups ~50 entries per .md file for efficient ingestion.
    Returns stats dict.
    """
    lang_dir = OUTPUT_DIR / language
    lang_dir.mkdir(parents=True, exist_ok=True)

    # Check if already converted (resume support)
    existing_files = list(lang_dir.glob("*.md"))
    if len(existing_files) > 10:
        entry_estimate = len(existing_files) * ENTRIES_PER_FILE
        print(f"[OK] {language} already has {len(existing_files)} files "
              f"(~{entry_estimate:,} entries), skipping conversion")
        return {
            "language": language,
            "files_written": len(existing_files),
            "entries_total": entry_estimate,
            "skipped": True,
        }

    stats = {
        "language": language,
        "files_written": 0,
        "entries_total": 0,
        "entries_skipped": 0,
        "skipped": False,
    }

    buffer = []
    file_index = 0

    for entry in _extract_jsonl_entries(zip_path, language):
        md = _entry_to_markdown(entry)
        if not md:
            stats["entries_skipped"] += 1
            continue

        buffer.append(md)
        stats["entries_total"] += 1

        if len(buffer) >= ENTRIES_PER_FILE:
            file_index += 1
            out_path = lang_dir / f"csn_{language}_{file_index:05d}.md"
            out_path.write_text("\n\n".join(buffer), encoding="utf-8")
            stats["files_written"] += 1
            buffer.clear()

            if file_index % 100 == 0:
                print(f"     {language}: {stats['entries_total']:,} entries -> "
                      f"{file_index} files")

    # Flush remaining entries
    if buffer:
        file_index += 1
        out_path = lang_dir / f"csn_{language}_{file_index:05d}.md"
        out_path.write_text("\n\n".join(buffer), encoding="utf-8")
        stats["files_written"] += 1

    print(f"[OK] {language}: {stats['entries_total']:,} entries -> "
          f"{stats['files_written']} files "
          f"({stats['entries_skipped']} skipped)")
    return stats


def download_codesearchnet() -> dict:
    """Download and convert all CodeSearchNet languages.

    Returns summary dict with per-language stats.
    """
    _ensure_dirs()

    print("=" * 60)
    print("CodeSearchNet Dataset Download + Conversion")
    print(f"Download dir: {DOWNLOAD_DIR}")
    print(f"Output dir:   {OUTPUT_DIR}")
    print(f"Languages:    {', '.join(LANGUAGES)}")
    print("=" * 60)

    t0 = time.time()
    all_stats = {}

    with DownloadManager(DOWNLOAD_DIR, max_retries=MAX_RETRIES, read_timeout_s=300.0) as downloader:
        for lang in LANGUAGES:
            print(f"\n--- {lang.upper()} ---")
            zip_path = _download_zip(lang, downloader)
            lang_stats = _convert_language(lang, zip_path)
            all_stats[lang] = lang_stats

    elapsed = time.time() - t0
    total_entries = sum(s["entries_total"] for s in all_stats.values())
    total_files = sum(s["files_written"] for s in all_stats.values())

    print("\n" + "=" * 60)
    print("CodeSearchNet Summary")
    print(f"  Languages: {len(LANGUAGES)}")
    print(f"  Total entries: {total_entries:,}")
    print(f"  Total files: {total_files:,}")
    print(f"  Elapsed: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print("=" * 60)

    return {
        "total_entries": total_entries,
        "total_files": total_files,
        "elapsed_s": elapsed,
        "languages": all_stats,
    }


if __name__ == "__main__":
    result = download_codesearchnet()
    sys.exit(0 if result["total_files"] > 0 else 1)
