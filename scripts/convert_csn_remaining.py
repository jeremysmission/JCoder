"""
Convert remaining CodeSearchNet JSONL files to markdown.

The overnight_download.py already converted Python (292K files).
This script converts the other 5 languages from the raw .jsonl
files at data/raw_downloads/codesearchnet/{lang}.jsonl

Usage:
    cd D:\JCoder
    python scripts/convert_csn_remaining.py
"""
from __future__ import annotations

import io
import json
import os
import sys
import time
from pathlib import Path

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

LANGUAGES = ["javascript", "java", "go", "php", "ruby"]
DATA_ROOT = Path(os.environ.get("JCODER_DATA", "data"))
RAW_DIR = DATA_ROOT / "raw_downloads" / "codesearchnet"
OUTPUT_DIR = DATA_ROOT / "clean_source" / "codesearchnet"
ENTRIES_PER_FILE = 50


def _entry_to_markdown(entry: dict) -> str:
    """Convert a single JSONL entry to markdown."""
    func_name = entry.get("func_name", entry.get("identifier", ""))
    docstring = entry.get("func_documentation_string", entry.get("docstring", ""))
    code = entry.get("whole_func_string", entry.get("original_string",
           entry.get("func_code_string", entry.get("code", ""))))
    language = entry.get("language", "")
    repo = entry.get("repository_name", entry.get("repo", entry.get("url", "")))

    if not code and not docstring:
        return ""

    parts = [f"# {func_name}" if func_name else "# (unnamed)"]
    if language:
        parts.append(f"- language: {language}")
    if repo:
        parts.append(f"- repo: {repo}")
    if docstring:
        parts.append(f"## Technical Explanation\n{docstring}")
    if code:
        parts.append(f"## Code\n{code}")
    parts.append("---")
    return "\n".join(parts)


def _convert_language(language: str) -> dict:
    """Convert one language JSONL to grouped markdown files."""
    jsonl_path = RAW_DIR / f"{language}.jsonl"
    if not jsonl_path.exists():
        print(f"[WARN] {jsonl_path} not found, skipping")
        return {"language": language, "files": 0, "entries": 0, "skipped": True}

    lang_dir = OUTPUT_DIR / language
    lang_dir.mkdir(parents=True, exist_ok=True)

    existing = list(lang_dir.glob("*.md"))
    if len(existing) > 10:
        est = len(existing) * ENTRIES_PER_FILE
        print(f"[OK] {language} already has {len(existing)} files (~{est:,} entries), skipping")
        return {"language": language, "files": len(existing), "entries": est, "skipped": True}

    stats = {"language": language, "files": 0, "entries": 0, "errors": 0, "skipped": False}
    buffer = []
    file_index = 0
    t0 = time.monotonic()

    with open(jsonl_path, "r", encoding="utf-8", errors="replace") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                stats["errors"] += 1
                continue

            md = _entry_to_markdown(entry)
            if not md:
                continue

            buffer.append(md)
            stats["entries"] += 1

            if len(buffer) >= ENTRIES_PER_FILE:
                file_index += 1
                out_path = lang_dir / f"csn_{language}_{file_index:05d}.md"
                out_path.write_text("\n\n".join(buffer), encoding="utf-8")
                stats["files"] += 1
                buffer.clear()

                if file_index % 500 == 0:
                    elapsed = time.monotonic() - t0
                    rate = stats["entries"] / elapsed if elapsed > 0 else 0
                    print(f"     {language}: {stats['entries']:,} entries -> "
                          f"{file_index} files ({rate:.0f} entries/s)")

    if buffer:
        file_index += 1
        out_path = lang_dir / f"csn_{language}_{file_index:05d}.md"
        out_path.write_text("\n\n".join(buffer), encoding="utf-8")
        stats["files"] += 1

    elapsed = time.monotonic() - t0
    print(f"[OK] {language}: {stats['entries']:,} entries -> {stats['files']} files "
          f"in {elapsed:.1f}s ({stats['errors']} errors)")
    return stats


def main():
    print("\nCodeSearchNet Remaining Language Conversion")
    print("=" * 50)
    all_stats = []
    t_total = time.monotonic()

    for lang in LANGUAGES:
        stats = _convert_language(lang)
        all_stats.append(stats)

    elapsed = time.monotonic() - t_total
    total_entries = sum(s["entries"] for s in all_stats)
    total_files = sum(s["files"] for s in all_stats)
    print("=" * 50)
    print(f"[OK] Total: {total_entries:,} entries -> {total_files:,} files in {elapsed:.1f}s")

    for s in all_stats:
        status = "SKIP" if s.get("skipped") else "OK"
        print(f"  [{status}] {s['language']}: {s['entries']:,} entries, {s['files']} files")


if __name__ == "__main__":
    main()
