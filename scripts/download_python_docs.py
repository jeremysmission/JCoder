"""
Download and convert Python standard library documentation to markdown.

Source: Official Python 3.13.3 text docs archive (tar.bz2).
Converts .txt doc files to markdown with module headers and preserved
code examples, compatible with corpus_pipeline.ingest_markdown_docs.

Output: D:\\JCoder_Data\\clean_source\\python_docs\\

Usage:
    cd D:\\JCoder
    .venv\\Scripts\\python scripts\\download_python_docs.py
"""
from __future__ import annotations

import io
import os
import re
import sys
import tarfile
import time
from pathlib import Path

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Wrap stdout for Windows Unicode safety
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DOCS_URL = "https://www.python.org/ftp/python/doc/3.13.3/python-3.13.3-docs-text.tar.bz2"

DATA_ROOT = Path(os.environ.get("JCODER_DATA", r"D:\JCoder_Data"))
DOWNLOAD_DIR = DATA_ROOT / "downloads" / "python_docs"
OUTPUT_DIR = DATA_ROOT / "clean_source" / "python_docs"

MAX_RETRIES = 3
CHUNK_SIZE = 1024 * 256  # 256 KB


def _ensure_dirs():
    """Create all required directories."""
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _download_archive() -> Path:
    """Download the Python docs tar.bz2 archive. Returns path to archive.

    Skips if already downloaded.
    """
    import httpx

    archive_path = DOWNLOAD_DIR / "python-3.13.3-docs-text.tar.bz2"

    if archive_path.exists() and archive_path.stat().st_size > 10000:
        size_mb = archive_path.stat().st_size / (1024 * 1024)
        print(f"[OK] Archive already exists ({size_mb:.1f} MB), skipping download")
        return archive_path

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"     Downloading Python docs archive (attempt {attempt}/{MAX_RETRIES})...")
            partial_path = DOWNLOAD_DIR / "python-docs.tar.bz2.partial"

            with httpx.stream("GET", DOCS_URL, follow_redirects=True,
                              timeout=httpx.Timeout(30.0, read=300.0)) as resp:
                resp.raise_for_status()
                total = int(resp.headers.get("content-length", 0))
                downloaded = 0
                with open(partial_path, "wb") as f:
                    for chunk in resp.iter_bytes(chunk_size=CHUNK_SIZE):
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total > 0 and downloaded % (1024 * 1024) < CHUNK_SIZE:
                            pct = downloaded * 100 // total
                            print(f"     {downloaded // (1024*1024)} / "
                                  f"{total // (1024*1024)} MB ({pct}%)")

            if partial_path.exists():
                if archive_path.exists():
                    archive_path.unlink()
                partial_path.rename(archive_path)

            size_mb = archive_path.stat().st_size / (1024 * 1024)
            print(f"[OK] Downloaded Python docs archive ({size_mb:.1f} MB)")
            return archive_path

        except Exception as exc:
            print(f"[WARN] Download attempt {attempt} failed: {exc}")
            if attempt < MAX_RETRIES:
                wait = 5 * attempt
                print(f"     Retrying in {wait}s...")
                time.sleep(wait)

    print(f"[FAIL] Could not download Python docs after {MAX_RETRIES} attempts")
    return archive_path


def _extract_archive(archive_path: Path) -> Path:
    """Extract the tar.bz2 archive. Returns path to extracted root."""
    if not archive_path.exists():
        print("[FAIL] Archive file not found")
        return DOWNLOAD_DIR

    extract_dir = DOWNLOAD_DIR / "extracted"

    # Check if already extracted
    if extract_dir.exists():
        txt_count = sum(1 for _ in extract_dir.rglob("*.txt"))
        if txt_count > 50:
            print(f"[OK] Already extracted ({txt_count} .txt files)")
            return extract_dir

    print("     Extracting archive...")
    extract_dir.mkdir(parents=True, exist_ok=True)

    try:
        with tarfile.open(archive_path, "r:bz2") as tar:
            # Safety: filter out any path traversal attempts
            members = []
            for member in tar.getmembers():
                # Skip anything with .. in path
                if ".." in member.name or member.name.startswith("/"):
                    continue
                members.append(member)
            tar.extractall(path=extract_dir, members=members)

        txt_count = sum(1 for _ in extract_dir.rglob("*.txt"))
        print(f"[OK] Extracted {txt_count} .txt files")
    except Exception as exc:
        print(f"[FAIL] Extraction failed: {exc}")

    return extract_dir


def _classify_doc(rel_path: str) -> str:
    """Classify a doc file into a category based on its path."""
    parts = rel_path.replace("\\", "/").lower()
    if "/library/" in parts:
        return "stdlib"
    if "/reference/" in parts:
        return "reference"
    if "/tutorial/" in parts:
        return "tutorial"
    if "/howto/" in parts:
        return "howto"
    if "/faq/" in parts:
        return "faq"
    if "/c-api/" in parts:
        return "c_api"
    if "/whatsnew/" in parts:
        return "whatsnew"
    if "/using/" in parts:
        return "using"
    return "other"


def _txt_to_markdown(content: str, rel_path: str) -> str:
    """Convert a Python docs .txt file to markdown.

    Preserves code examples, adds module header, and formats
    section headings for corpus_pipeline._split_by_headings.
    """
    # Derive module/topic name from file path
    fname = Path(rel_path).stem
    category = _classify_doc(rel_path)

    lines = content.splitlines()
    md_lines = []

    # Add top-level heading
    md_lines.append(f"# Module: {fname}")
    md_lines.append(f"- source_kind: python_docs")
    md_lines.append(f"- category: {category}")
    md_lines.append(f"- path: {rel_path}")
    md_lines.append("")

    i = 0
    in_code_block = False

    while i < len(lines):
        line = lines[i]

        # Detect RST-style underline headings: line followed by ===, ---, ~~~
        if (i + 1 < len(lines)
                and len(lines[i + 1].strip()) > 0
                and len(set(lines[i + 1].strip())) == 1
                and lines[i + 1].strip()[0] in "=-~`^"
                and len(lines[i + 1].strip()) >= max(3, len(line.strip()) - 2)):
            heading_text = line.strip()
            underline_char = lines[i + 1].strip()[0]
            if heading_text:
                # Map underline char to heading level
                if underline_char == "=":
                    md_lines.append(f"## {heading_text}")
                elif underline_char == "-":
                    md_lines.append(f"### {heading_text}")
                else:
                    md_lines.append(f"#### {heading_text}")
                i += 2
                continue

        # Detect code blocks: lines indented with 3+ spaces after a blank line
        # or lines starting with >>> (doctest)
        stripped = line.lstrip()
        indent = len(line) - len(stripped)

        if stripped.startswith(">>>") and not in_code_block:
            md_lines.append("```python")
            md_lines.append(line)
            in_code_block = True
            i += 1
            continue

        if in_code_block:
            if stripped.startswith(">>>") or stripped.startswith("...") or indent >= 3:
                md_lines.append(line)
                i += 1
                continue
            if stripped == "":
                # Check if next non-empty line continues the block
                j = i + 1
                while j < len(lines) and lines[j].strip() == "":
                    j += 1
                if j < len(lines) and (lines[j].startswith(">>>")
                                        or lines[j].startswith("   ")):
                    md_lines.append(line)
                    i += 1
                    continue
            md_lines.append("```")
            md_lines.append("")
            in_code_block = False
            # Fall through to process current line

        md_lines.append(line)
        i += 1

    # Close any open code block
    if in_code_block:
        md_lines.append("```")

    return "\n".join(md_lines)


def _convert_docs(extract_dir: Path) -> dict:
    """Convert all extracted .txt docs to markdown files.

    One output file per source doc file.
    Returns stats dict.
    """
    stats = {
        "files_converted": 0,
        "files_skipped": 0,
        "categories": {},
    }

    # Find the actual docs root (usually python-3.13.3-docs-text/)
    docs_root = extract_dir
    subdirs = [d for d in extract_dir.iterdir() if d.is_dir()]
    if len(subdirs) == 1 and subdirs[0].name.startswith("python"):
        docs_root = subdirs[0]

    # Check if already converted
    existing = list(OUTPUT_DIR.glob("**/*.md"))
    if len(existing) > 50:
        print(f"[OK] Already converted ({len(existing)} files), skipping")
        stats["files_converted"] = len(existing)
        return stats

    txt_files = sorted(docs_root.rglob("*.txt"))
    if not txt_files:
        print("[WARN] No .txt files found in extracted docs")
        return stats

    print(f"     Converting {len(txt_files)} doc files to markdown...")

    for txt_path in txt_files:
        try:
            rel = txt_path.relative_to(docs_root)
            rel_str = str(rel).replace("\\", "/")

            # Skip non-content files
            if rel.name in ("contents.txt", "copyright.txt", "bugs.txt",
                            "about.txt", "license.txt"):
                stats["files_skipped"] += 1
                continue

            content = txt_path.read_text(encoding="utf-8", errors="replace")
            if not content.strip() or len(content.strip()) < 50:
                stats["files_skipped"] += 1
                continue

            md_content = _txt_to_markdown(content, rel_str)

            # Determine output path: preserve directory structure
            category = _classify_doc(rel_str)
            out_dir = OUTPUT_DIR / category
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / (txt_path.stem + ".md")

            out_path.write_text(md_content, encoding="utf-8")
            stats["files_converted"] += 1
            stats["categories"][category] = stats["categories"].get(category, 0) + 1

        except Exception as exc:
            print(f"[WARN] Error converting {txt_path.name}: {exc}")
            stats["files_skipped"] += 1

    print(f"[OK] Converted {stats['files_converted']} doc files")
    for cat, count in sorted(stats["categories"].items()):
        print(f"     {cat}: {count} files")

    return stats


def download_python_docs() -> dict:
    """Download, extract, and convert Python docs.

    Returns summary dict.
    """
    _ensure_dirs()

    print("=" * 60)
    print("Python Standard Library Documentation Download")
    print(f"Source:  {DOCS_URL}")
    print(f"Output:  {OUTPUT_DIR}")
    print("=" * 60)

    t0 = time.time()

    archive_path = _download_archive()
    extract_dir = _extract_archive(archive_path)
    stats = _convert_docs(extract_dir)

    elapsed = time.time() - t0

    print("\n" + "=" * 60)
    print("Python Docs Summary")
    print(f"  Files converted: {stats['files_converted']}")
    print(f"  Files skipped:   {stats['files_skipped']}")
    print(f"  Elapsed: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print("=" * 60)

    return {
        "files_converted": stats["files_converted"],
        "elapsed_s": elapsed,
    }


if __name__ == "__main__":
    result = download_python_docs()
    sys.exit(0 if result["files_converted"] > 0 else 1)
