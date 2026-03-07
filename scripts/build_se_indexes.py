"""
Build FTS5 indexes from StackExchange 7z archives.

Extracts Posts.xml from each archive, parses Q&A pairs (score >= 3),
builds per-site FTS5 indexes.

Usage:
    cd D:\\JCoder
    .venv\\Scripts\\python scripts\\build_se_indexes.py

Sites processed (programming-relevant only):
  codereview, dba, security, serverfault, superuser, unix,
  softwareengineering, askubuntu, arduino, reverseengineering,
  networkengineering, codegolf, gamedev, webapps, webmasters

Safe to interrupt and resume -- skips sites with existing indexes.
"""
from __future__ import annotations

import hashlib
import io
import os
import re
import sqlite3
import sys
import time
import xml.etree.ElementTree as ET
from html import unescape
from pathlib import Path

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace"
    )

DATA_ROOT = Path(os.environ.get("JCODER_DATA", r"D:\JCoder_Data"))
INDEX_DIR = DATA_ROOT / "indexes"
INDEX_DIR.mkdir(parents=True, exist_ok=True)

# Two possible archive locations
ARCHIVE_DIRS = [
    Path(r"D:\Projects\KnowledgeBase\stackexchange_20251231"),
    Path(r"D:\RAG Source Data\stackexchange_20251231"),
]

# Programming-relevant SE sites (skip meta sites)
TARGET_SITES = [
    "codereview.stackexchange.com",
    "dba.stackexchange.com",
    "security.stackexchange.com",
    "serverfault.com",
    "superuser.com",
    "unix.stackexchange.com",
    "softwareengineering.stackexchange.com",
    "askubuntu.com",
    "arduino.stackexchange.com",
    "reverseengineering.stackexchange.com",
    "networkengineering.stackexchange.com",
    "codegolf.stackexchange.com",
    "gamedev.stackexchange.com",
    "webapps.stackexchange.com",
    "webmasters.stackexchange.com",
    "crypto.stackexchange.com",
    "raspberrypi.stackexchange.com",
    "ai.stackexchange.com",
]

BATCH_SIZE = 5000
MAX_CHARS = 4000
MIN_SCORE = 3  # quality gate

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
_NORMALIZE_RE = re.compile(r"[_\-./\\:]")
_CAMEL_RE1 = re.compile(r"([a-z])([A-Z])")
_CAMEL_RE2 = re.compile(r"([A-Z]+)([A-Z][a-z])")


def _strip_html(text: str) -> str:
    """Remove HTML tags and decode entities."""
    text = unescape(text)
    text = _TAG_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text)
    return text.strip()


def _normalize(text: str) -> str:
    out = _NORMALIZE_RE.sub(" ", text)
    out = _CAMEL_RE1.sub(r"\1 \2", out)
    out = _CAMEL_RE2.sub(r"\1 \2", out)
    return out.lower()


def _site_key(site_name: str) -> str:
    """Convert site name to index key: codereview.stackexchange.com -> se_codereview"""
    key = site_name.split(".")[0]
    return f"se_{key}"


def _find_archive(site_name: str) -> Path | None:
    """Find the 7z archive for a site across known directories."""
    filename = f"{site_name}.7z"
    for d in ARCHIVE_DIRS:
        p = d / filename
        if p.exists():
            return p
    return None


def _extract_posts_xml(archive_path: Path) -> bytes | None:
    """Extract Posts.xml from a 7z archive to a temp dir, read it, clean up."""
    try:
        import py7zr
    except ImportError:
        print("  [FAIL] py7zr not installed. Run: pip install py7zr")
        return None

    import tempfile
    import shutil

    tmp_dir = None
    try:
        with py7zr.SevenZipFile(str(archive_path), "r") as z:
            names = z.getnames()
            posts_name = None
            for n in names:
                if n.lower() == "posts.xml" or n.lower().endswith("/posts.xml"):
                    posts_name = n
                    break
            if not posts_name:
                print(f"  [WARN] No Posts.xml found in {archive_path.name}")
                return None

            # Extract to temp dir (py7zr 1.x doesn't have .read())
            tmp_dir = tempfile.mkdtemp(prefix="se_extract_")
            z.extract(path=tmp_dir, targets=[posts_name])

        # Read the extracted file
        extracted = Path(tmp_dir) / posts_name
        if extracted.exists():
            return extracted.read_bytes()
        print(f"  [WARN] Posts.xml not found after extraction")
        return None
    except Exception as exc:
        print(f"  [FAIL] Could not extract {archive_path.name}: {exc}")
        return None
    finally:
        if tmp_dir and Path(tmp_dir).exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)


def _parse_and_index(xml_data: bytes, site_key: str, db_path: Path) -> tuple:
    """Parse Posts.xml and build FTS5 index. Returns (entries, chunks)."""
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute(
        "CREATE VIRTUAL TABLE IF NOT EXISTS chunks "
        "USING fts5(search_content, source_path, chunk_id)"
    )

    batch = []
    total_entries = 0
    total_chunks = 0
    skipped = 0

    # Collect questions and answers
    questions = {}  # id -> {title, body, score, tags}
    answers = {}    # parent_id -> [answer_bodies]

    # Parse iteratively to handle large XML
    context = ET.iterparse(io.BytesIO(xml_data), events=("end",))
    for event, elem in context:
        if elem.tag != "row":
            continue

        post_type = elem.get("PostTypeId", "")
        score = int(elem.get("Score", "0"))
        body = elem.get("Body", "")
        post_id = elem.get("Id", "")

        if post_type == "1":  # Question
            if score >= MIN_SCORE and body:
                title = elem.get("Title", "")
                tags = elem.get("Tags", "")
                questions[post_id] = {
                    "title": title,
                    "body": body,
                    "score": score,
                    "tags": tags,
                }
        elif post_type == "2":  # Answer
            parent = elem.get("ParentId", "")
            if score >= MIN_SCORE and body and parent:
                if parent not in answers:
                    answers[parent] = []
                answers[parent].append(body)

        elem.clear()

    # Build Q+A pairs
    for qid, q in questions.items():
        title = _strip_html(q["title"])
        q_body = _strip_html(q["body"])
        tags = q["tags"].replace("><", ", ").strip("<>")

        parts = [f"Q: {title}"]
        if tags:
            parts.append(f"Tags: {tags}")
        parts.append(q_body)

        ans_list = answers.get(qid, [])
        for ans_body in ans_list[:3]:  # top 3 answers
            parts.append(f"\nA: {_strip_html(ans_body)}")

        text = "\n\n".join(parts)
        if len(text) < 50:
            skipped += 1
            continue

        total_entries += 1
        source_id = f"{site_key}_{total_entries:07d}"

        # Chunk
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
                cid = hashlib.sha256(f"{source_id}:{cidx}".encode()).hexdigest()
                batch.append((_normalize(chunk), source_id, cid))
                cidx += 1
                total_chunks += 1
            pos = end

        if len(batch) >= BATCH_SIZE:
            conn.executemany(
                "INSERT INTO chunks(search_content, source_path, chunk_id) "
                "VALUES (?, ?, ?)", batch)
            conn.commit()
            batch = []

    # Flush
    if batch:
        conn.executemany(
            "INSERT INTO chunks(search_content, source_path, chunk_id) "
            "VALUES (?, ?, ?)", batch)
        conn.commit()

    conn.close()
    return total_entries, total_chunks, skipped


def main():
    print("=" * 60)
    print("StackExchange FTS5 Index Builder")
    print(f"Index dir: {INDEX_DIR}")
    print(f"Sites: {len(TARGET_SITES)}")
    print(f"Min score: {MIN_SCORE}")
    print("=" * 60)

    t0 = time.time()
    results = []

    for site in TARGET_SITES:
        site_key = _site_key(site)
        db_path = INDEX_DIR / f"{site_key}.fts5.db"

        print(f"\n--- {site} -> {site_key} ---")

        # Skip if already built
        if db_path.exists() and db_path.stat().st_size > 100_000:
            size_mb = db_path.stat().st_size / 1e6
            print(f"  [OK] Already exists ({size_mb:.0f} MB)")
            results.append((site_key, "exists", size_mb))
            continue

        # Find archive
        archive = _find_archive(site)
        if not archive:
            print(f"  [WARN] No archive found, skipping")
            results.append((site_key, "missing", 0))
            continue

        size_mb = archive.stat().st_size / 1e6
        print(f"  Archive: {archive.name} ({size_mb:.0f} MB)")

        # Extract Posts.xml
        print(f"  Extracting Posts.xml...", end=" ", flush=True)
        t1 = time.time()
        xml_data = _extract_posts_xml(archive)
        if not xml_data:
            results.append((site_key, "extract_fail", 0))
            continue
        print(f"{len(xml_data)/1e6:.0f} MB in {time.time()-t1:.0f}s")

        # Parse and index
        print(f"  Indexing (score >= {MIN_SCORE})...", end=" ", flush=True)
        t1 = time.time()
        entries, chunks, skipped = _parse_and_index(xml_data, site_key, db_path)
        elapsed = time.time() - t1
        idx_size = db_path.stat().st_size / 1e6 if db_path.exists() else 0
        print(f"{entries:,} entries, {chunks:,} chunks ({idx_size:.0f} MB) "
              f"in {elapsed:.0f}s ({skipped:,} skipped)")
        results.append((site_key, "built", idx_size))

    elapsed = time.time() - t0
    print("\n" + "=" * 60)
    print("Summary")
    for key, status, size in results:
        print(f"  {key:30s}: {status:12s} ({size:.0f} MB)")
    print(f"\n  Total time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print("=" * 60)


if __name__ == "__main__":
    main()
