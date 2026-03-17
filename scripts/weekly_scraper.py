"""
Weekly Knowledge Scraper (Sprint 14)
-------------------------------------
Pulls fresh coding knowledge from public sources and ingests into FTS5.

Three scrapers:
  1. RSS/Atom feeds (Python blog, Real Python, PEP index, HN top stories)
  2. Changelog monitor (Python releases, top pip packages)
  3. Stack Exchange recent high-quality answers

Output: markdown chunks for FTS5 indexing into fresh_knowledge.fts5.db.

Usage:
    python scripts/weekly_scraper.py                      # all scrapers
    python scripts/weekly_scraper.py --scraper rss         # RSS only
    python scripts/weekly_scraper.py --scraper changelog   # changelogs only
    python scripts/weekly_scraper.py --scraper se          # SE answers only
    python scripts/weekly_scraper.py --dry-run              # preview without ingesting
    python scripts/weekly_scraper.py --max-chunks 500       # cap total chunks
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from xml.etree import ElementTree

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))


def _fix_stdout():
    if sys.platform == "win32" and hasattr(sys.stdout, "buffer"):
        import io
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace")


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _chunk_id(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _strip_html(html: str) -> str:
    """Remove HTML tags, collapse whitespace."""
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ---------------------------------------------------------------------------
# RSS/Atom Feed Scraper
# ---------------------------------------------------------------------------

RSS_FEEDS = [
    ("Python Blog", "https://blog.python.org/feeds/posts/default?alt=rss"),
    ("Real Python", "https://realpython.com/atom.xml"),
    ("PEP Index", "https://peps.python.org/peps.rss"),
]


def scrape_rss_feeds(max_per_feed: int = 20) -> List[Dict[str, str]]:
    """Fetch and parse RSS/Atom feeds into text chunks."""
    try:
        import httpx
    except ImportError:
        print("[WARN] httpx not installed -- RSS scraping unavailable")
        return []

    chunks: List[Dict[str, str]] = []
    client = httpx.Client(timeout=30, follow_redirects=True)

    for feed_name, url in RSS_FEEDS:
        try:
            resp = client.get(url)
            if resp.status_code != 200:
                print(f"[WARN] {feed_name}: HTTP {resp.status_code}")
                continue

            root = ElementTree.fromstring(resp.text)

            # Handle both RSS and Atom
            items = root.findall(".//item") or root.findall(
                ".//{http://www.w3.org/2005/Atom}entry")

            count = 0
            for item in items[:max_per_feed]:
                title = _get_text(item, "title")
                desc = _get_text(item, "description") or _get_text(
                    item, "{http://www.w3.org/2005/Atom}content")
                link = _get_text(item, "link") or _get_attr(
                    item, "{http://www.w3.org/2005/Atom}link", "href")

                if not title:
                    continue

                text = f"# {title}\n\n"
                if desc:
                    text += _strip_html(desc)[:2000]
                if link:
                    text += f"\n\nSource: {link}"

                chunks.append({
                    "content": text,
                    "source": f"rss/{feed_name}",
                    "title": title,
                })
                count += 1

            print(f"[OK] {feed_name}: {count} entries")
        except Exception as exc:
            print(f"[WARN] {feed_name}: {exc}")

    client.close()
    return chunks


def _get_text(elem: Any, tag: str) -> str:
    """Get text content of a child element."""
    child = elem.find(tag)
    if child is None:
        child = elem.find(f"{{http://www.w3.org/2005/Atom}}{tag}")
    return (child.text or "").strip() if child is not None else ""


def _get_attr(elem: Any, tag: str, attr: str) -> str:
    """Get attribute of a child element."""
    child = elem.find(tag)
    return child.get(attr, "") if child is not None else ""


# ---------------------------------------------------------------------------
# Changelog Monitor
# ---------------------------------------------------------------------------

PYPI_PACKAGES = [
    "requests", "flask", "django", "fastapi", "pydantic",
    "httpx", "pytest", "sqlalchemy", "uvicorn", "pandas",
    "numpy", "click", "typer", "rich", "black",
    "ruff", "mypy", "celery", "redis", "boto3",
]


def scrape_changelogs(max_packages: int = 20) -> List[Dict[str, str]]:
    """Check PyPI for recent releases of top packages."""
    try:
        import httpx
    except ImportError:
        print("[WARN] httpx not installed -- changelog scraping unavailable")
        return []

    chunks: List[Dict[str, str]] = []
    client = httpx.Client(timeout=15, follow_redirects=True)

    for pkg in PYPI_PACKAGES[:max_packages]:
        try:
            resp = client.get(f"https://pypi.org/pypi/{pkg}/json")
            if resp.status_code != 200:
                continue
            data = resp.json()
            info = data.get("info", {})
            version = info.get("version", "")
            summary = info.get("summary", "")
            requires_python = info.get("requires_python", "")

            # Get recent releases (last 3)
            releases = list(data.get("releases", {}).keys())[-3:]

            text = (
                f"# {pkg} v{version}\n\n"
                f"Summary: {summary}\n"
                f"Python: {requires_python or 'any'}\n"
                f"Recent versions: {', '.join(releases)}\n"
            )

            chunks.append({
                "content": text,
                "source": f"pypi/{pkg}",
                "title": f"{pkg} v{version}",
            })
        except Exception:
            continue

    client.close()
    print(f"[OK] PyPI changelogs: {len(chunks)} packages")
    return chunks


# ---------------------------------------------------------------------------
# Stack Exchange Recent Answers
# ---------------------------------------------------------------------------

SE_API_BASE = "https://api.stackexchange.com/2.3"
SE_SITES = ["stackoverflow"]


def scrape_se_answers(
    min_score: int = 10,
    max_per_site: int = 30,
    days: int = 30,
) -> List[Dict[str, str]]:
    """Pull highest-voted recent answers from Stack Exchange."""
    try:
        import httpx
    except ImportError:
        print("[WARN] httpx not installed -- SE scraping unavailable")
        return []

    from_date = int(time.time()) - (days * 86400)
    chunks: List[Dict[str, str]] = []
    client = httpx.Client(timeout=30, follow_redirects=True)

    for site in SE_SITES:
        try:
            resp = client.get(
                f"{SE_API_BASE}/answers",
                params={
                    "order": "desc",
                    "sort": "votes",
                    "site": site,
                    "filter": "withbody",
                    "fromdate": from_date,
                    "min": min_score,
                    "pagesize": min(max_per_site, 30),
                    "tagged": "python;javascript;typescript;go;rust",
                },
            )
            if resp.status_code != 200:
                print(f"[WARN] SE {site}: HTTP {resp.status_code}")
                continue

            data = resp.json()
            items = data.get("items", [])

            for item in items:
                body = _strip_html(item.get("body", ""))
                score = item.get("score", 0)
                qid = item.get("question_id", "")

                if len(body) < 100:
                    continue

                text = (
                    f"# SE Answer (score: {score})\n\n"
                    f"{body[:2000]}\n\n"
                    f"Source: https://{site}.com/q/{qid}"
                )

                chunks.append({
                    "content": text,
                    "source": f"se/{site}/{qid}",
                    "title": f"SE answer #{qid} (score {score})",
                })

            print(f"[OK] SE {site}: {len(items)} answers")
        except Exception as exc:
            print(f"[WARN] SE {site}: {exc}")

    client.close()
    return chunks


# ---------------------------------------------------------------------------
# FTS5 Ingestion
# ---------------------------------------------------------------------------

def ingest_chunks(
    chunks: List[Dict[str, str]],
    db_path: str = "data/fresh_knowledge.fts5.db",
    max_chunks: int = 10000,
) -> int:
    """Ingest chunks into a rotating FTS5 database."""
    if not chunks:
        return 0

    db = Path(db_path)
    db.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db))
    conn.execute(
        "CREATE VIRTUAL TABLE IF NOT EXISTS chunks "
        "USING fts5(content, source, title, chunk_id UNINDEXED, ingested_at UNINDEXED)"
    )

    # Check current size
    try:
        cur = conn.execute("SELECT COUNT(*) FROM chunks")
        current_count = cur.fetchone()[0]
    except Exception:
        current_count = 0

    # Rotate: delete oldest if over limit
    if current_count + len(chunks) > max_chunks:
        excess = current_count + len(chunks) - max_chunks
        conn.execute(
            "DELETE FROM chunks WHERE rowid IN "
            "(SELECT rowid FROM chunks ORDER BY ingested_at ASC LIMIT ?)",
            (excess,),
        )

    ingested = 0
    now = time.time()
    for chunk in chunks:
        cid = _chunk_id(chunk["content"])
        # Skip duplicates
        existing = conn.execute(
            "SELECT 1 FROM chunks WHERE chunk_id = ?", (cid,)
        ).fetchone()
        if existing:
            continue

        conn.execute(
            "INSERT INTO chunks (content, source, title, chunk_id, ingested_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (chunk["content"], chunk.get("source", ""),
             chunk.get("title", ""), cid, now),
        )
        ingested += 1

    conn.commit()
    conn.close()
    return ingested


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_weekly_scrape(
    scrapers: Optional[List[str]] = None,
    max_chunks: int = 500,
    dry_run: bool = False,
    output_dir: str = "logs/weekly_scraper",
) -> Dict[str, Any]:
    """Run all (or selected) scrapers and ingest results."""
    all_scrapers = scrapers or ["rss", "changelog", "se"]
    all_chunks: List[Dict[str, str]] = []

    print(f"[OK] Weekly scraper starting at {_timestamp()}")
    print(f"[OK] Scrapers: {', '.join(all_scrapers)}")
    t0 = time.time()

    if "rss" in all_scrapers:
        all_chunks.extend(scrape_rss_feeds())

    if "changelog" in all_scrapers:
        all_chunks.extend(scrape_changelogs())

    if "se" in all_scrapers:
        all_chunks.extend(scrape_se_answers())

    all_chunks = all_chunks[:max_chunks]
    print(f"[OK] Total chunks collected: {len(all_chunks)}")

    ingested = 0
    if not dry_run and all_chunks:
        ingested = ingest_chunks(all_chunks)
        print(f"[OK] Ingested {ingested} new chunks into fresh_knowledge.fts5.db")
    elif dry_run:
        print("[OK] Dry run -- skipping ingestion")

    elapsed = time.time() - t0

    # Save report
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    report = {
        "timestamp": _timestamp(),
        "scrapers": all_scrapers,
        "total_collected": len(all_chunks),
        "ingested": ingested,
        "elapsed_s": round(elapsed, 1),
        "sources": {},
    }
    for chunk in all_chunks:
        src = chunk.get("source", "unknown").split("/")[0]
        report["sources"][src] = report["sources"].get(src, 0) + 1

    report_path = out_dir / f"scrape_{ts}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"[OK] Scrape complete in {elapsed:.1f}s")
    print(f"[OK] Report: {report_path}")
    return report


def main():
    _fix_stdout()
    parser = argparse.ArgumentParser(description="Weekly knowledge scraper")
    parser.add_argument(
        "--scraper", action="append", choices=["rss", "changelog", "se"],
        help="Which scrapers to run (default: all)")
    parser.add_argument(
        "--max-chunks", type=int, default=500,
        help="Maximum chunks to collect/ingest")
    parser.add_argument(
        "--output-dir", default="logs/weekly_scraper",
        help="Output directory for reports")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Collect but don't ingest")
    args = parser.parse_args()

    run_weekly_scrape(
        scrapers=args.scraper,
        max_chunks=args.max_chunks,
        dry_run=args.dry_run,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
