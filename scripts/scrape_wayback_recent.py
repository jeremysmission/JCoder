"""
Wayback Machine Recent Content Scraper
-----------------------------------------
Uses Internet Archive's Wayback Machine CDX API to fetch cached versions
of paywalled/blocked sites from the past 3 weeks. Free access to content
that would otherwise require subscriptions.

Also scrapes free tech blog archives and documentation sites.

Usage:
    python scripts/scrape_wayback_recent.py [--weeks 3] [--output data/raw_downloads/wayback]
"""

import json
import os
import sys
import time
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

UA = "JCoder-Research/1.0 (AI code assistant research)"

# Sites we want recent content from (Wayback has cached versions)
WAYBACK_TARGETS = [
    # AI/ML research blogs (often paywalled or rate-limited)
    {"url": "openai.com/blog/*", "label": "OpenAI Blog", "max": 15},
    {"url": "anthropic.com/research/*", "label": "Anthropic Research", "max": 10},
    {"url": "deepmind.google/research/*", "label": "DeepMind Research", "max": 10},
    {"url": "ai.meta.com/blog/*", "label": "Meta AI Blog", "max": 10},
    {"url": "blog.google/technology/ai/*", "label": "Google AI Blog", "max": 10},
    # Dev tools and frameworks
    {"url": "blog.rust-lang.org/*", "label": "Rust Blog", "max": 5},
    {"url": "go.dev/blog/*", "label": "Go Blog", "max": 5},
    {"url": "devblogs.microsoft.com/python/*", "label": "MS Python Blog", "max": 5},
    # Security
    {"url": "blog.cloudflare.com/*", "label": "Cloudflare Blog", "max": 5},
    {"url": "security.googleblog.com/*", "label": "Google Security Blog", "max": 5},
]

# Free direct-access blogs and docs (no Wayback needed)
DIRECT_SOURCES = [
    # Hugging Face daily papers (free, structured)
    {"url": "https://huggingface.co/papers?date={date}", "label": "HF Papers", "type": "daily"},
    # Python insider blog
    {"url": "https://blog.python.org/feeds/posts/default?alt=rss&max-results=20", "label": "Python Blog", "type": "rss"},
    # GitHub changelog
    {"url": "https://github.blog/changelog/feed/", "label": "GitHub Changelog", "type": "rss"},
]

CDX_API = "https://web.archive.org/cdx/search/cdx"


def fetch_wayback_urls(site_pattern: str, weeks: int = 3, max_results: int = 10):
    """Query Wayback Machine CDX API for recent cached pages."""
    since = (datetime.now() - timedelta(weeks=weeks)).strftime("%Y%m%d")
    params = (
        f"?url={site_pattern}&output=json&limit={max_results}"
        f"&from={since}&filter=statuscode:200&filter=mimetype:text/html"
        f"&collapse=urlkey&fl=timestamp,original,length"
    )
    try:
        req = urllib.request.Request(CDX_API + params, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
        # First row is headers, rest is data
        if len(data) > 1:
            return [{"timestamp": row[0], "url": row[1], "length": row[2]}
                    for row in data[1:]]
    except Exception as e:
        print(f"  [CDX ERROR] {site_pattern}: {e}")
    return []


def fetch_wayback_page(timestamp: str, url: str, dest: Path) -> bool:
    """Download a cached page from Wayback Machine."""
    if dest.exists() and dest.stat().st_size > 500:
        return True
    wayback_url = f"https://web.archive.org/web/{timestamp}id_/{url}"
    try:
        req = urllib.request.Request(wayback_url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
            if len(data) > 200:
                dest.write_bytes(data)
                return True
    except Exception as e:
        print(f"  [FAIL] {url}: {e}")
    return False


def fetch_direct_page(url: str, dest: Path) -> bool:
    """Download a page directly."""
    if dest.exists() and dest.stat().st_size > 500:
        return True
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
            if len(data) > 200:
                dest.write_bytes(data)
                return True
    except Exception as e:
        print(f"  [FAIL] {url}: {e}")
    return False


def scrape_wayback_targets(out: Path, weeks: int = 3):
    """Scrape all Wayback targets."""
    total = 0
    for target in WAYBACK_TARGETS:
        label = target["label"]
        safe_label = label.lower().replace(" ", "_")
        target_dir = out / "wayback" / safe_label
        target_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n  [{label}] Querying CDX API...")
        results = fetch_wayback_urls(target["url"], weeks, target["max"])
        print(f"  [{label}] Found {len(results)} cached pages")

        for i, r in enumerate(results):
            safe_url = r["url"].split("/")[-1][:60] or f"page_{i}"
            safe_name = "".join(c if c.isalnum() or c in "-_." else "_" for c in safe_url)
            dest = target_dir / f"{r['timestamp']}_{safe_name}.html"
            if fetch_wayback_page(r["timestamp"], r["url"], dest):
                total += 1
                print(f"    [OK] {dest.name}")
            time.sleep(1.0)  # Be polite to Archive.org
    return total


def scrape_direct_sources(out: Path):
    """Scrape free direct-access sources."""
    total = 0
    direct_dir = out / "direct"
    direct_dir.mkdir(parents=True, exist_ok=True)

    # HuggingFace daily papers for past 21 days
    print("\n  [HuggingFace Papers] Fetching past 21 days...")
    for days_ago in range(21):
        date = (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d")
        url = f"https://huggingface.co/papers?date={date}"
        dest = direct_dir / f"hf_papers_{date}.html"
        if fetch_direct_page(url, dest):
            total += 1
        time.sleep(0.5)

    # RSS feeds
    for source in DIRECT_SOURCES:
        if source["type"] == "rss":
            safe = source["label"].lower().replace(" ", "_")
            dest = direct_dir / f"{safe}_feed.xml"
            if fetch_direct_page(source["url"], dest):
                total += 1
                print(f"    [OK] {source['label']} RSS")
            time.sleep(0.3)

    return total


def main():
    weeks = 3
    output = Path("data/raw_downloads/wayback_research")

    for i, arg in enumerate(sys.argv[1:]):
        if arg == "--weeks" and i + 1 < len(sys.argv) - 1:
            weeks = int(sys.argv[i + 2])
        elif arg == "--output" and i + 1 < len(sys.argv) - 1:
            output = Path(sys.argv[i + 2])

    output.mkdir(parents=True, exist_ok=True)
    total = 0

    print(f"=== Wayback + Direct Research Scraper (past {weeks} weeks) ===")

    print("\n[1/2] Wayback Machine cached content...")
    total += scrape_wayback_targets(output, weeks)

    print("\n[2/2] Direct free sources...")
    total += scrape_direct_sources(output)

    print(f"\n=== DONE: {total} pages saved to {output} ===")


if __name__ == "__main__":
    os.chdir(os.environ.get("JCODER_ROOT", str(Path(__file__).resolve().parent.parent)))
    main()
