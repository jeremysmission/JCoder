"""
Download and index arXiv papers on self-evolving agents, agentic AI,
tool learning, memory systems, and prompt optimization.

Sources: Awesome-Self-Evolving-Agents survey (120+ papers).
Downloads abstracts + full text where available via arXiv API.

Usage:
    cd D:\\JCoder
    .venv\\Scripts\\python scripts\\download_arxiv_agentic.py
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
import sqlite3
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

from core.download_manager import DownloadManager

DATA_ROOT = Path(os.environ.get("JCODER_DATA", r"D:\JCoder_Data"))
INDEX_DIR = DATA_ROOT / "indexes"
DOWNLOAD_DIR = DATA_ROOT / "downloads" / "arxiv_agentic"
INDEX_DIR.mkdir(parents=True, exist_ok=True)
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 500
MAX_CHARS = 4000

_NORMALIZE_RE = re.compile(r"[_\-./\\:]")
_CAMEL_RE1 = re.compile(r"([a-z])([A-Z])")
_CAMEL_RE2 = re.compile(r"([A-Z]+)([A-Z][a-z])")
# TODO(DRY): The _DOWNLOADER singleton + _get_downloader() boilerplate below is
# duplicated across 8 download scripts (download_arxiv_agentic, download_code_corpora,
# download_expansion_tier1, download_instruction_corpora, download_new_datasets_2026_03_10,
# download_phase6_datasets, download_subject_repos, reacquire_se_archives). Consider
# extracting a shared base class or helper function in core/download_manager.py.
_DOWNLOADER: DownloadManager | None = None


def _normalize(text: str) -> str:
    out = _NORMALIZE_RE.sub(" ", text)
    out = _CAMEL_RE1.sub(r"\1 \2", out)
    out = _CAMEL_RE2.sub(r"\1 \2", out)
    return out.lower()


def _configure_stdout() -> None:
    if sys.platform != "win32":
        return
    buffer = getattr(sys.stdout, "buffer", None)
    if buffer is None:
        return
    sys.stdout = io.TextIOWrapper(buffer, encoding="utf-8", errors="replace")


def _get_downloader() -> DownloadManager:
    global _DOWNLOADER
    if _DOWNLOADER is None:
        _DOWNLOADER = DownloadManager(DOWNLOAD_DIR, read_timeout_s=60.0)
    return _DOWNLOADER


def _close_downloader() -> None:
    global _DOWNLOADER
    if _DOWNLOADER is not None:
        _DOWNLOADER.close()
        _DOWNLOADER = None


def _fetch_arxiv_feed(url: str) -> str:
    return _get_downloader().fetch_text(
        url,
        headers={"Accept": "application/atom+xml"},
    )


# All arXiv IDs from Awesome-Self-Evolving-Agents survey
ARXIV_IDS = [
    # Frameworks
    "2507.03616", "2505.16988",
    # SFT approaches
    "2309.17452", "2203.14465", "2404.14662", "2405.07551", "2503.03686",
    # RL approaches
    "2401.10020", "2411.15124", "2402.00658", "2408.07199", "2405.14333",
    "2412.17451", "2505.03335", "2508.05004", "2506.24119", "2507.13833",
    "2509.07980", "2508.10874", "2505.20347",
    # Feedback-based
    "2207.10397", "2302.08468", "2303.04910", "2312.08935", "2410.18451",
    "2410.08146", "2507.01352",
    # Search-based
    "2203.11171", "2210.16257", "2305.10601", "2406.04271", "2401.17686",
    "2308.09687", "2412.09078",
    # Reasoning
    "2503.04625", "2506.09820",
    # Prompt optimization - edit
    "2210.17041", "2203.07281", "2211.11890", "2311.08364",
    # Prompt optimization - evolutionary
    "2309.08532", "2309.16797", "2507.19457",
    # Prompt optimization - generative
    "2211.01910", "2310.16427", "2309.03409", "2308.02151", "2406.11695",
    "2405.17346", "2410.08601", "2502.06855",
    # Text gradient prompt optimization
    "2305.03495", "2406.07496", "2412.03624", "2407.12865", "2402.17564",
    "2412.03092", "2506.06254",
    # Memory optimization
    "2402.09727", "2409.07429", "2305.10250", "2406.14550", "2404.00573",
    "2402.11975", "2410.08815", "2412.18069", "2502.12110", "2504.19413",
    "2508.19828", "2508.09736",
    # Tool learning - training
    "2305.18752", "2307.16789", "2403.04746", "2308.14034", "2410.06617",
    "2410.12952", "2412.15606", "2503.07826", "2411.00412",
    # Tool learning - RL
    "2504.11536", "2504.13958", "2505.00024", "2504.04736", "2504.21561",
    "2505.16410", "2507.19849", "2507.21836",
    # Tool learning - inference
    "2401.06201", "2410.08197", "2503.14432", "2310.13227", "2406.03807",
    "2506.01056",
    # Tool creation
    "2305.14318", "2402.11359", "2312.10908", "2505.20286",
    # Unified optimization
    "2508.19005", "2502.05907",
    # Multi-agent
    "2507.22606", "2505.14738", "2503.03205", "2410.10762", "2408.08435",
    "2504.00587", "2502.04306", "2502.02533", "2505.14996", "2505.22967",
    "2402.16823",
    # Classic agentic papers
    "2303.11366",  # Reflexion
    "2305.16291",  # Voyager
    "2308.10144",  # ExpeL
    "2310.12931",  # MemGPT
    "2401.14656",  # MCP (Model Context Protocol concepts)
    "2305.17126",  # ReAct (reasoning + acting)
    "2210.03629",  # ReAct original
    # Additional self-learning / meta-learning
    "2310.04406",  # Self-RAG
    "2305.20050",  # Gorilla (tool calling)
    "2306.06070",  # Llemma (math reasoning)
    "2401.02954",  # RLHF survey
    "2308.12950",  # AgentBench
]


def fetch_arxiv_metadata(arxiv_id: str) -> dict | None:
    """Fetch title + abstract from arXiv API."""
    cache_file = DOWNLOAD_DIR / f"{arxiv_id.replace('/', '_')}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text(encoding="utf-8"))

    url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
    try:
        feed_text = _fetch_arxiv_feed(url)
    except Exception as exc:
        print(f"    [WARN] API error for {arxiv_id}: {exc}")
        return None

    # Parse XML
    try:
        root = ET.fromstring(feed_text)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        entry = root.find("atom:entry", ns)
        if entry is None:
            return None

        title = entry.findtext("atom:title", "", ns).strip()
        title = re.sub(r"\s+", " ", title)
        abstract = entry.findtext("atom:summary", "", ns).strip()
        abstract = re.sub(r"\s+", " ", abstract)

        authors = []
        for author in entry.findall("atom:author", ns):
            name = author.findtext("atom:name", "", ns)
            if name:
                authors.append(name)

        categories = []
        for cat in entry.findall("atom:category", ns):
            term = cat.get("term", "")
            if term:
                categories.append(term)

        published = entry.findtext("atom:published", "", ns)[:10]

        result = {
            "arxiv_id": arxiv_id,
            "title": title,
            "abstract": abstract,
            "authors": ", ".join(authors[:10]),
            "categories": ", ".join(categories),
            "published": published,
        }

        # Cache
        cache_file.write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        return result

    except Exception as exc:
        print(f"    [WARN] Parse error for {arxiv_id}: {exc}")
        return None


def build_agentic_index():
    """Download all papers and build FTS5 index."""
    db_path = INDEX_DIR / "arxiv_agentic_ai.fts5.db"

    if db_path.exists() and db_path.stat().st_size > 100_000:
        print(f"  [OK] {db_path.name} exists "
              f"({db_path.stat().st_size/1e6:.0f} MB)")
        return

    print(f"  Fetching {len(ARXIV_IDS)} papers from arXiv API...")
    print(f"  (rate-limited to 1 req/3s per arXiv policy)")

    papers = []
    for i, aid in enumerate(ARXIV_IDS):
        print(f"    [{i+1}/{len(ARXIV_IDS)}] {aid}... ", end="", flush=True)
        meta = fetch_arxiv_metadata(aid)
        if meta:
            papers.append(meta)
            title_short = meta["title"][:60]
            print(f"{title_short}")
        else:
            print("FAILED")

        # arXiv rate limit: max 1 request per 3 seconds
        if i < len(ARXIV_IDS) - 1:
            time.sleep(3.1)

    print(f"\n  {len(papers)} papers fetched. Building index...")

    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute(
        "CREATE VIRTUAL TABLE IF NOT EXISTS chunks "
        "USING fts5(search_content, source_path, chunk_id)"
    )

    for p in papers:
        text = (
            f"Title: {p['title']}\n"
            f"Authors: {p['authors']}\n"
            f"Published: {p['published']}\n"
            f"Categories: {p['categories']}\n"
            f"arXiv: {p['arxiv_id']}\n\n"
            f"Abstract:\n{p['abstract']}"
        )
        normalized = _normalize(text)
        cid = hashlib.sha256(
            p["arxiv_id"].encode()
        ).hexdigest()
        conn.execute(
            "INSERT INTO chunks(search_content, source_path, chunk_id) "
            "VALUES (?, ?, ?)",
            (normalized, f"arxiv:{p['arxiv_id']}", cid)
        )

    conn.commit()
    conn.close()

    size_mb = db_path.stat().st_size / 1e6
    print(f"  [OK] {db_path.name}: {len(papers)} papers ({size_mb:.1f} MB)")


def build_broad_ml_search():
    """Also fetch broader ML/AI topics via arXiv search API."""
    db_path = INDEX_DIR / "arxiv_ml_broad.fts5.db"

    if db_path.exists() and db_path.stat().st_size > 1_000_000:
        print(f"  [OK] {db_path.name} exists "
              f"({db_path.stat().st_size/1e6:.0f} MB)")
        return

    queries = [
        "cat:cs.AI+AND+ti:agent+AND+ti:learning",
        "cat:cs.LG+AND+ti:self-improving",
        "cat:cs.CL+AND+ti:code+generation",
        "cat:cs.SE+AND+ti:large+language+model",
        "cat:cs.AI+AND+ti:retrieval+augmented",
        "cat:cs.CL+AND+ti:tool+use+AND+ti:language+model",
        "cat:cs.AI+AND+ti:memory+AND+ti:agent",
        "cat:cs.LG+AND+ti:reinforcement+AND+ti:reasoning",
    ]

    all_papers = {}
    for q in queries:
        url = (
            f"http://export.arxiv.org/api/query?"
            f"search_query={q}&start=0&max_results=200"
            f"&sortBy=submittedDate&sortOrder=descending"
        )
        print(f"  Searching: {q[:50]}...", flush=True)
        try:
            root = ET.fromstring(_fetch_arxiv_feed(url))
            ns = {"atom": "http://www.w3.org/2005/Atom"}

            for entry in root.findall("atom:entry", ns):
                aid_url = entry.findtext("atom:id", "", ns)
                aid = aid_url.split("/abs/")[-1].split("v")[0] if aid_url else ""
                if not aid or aid in all_papers:
                    continue

                title = entry.findtext("atom:title", "", ns).strip()
                title = re.sub(r"\s+", " ", title)
                abstract = entry.findtext("atom:summary", "", ns).strip()
                abstract = re.sub(r"\s+", " ", abstract)

                authors = []
                for author in entry.findall("atom:author", ns):
                    name = author.findtext("atom:name", "", ns)
                    if name:
                        authors.append(name)

                categories = []
                for cat in entry.findall("atom:category", ns):
                    term = cat.get("term", "")
                    if term:
                        categories.append(term)

                published = entry.findtext("atom:published", "", ns)[:10]

                all_papers[aid] = {
                    "arxiv_id": aid,
                    "title": title,
                    "abstract": abstract,
                    "authors": ", ".join(authors[:10]),
                    "categories": ", ".join(categories),
                    "published": published,
                }

            print(f"    {len(all_papers)} total papers so far")
        except Exception as exc:
            print(f"    [WARN] Search failed: {exc}")

        time.sleep(3.1)

    print(f"\n  {len(all_papers)} unique papers. Building index...")

    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute(
        "CREATE VIRTUAL TABLE IF NOT EXISTS chunks "
        "USING fts5(search_content, source_path, chunk_id)"
    )

    for p in all_papers.values():
        text = (
            f"Title: {p['title']}\n"
            f"Authors: {p['authors']}\n"
            f"Published: {p['published']}\n"
            f"Categories: {p['categories']}\n"
            f"arXiv: {p['arxiv_id']}\n\n"
            f"Abstract:\n{p['abstract']}"
        )
        normalized = _normalize(text)
        cid = hashlib.sha256(p["arxiv_id"].encode()).hexdigest()
        conn.execute(
            "INSERT INTO chunks(search_content, source_path, chunk_id) "
            "VALUES (?, ?, ?)",
            (normalized, f"arxiv:{p['arxiv_id']}", cid)
        )

    conn.commit()
    conn.close()

    size_mb = db_path.stat().st_size / 1e6
    print(f"  [OK] {db_path.name}: {len(all_papers)} papers "
          f"({size_mb:.1f} MB)")


def main():
    _configure_stdout()
    print("=" * 60)
    print("JCoder arXiv Agentic AI Paper Downloader")
    print(f"Index dir: {INDEX_DIR}")
    print("=" * 60)

    t0 = time.time()

    print("\n--- CURATED AGENTIC PAPERS (120+) ---")
    build_agentic_index()

    print("\n--- BROAD ML/AI SEARCH (up to 1600) ---")
    build_broad_ml_search()

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.0f}s ({elapsed / 60:.1f} min)")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    finally:
        _close_downloader()
