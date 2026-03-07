"""
Research Scanner -- Crawl and index the latest AI/ML advances.

Scans curated sources for cutting-edge content on:
  - Agentic AI (ReAct, reflection, tool use, planning)
  - Code assistance (completion, debugging, review, generation)
  - RAG techniques (hybrid search, reranking, GraphRAG, late chunking)
  - Distillation (knowledge transfer, synthetic data, self-play)
  - Self-learning / reasoning (MCTS, chain-of-thought, self-improvement)

Sources:
  - ArXiv (cs.AI, cs.CL, cs.SE papers via API)
  - HuggingFace papers (trending/recent)
  - GitHub trending repos (AI/ML tools)

Outputs to: research_feed.fts5.db (searchable knowledge base)

Usage:
    cd D:\\JCoder
    .venv\\Scripts\\python scripts\\research_scanner.py [--days 30]

Safe to re-run -- deduplicates by paper ID / URL.
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
from datetime import datetime, timedelta
from pathlib import Path

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace"
    )

import httpx

DATA_ROOT = Path(os.environ.get("JCODER_DATA", r"D:\JCoder_Data"))
INDEX_DIR = DATA_ROOT / "indexes"
INDEX_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = INDEX_DIR / "research_feed.fts5.db"

# Research topics to track
ARXIV_QUERIES = [
    # Agentic AI
    '(ti:"code agent" OR ti:"coding agent" OR ti:"agentic") AND cat:cs.AI',
    'ti:"tool use" AND ti:"language model" AND cat:cs.CL',
    'ti:"self-improving" AND ti:"language model"',
    'ti:"reflection" AND ti:"reasoning" AND cat:cs.AI',
    # RAG
    'ti:"retrieval augmented" AND cat:cs.CL',
    'ti:"RAG" AND (ti:"code" OR ti:"programming") AND cat:cs.CL',
    'ti:"hybrid search" AND ti:"retrieval"',
    'ti:"reranking" AND ti:"retrieval"',
    # Code generation
    'ti:"code generation" AND cat:cs.SE',
    'ti:"code completion" AND cat:cs.SE',
    'ti:"program synthesis" AND cat:cs.PL',
    'ti:"code review" AND ti:"automated"',
    # Distillation
    'ti:"knowledge distillation" AND ti:"code"',
    'ti:"synthetic data" AND ti:"code"',
    'ti:"self-play" AND ti:"language model"',
    # Reasoning
    'ti:"chain of thought" AND cat:cs.AI',
    '(ti:"tree of thought" OR ti:"monte carlo") AND ti:"reasoning"',
    'ti:"self-learning" AND ti:"language model"',
]

# HuggingFace daily papers endpoint
HF_PAPERS_URL = "https://huggingface.co/api/daily_papers"

BATCH_SIZE = 500

_TAG_RE = re.compile(r"<[^>]+>")
_NORMALIZE_RE = re.compile(r"[_\-./\\:]")
_CAMEL_RE1 = re.compile(r"([a-z])([A-Z])")
_CAMEL_RE2 = re.compile(r"([A-Z]+)([A-Z][a-z])")


def _normalize(text: str) -> str:
    out = _NORMALIZE_RE.sub(" ", text)
    out = _CAMEL_RE1.sub(r"\1 \2", out)
    out = _CAMEL_RE2.sub(r"\1 \2", out)
    return out.lower()


def _init_db():
    """Initialize the research feed database."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute(
        "CREATE VIRTUAL TABLE IF NOT EXISTS chunks "
        "USING fts5(search_content, source_path, chunk_id)"
    )
    # Dedup tracking table
    conn.execute(
        "CREATE TABLE IF NOT EXISTS seen_ids "
        "(paper_id TEXT PRIMARY KEY, added_date TEXT)"
    )
    conn.commit()
    return conn


def _is_seen(conn, paper_id: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM seen_ids WHERE paper_id = ?", (paper_id,)
    ).fetchone()
    return row is not None


def _mark_seen(conn, paper_id: str):
    conn.execute(
        "INSERT OR IGNORE INTO seen_ids (paper_id, added_date) VALUES (?, ?)",
        (paper_id, datetime.now().isoformat()),
    )


def _add_paper(conn, paper_id: str, title: str, abstract: str,
               authors: str, source: str, url: str, date: str):
    """Add a research paper to the FTS5 index."""
    text = f"# {title}\n\nAuthors: {authors}\nDate: {date}\nSource: {source}\nURL: {url}\n\n{abstract}"
    normalized = _normalize(text)
    cid = hashlib.sha256(paper_id.encode()).hexdigest()

    conn.execute(
        "INSERT INTO chunks(search_content, source_path, chunk_id) "
        "VALUES (?, ?, ?)",
        (normalized, f"research_{paper_id}", cid),
    )
    _mark_seen(conn, paper_id)


# -----------------------------------------------------------------------
# ArXiv scanner
# -----------------------------------------------------------------------

def scan_arxiv(conn, days: int = 30) -> int:
    """Scan ArXiv for recent papers matching our research queries."""
    print("\n--- ARXIV SCAN ---")
    added = 0
    client = httpx.Client(timeout=30, follow_redirects=True)

    for query in ARXIV_QUERIES:
        safe_query = query[:60] + "..." if len(query) > 60 else query
        print(f"  Query: {safe_query}", end=" ", flush=True)

        try:
            # ArXiv API
            params = {
                "search_query": query,
                "start": 0,
                "max_results": 50,
                "sortBy": "submittedDate",
                "sortOrder": "descending",
            }
            r = client.get("https://export.arxiv.org/api/query", params=params,
                          follow_redirects=True)
            r.raise_for_status()

            # Parse Atom XML
            import xml.etree.ElementTree as ET
            root = ET.fromstring(r.text)
            ns = {"atom": "http://www.w3.org/2005/Atom"}

            entries = root.findall("atom:entry", ns)
            query_added = 0

            for entry in entries:
                arxiv_id = entry.find("atom:id", ns)
                title = entry.find("atom:title", ns)
                summary = entry.find("atom:summary", ns)
                published = entry.find("atom:published", ns)

                if arxiv_id is None or title is None or summary is None:
                    continue

                paper_id = arxiv_id.text.strip().split("/")[-1]
                if _is_seen(conn, paper_id):
                    continue

                # Check date filter
                if published is not None:
                    pub_date = published.text[:10]
                    try:
                        pub_dt = datetime.fromisoformat(pub_date)
                        if pub_dt < datetime.now() - timedelta(days=days):
                            continue
                    except ValueError:
                        pub_date = "unknown"
                else:
                    pub_date = "unknown"

                # Get authors
                author_elems = entry.findall("atom:author/atom:name", ns)
                authors = ", ".join(a.text for a in author_elems[:5])
                if len(author_elems) > 5:
                    authors += f" et al. ({len(author_elems)} authors)"

                _add_paper(
                    conn, paper_id,
                    title=title.text.strip().replace("\n", " "),
                    abstract=summary.text.strip().replace("\n", " "),
                    authors=authors,
                    source="arxiv",
                    url=f"https://arxiv.org/abs/{paper_id}",
                    date=pub_date,
                )
                query_added += 1
                added += 1

            print(f"-> {query_added} new papers")

            # Be polite to ArXiv API
            time.sleep(3)

        except Exception as exc:
            print(f"-> [FAIL] {exc}")

    client.close()
    conn.commit()
    return added


# -----------------------------------------------------------------------
# HuggingFace papers scanner
# -----------------------------------------------------------------------

def scan_hf_papers(conn, days: int = 30) -> int:
    """Scan HuggingFace daily papers feed."""
    print("\n--- HUGGINGFACE PAPERS ---")
    added = 0

    # Keywords to filter for relevant papers
    keywords = [
        "agent", "code", "rag", "retrieval", "distill",
        "reasoning", "self-learn", "self-improv", "self-play",
        "tool use", "function call", "chain of thought",
        "tree of thought", "reflection", "synthetic data",
        "code generation", "code review", "programming",
        "monte carlo", "search", "rerank",
    ]

    try:
        client = httpx.Client(timeout=30, follow_redirects=True)
        r = client.get(HF_PAPERS_URL)
        r.raise_for_status()
        papers = r.json()
        client.close()

        print(f"  Fetched {len(papers)} papers from HF daily feed")

        for paper in papers:
            paper_data = paper.get("paper", {})
            paper_id = paper_data.get("id", "")
            title = paper_data.get("title", "")
            summary = paper_data.get("summary", "")
            pub_date = paper.get("publishedAt", "")[:10]
            authors_list = paper_data.get("authors", [])

            if not paper_id or not title:
                continue
            if _is_seen(conn, f"hf_{paper_id}"):
                continue

            # Filter by keywords
            text_lower = (title + " " + summary).lower()
            if not any(kw in text_lower for kw in keywords):
                continue

            authors = ", ".join(
                a.get("name", "") for a in authors_list[:5]
            )

            _add_paper(
                conn, f"hf_{paper_id}",
                title=title,
                abstract=summary,
                authors=authors,
                source="huggingface",
                url=f"https://huggingface.co/papers/{paper_id}",
                date=pub_date,
            )
            added += 1

        print(f"  Added {added} relevant papers")

    except Exception as exc:
        print(f"  [FAIL] HF papers: {exc}")

    conn.commit()
    return added


# -----------------------------------------------------------------------
# GitHub trending scanner
# -----------------------------------------------------------------------

def scan_github_trending(conn) -> int:
    """Scan GitHub trending repos for AI/ML tools."""
    print("\n--- GITHUB TRENDING ---")
    added = 0

    # Use GitHub search API for recent popular repos
    queries = [
        "code+agent+language:python stars:>100",
        "rag+retrieval+augmented stars:>50",
        "code+assistant+llm stars:>100",
        "distillation+code stars:>50",
        "agentic+framework stars:>50",
    ]

    client = httpx.Client(timeout=30)

    for query in queries:
        print(f"  Query: {query[:50]}...", end=" ", flush=True)
        try:
            r = client.get(
                "https://api.github.com/search/repositories",
                params={
                    "q": query + " pushed:>2026-01-01",
                    "sort": "stars",
                    "order": "desc",
                    "per_page": 20,
                },
                headers={"Accept": "application/vnd.github.v3+json"},
            )
            r.raise_for_status()
            data = r.json()

            query_added = 0
            for repo in data.get("items", []):
                repo_id = f"gh_{repo['full_name']}"
                if _is_seen(conn, repo_id):
                    continue

                name = repo.get("full_name", "")
                desc = repo.get("description", "") or ""
                stars = repo.get("stargazers_count", 0)
                lang = repo.get("language", "")
                topics = ", ".join(repo.get("topics", []))
                url = repo.get("html_url", "")
                updated = repo.get("updated_at", "")[:10]

                text = (
                    f"# {name}\n\n"
                    f"Stars: {stars} | Language: {lang}\n"
                    f"Topics: {topics}\n"
                    f"URL: {url}\n"
                    f"Updated: {updated}\n\n"
                    f"{desc}"
                )

                normalized = _normalize(text)
                cid = hashlib.sha256(repo_id.encode()).hexdigest()
                conn.execute(
                    "INSERT INTO chunks(search_content, source_path, chunk_id) "
                    "VALUES (?, ?, ?)",
                    (normalized, f"research_{repo_id}", cid),
                )
                _mark_seen(conn, repo_id)
                query_added += 1
                added += 1

            print(f"-> {query_added} new repos")
            time.sleep(2)  # Rate limit

        except Exception as exc:
            print(f"-> [FAIL] {exc}")

    client.close()
    conn.commit()
    return added


# -----------------------------------------------------------------------
# Hacker News scanner (dev culture / trends pulse)
# -----------------------------------------------------------------------

def scan_hacker_news(conn) -> int:
    """Scan Hacker News for AI/coding culture and trends."""
    print("\n--- HACKER NEWS (dev culture) ---")
    added = 0

    # HN Algolia search API -- free, no auth needed
    keywords = [
        "agentic AI",
        "coding assistant",
        "RAG retrieval",
        "code generation LLM",
        "AI distillation",
        "self-improving AI",
        "chain of thought reasoning",
        "developer tools AI",
    ]

    client = httpx.Client(timeout=30, follow_redirects=True)

    for kw in keywords:
        print(f"  HN search: {kw}...", end=" ", flush=True)
        try:
            r = client.get(
                "https://hn.algolia.com/api/v1/search",
                params={
                    "query": kw,
                    "tags": "story",
                    "numericFilters": "points>20",
                    "hitsPerPage": 20,
                },
            )
            r.raise_for_status()
            data = r.json()

            kw_added = 0
            for hit in data.get("hits", []):
                hn_id = f"hn_{hit.get('objectID', '')}"
                if _is_seen(conn, hn_id):
                    continue

                title = hit.get("title", "")
                url = hit.get("url", "") or f"https://news.ycombinator.com/item?id={hit.get('objectID', '')}"
                points = hit.get("points", 0)
                comments = hit.get("num_comments", 0)
                date = hit.get("created_at", "")[:10]
                author = hit.get("author", "")

                text = (
                    f"# {title}\n\n"
                    f"Source: Hacker News | Points: {points} | Comments: {comments}\n"
                    f"Author: {author} | Date: {date}\n"
                    f"URL: {url}\n"
                )

                normalized = _normalize(text)
                cid = hashlib.sha256(hn_id.encode()).hexdigest()
                conn.execute(
                    "INSERT INTO chunks(search_content, source_path, chunk_id) "
                    "VALUES (?, ?, ?)",
                    (normalized, f"research_{hn_id}", cid),
                )
                _mark_seen(conn, hn_id)
                kw_added += 1
                added += 1

            print(f"-> {kw_added} stories")
            time.sleep(1)

        except Exception as exc:
            print(f"-> [FAIL] {exc}")

    client.close()
    conn.commit()
    return added


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    days = 30
    if len(sys.argv) > 2 and sys.argv[1] == "--days":
        days = int(sys.argv[2])

    print("=" * 60)
    print("JCoder Research Scanner")
    print(f"Index: {DB_PATH}")
    print(f"Lookback: {days} days")
    print(f"Topics: agentic AI, code assist, RAG, distillation, reasoning")
    print("=" * 60)

    t0 = time.time()
    conn = _init_db()

    # Count existing
    try:
        existing = conn.execute("SELECT COUNT(*) FROM seen_ids").fetchone()[0]
    except Exception:
        existing = 0
    print(f"\nExisting papers in index: {existing}")

    # Scan sources
    arxiv_added = scan_arxiv(conn, days=days)
    hf_added = scan_hf_papers(conn, days=days)
    gh_added = scan_github_trending(conn)
    hn_added = scan_hacker_news(conn)

    total_added = arxiv_added + hf_added + gh_added + hn_added
    elapsed = time.time() - t0

    # Final count
    try:
        total = conn.execute("SELECT COUNT(*) FROM seen_ids").fetchone()[0]
    except Exception:
        total = existing + total_added

    conn.close()

    size_mb = DB_PATH.stat().st_size / 1e6 if DB_PATH.exists() else 0

    print("\n" + "=" * 60)
    print("Summary")
    print(f"  ArXiv papers added:     {arxiv_added}")
    print(f"  HuggingFace papers:     {hf_added}")
    print(f"  GitHub repos:           {gh_added}")
    print(f"  Hacker News stories:    {hn_added}")
    print(f"  Total new:              {total_added}")
    print(f"  Total in index:         {total}")
    print(f"  Index size:             {size_mb:.1f} MB")
    print(f"  Elapsed:                {elapsed:.0f}s")
    print("=" * 60)
    print("\nRe-run periodically to stay current. Use --days N to adjust lookback.")


if __name__ == "__main__":
    main()
