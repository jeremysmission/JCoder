"""
Research Source Registry & Novelty Scorer
-------------------------------------------
Ranked registry of the world's best free academic and research sources,
with programmatic extraction strategies and novelty scoring.

All sources are FREE (no API keys required for basic access).
Each source has a dedicated parser and relevance scorer.

Tier 1 -- Primary Research (highest signal, peer-reviewed):
  1. arXiv (cs.AI, cs.CL, cs.LG, cs.SE, cs.IR)
  2. Semantic Scholar (API: 100 req/5min free)
  3. Papers With Code (API: free, links papers to implementations)
  4. OpenReview (ICLR/NeurIPS/ICML submissions, with reviews)
  5. ACL Anthology (NLP-specific, all open access)

Tier 2 -- Implementation & Code (papers + working code):
  6. GitHub (trending, topics, releases API)
  7. PyPI (new packages, version bumps)
  8. HuggingFace Daily Papers (curated by community)

Tier 3 -- Community Intelligence (fast, noisy):
  9. Hacker News (Algolia API: free, real-time)
  10. Reddit (pushshift API, r/MachineLearning)

Tier 4 -- Corporate Research (high quality, delayed):
  11. Google DeepMind blog
  12. Microsoft Research blog

Tier 5 -- Aggregators (pre-filtered):
  13. Connected Papers (citation graph API)
  14. AK's Daily Papers (@_akhaliq summaries)
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Source definitions
# ---------------------------------------------------------------------------

@dataclass
class ResearchSource:
    """A registered research source with metadata."""
    name: str
    tier: int  # 1-5 (lower = higher quality)
    url_template: str  # URL with {query} placeholder
    update_freq_hours: float  # how often new content appears
    signal_to_noise: float  # 0.0-1.0 (higher = better)
    parser_name: str  # which parser function to use
    requires_key: bool = False
    rate_limit_per_min: int = 10
    relevance_keywords: List[str] = field(default_factory=list)


# Pre-configured source registry (ranked by value)
SOURCES = [
    ResearchSource(
        name="arxiv_cs_ai",
        tier=1,
        url_template="https://export.arxiv.org/api/query?search_query=cat:cs.AI+AND+{query}&sortBy=submittedDate&sortOrder=descending&max_results=25",
        update_freq_hours=1.0,
        signal_to_noise=0.7,
        parser_name="arxiv_atom",
        rate_limit_per_min=5,
        relevance_keywords=["retrieval", "RAG", "code generation", "self-improving"],
    ),
    ResearchSource(
        name="arxiv_cs_cl",
        tier=1,
        url_template="https://export.arxiv.org/api/query?search_query=cat:cs.CL+AND+{query}&sortBy=submittedDate&sortOrder=descending&max_results=25",
        update_freq_hours=1.0,
        signal_to_noise=0.65,
        parser_name="arxiv_atom",
        rate_limit_per_min=5,
        relevance_keywords=["language model", "prompt", "self-learning"],
    ),
    ResearchSource(
        name="arxiv_cs_ir",
        tier=1,
        url_template="https://export.arxiv.org/api/query?search_query=cat:cs.IR+AND+{query}&sortBy=submittedDate&sortOrder=descending&max_results=25",
        update_freq_hours=2.0,
        signal_to_noise=0.8,
        parser_name="arxiv_atom",
        rate_limit_per_min=5,
        relevance_keywords=["retrieval augmented", "embedding", "reranking"],
    ),
    ResearchSource(
        name="semantic_scholar",
        tier=1,
        url_template="https://api.semanticscholar.org/graph/v1/paper/search?query={query}&limit=20&fields=title,abstract,year,citationCount,openAccessPdf,externalIds",
        update_freq_hours=24.0,
        signal_to_noise=0.9,
        parser_name="semantic_scholar",
        rate_limit_per_min=20,
        relevance_keywords=["RAG", "code", "self-learning"],
    ),
    ResearchSource(
        name="papers_with_code",
        tier=1,
        url_template="https://paperswithcode.com/api/v1/papers/?q={query}&ordering=-proceeding&items_per_page=20",
        update_freq_hours=12.0,
        signal_to_noise=0.95,
        parser_name="papers_with_code",
        rate_limit_per_min=10,
        relevance_keywords=["benchmark", "implementation", "code"],
    ),
    ResearchSource(
        name="github_trending",
        tier=2,
        url_template="https://api.github.com/search/repositories?q={query}+language:python&sort=stars&order=desc&per_page=20",
        update_freq_hours=6.0,
        signal_to_noise=0.5,
        parser_name="github_search",
        rate_limit_per_min=10,
        relevance_keywords=["rag", "llm", "retrieval", "self-improving"],
    ),
    ResearchSource(
        name="pypi_new",
        tier=2,
        url_template="https://pypi.org/pypi/{query}/json",
        update_freq_hours=24.0,
        signal_to_noise=0.4,
        parser_name="pypi_package",
        rate_limit_per_min=30,
    ),
    ResearchSource(
        name="hf_daily_papers",
        tier=2,
        url_template="https://huggingface.co/api/daily_papers?limit=30",
        update_freq_hours=24.0,
        signal_to_noise=0.85,
        parser_name="hf_daily",
        rate_limit_per_min=10,
    ),
    ResearchSource(
        name="hacker_news",
        tier=3,
        url_template="https://hn.algolia.com/api/v1/search_by_date?query={query}&tags=story&hitsPerPage=25",
        update_freq_hours=0.5,
        signal_to_noise=0.3,
        parser_name="hn_algolia",
        rate_limit_per_min=30,
    ),
]


# ---------------------------------------------------------------------------
# Novelty scoring
# ---------------------------------------------------------------------------

def novelty_score(
    title: str,
    abstract: str,
    year: int,
    citation_count: int,
    known_hashes: set,
) -> float:
    """
    Score how novel/important a paper is. Higher = more novel.

    Signals:
    1. Recency (2026 > 2025 > 2024)
    2. Keyword density for target topics
    3. Citation velocity (citations / age in months)
    4. Deduplication (seen before = 0)
    """
    # Dedup check
    content_hash = hashlib.sha256(
        f"{title}:{abstract[:200]}".encode()).hexdigest()[:16]
    if content_hash in known_hashes:
        return 0.0
    known_hashes.add(content_hash)

    score = 0.0

    # Recency (0.0-0.3)
    current_year = 2026
    age_years = max(0, current_year - year)
    recency = max(0.0, 0.3 - age_years * 0.1)
    score += recency

    # Keyword relevance (0.0-0.4)
    target_keywords = {
        "self-improving": 0.08, "self-learning": 0.08,
        "retrieval augmented": 0.06, "rag": 0.06,
        "code generation": 0.05, "evolutionary": 0.05,
        "curriculum": 0.04, "meta-learning": 0.04,
        "reflection": 0.03, "corrective": 0.03,
        "adversarial": 0.03, "self-play": 0.03,
        "prompt optimization": 0.04, "quality diversity": 0.03,
    }
    text = f"{title} {abstract}".lower()
    kw_score = sum(
        weight for kw, weight in target_keywords.items()
        if kw in text
    )
    score += min(0.4, kw_score)

    # Citation velocity (0.0-0.2)
    if year >= 2025 and citation_count > 0:
        months = max(1, (current_year - year) * 12 + 6)
        velocity = citation_count / months
        score += min(0.2, velocity * 0.02)
    elif citation_count > 50:
        score += 0.1

    # Title quality bonus (0.0-0.1)
    if any(w in title.lower() for w in
           ["breakthrough", "novel", "state-of-the-art", "sota", "surpass"]):
        score += 0.05

    return min(1.0, score)


# ---------------------------------------------------------------------------
# Parsed result
# ---------------------------------------------------------------------------

@dataclass
class ParsedPaper:
    """A parsed research paper/item from any source."""
    source: str
    title: str
    abstract: str
    url: str
    year: int = 2026
    citation_count: int = 0
    has_code: bool = False
    code_url: str = ""
    novelty: float = 0.0
    relevance_keywords_hit: List[str] = field(default_factory=list)
    fetched_at: float = 0.0


# ---------------------------------------------------------------------------
# Result store
# ---------------------------------------------------------------------------

class ResearchStore:
    """SQLite store for harvested research papers with novelty tracking."""

    def __init__(self, db_path: str = "_research/research.db"):
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self._db_path))

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS papers (
                    content_hash TEXT PRIMARY KEY,
                    source TEXT,
                    title TEXT,
                    abstract TEXT,
                    url TEXT,
                    year INTEGER,
                    citation_count INTEGER,
                    has_code INTEGER,
                    code_url TEXT,
                    novelty REAL,
                    keywords_json TEXT,
                    fetched_at REAL,
                    processed INTEGER DEFAULT 0
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_papers_novelty
                ON papers(novelty DESC)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_papers_source
                ON papers(source)
            """)
            conn.commit()

    def store(self, paper: ParsedPaper) -> bool:
        """Store a paper. Returns True if new, False if duplicate."""
        content_hash = hashlib.sha256(
            f"{paper.title}:{paper.abstract[:200]}".encode()
        ).hexdigest()[:16]

        with self._connect() as conn:
            cur = conn.execute(
                "SELECT content_hash FROM papers WHERE content_hash = ?",
                (content_hash,),
            )
            if cur.fetchone():
                return False  # duplicate

            conn.execute("""
                INSERT INTO papers
                (content_hash, source, title, abstract, url, year,
                 citation_count, has_code, code_url, novelty,
                 keywords_json, fetched_at, processed)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
            """, (
                content_hash, paper.source, paper.title,
                paper.abstract[:2000], paper.url, paper.year,
                paper.citation_count, 1 if paper.has_code else 0,
                paper.code_url, paper.novelty,
                json.dumps(paper.relevance_keywords_hit),
                paper.fetched_at,
            ))
            conn.commit()
            return True

    def top_novel(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Return most novel unprocessed papers."""
        with self._connect() as conn:
            cur = conn.execute(
                "SELECT source, title, abstract, url, novelty, has_code, code_url "
                "FROM papers WHERE processed = 0 "
                "ORDER BY novelty DESC LIMIT ?",
                (limit,),
            )
            return [
                {
                    "source": r[0], "title": r[1],
                    "abstract": r[2][:300], "url": r[3],
                    "novelty": round(r[4], 3),
                    "has_code": bool(r[5]), "code_url": r[6],
                }
                for r in cur.fetchall()
            ]

    def mark_processed(self, title: str) -> None:
        content_hash = hashlib.sha256(title.encode()).hexdigest()[:16]
        with self._connect() as conn:
            conn.execute(
                "UPDATE papers SET processed = 1 WHERE content_hash = ?",
                (content_hash,),
            )
            conn.commit()

    def stats(self) -> Dict[str, Any]:
        with self._connect() as conn:
            cur = conn.execute("""
                SELECT source, COUNT(*), AVG(novelty),
                       SUM(CASE WHEN has_code = 1 THEN 1 ELSE 0 END)
                FROM papers GROUP BY source ORDER BY AVG(novelty) DESC
            """)
            rows = cur.fetchall()

            total = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
            unprocessed = conn.execute(
                "SELECT COUNT(*) FROM papers WHERE processed = 0"
            ).fetchone()[0]

        return {
            "total_papers": total,
            "unprocessed": unprocessed,
            "by_source": [
                {
                    "source": r[0], "count": r[1],
                    "avg_novelty": round(r[2], 3),
                    "with_code": r[3],
                }
                for r in rows
            ],
        }
