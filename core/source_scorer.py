"""
Source Credibility Scorer (CRAAP Framework)
-------------------------------------------
Scores research sources on 5 dimensions: Currency, Relevance,
Authority, Accuracy, Purpose. Outputs 0.0-1.0 composite score.

Based on: Illinois State University CRAAP Test + UVA P.R.O.V.E.N. method.
Speed-optimized: 3 of 5 dimensions are pure heuristics (no LLM).
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Dict, List, Optional

from .runtime import Runtime


# ---------------------------------------------------------------------------
# Composite weights
# ---------------------------------------------------------------------------

_W_CURRENCY = 0.15
_W_RELEVANCE = 0.30
_W_AUTHORITY = 0.20
_W_ACCURACY = 0.20
_W_PURPOSE = 0.15


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class CredibilityScore:
    """Per-source credibility breakdown."""
    currency: float     # 0-1: recency relative to field pace
    relevance: float    # 0-1: semantic match to research question
    authority: float    # 0-1: domain TLD, citation count, known publisher
    accuracy: float     # 0-1: has references, peer-reviewed, code available
    purpose: float      # 0-1: academic vs commercial vs promotional
    composite: float    # weighted average


# ---------------------------------------------------------------------------
# Source tier mappings
# ---------------------------------------------------------------------------

_SOURCE_TIERS: Dict[str, float] = {
    "arxiv": 0.9,
    "arxiv_cs_ai": 0.9,
    "arxiv_cs_cl": 0.9,
    "arxiv_cs_ir": 0.9,
    "semantic_scholar": 0.9,
    "papers_with_code": 0.85,
    "openreview": 0.85,
    "acl_anthology": 0.9,
    "github": 0.5,
    "github_trending": 0.5,
    "pypi": 0.4,
    "pypi_new": 0.4,
    "hf_daily_papers": 0.6,
    "hacker_news": 0.3,
    "hn": 0.3,
    "reddit": 0.3,
    "blog": 0.3,
}

_ACADEMIC_URL_PATTERNS = ("arxiv", "acl", "neurips", "openreview", "icml",
                          "iclr", "aaai", "aclweb")
_IMPL_URL_PATTERNS = ("github", "pypi", "gitlab", "huggingface")
_OPINION_URL_PATTERNS = ("blog", "medium", "substack", "wordpress", "dev.to")
_CURRENT_YEAR = datetime.now(timezone.utc).year


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------

class SourceScorer:
    """
    CRAAP-based credibility scorer for research sources.

    Three of five dimensions (currency, authority, accuracy) are pure
    heuristics -- no network or LLM calls required.  Relevance uses
    keyword overlap as a fallback when no Runtime is provided, or an
    LLM prompt when one is available.  Purpose is always heuristic.
    """

    def __init__(self, runtime: Optional[Runtime] = None) -> None:
        self._runtime = runtime

    # -- public API --------------------------------------------------------

    def score(self, paper: Dict, query: str) -> CredibilityScore:
        """Score a single paper dict against *query*."""
        title = paper.get("title", "")
        abstract = paper.get("abstract", "")
        try:
            year = int(paper.get("year", 0) or 0)
        except (ValueError, TypeError, OverflowError):
            year = 0
        try:
            raw_cites = paper.get("citation_count", 0) or 0
            if isinstance(raw_cites, float) and (raw_cites != raw_cites or raw_cites == float('inf') or raw_cites == float('-inf')):
                citations = 0
            else:
                citations = int(raw_cites)
        except (ValueError, TypeError, OverflowError):
            citations = 0
        source = paper.get("source", "")
        url = paper.get("url", "")
        has_refs = bool(paper.get("has_refs", False))
        has_code = bool(paper.get("has_code", False))
        peer_reviewed = bool(paper.get("peer_reviewed", False))

        currency = self._score_currency(year)
        authority = self._score_authority(source, citations, url)
        accuracy = self._score_accuracy(has_refs, has_code, peer_reviewed)
        purpose = self._score_purpose(abstract, url)

        if self._runtime is not None:
            relevance = self._score_relevance_llm(title, abstract, query)
        else:
            relevance = self._score_relevance_heuristic(title, abstract, query)

        composite = max(0.0, min(1.0,
            _W_CURRENCY * currency
            + _W_RELEVANCE * relevance
            + _W_AUTHORITY * authority
            + _W_ACCURACY * accuracy
            + _W_PURPOSE * purpose
        ))

        return CredibilityScore(
            currency=round(max(0.0, min(1.0, currency)), 4),
            relevance=round(max(0.0, min(1.0, relevance)), 4),
            authority=round(max(0.0, min(1.0, authority)), 4),
            accuracy=round(max(0.0, min(1.0, accuracy)), 4),
            purpose=round(max(0.0, min(1.0, purpose)), 4),
            composite=round(composite, 4),
        )

    def score_batch(
        self, papers: List[Dict], query: str
    ) -> List[CredibilityScore]:
        """Score multiple papers. Returns one CredibilityScore per paper."""
        return [self.score(p, query) for p in papers]

    # -- heuristic dimensions (no LLM) ------------------------------------

    @staticmethod
    def _score_currency(year: int) -> float:
        """Recency score anchored to current year."""
        table = {
            _CURRENT_YEAR: 1.0,
            _CURRENT_YEAR - 1: 0.85,
            _CURRENT_YEAR - 2: 0.7,
            _CURRENT_YEAR - 3: 0.5,
            _CURRENT_YEAR - 4: 0.3,
        }
        if year > _CURRENT_YEAR:
            return 1.0
        return table.get(year, 0.1)

    @staticmethod
    def _score_authority(source: str, citations: int, url: str) -> float:
        """Source tier + citation boost + TLD boost."""
        base = _SOURCE_TIERS.get(source.lower(), 0.4)

        # Citation boost: up to 0.3, scaled by citations/1000
        citation_boost = min(0.3, max(0, citations) / 1000)

        # TLD boost
        tld_boost = 0.0
        if url:
            lower_url = url.lower()
            if ".edu" in lower_url:
                tld_boost = 0.1
            elif ".gov" in lower_url:
                tld_boost = 0.1
            elif ".org" in lower_url:
                tld_boost = 0.05

        return min(1.0, base + citation_boost + tld_boost)

    @staticmethod
    def _score_accuracy(
        has_refs: bool, has_code: bool, peer_reviewed: bool
    ) -> float:
        """Evidence quality.  Base 0.3 + optional bonuses."""
        return 0.3 + has_refs * 0.3 + has_code * 0.2 + peer_reviewed * 0.2

    @staticmethod
    def _score_purpose(abstract: str, url: str) -> float:
        """Infer intent from URL patterns and abstract language."""
        lower_url = url.lower() if url else ""

        if any(p in lower_url for p in _ACADEMIC_URL_PATTERNS):
            return 0.9

        if any(p in lower_url for p in _IMPL_URL_PATTERNS):
            return 0.7

        if any(p in lower_url for p in _OPINION_URL_PATTERNS):
            return 0.4

        return 0.5

    # -- relevance (heuristic fallback) ------------------------------------

    @staticmethod
    def _score_relevance_heuristic(
        title: str, abstract: str, query: str
    ) -> float:
        """Keyword overlap ratio -- used when no Runtime is available."""
        query_tokens = set(re.findall(r"\w+", query.lower()))
        if not query_tokens:
            return 0.0
        doc_tokens = set(re.findall(r"\w+", f"{title} {abstract}".lower()))
        overlap = query_tokens & doc_tokens
        return len(overlap) / len(query_tokens)

    # -- relevance (LLM) --------------------------------------------------

    def _score_relevance_llm(
        self, title: str, abstract: str, query: str
    ) -> float:
        """Ask the LLM to rate relevance 0.0-1.0."""
        prompt_text = (
            "Rate 0.0 to 1.0 how relevant this paper is to the query. "
            "Reply with only a number.\n\n"
            f"Paper title: {title}\n"
            f"Abstract: {abstract[:500]}\n\n"
            f"Query: {query}"
        )
        try:
            raw = self._runtime.generate(
                question=prompt_text,
                context_chunks=[],
                system_prompt="You are a relevance scorer. Reply with a single float.",
                temperature=0.0,
                max_tokens=8,
            )
            value = float(raw.strip())
            return max(0.0, min(1.0, value))
        except (ValueError, TypeError, AttributeError):
            # LLM returned non-numeric or call failed -- fall back
            return self._score_relevance_heuristic(title, abstract, query)
