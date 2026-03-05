"""
Layered Triage Engine (Satellite / Drone / Deep Dive)
------------------------------------------------------
3-layer speed reading that replaces flat triage with a
progressive filtering pipeline. Based on MIT/Oxford Layered
Preview method which cuts paper review from 60 to 22 minutes.

Layer 1 - Satellite (heuristic only, <1 sec per 100 papers):
  Title keywords, year, source tier, citation count.

Layer 2 - Drone (1 batched LLM call on abstracts):
  Relevance scoring + 1-sentence summary.

Layer 3 - Deep Dive (only top N survive):
  Flag for full digest by RapidDigester.

Speed principle: eliminate 70% of papers before any LLM call.
"""

from __future__ import annotations

import hashlib
import re
import time
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from core.runtime import Runtime


# ---------------------------------------------------------------------------
# Keyword banks for satellite-pass heuristic scoring
# ---------------------------------------------------------------------------

_HIGH_SIGNAL_KEYWORDS = {
    "novel", "state-of-the-art", "sota", "breakthrough", "surpass",
    "outperform", "self-improving", "self-learning", "retrieval augmented",
    "rag", "meta-learning", "evolutionary", "benchmark", "empirical",
}

_METHODOLOGY_KEYWORDS = {
    "systematic review", "meta-analysis", "survey", "framework",
    "methodology", "pipeline", "architecture", "ablation",
    "reproducible", "open source", "replication",
}

# Source tier defaults (mirrors research_sources.py tiers)
_SOURCE_TIER_SCORES = {
    "arxiv_cs_ai": 0.90, "arxiv_cs_cl": 0.80, "arxiv_cs_ir": 0.90,
    "arxiv_cs_se": 0.80, "arxiv_cs_lg": 0.85,
    "semantic_scholar": 0.90, "papers_with_code": 0.85,
    "openreview": 0.80, "acl_anthology": 0.85,
    "github_trending": 0.50, "pypi_new": 0.40,
    "hf_daily_papers": 0.75,
    "hacker_news": 0.30, "reddit": 0.25,
    "deepmind_blog": 0.60, "ms_research_blog": 0.60,
    "connected_papers": 0.70,
}
_CURRENT_YEAR = datetime.now(timezone.utc).year


@dataclass
class TriageResult:
    """Result of layered triage for a single paper."""
    content_hash: str
    title: str
    # Satellite pass (heuristic only)
    satellite_score: float
    satellite_pass: bool
    # Drone pass (LLM on abstract)
    drone_score: float = 0.0
    drone_summary: str = ""
    drone_pass: bool = False
    # Deep dive flag
    deep_dive: bool = False


class LayeredTriage:
    """
    Progressive paper filtering: Satellite -> Drone -> Deep Dive.

    Satellite eliminates ~70% with zero LLM cost.
    Drone scores the survivors in one batched LLM call.
    Deep Dive flags the top N for full digest.
    """

    def __init__(
        self,
        runtime: Optional[Runtime] = None,
    ):
        self.runtime = runtime

    def triage_batch(
        self,
        papers: List[Dict[str, Any]],
        query: str,
        satellite_cutoff: float = 0.3,
        drone_cutoff: float = 0.5,
        max_deep_dive: int = 5,
    ) -> List[TriageResult]:
        """
        Run all three triage layers on a batch of papers.

        Args:
            papers: list of dicts with keys: title, abstract, year,
                    citation_count, source, url (all optional)
            query: the research question driving this sprint
            satellite_cutoff: minimum satellite score to survive (0-1)
            drone_cutoff: minimum drone score to survive (0-1)
            max_deep_dive: max papers flagged for full digest

        Returns:
            All TriageResults, sorted by drone_score descending.
            Only top papers have deep_dive=True.
        """
        if not papers:
            return []

        # Layer 1: Satellite (pure heuristic)
        results = self._satellite_pass(papers, query, cutoff=satellite_cutoff)

        # Filter
        survivors = [r for r in results if r.satellite_pass]
        survivor_papers = []
        hash_to_paper = {}
        for p in papers:
            h = _content_hash(p)
            hash_to_paper[h] = p
        for r in survivors:
            if r.content_hash in hash_to_paper:
                survivor_papers.append(hash_to_paper[r.content_hash])

        # Layer 2: Drone (LLM on abstracts)
        if survivors and self.runtime:
            survivors = self._drone_pass(survivors, survivor_papers, query)
        elif survivors:
            # No runtime: promote all satellite survivors with satellite score
            for r in survivors:
                r.drone_score = r.satellite_score
                r.drone_summary = ""
                r.drone_pass = r.satellite_score >= drone_cutoff

        # Apply drone cutoff
        for r in results:
            if r.content_hash not in {s.content_hash for s in survivors}:
                continue
            match = next((s for s in survivors if s.content_hash == r.content_hash), None)
            if match:
                r.drone_score = match.drone_score
                r.drone_summary = match.drone_summary
                r.drone_pass = match.drone_score >= drone_cutoff

        # Layer 3: Select deep dive candidates
        drone_survivors = sorted(
            [r for r in results if r.drone_pass],
            key=lambda r: r.drone_score,
            reverse=True,
        )
        for i, r in enumerate(drone_survivors):
            if i < max_deep_dive:
                r.deep_dive = True

        # Sort all results by best available score
        results.sort(
            key=lambda r: r.drone_score if r.drone_pass else r.satellite_score,
            reverse=True,
        )
        return results

    def _satellite_pass(
        self,
        papers: List[Dict[str, Any]],
        query: str,
        cutoff: float = 0.3,
    ) -> List[TriageResult]:
        """
        Layer 1: Pure heuristic scoring. No LLM calls.

        Scores based on: title keyword relevance, year recency,
        source tier, citation count, and query keyword overlap.
        """
        query_words = set(re.findall(r"\w{3,}", query.lower()))
        results = []

        for paper in papers:
            title = str(paper.get("title") or "")
            title_lower = title.lower()
            try:
                year = int(paper.get("year") or 2020)
            except (ValueError, TypeError, OverflowError):
                year = 2020
            try:
                raw_cites = paper.get("citation_count") or 0
                citations = int(raw_cites) if not (isinstance(raw_cites, float) and (raw_cites != raw_cites or raw_cites == float('inf') or raw_cites == float('-inf'))) else 0
            except (ValueError, TypeError, OverflowError):
                citations = 0
            source = str(paper.get("source") or "")
            abstract = str(paper.get("abstract") or "")

            score = 0.0

            # Recency (0-0.25)
            age = max(0, _CURRENT_YEAR - year)
            score += max(0.0, 0.25 - age * 0.06)

            # Source tier (0-0.20)
            tier_score = _SOURCE_TIER_SCORES.get(source, 0.5)
            score += tier_score * 0.20

            # High-signal keywords in title (0-0.20)
            title_words = set(re.findall(r"\w{3,}", title_lower))
            hit_count = len(title_words & _HIGH_SIGNAL_KEYWORDS)
            score += min(0.20, hit_count * 0.07)

            # Methodology keywords in title or abstract (0-0.10)
            combined = title_lower + " " + abstract[:300].lower()
            method_hits = sum(1 for kw in _METHODOLOGY_KEYWORDS if kw in combined)
            score += min(0.10, method_hits * 0.04)

            # Query keyword overlap (0-0.15)
            if query_words:
                all_words = set(re.findall(r"\w{3,}", combined))
                overlap = len(query_words & all_words) / len(query_words)
                score += overlap * 0.15

            # Citation boost (0-0.10)
            if citations > 0:
                score += min(0.10, citations / 500 * 0.10)

            score = min(1.0, score)
            content_h = _content_hash(paper)

            results.append(TriageResult(
                content_hash=content_h,
                title=title,
                satellite_score=round(score, 4),
                satellite_pass=score >= cutoff,
            ))

        return results

    def _drone_pass(
        self,
        survivors: List[TriageResult],
        papers: List[Dict[str, Any]],
        query: str,
    ) -> List[TriageResult]:
        """
        Layer 2: Single batched LLM call on abstracts of survivors.

        Asks LLM to rate relevance 0.0-1.0 and provide 1-sentence summary
        for each paper.
        """
        # Build batch prompt
        entries = []
        for i, paper in enumerate(papers):
            title = str(paper.get("title") or "Unknown")
            abstract = str(paper.get("abstract") or "")[:500]
            entries.append(f"[{i}] {title}\n{abstract}")

        batch_text = "\n---\n".join(entries)

        prompt = (
            f"Research query: {query}\n\n"
            f"Rate each paper's relevance to the query (0.0 to 1.0) "
            f"and write a 1-sentence summary.\n\n"
            f"Papers:\n{batch_text}\n\n"
            f"Reply as one line per paper: INDEX|SCORE|SUMMARY\n"
            f"Example: 0|0.85|Proposes a novel method for X."
        )

        try:
            response = self.runtime.generate(
                question=prompt,
                context_chunks=[],
                system_prompt="You are a research triage assistant. Be precise with scores.",
                temperature=0.1,
                max_tokens=len(papers) * 80,
            )
            parsed = _parse_drone_response(response, len(papers))
        except Exception:
            # LLM failure: fall back to satellite scores
            parsed = {}

        for i, result in enumerate(survivors):
            if i in parsed:
                result.drone_score = parsed[i]["score"]
                result.drone_summary = parsed[i]["summary"]
            else:
                result.drone_score = result.satellite_score
                result.drone_summary = ""
            result.drone_pass = True  # cutoff applied by caller

        return survivors


def _content_hash(paper: Dict) -> str:
    """SHA-256 hash of title + first 200 chars of abstract."""
    title = str(paper.get("title") or "")
    abstract = str(paper.get("abstract") or "")[:200]
    raw = f"{title}:{abstract}".encode("utf-8", errors="replace")
    return hashlib.sha256(raw).hexdigest()[:16]


def _parse_drone_response(text: str, expected: int) -> Dict[int, Dict]:
    """Parse INDEX|SCORE|SUMMARY lines from LLM response."""
    results = {}
    for line in text.strip().splitlines():
        line = line.strip()
        if not line or "|" not in line:
            continue
        parts = line.split("|", 2)
        if len(parts) < 3:
            continue
        try:
            idx = int(parts[0].strip().strip("[]"))
            score = float(parts[1].strip())
            summary = parts[2].strip()
            if 0 <= idx < expected and 0.0 <= score <= 1.0:
                results[idx] = {"score": score, "summary": summary}
        except (ValueError, IndexError):
            continue
    return results
