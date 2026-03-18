"""
Devil's Advocate Engine
-----------------------
Given a claim or finding, automatically generates counter-queries,
retrieves opposing evidence, and computes a balance score.

Based on: Confirmation bias research (ATLAS.ti, PMC 2024),
pre-registration principles, and steel-man argumentation.

The single most effective anti-bias technique: actively seek
disconfirming evidence. This module automates that.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from core.runtime import Runtime

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LLM prompts
# ---------------------------------------------------------------------------

_COUNTER_QUERY_PROMPT = (
    "Generate {max_n} search queries that would find evidence AGAINST "
    "this claim: '{claim}'. Include queries like '[claim] debunked', "
    "'[claim] criticism', '[claim] limitations'. "
    "Return one query per line."
)

_BLIND_SPOT_PROMPT = (
    "Given claim '{claim}' and these source titles, what perspectives "
    "or subtopics are NOT covered? List 1-3 blind spots, one per line."
)

# Heuristic suffixes used when no LLM is available.
_HEURISTIC_SUFFIXES = (" criticism", " limitations", " debunked")


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class BalanceReport:
    """Output of a devil's-advocate challenge against a claim."""

    original_claim: str
    counter_queries: List[str]
    supporting_count: int
    opposing_count: int
    balance_ratio: float            # 0.0 = only supporting, 1.0 = balanced
    strongest_counter: str          # title of best opposing evidence
    blind_spots: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class DevilsAdvocate:
    """
    Anti-bias engine that actively seeks disconfirming evidence.

    Works in three modes depending on what is provided:
      * runtime + fetch_fn  -- full LLM-driven counter-query + retrieval
      * fetch_fn only       -- heuristic counter-queries + retrieval
      * neither             -- heuristic queries, empty opposing list
    """

    def __init__(
        self,
        runtime: Optional[Runtime] = None,
        fetch_fn: Optional[Callable] = None,
    ) -> None:
        self._runtime = runtime
        self._fetch_fn = fetch_fn

    # -- public API --------------------------------------------------------

    def challenge(
        self,
        claim: str,
        supporting_papers: List[Dict],
        max_counter_queries: int = 3,
    ) -> BalanceReport:
        """
        Run a full devil's-advocate challenge.

        Steps:
          1. Generate counter-queries (LLM or heuristic).
          2. Fetch opposing evidence via *fetch_fn*.
          3. Compute a balance ratio.
          4. Identify blind spots (LLM only).
        """
        counter_queries = self._generate_counter_queries(
            claim, max_n=max_counter_queries,
        )
        opposing = self._fetch_opposing(counter_queries)

        balance = self._compute_balance(supporting_papers, opposing)

        all_papers = supporting_papers + opposing
        blind_spots = self._identify_blind_spots(claim, all_papers)

        strongest = ""
        if opposing:
            strongest = opposing[0].get("title", "")

        return BalanceReport(
            original_claim=claim,
            counter_queries=counter_queries,
            supporting_count=len(supporting_papers),
            opposing_count=len(opposing),
            balance_ratio=balance,
            strongest_counter=strongest,
            blind_spots=blind_spots,
        )

    # -- counter-query generation ------------------------------------------

    def _generate_counter_queries(
        self, claim: str, max_n: int = 3,
    ) -> List[str]:
        """
        Build search queries designed to surface disconfirming evidence.

        When a Runtime is available the LLM crafts nuanced queries.
        Otherwise we fall back to simple heuristic suffixes.
        """
        if self._runtime is not None:
            return self._counter_queries_llm(claim, max_n)
        return self._counter_queries_heuristic(claim, max_n)

    def _counter_queries_llm(
        self, claim: str, max_n: int,
    ) -> List[str]:
        """Ask the LLM for counter-queries."""
        prompt_text = _COUNTER_QUERY_PROMPT.format(
            max_n=max_n, claim=claim,
        )
        try:
            raw = self._runtime.generate(
                question=prompt_text,
                context_chunks=[],
                system_prompt=(
                    "You generate search queries. "
                    "Return one query per line, nothing else."
                ),
                temperature=0.3,
                max_tokens=256,
            )
            lines = [
                ln.strip() for ln in raw.strip().splitlines() if ln.strip()
            ]
            return lines[:max_n]
        except Exception:
            log.warning("LLM counter-query generation failed, using heuristic", exc_info=True)
            return self._counter_queries_heuristic(claim, max_n)

    @staticmethod
    def _counter_queries_heuristic(
        claim: str, max_n: int,
    ) -> List[str]:
        """Append standard critical suffixes to the claim text."""
        queries = [claim + suffix for suffix in _HEURISTIC_SUFFIXES]
        return queries[:max_n]

    # -- opposing evidence retrieval ---------------------------------------

    def _fetch_opposing(self, queries: List[str]) -> List[Dict]:
        """
        Call *fetch_fn* for each counter-query and deduplicate results.

        Deduplication uses the 'title' key when present.
        """
        if self._fetch_fn is None:
            return []

        seen_titles: set = set()
        results: List[Dict] = []

        for query in queries:
            papers = self._fetch_fn(query)
            for paper in papers:
                title = paper.get("title", "")
                if title and title in seen_titles:
                    continue
                if title:
                    seen_titles.add(title)
                results.append(paper)

        return results

    # -- balance computation -----------------------------------------------

    @staticmethod
    def _compute_balance(
        supporting: List[Dict],
        opposing: List[Dict],
    ) -> float:
        """
        Ratio of opposing to total sources.

        Returns 0.0 when no sources exist or all are supporting.
        Returns 1.0 when evidence is perfectly balanced.
        Clamped to [0.0, 1.0].
        """
        total = len(supporting) + len(opposing)
        if total == 0:
            return 0.0
        ratio = len(opposing) / total
        return max(0.0, min(1.0, ratio))

    # -- blind spot detection ----------------------------------------------

    def _identify_blind_spots(
        self, claim: str, all_papers: List[Dict],
    ) -> List[str]:
        """
        Use the LLM to identify perspectives not covered by any source.

        Returns an empty list when no Runtime is configured.
        """
        if self._runtime is None:
            return []

        titles = [p.get("title", "") for p in all_papers if p.get("title")]
        title_block = "\n".join(titles) if titles else "(no sources)"
        prompt_text = _BLIND_SPOT_PROMPT.format(claim=claim)
        context = f"Source titles:\n{title_block}"

        try:
            raw = self._runtime.generate(
                question=prompt_text,
                context_chunks=[context],
                system_prompt=(
                    "You identify research blind spots. "
                    "Return 1-3 items, one per line."
                ),
                temperature=0.3,
                max_tokens=256,
            )
            lines = [
                ln.strip() for ln in raw.strip().splitlines() if ln.strip()
            ]
            return lines[:3]
        except Exception:
            log.warning("Blind spot identification failed", exc_info=True)
            return []
