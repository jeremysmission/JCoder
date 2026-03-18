"""
Claim Verifier (Lateral Reading)
---------------------------------
For key claims, auto-searches 2+ independent sources and compares.
Based on Stanford SHEG research: professional fact-checkers achieve
100% accuracy by leaving the page to verify; PhD historians only 50%.

The principle: never trust a single source. Verify through independent
cross-reference.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from .runtime import Runtime

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class VerificationResult:
    """Outcome of cross-source verification for a single claim."""
    claim: str
    verified: bool                          # True if 2+ independent sources corroborate
    corroborating_sources: List[Dict]       # sources that agree
    contradicting_sources: List[Dict]       # sources that disagree
    confidence: float                       # 0.0-1.0
    verification_method: str                # "cross_reference" | "citation_check" | "unverifiable"


# ---------------------------------------------------------------------------
# Stop words for heuristic query generation
# ---------------------------------------------------------------------------

_STOP_WORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but",
    "by", "can", "could", "did", "do", "does", "for", "from", "had",
    "has", "have", "he", "her", "him", "his", "how", "i", "if", "in",
    "into", "is", "it", "its", "may", "me", "might", "my", "no", "not",
    "of", "on", "or", "our", "shall", "she", "should", "so", "some",
    "that", "the", "their", "them", "then", "there", "these", "they",
    "this", "to", "us", "was", "we", "were", "what", "when", "where",
    "which", "who", "whom", "why", "will", "with", "would", "you",
    "your",
})


# ---------------------------------------------------------------------------
# Claim Verifier
# ---------------------------------------------------------------------------

class ClaimVerifier:
    """
    Cross-source fact verification via lateral reading.

    When a Runtime is available, uses the LLM to generate diverse search
    queries and to classify candidate sources.  Falls back to keyword
    heuristics when running offline (no Runtime).

    ``fetch_fn`` is the retrieval callback: (query: str) -> List[Dict].
    Each returned dict should contain at least ``title`` and ``abstract``
    (or ``snippet``) keys.  Without a fetch_fn the verifier immediately
    returns ``verification_method="unverifiable"``.
    """

    def __init__(
        self,
        runtime: Optional[Runtime] = None,
        fetch_fn: Optional[Callable[[str], List[Dict]]] = None,
    ) -> None:
        self._runtime = runtime
        self._fetch_fn = fetch_fn

    # -- public API --------------------------------------------------------

    def verify(
        self,
        claim: str,
        original_source: Optional[Dict] = None,
    ) -> VerificationResult:
        """
        Verify a single claim through lateral reading.

        Steps:
            1. Generate verification queries (rephrase for independent search).
            2. Fetch candidate sources via ``fetch_fn``.
            3. Classify each candidate as corroborating or contradicting.
            4. Compute confidence and verification status.
        """
        if self._fetch_fn is None:
            return VerificationResult(
                claim=claim,
                verified=False,
                corroborating_sources=[],
                contradicting_sources=[],
                confidence=0.0,
                verification_method="unverifiable",
            )

        # Step 1 -- generate diverse verification queries
        queries = self._generate_verification_queries(claim)

        # Step 2 -- fetch candidates from each query
        candidates: List[Dict] = []
        seen_titles: set[str] = set()
        for q in queries:
            for item in self._fetch_fn(q):
                title = str(item.get("title") or "").lower().strip()
                if title and title not in seen_titles:
                    candidates.append(item)
                    seen_titles.add(title)

        if not candidates:
            return VerificationResult(
                claim=claim,
                verified=False,
                corroborating_sources=[],
                contradicting_sources=[],
                confidence=0.0,
                verification_method="cross_reference",
            )

        # Step 3 -- classify each candidate
        corroborating, contradicting = self._cross_reference(claim, candidates)

        # Step 4 -- compute confidence and status
        confidence = self._compute_confidence(corroborating, contradicting)
        verified = len(corroborating) >= 2 and len(corroborating) > len(contradicting)

        return VerificationResult(
            claim=claim,
            verified=verified,
            corroborating_sources=corroborating,
            contradicting_sources=contradicting,
            confidence=confidence,
            verification_method="cross_reference",
        )

    def verify_batch(
        self,
        claims: List[str],
        original_sources: Optional[List[Dict]] = None,
    ) -> List[VerificationResult]:
        """Verify multiple claims.  Returns one result per claim."""
        sources_list = original_sources or [None] * len(claims)
        return [
            self.verify(claim, src)
            for claim, src in zip(claims, sources_list)
        ]

    # -- query generation --------------------------------------------------

    def _generate_verification_queries(self, claim: str) -> List[str]:
        """
        Produce 2-3 independent search queries for the claim.

        With a Runtime the LLM rephrases using different terminology.
        Without one, key nouns are extracted as a simple heuristic.
        """
        if self._runtime is not None:
            return self._generate_queries_llm(claim)
        return self._generate_queries_heuristic(claim)

    def _generate_queries_llm(self, claim: str) -> List[str]:
        """Use the LLM to rephrase the claim into diverse queries."""
        prompt_text = (
            "Rephrase this claim as 2-3 different search queries to find "
            "independent verification. Do NOT search for the claim itself. "
            "Use different terminology. One query per line.\n\n"
            f"Claim: {claim}"
        )
        try:
            raw = self._runtime.generate(
                question=prompt_text,
                context_chunks=[],
                system_prompt="You generate search queries. One per line, nothing else.",
                temperature=0.3,
                max_tokens=128,
            )
            lines = [ln.strip() for ln in raw.strip().splitlines() if ln.strip()]
            return lines[:3] if lines else [claim]
        except (AttributeError, Exception):
            log.warning("LLM query generation failed for claim verification, using heuristic", exc_info=True)
            return self._generate_queries_heuristic(claim)

    @staticmethod
    def _generate_queries_heuristic(claim: str) -> List[str]:
        """Extract key nouns from the claim and build simple queries."""
        words = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", claim)
        keywords = [w for w in words if w.lower() not in _STOP_WORDS and len(w) > 2]
        if not keywords:
            return [claim]
        # Build two queries: full keyword set and first-half subset
        full_query = " ".join(keywords)
        half = max(1, len(keywords) // 2)
        partial_query = " ".join(keywords[:half])
        queries = [full_query]
        if partial_query != full_query:
            queries.append(partial_query)
        return queries

    # -- cross-reference classification ------------------------------------

    def _cross_reference(
        self, claim: str, candidates: List[Dict]
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Classify candidates as corroborating or contradicting.

        Returns (corroborating, contradicting).
        """
        if self._runtime is not None:
            return self._cross_reference_llm(claim, candidates)
        return self._cross_reference_heuristic(claim, candidates)

    def _cross_reference_llm(
        self, claim: str, candidates: List[Dict]
    ) -> Tuple[List[Dict], List[Dict]]:
        """Use the LLM to judge each candidate against the claim."""
        corroborating: List[Dict] = []
        contradicting: List[Dict] = []

        for candidate in candidates:
            abstract = candidate.get("abstract", candidate.get("snippet", ""))
            title = candidate.get("title", "")
            prompt_text = (
                "Does this source support or contradict the claim? "
                "Reply SUPPORTS or CONTRADICTS or IRRELEVANT.\n\n"
                f"Claim: {claim}\n\n"
                f"Source title: {title}\n"
                f"Source content: {abstract[:500]}"
            )
            try:
                raw = self._runtime.generate(
                    question=prompt_text,
                    context_chunks=[],
                    system_prompt="Reply with one word: SUPPORTS, CONTRADICTS, or IRRELEVANT.",
                    temperature=0.0,
                    max_tokens=8,
                )
                label = raw.strip().upper()
                if "SUPPORT" in label:
                    corroborating.append(candidate)
                elif "CONTRADICT" in label:
                    contradicting.append(candidate)
                # IRRELEVANT sources are dropped silently
            except (AttributeError, Exception):
                log.debug("LLM cross-reference call failed for candidate", exc_info=True)
                continue

        return corroborating, contradicting

    @staticmethod
    def _cross_reference_heuristic(
        claim: str, candidates: List[Dict]
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Keyword-overlap heuristic for classification.

        High overlap with the claim tokens = corroborating.
        Very low overlap = contradicting (different perspective on same topic).
        """
        claim_tokens = set(re.findall(r"\w+", claim.lower()))
        if not claim_tokens:
            return [], []

        corroborating: List[Dict] = []
        contradicting: List[Dict] = []

        for candidate in candidates:
            text = "{} {}".format(
                candidate.get("title", ""),
                candidate.get("abstract", candidate.get("snippet", "")),
            )
            doc_tokens = set(re.findall(r"\w+", text.lower()))
            overlap = len(claim_tokens & doc_tokens) / len(claim_tokens)

            if overlap >= 0.4:
                corroborating.append(candidate)
            elif overlap < 0.15:
                contradicting.append(candidate)
            # 0.15 <= overlap < 0.4 treated as irrelevant

        return corroborating, contradicting

    # -- confidence computation --------------------------------------------

    @staticmethod
    def _compute_confidence(
        corroborating: List[Dict], contradicting: List[Dict]
    ) -> float:
        """
        Confidence score based on corroborating/contradicting counts.

        Scale: 0 corroborating = 0.0, 1 = 0.4, 2 = 0.7, 3+ = 0.9.
        Reduction: -0.15 per contradicting source.
        Clamped to [0.0, 1.0].
        """
        count = len(corroborating)
        if count == 0:
            base = 0.0
        elif count == 1:
            base = 0.4
        elif count == 2:
            base = 0.7
        else:
            base = 0.9

        reduction = 0.15 * len(contradicting)
        return max(0.0, min(1.0, base - reduction))
