"""
Evidence Weighter (Inverse-Variance Method)
--------------------------------------------
When multiple sources discuss the same claim, weight by quality
instead of treating equally. Based on Hedges & Olkin (1985)
Statistical Methods for Meta-Analysis.

Pure computation -- no LLM calls. Runs in microseconds.
"""

from __future__ import annotations

from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Dict, List

from core.source_scorer import CredibilityScore


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class WeightedEvidence:
    """Combined evidence for a single claim across multiple sources."""
    claim: str
    sources: List[Dict]         # papers supporting this claim
    weights: List[float]        # per-source weight (0-1, normalized to sum 1.0)
    weighted_confidence: float  # combined confidence 0-1
    agreement_ratio: float      # fraction of sources that agree (1.0 = unanimous)


@dataclass
class EvidenceSummary:
    """Aggregate statistics across all evaluated claims."""
    total_claims: int
    avg_confidence: float
    high_confidence_claims: int   # confidence > 0.7
    contested_claims: int         # agreement_ratio < 0.6
    evidence_items: List[WeightedEvidence] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Current reference year for recency decay
# ---------------------------------------------------------------------------

_CURRENT_YEAR = datetime.now(timezone.utc).year


# ---------------------------------------------------------------------------
# Weighter
# ---------------------------------------------------------------------------

class EvidenceWeighter:
    """
    Inverse-variance evidence weighter.

    Given a claim with N supporting sources, computes per-source weights
    based on credibility composite, publication recency, and citation count.
    Normalises weights to sum to 1.0 so the result is a proper weighted
    average -- not inflated by adding more low-quality sources.
    """

    def __init__(self) -> None:
        pass

    # -- public API --------------------------------------------------------

    def weight(
        self,
        claim: str,
        sources: List[Dict],
        credibility_scores: List[CredibilityScore],
    ) -> WeightedEvidence:
        """
        Compute weighted evidence for *claim* across *sources*.

        Each source dict must contain at least a ``"year"`` key (int).
        Optional ``"citation_count"`` key (int, default 0).

        Parameters
        ----------
        claim : str
            The factual claim being evaluated.
        sources : list[dict]
            Paper metadata dicts (one per source).
        credibility_scores : list[CredibilityScore]
            Matching credibility scores (same order as *sources*).

        Returns
        -------
        WeightedEvidence
        """
        if len(sources) != len(credibility_scores):
            raise ValueError(
                f"sources ({len(sources)}) and credibility_scores "
                f"({len(credibility_scores)}) must have equal length"
            )

        # Compute raw weights
        raw_weights: List[float] = []
        for src, cred in zip(sources, credibility_scores):
            try:
                year = int(src.get("year") or _CURRENT_YEAR)
            except (ValueError, TypeError, OverflowError):
                year = _CURRENT_YEAR
            try:
                raw_cites = src.get("citation_count") or 0
                if isinstance(raw_cites, float) and (raw_cites != raw_cites or raw_cites == float('inf') or raw_cites == float('-inf')):
                    citations = 0
                else:
                    citations = max(0, int(raw_cites))
            except (ValueError, TypeError, OverflowError):
                citations = 0
            raw_weights.append(self._compute_weight(cred, year, citations))

        # Normalise to sum to 1.0 (guard against all-zero edge case)
        total = sum(raw_weights)
        if total > 0.0:
            weights = [w / total for w in raw_weights]
        else:
            # Uniform fallback when every source has zero weight
            n = len(raw_weights)
            weights = [1.0 / n for _ in raw_weights]

        # Weighted confidence
        weighted_confidence = sum(
            w * cred.composite
            for w, cred in zip(weights, credibility_scores)
        )

        # Agreement ratio: fraction of sources with relevance > 0.5
        n_total = len(credibility_scores)
        n_agree = sum(1 for cred in credibility_scores if cred.relevance > 0.5)
        agreement_ratio = n_agree / n_total if n_total > 0 else 0.0

        return WeightedEvidence(
            claim=claim,
            sources=sources,
            weights=weights,
            weighted_confidence=round(weighted_confidence, 6),
            agreement_ratio=round(agreement_ratio, 6),
        )

    def combine_evidence(
        self, evidences: List[WeightedEvidence]
    ) -> EvidenceSummary:
        """
        Aggregate multiple WeightedEvidence items into a summary.

        Parameters
        ----------
        evidences : list[WeightedEvidence]

        Returns
        -------
        EvidenceSummary
        """
        total = len(evidences)
        if total == 0:
            return EvidenceSummary(
                total_claims=0,
                avg_confidence=0.0,
                high_confidence_claims=0,
                contested_claims=0,
                evidence_items=[],
            )

        avg_conf = sum(e.weighted_confidence for e in evidences) / total
        high = sum(1 for e in evidences if e.weighted_confidence > 0.7)
        contested = sum(1 for e in evidences if e.agreement_ratio < 0.6)

        return EvidenceSummary(
            total_claims=total,
            avg_confidence=round(avg_conf, 6),
            high_confidence_claims=high,
            contested_claims=contested,
            evidence_items=list(evidences),
        )

    # -- internals ---------------------------------------------------------

    @staticmethod
    def _compute_weight(
        cred: CredibilityScore, year: int, citations: int
    ) -> float:
        """
        Raw per-source weight before normalisation.

        weight = composite * recency_factor * citation_factor

        recency_factor  decays 0.15 per year from current year, floor 0.1.
        citation_factor boosts up to 1.5x based on citation count.
        """
        recency_factor = max(0.1, 1.0 - (_CURRENT_YEAR - year) * 0.15)
        citation_factor = min(1.5, 1.0 + citations / 500)
        return cred.composite * recency_factor * citation_factor
