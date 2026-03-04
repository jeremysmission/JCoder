"""Tests for the inverse-variance evidence weighter."""

from __future__ import annotations

from core.evidence_weighter import EvidenceWeighter, WeightedEvidence
from core.source_scorer import CredibilityScore


def _make_cred(
    composite: float = 0.8,
    relevance: float = 0.8,
    currency: float = 0.8,
    authority: float = 0.8,
    accuracy: float = 0.8,
    purpose: float = 0.8,
) -> CredibilityScore:
    """Build a CredibilityScore with sensible defaults."""
    return CredibilityScore(
        currency=currency,
        relevance=relevance,
        authority=authority,
        accuracy=accuracy,
        purpose=purpose,
        composite=composite,
    )


def _make_paper(year: int = 2026, citation_count: int = 0) -> dict:
    """Build a minimal paper dict."""
    return {"year": year, "citation_count": citation_count, "title": "paper"}


# -----------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------


def test_single_source():
    """One source gets weight 1.0."""
    ew = EvidenceWeighter()
    result = ew.weight(
        claim="X is true",
        sources=[_make_paper()],
        credibility_scores=[_make_cred()],
    )
    assert len(result.weights) == 1
    assert result.weights[0] == 1.0


def test_equal_sources():
    """Two identical sources get equal weights (~0.5 each)."""
    ew = EvidenceWeighter()
    cred = _make_cred(composite=0.7, relevance=0.9)
    result = ew.weight(
        claim="shared claim",
        sources=[_make_paper(2024), _make_paper(2024)],
        credibility_scores=[cred, cred],
    )
    assert len(result.weights) == 2
    assert abs(result.weights[0] - 0.5) < 1e-9
    assert abs(result.weights[1] - 0.5) < 1e-9


def test_weighted_by_credibility():
    """High-credibility source gets a higher weight than low-credibility."""
    ew = EvidenceWeighter()
    high = _make_cred(composite=0.9, relevance=0.9)
    low = _make_cred(composite=0.2, relevance=0.9)
    result = ew.weight(
        claim="contested claim",
        sources=[_make_paper(2026), _make_paper(2026)],
        credibility_scores=[high, low],
    )
    assert result.weights[0] > result.weights[1]


def test_recency_factor():
    """A 2026 paper is weighted more than a 2020 paper (same credibility)."""
    ew = EvidenceWeighter()
    cred = _make_cred(composite=0.8, relevance=0.9)
    result = ew.weight(
        claim="recency matters",
        sources=[_make_paper(year=2026), _make_paper(year=2020)],
        credibility_scores=[cred, cred],
    )
    # 2026 source should dominate
    assert result.weights[0] > result.weights[1]


def test_citation_boost():
    """Highly cited paper weighted more (same year, same credibility)."""
    ew = EvidenceWeighter()
    cred = _make_cred(composite=0.8, relevance=0.9)
    result = ew.weight(
        claim="citations help",
        sources=[
            _make_paper(year=2024, citation_count=1000),
            _make_paper(year=2024, citation_count=0),
        ],
        credibility_scores=[cred, cred],
    )
    assert result.weights[0] > result.weights[1]


def test_weights_sum_to_one():
    """Any combination of sources produces weights that sum to 1.0."""
    ew = EvidenceWeighter()
    sources = [
        _make_paper(2026, 500),
        _make_paper(2023, 10),
        _make_paper(2019, 3000),
        _make_paper(2025, 0),
    ]
    creds = [
        _make_cred(composite=0.95, relevance=0.9),
        _make_cred(composite=0.3, relevance=0.2),
        _make_cred(composite=0.6, relevance=0.8),
        _make_cred(composite=0.1, relevance=0.6),
    ]
    result = ew.weight(
        claim="sum invariant",
        sources=sources,
        credibility_scores=creds,
    )
    assert abs(sum(result.weights) - 1.0) < 1e-9


def test_combine_evidence():
    """Three claims aggregated: verify avg_confidence and contested count."""
    ew = EvidenceWeighter()

    # Claim 1: high confidence, unanimous agreement
    e1 = ew.weight(
        claim="claim A",
        sources=[_make_paper(2026), _make_paper(2025)],
        credibility_scores=[
            _make_cred(composite=0.9, relevance=0.9),
            _make_cred(composite=0.85, relevance=0.8),
        ],
    )

    # Claim 2: low confidence, contested (relevance <= 0.5 for all)
    e2 = ew.weight(
        claim="claim B",
        sources=[_make_paper(2020), _make_paper(2019)],
        credibility_scores=[
            _make_cred(composite=0.2, relevance=0.3),
            _make_cred(composite=0.15, relevance=0.4),
        ],
    )

    # Claim 3: medium confidence, partially contested
    e3 = ew.weight(
        claim="claim C",
        sources=[_make_paper(2024), _make_paper(2023), _make_paper(2022)],
        credibility_scores=[
            _make_cred(composite=0.6, relevance=0.7),
            _make_cred(composite=0.5, relevance=0.4),
            _make_cred(composite=0.55, relevance=0.3),
        ],
    )

    summary = ew.combine_evidence([e1, e2, e3])

    assert summary.total_claims == 3
    # avg_confidence should be the mean of the three weighted_confidence values
    expected_avg = (
        e1.weighted_confidence + e2.weighted_confidence + e3.weighted_confidence
    ) / 3
    assert abs(summary.avg_confidence - expected_avg) < 1e-4

    # e1 has weighted_confidence close to 0.9 -> high confidence
    assert summary.high_confidence_claims >= 1

    # e2 has agreement_ratio 0.0 (both relevance <= 0.5) -> contested
    assert e2.agreement_ratio < 0.6
    assert summary.contested_claims >= 1


def test_zero_credibility():
    """All sources with 0 credibility still produce valid output (no div by zero)."""
    ew = EvidenceWeighter()
    zero = _make_cred(
        composite=0.0, relevance=0.0,
        currency=0.0, authority=0.0, accuracy=0.0, purpose=0.0,
    )
    result = ew.weight(
        claim="nothing credible",
        sources=[_make_paper(2026), _make_paper(2025)],
        credibility_scores=[zero, zero],
    )
    # Weights should still sum to 1.0 via uniform fallback
    assert abs(sum(result.weights) - 1.0) < 1e-9
    # Confidence should be 0.0 (all composites are 0)
    assert result.weighted_confidence == 0.0
    # Agreement ratio should be 0.0 (all relevance are 0)
    assert result.agreement_ratio == 0.0
