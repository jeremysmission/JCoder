"""Source credibility scorer must produce valid CRAAP scores."""

from __future__ import annotations

import math

import pytest

from core.source_scorer import CredibilityScore, SourceScorer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _paper(**overrides) -> dict:
    """Build a paper dict with sane defaults, overridden by caller."""
    base = {
        "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP",
        "abstract": "We combine retrieval with generation for open-domain QA.",
        "year": 2025,
        "citation_count": 0,
        "source": "arxiv",
        "url": "https://arxiv.org/abs/2005.11401",
        "has_code": False,
        "has_refs": False,
        "peer_reviewed": False,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Currency
# ---------------------------------------------------------------------------

def test_currency_recent():
    scorer = SourceScorer()
    result = scorer.score(_paper(year=2026), "RAG")
    assert result.currency == pytest.approx(1.0)


def test_currency_old():
    scorer = SourceScorer()
    result = scorer.score(_paper(year=2020), "RAG")
    assert result.currency == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# Authority
# ---------------------------------------------------------------------------

def test_authority_arxiv():
    scorer = SourceScorer()
    result = scorer.score(_paper(source="arxiv", citation_count=0), "RAG")
    assert result.authority >= 0.85


def test_authority_hn():
    scorer = SourceScorer()
    result = scorer.score(
        _paper(source="hacker_news", url="https://news.ycombinator.com", citation_count=0),
        "RAG",
    )
    assert result.authority <= 0.5


# ---------------------------------------------------------------------------
# Accuracy
# ---------------------------------------------------------------------------

def test_accuracy_full():
    scorer = SourceScorer()
    result = scorer.score(
        _paper(has_refs=True, has_code=True, peer_reviewed=True), "RAG"
    )
    assert result.accuracy == pytest.approx(1.0)


def test_accuracy_minimal():
    scorer = SourceScorer()
    result = scorer.score(
        _paper(has_refs=False, has_code=False, peer_reviewed=False), "RAG"
    )
    assert result.accuracy == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# Composite
# ---------------------------------------------------------------------------

def test_composite_weights():
    """Composite must equal the documented weighted average."""
    scorer = SourceScorer()
    result = scorer.score(_paper(year=2025, source="arxiv"), "RAG")

    expected = (
        0.15 * result.currency
        + 0.30 * result.relevance
        + 0.20 * result.authority
        + 0.20 * result.accuracy
        + 0.15 * result.purpose
    )
    assert result.composite == pytest.approx(expected, abs=1e-3)


# ---------------------------------------------------------------------------
# Heuristic-only mode
# ---------------------------------------------------------------------------

def test_heuristic_only_mode():
    """SourceScorer(runtime=None) must produce valid scores without LLM."""
    scorer = SourceScorer(runtime=None)
    result = scorer.score(_paper(), "retrieval augmented generation")

    assert 0.0 <= result.currency <= 1.0
    assert 0.0 <= result.relevance <= 1.0
    assert 0.0 <= result.authority <= 1.0
    assert 0.0 <= result.accuracy <= 1.0
    assert 0.0 <= result.purpose <= 1.0
    assert 0.0 <= result.composite <= 1.0


# ---------------------------------------------------------------------------
# Batch scoring
# ---------------------------------------------------------------------------

def test_score_batch():
    scorer = SourceScorer()
    papers = [
        _paper(year=2026, source="arxiv"),
        _paper(year=2023, source="github"),
        _paper(year=2020, source="hacker_news"),
    ]
    results = scorer.score_batch(papers, "RAG pipeline")

    assert len(results) == 3
    assert all(isinstance(r, CredibilityScore) for r in results)
    # Most-recent arxiv paper should outscore the old HN link
    assert results[0].composite > results[2].composite
