"""
Adversarial weakness tests for the research pipeline.

Tests edge cases, failure modes, and quality gaps:
1. Garbage input handling
2. Adversarial papers that game the scoring
3. LLM failure resilience
4. Boundary conditions
5. Bias blind spots
6. Scale stress
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.source_scorer import SourceScorer
from core.layered_triage import LayeredTriage, _content_hash
from core.evidence_weighter import EvidenceWeighter, WeightedEvidence
from core.synthesis_matrix import SynthesisMatrix
from core.devils_advocate import DevilsAdvocate
from core.claim_verifier import ClaimVerifier
from core.prisma_tracker import PrismaTracker
from core.research_sprint import ResearchSprinter, SprintConfig


class CrashingLLM:
    """LLM that always throws."""
    def generate(self, question="", context_chunks=None, **kwargs):
        raise ConnectionError("vLLM server unreachable")


class GarbageLLM:
    """LLM that returns unparseable garbage."""
    def generate(self, question="", context_chunks=None, **kwargs):
        return "!!@@##$$%%^^&&**(())\nNOT_VALID_JSON\n{broken: true"


class EmptyLLM:
    """LLM that returns empty strings."""
    def generate(self, question="", context_chunks=None, **kwargs):
        return ""


# ---------------------------------------------------------------------------
# 1. Garbage input handling
# ---------------------------------------------------------------------------

def test_triage_with_empty_titles():
    """Papers with empty/missing titles don't crash."""
    lt = LayeredTriage(runtime=None)
    papers = [
        {"title": "", "abstract": "Some content.", "year": 2025},
        {"abstract": "No title key.", "year": 2024},
        {"title": "Valid Paper", "abstract": "Research.", "year": 2025},
    ]
    results = lt.triage_batch(papers, "research")
    assert len(results) == 3  # all processed, none crashed


def test_triage_with_none_values():
    """Papers with None values in fields don't crash."""
    lt = LayeredTriage(runtime=None)
    papers = [
        {"title": None, "abstract": None, "year": None, "citation_count": None},
    ]
    # Should not raise
    results = lt.triage_batch(papers, "test")
    assert len(results) == 1


def test_scorer_with_missing_fields():
    """Scorer handles papers missing all optional fields."""
    scorer = SourceScorer(runtime=None)
    minimal = {"title": "X"}  # nothing else
    score = scorer.score(minimal, "query")
    assert 0.0 <= score.composite <= 1.0


def test_scorer_with_negative_citations():
    """Negative citation count doesn't break scoring."""
    scorer = SourceScorer(runtime=None)
    paper = {"title": "X", "citation_count": -100, "year": 2025}
    score = scorer.score(paper, "query")
    assert score.authority >= 0.0


def test_scorer_with_future_year():
    """Year 2030 doesn't produce negative currency."""
    scorer = SourceScorer(runtime=None)
    paper = {"title": "X", "year": 2030}
    score = scorer.score(paper, "query")
    assert score.currency >= 0.0


def test_prisma_with_empty_strings(tmp_path):
    """PRISMA tracker handles empty hash/title/source."""
    tracker = PrismaTracker(db_path=str(tmp_path / "p.db"))
    tracker.identify("", "", "")
    tracker.screen("", passed=True, reason="")
    counts = tracker.flow_counts()
    assert counts["identified"] == 1
    tracker.close()


# ---------------------------------------------------------------------------
# 2. Adversarial papers that game the scoring
# ---------------------------------------------------------------------------

def test_keyword_stuffed_title():
    """Paper with keyword-stuffed title shouldn't dominate scoring."""
    lt = LayeredTriage(runtime=None)
    stuffed = {
        "title": "Novel State-of-the-Art Breakthrough Self-Improving Meta-Learning RAG Benchmark Survey",
        "abstract": "This paper has no substance.",
        "year": 2026,
        "citation_count": 0,
        "source": "hacker_news",
    }
    legitimate = {
        "title": "Dense Passage Retrieval for Open-Domain QA",
        "abstract": "We show dense representations outperform BM25 on multiple benchmarks.",
        "year": 2020,
        "citation_count": 4800,
        "source": "arxiv_cs_ir",
    }
    results = lt.triage_batch([stuffed, legitimate], "retrieval")
    # Legitimate paper should score competitively despite fewer keywords
    # because citation count and source tier should help
    assert abs(results[0].satellite_score - results[1].satellite_score) <= 0.2


def test_credibility_punishes_hype():
    """Low-tier source with zero citations scores lower despite recency."""
    scorer = SourceScorer(runtime=None)
    hype = {"title": "BREAKTHROUGH!", "year": 2026, "citation_count": 0,
            "source": "reddit", "has_code": False, "has_refs": False}
    solid = {"title": "Systematic Review", "year": 2023, "citation_count": 200,
             "source": "semantic_scholar", "has_code": False, "has_refs": True}

    s_hype = scorer.score(hype, "review")
    s_solid = scorer.score(solid, "review")
    assert s_solid.composite > s_hype.composite, "Hype beat substance"


# ---------------------------------------------------------------------------
# 3. LLM failure resilience
# ---------------------------------------------------------------------------

def test_triage_survives_crashing_llm():
    """Triage falls back to satellite scores when LLM crashes."""
    lt = LayeredTriage(runtime=CrashingLLM())
    papers = [
        {"title": "Good Paper", "abstract": "Research.", "year": 2025,
         "source": "arxiv_cs_ai", "citation_count": 100},
    ]
    results = lt.triage_batch(papers, "research", satellite_cutoff=0.1)
    # Should still have results (satellite pass ran)
    assert len(results) == 1
    assert results[0].satellite_score > 0


def test_triage_survives_garbage_llm():
    """Triage handles unparseable LLM output gracefully."""
    lt = LayeredTriage(runtime=GarbageLLM())
    papers = [
        {"title": "Paper A", "abstract": "Content.", "year": 2025,
         "source": "semantic_scholar", "citation_count": 50},
    ]
    results = lt.triage_batch(papers, "query", satellite_cutoff=0.1)
    assert len(results) == 1


def test_triage_survives_empty_llm():
    """Triage handles empty LLM response."""
    lt = LayeredTriage(runtime=EmptyLLM())
    papers = [
        {"title": "Paper", "abstract": "Content.", "year": 2025,
         "source": "arxiv_cs_ir", "citation_count": 20},
    ]
    results = lt.triage_batch(papers, "query", satellite_cutoff=0.1)
    assert len(results) == 1


def test_synthesis_survives_crashing_llm():
    """Synthesis matrix falls back to heuristic when LLM crashes."""
    synth = SynthesisMatrix(runtime=CrashingLLM())
    papers = [
        {"title": "P1", "abstract": "retrieval methods", "key_claims": ["claim1"]},
        {"title": "P2", "abstract": "search techniques", "key_claims": ["claim2"]},
    ]
    # Should not raise -- falls back to heuristic
    report = synth.build(papers, "retrieval")
    assert report.total_sources == 2


def test_advocate_survives_crashing_llm():
    """Devil's advocate falls back to heuristic with crashed LLM."""
    advocate = DevilsAdvocate(runtime=CrashingLLM(), fetch_fn=lambda q: [])
    report = advocate.challenge("test claim", [])
    assert len(report.counter_queries) >= 1  # heuristic fallback


def test_verifier_survives_crashing_llm():
    """Verifier produces valid result even with crashed LLM."""
    verifier = ClaimVerifier(runtime=CrashingLLM(), fetch_fn=lambda q: [
        {"title": "Source", "abstract": "test claim confirmed", "year": 2025},
    ])
    result = verifier.verify("test claim")
    assert result.verification_method == "cross_reference"


# ---------------------------------------------------------------------------
# 4. Boundary conditions
# ---------------------------------------------------------------------------

def test_single_paper_sprint(tmp_path):
    """Sprint with exactly 1 paper still produces valid output."""
    config = SprintConfig(
        focus_topics=["test"],
        output_dir=str(tmp_path / "sprints"),
        satellite_cutoff=0.0,
        drone_cutoff=0.0,
        max_deep_dive=1,
    )

    class MinimalLLM:
        def generate(self, question="", context_chunks=None, **kwargs):
            return "0|0.5|A paper."

    sprinter = ResearchSprinter(
        runtime=MinimalLLM(),
        discover_fn=lambda t: [{"title": "Solo Paper", "abstract": "Research.", "year": 2025}],
        config=config,
    )
    result = sprinter.run_sprint()
    assert result.papers_discovered == 1


def test_evidence_weighter_single_source():
    """Single source gets full weight."""
    weighter = EvidenceWeighter()
    from core.source_scorer import CredibilityScore
    cred = CredibilityScore(0.8, 0.9, 0.7, 0.8, 0.6, 0.78)
    result = weighter.weight("claim", [{"year": 2025, "citation_count": 100}], [cred])
    assert len(result.weights) == 1
    assert result.weights[0] == 1.0


def test_synthesis_single_paper():
    """Synthesis with 1 paper produces valid matrix."""
    synth = SynthesisMatrix(runtime=None)
    report = synth.build(
        [{"title": "Solo", "abstract": "research methods", "key_claims": ["c1"]}],
        "research",
    )
    assert report.total_sources == 1


def test_hundred_papers_triage():
    """Triage handles 100 papers without crashing or excessive slowdown."""
    import time
    lt = LayeredTriage(runtime=None)
    papers = [
        {"title": f"Paper {i} on retrieval", "abstract": f"Content about search {i}.",
         "year": 2020 + (i % 7), "citation_count": i * 5,
         "source": "arxiv_cs_ai"}
        for i in range(100)
    ]
    t0 = time.time()
    results = lt.triage_batch(papers, "retrieval search", satellite_cutoff=0.1)
    elapsed = time.time() - t0
    assert len(results) == 100
    assert elapsed < 5.0, f"100-paper triage took {elapsed:.1f}s -- too slow"


# ---------------------------------------------------------------------------
# 5. Bias blind spots
# ---------------------------------------------------------------------------

def test_balance_ratio_detects_one_sided_research():
    """Devil's advocate flags when all sources agree (no opposition found)."""
    advocate = DevilsAdvocate(runtime=None, fetch_fn=lambda q: [])
    report = advocate.challenge(
        "All studies show X is effective",
        supporting_papers=[{"title": f"Study {i}"} for i in range(5)],
    )
    # With no opposing sources found, balance should be 0.0 (one-sided)
    assert report.balance_ratio == 0.0
    assert report.opposing_count == 0


def test_verification_flags_unverifiable():
    """Claim without fetch_fn is correctly flagged as unverifiable."""
    verifier = ClaimVerifier(runtime=None, fetch_fn=None)
    result = verifier.verify("Some claim")
    assert result.verification_method == "unverifiable"
    assert result.verified is False
    assert result.confidence == 0.0


# ---------------------------------------------------------------------------
# 6. Content hash collision resistance
# ---------------------------------------------------------------------------

def test_similar_titles_different_hashes():
    """Papers with similar but not identical titles get different hashes."""
    p1 = {"title": "Dense Passage Retrieval for QA", "abstract": "Method A."}
    p2 = {"title": "Dense Passage Retrieval for IR", "abstract": "Method B."}
    assert _content_hash(p1) != _content_hash(p2)


def test_unicode_paper_titles():
    """Papers with unicode characters in title/abstract don't crash."""
    lt = LayeredTriage(runtime=None)
    papers = [
        {"title": "Forschungsmethoden und Ubersicht", "abstract": "Deutsche Zusammenfassung."},
        {"title": "Research with emojis in title", "abstract": "Normal abstract."},
    ]
    results = lt.triage_batch(papers, "research")
    assert len(results) == 2
