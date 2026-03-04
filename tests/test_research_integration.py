"""
Integration test: Full enhanced research pipeline end-to-end.

Exercises all 7 phases with realistic paper data and a mock LLM
that returns structured responses matching what each module expects.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.prisma_tracker import PrismaTracker
from core.source_scorer import SourceScorer, CredibilityScore
from core.layered_triage import LayeredTriage, _content_hash
from core.evidence_weighter import EvidenceWeighter
from core.synthesis_matrix import SynthesisMatrix
from core.devils_advocate import DevilsAdvocate
from core.claim_verifier import ClaimVerifier
from core.research_sprint import ResearchSprinter, SprintConfig


# ---------------------------------------------------------------------------
# Realistic sample papers (meta-research about research methodology)
# ---------------------------------------------------------------------------

SAMPLE_PAPERS = [
    {
        "title": "PRISMA 2020: Updated Guidelines for Systematic Reviews",
        "abstract": (
            "Systematic reviews are essential for evidence-based practice. "
            "This paper updates the PRISMA statement with 27 items covering "
            "eligibility criteria, information sources, search strategy, "
            "study selection, data collection, risk of bias assessment, "
            "and synthesis of results. The flow diagram now distinguishes "
            "between database and register searches."
        ),
        "year": 2021,
        "citation_count": 45000,
        "source": "semantic_scholar",
        "url": "https://doi.org/10.1136/bmj.n71",
        "has_code": False,
        "has_refs": True,
    },
    {
        "title": "Dense Passage Retrieval for Open-Domain Question Answering",
        "abstract": (
            "We show that retrieval based on dense representations alone "
            "can outperform strong sparse retrieval methods like BM25. "
            "Using a dual-encoder architecture with BERT, we achieve "
            "state-of-the-art results on multiple QA benchmarks. "
            "FAISS enables sub-second retrieval over millions of passages."
        ),
        "year": 2020,
        "citation_count": 4800,
        "source": "arxiv_cs_ir",
        "url": "https://arxiv.org/abs/2004.04906",
        "has_code": True,
        "has_refs": True,
    },
    {
        "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP",
        "abstract": (
            "We propose RAG, a model that combines parametric and "
            "non-parametric memory. The retriever fetches relevant passages, "
            "and the generator conditions on both input and retrieved context. "
            "RAG achieves state-of-the-art on open-domain QA and generates "
            "more factual, specific text than baselines."
        ),
        "year": 2020,
        "citation_count": 10200,
        "source": "arxiv_cs_cl",
        "url": "https://arxiv.org/abs/2005.11401",
        "has_code": True,
        "has_refs": True,
    },
    {
        "title": "Using Text Mining for Study Identification in Systematic Reviews",
        "abstract": (
            "This systematic review of text mining approaches for screening "
            "found that workload savings of 30-70% are achievable. "
            "Active learning outperforms static classification. "
            "The major barrier is trust and adoption, not algorithmic performance."
        ),
        "year": 2015,
        "citation_count": 650,
        "source": "semantic_scholar",
        "url": "https://doi.org/10.1186/2046-4053-4-5",
        "has_code": False,
        "has_refs": True,
    },
    {
        "title": "Attention Is All You Need",
        "abstract": (
            "We propose the Transformer architecture based on self-attention. "
            "Multi-head attention captures different relationship types. "
            "Position encodings inject sequence order without recurrence. "
            "The architecture enables massive parallelization."
        ),
        "year": 2017,
        "citation_count": 160000,
        "source": "arxiv_cs_ai",
        "url": "https://arxiv.org/abs/1706.03762",
        "has_code": True,
        "has_refs": True,
    },
    {
        "title": "Random Blog Post About Cooking",
        "abstract": "How to make the perfect sourdough bread at home.",
        "year": 2023,
        "citation_count": 0,
        "source": "hacker_news",
        "url": "https://example.com/cooking",
        "has_code": False,
        "has_refs": False,
    },
    {
        "title": "BM25 Is All You Need for First-Stage Retrieval",
        "abstract": (
            "We argue that BM25 remains a strong baseline for first-stage "
            "retrieval. Through systematic benchmarking across 18 datasets, "
            "we show that carefully tuned BM25 matches or outperforms many "
            "learned dense retrievers. The simplicity and efficiency of BM25 "
            "make it the pragmatic choice for production systems."
        ),
        "year": 2024,
        "citation_count": 120,
        "source": "arxiv_cs_ir",
        "url": "https://arxiv.org/abs/2402.12345",
        "has_code": True,
        "has_refs": True,
    },
    {
        "title": "Meta-Analysis of Research Automation Tools: A Survey",
        "abstract": (
            "We survey 44 tools for automated systematic review. "
            "Machine learning screening saves 30-70% of manual effort. "
            "Active learning with human feedback is the most promising approach. "
            "Key challenges: trust, reproducibility, and integration with workflows."
        ),
        "year": 2025,
        "citation_count": 35,
        "source": "papers_with_code",
        "url": "https://paperswithcode.com/paper/meta-analysis-automation",
        "has_code": True,
        "has_refs": True,
    },
]


class SmartMockLLM:
    """Mock LLM that returns structured responses for each module's prompts."""

    def generate(self, question="", context_chunks=None, **kwargs):
        text = question.lower() if question else ""

        # Drone triage: return INDEX|SCORE|SUMMARY format
        if "rate each paper" in text and "index|score|summary" in text:
            lines = []
            for i in range(10):
                lines.append(f"{i}|0.{70 + i}|Relevant paper on research methods.")
            return "\n".join(lines)

        # Relevance scoring: return a number
        if "rate 0.0 to 1.0 how relevant" in text:
            return "0.82"

        # Purpose classification
        if "classify the purpose" in text:
            return "academic"

        # Theme extraction: return themes
        if "identify" in text and "theme" in text:
            return (
                "information retrieval methods\n"
                "systematic review automation\n"
                "neural vs sparse retrieval\n"
                "evidence synthesis techniques\n"
                "research methodology frameworks"
            )

        # Position classification (JSON)
        if "classify" in text and "sentiment" in text:
            return json.dumps([
                {"source": "paper1", "theme": "t1", "sentiment": "supports",
                 "position": "Agrees strongly", "strength": 0.9},
            ])

        # Counter-query generation
        if "generate" in text and "against" in text:
            return (
                "research automation fails in practice\n"
                "limitations of systematic review tools\n"
                "manual research outperforms automated"
            )

        # Claim verification rephrasing
        if "rephrase" in text and "verification" in text:
            return (
                "evidence for systematic review effectiveness\n"
                "automated screening accuracy benchmarks"
            )

        # Supports/contradicts classification
        if "supports or contradicts" in text.lower():
            return "SUPPORTS"

        # Blind spots identification
        if "blind spot" in text or "not covered" in text:
            return (
                "qualitative research methodology\n"
                "domain-specific limitations"
            )

        # Default: return something useful
        if context_chunks:
            return f"Analysis based on {len(context_chunks)} sources."
        return "No relevant context found."


def _mock_fetch_fn(query):
    """Mock fetch function returning 2 papers per query."""
    return [
        {
            "title": f"Independent Source on {query[:30]}",
            "abstract": f"This paper independently confirms findings about {query}.",
            "year": 2025,
            "citation_count": 50,
            "source": "semantic_scholar",
        },
        {
            "title": f"Critical Review of {query[:30]}",
            "abstract": f"This review questions assumptions about {query}.",
            "year": 2024,
            "citation_count": 30,
            "source": "arxiv_cs_ai",
        },
    ]


def _mock_discover_fn(topic):
    """Return sample papers for any topic."""
    return SAMPLE_PAPERS


# ---------------------------------------------------------------------------
# Phase 1: PRISMA Tracking
# ---------------------------------------------------------------------------

def test_prisma_full_pipeline(tmp_path):
    """PRISMA tracker logs papers through all stages correctly."""
    tracker = PrismaTracker(db_path=str(tmp_path / "prisma.db"))

    # Simulate a real pipeline
    for p in SAMPLE_PAPERS:
        h = _content_hash(p)
        tracker.identify(p["title"], p.get("source", ""), h)

    # Screen: pass 6, fail 2
    for i, p in enumerate(SAMPLE_PAPERS):
        h = _content_hash(p)
        passed = p["citation_count"] > 0
        tracker.screen(h, passed=passed, reason="citation_count > 0" if passed else "zero citations")

    counts = tracker.flow_counts()
    assert counts["identified"] == 8
    assert counts["screened"] >= 5
    assert counts["excluded"] >= 1

    diagram = tracker.flow_diagram_text()
    assert "Identified" in diagram
    assert "Screened" in diagram

    tracker.close()


# ---------------------------------------------------------------------------
# Phase 2: Layered Triage
# ---------------------------------------------------------------------------

def test_layered_triage_filters_irrelevant():
    """Satellite pass eliminates cooking blog, keeps research papers."""
    lt = LayeredTriage(runtime=None)
    results = lt.triage_batch(
        SAMPLE_PAPERS,
        query="systematic review retrieval augmented generation",
        satellite_cutoff=0.25,
        drone_cutoff=0.2,
        max_deep_dive=3,
    )

    # Cooking post should score lowest
    cooking = [r for r in results if "Cooking" in r.title]
    assert len(cooking) == 1
    assert cooking[0].satellite_score < 0.25
    assert cooking[0].deep_dive is False

    # High-signal papers should pass
    deep = [r for r in results if r.deep_dive]
    assert len(deep) <= 3
    assert all(r.satellite_score >= 0.25 for r in deep)


def test_layered_triage_with_llm():
    """Drone pass produces scores and summaries with LLM."""
    lt = LayeredTriage(runtime=SmartMockLLM())
    results = lt.triage_batch(
        SAMPLE_PAPERS,
        query="research methodology automation",
        satellite_cutoff=0.2,
        drone_cutoff=0.3,
        max_deep_dive=3,
    )

    # Drone pass should have run on satellite survivors
    survivors = [r for r in results if r.satellite_pass]
    with_drone = [r for r in survivors if r.drone_score > 0]
    assert len(with_drone) >= 1


# ---------------------------------------------------------------------------
# Phase 3: Credibility Scoring
# ---------------------------------------------------------------------------

def test_credibility_differentiates_sources():
    """CRAAP scoring ranks academic papers above blog posts."""
    scorer = SourceScorer(runtime=None)
    scores = scorer.score_batch(SAMPLE_PAPERS, "research methodology")

    # Find cooking post vs PRISMA paper
    prisma_idx = next(i for i, p in enumerate(SAMPLE_PAPERS) if "PRISMA" in p["title"])
    cooking_idx = next(i for i, p in enumerate(SAMPLE_PAPERS) if "Cooking" in p["title"])

    assert scores[prisma_idx].composite > scores[cooking_idx].composite
    assert scores[prisma_idx].authority > scores[cooking_idx].authority
    assert scores[prisma_idx].accuracy > scores[cooking_idx].accuracy


def test_credibility_all_dimensions_bounded():
    """All credibility dimensions are in [0, 1]."""
    scorer = SourceScorer(runtime=None)
    for paper in SAMPLE_PAPERS:
        score = scorer.score(paper, "retrieval")
        for dim in [score.currency, score.relevance, score.authority,
                    score.accuracy, score.purpose, score.composite]:
            assert 0.0 <= dim <= 1.0, f"Out of bounds: {dim} for {paper['title']}"


# ---------------------------------------------------------------------------
# Phase 4: Evidence Weighting
# ---------------------------------------------------------------------------

def test_evidence_weighting_favors_quality():
    """High-citation recent paper weighted more than old uncited one."""
    scorer = SourceScorer(runtime=None)
    weighter = EvidenceWeighter()

    papers = [SAMPLE_PAPERS[0], SAMPLE_PAPERS[5]]  # PRISMA vs Cooking
    creds = scorer.score_batch(papers, "systematic review")

    result = weighter.weight("systematic reviews are useful", papers, creds)
    # PRISMA should get higher weight
    assert result.weights[0] > result.weights[1]
    assert abs(sum(result.weights) - 1.0) < 0.001


# ---------------------------------------------------------------------------
# Phase 5: Verification
# ---------------------------------------------------------------------------

def test_claim_verification_with_sources():
    """Claim verifier finds corroborating sources via fetch_fn."""
    verifier = ClaimVerifier(runtime=None, fetch_fn=_mock_fetch_fn)
    result = verifier.verify("Active learning saves 30-70% screening effort")

    assert result.verification_method == "cross_reference"
    assert result.confidence >= 0.0
    assert isinstance(result.corroborating_sources, list)


def test_devils_advocate_generates_counters():
    """Devil's advocate generates counter-queries and computes balance."""
    advocate = DevilsAdvocate(runtime=None, fetch_fn=_mock_fetch_fn)
    report = advocate.challenge(
        "Dense retrieval always outperforms BM25",
        supporting_papers=SAMPLE_PAPERS[:3],
    )

    assert len(report.counter_queries) >= 1
    assert 0.0 <= report.balance_ratio <= 1.0
    assert report.original_claim == "Dense retrieval always outperforms BM25"


def test_devils_advocate_with_llm():
    """Devil's advocate uses LLM for smarter counter-queries."""
    advocate = DevilsAdvocate(runtime=SmartMockLLM(), fetch_fn=_mock_fetch_fn)
    report = advocate.challenge(
        "Automated systematic reviews are as good as manual ones",
        supporting_papers=SAMPLE_PAPERS[:3],
        max_counter_queries=3,
    )

    assert len(report.counter_queries) >= 1
    # Should find opposing evidence via fetch_fn
    assert report.opposing_count >= 0


# ---------------------------------------------------------------------------
# Phase 6: Synthesis Matrix
# ---------------------------------------------------------------------------

def test_synthesis_matrix_builds_grid():
    """Synthesis matrix produces theme x source grid."""
    synth = SynthesisMatrix(runtime=None)
    digested = [
        {
            "title": p["title"],
            "abstract": p["abstract"],
            "key_claims": [f"Claim from {p['title'][:20]}"],
            "triage_summary": p["abstract"][:100],
        }
        for p in SAMPLE_PAPERS[:5]
    ]

    report = synth.build(digested, "research methodology")
    assert report.total_sources == 5
    assert len(report.themes) >= 1

    md = synth.to_markdown_table(report)
    assert "|" in md  # has table structure
    assert "Agreements" in md or "Contradictions" in md or "Gaps" in md


def test_synthesis_matrix_with_llm():
    """Synthesis matrix uses LLM for theme extraction."""
    synth = SynthesisMatrix(runtime=SmartMockLLM())
    digested = [
        {
            "title": p["title"],
            "abstract": p["abstract"],
            "key_claims": [f"Claim about {p['title'][:20]}"],
        }
        for p in SAMPLE_PAPERS[:4]
    ]

    report = synth.build(digested, "retrieval augmented generation")
    assert len(report.themes) >= 1
    assert report.total_sources == 4


# ---------------------------------------------------------------------------
# Full pipeline integration
# ---------------------------------------------------------------------------

def test_full_sprint_pipeline(tmp_path):
    """End-to-end sprint with all 7 phases using mock LLM."""
    config = SprintConfig(
        focus_topics=["research methodology automation"],
        max_papers_to_triage=8,
        max_papers_to_digest=3,
        output_dir=str(tmp_path / "sprints"),
        prisma_enabled=True,
        credibility_scoring=True,
        devils_advocate=False,  # keep fast
        claim_verification=False,  # keep fast
        satellite_cutoff=0.2,
        drone_cutoff=0.2,
        max_deep_dive=3,
    )

    sprinter = ResearchSprinter(
        runtime=SmartMockLLM(),
        discover_fn=_mock_discover_fn,
        config=config,
    )

    result = sprinter.run_sprint()

    # Phase 1: Discovery
    assert result.papers_discovered == len(SAMPLE_PAPERS)

    # Phase 2: Screening
    assert result.papers_triaged >= 1

    # Phase 7: Report
    assert result.report_path
    report_path = Path(result.report_path)
    assert report_path.exists()
    report_text = report_path.read_text(encoding="utf-8")
    assert "PRISMA" in report_text
    assert "Research Sprint" in report_text

    # PRISMA flow should have data
    assert result.prisma_flow.get("identified", 0) == len(SAMPLE_PAPERS)

    # Sprint dir should have artifacts
    sprint_dir = report_path.parent
    assert (sprint_dir / "digests.json").exists()
    assert (sprint_dir / "prisma.db").exists()


def test_full_sprint_with_verification(tmp_path):
    """Sprint with devil's advocate + claim verification enabled."""
    config = SprintConfig(
        focus_topics=["systematic review automation"],
        max_papers_to_triage=8,
        max_papers_to_digest=3,
        output_dir=str(tmp_path / "sprints"),
        prisma_enabled=True,
        credibility_scoring=True,
        devils_advocate=True,
        claim_verification=True,
        satellite_cutoff=0.15,
        drone_cutoff=0.15,
        max_deep_dive=3,
        max_verify_claims=3,
        max_counter_queries=2,
    )

    sprinter = ResearchSprinter(
        runtime=SmartMockLLM(),
        discover_fn=_mock_discover_fn,
        fetch_fn=_mock_fetch_fn,
        config=config,
    )

    result = sprinter.run_sprint()

    assert result.papers_discovered == len(SAMPLE_PAPERS)
    assert result.report_path
    # balance_ratio should be set (devil's advocate ran)
    # Note: may be -1 if no claims were found in digested papers
    assert isinstance(result.balance_ratio, float)


def test_sprint_empty_discovery(tmp_path):
    """Sprint handles empty discovery gracefully."""
    config = SprintConfig(
        focus_topics=["nonexistent topic"],
        output_dir=str(tmp_path / "sprints"),
    )

    sprinter = ResearchSprinter(
        runtime=SmartMockLLM(),
        discover_fn=lambda t: [],
        config=config,
    )

    result = sprinter.run_sprint()
    assert result.papers_discovered == 0
    assert result.papers_digested == 0


def test_sprint_no_discover_fn(tmp_path):
    """Sprint with pre-supplied papers (no discover_fn)."""
    config = SprintConfig(
        output_dir=str(tmp_path / "sprints"),
        satellite_cutoff=0.15,
        drone_cutoff=0.15,
        max_deep_dive=2,
    )

    sprinter = ResearchSprinter(
        runtime=SmartMockLLM(),
        config=config,
    )

    result = sprinter.run_sprint(papers=SAMPLE_PAPERS[:3])
    assert result.papers_discovered == 3


# ---------------------------------------------------------------------------
# Research quality assertions
# ---------------------------------------------------------------------------

def test_triage_ranks_research_above_noise():
    """The full pipeline consistently ranks research papers above noise."""
    lt = LayeredTriage(runtime=None)
    results = lt.triage_batch(
        SAMPLE_PAPERS,
        query="information retrieval systematic review meta-analysis",
        satellite_cutoff=0.1,
        max_deep_dive=5,
    )

    # Sort by best score
    sorted_results = sorted(results, key=lambda r: r.satellite_score, reverse=True)

    # Cooking post should be in bottom half
    cooking_rank = next(
        i for i, r in enumerate(sorted_results) if "Cooking" in r.title
    )
    assert cooking_rank >= len(sorted_results) // 2, (
        f"Cooking post ranked {cooking_rank} out of {len(sorted_results)} -- too high"
    )


def test_credibility_scores_correlate_with_citations():
    """Papers with more citations and code should score higher on authority + accuracy."""
    scorer = SourceScorer(runtime=None)

    high = {"title": "Good Paper", "abstract": "Research.", "year": 2025,
            "citation_count": 5000, "source": "semantic_scholar",
            "url": "https://doi.org/x", "has_code": True, "has_refs": True}
    low = {"title": "Weak Paper", "abstract": "Thoughts.", "year": 2023,
           "citation_count": 2, "source": "hacker_news",
           "url": "https://hn.com/x", "has_code": False, "has_refs": False}

    s_high = scorer.score(high, "research")
    s_low = scorer.score(low, "research")

    assert s_high.composite > s_low.composite
    assert s_high.authority > s_low.authority
    assert s_high.accuracy > s_low.accuracy
