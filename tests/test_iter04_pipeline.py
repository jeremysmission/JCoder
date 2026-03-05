"""Iteration 4: Pipeline stage isolation and data flow tests."""

import pytest
from core.layered_triage import LayeredTriage, _content_hash
from core.source_scorer import SourceScorer, CredibilityScore
from core.evidence_weighter import EvidenceWeighter
from core.synthesis_matrix import SynthesisMatrix
from core.devils_advocate import DevilsAdvocate
from core.claim_verifier import ClaimVerifier
from core.prisma_tracker import PrismaTracker


# ---------------------------------------------------------------------------
# Stage isolation: each module works independently
# ---------------------------------------------------------------------------

def test_triage_no_side_effects():
    """Triage doesn't mutate the input paper dicts."""
    lt = LayeredTriage(runtime=None)
    papers = [{"title": "Paper A", "abstract": "Abstract A", "year": 2025}]
    import copy
    original = copy.deepcopy(papers)
    lt.triage_batch(papers, "test")
    assert papers == original, "Triage mutated input papers"


def test_scorer_no_side_effects():
    """Scorer doesn't mutate the input paper dict."""
    scorer = SourceScorer(runtime=None)
    paper = {"title": "Paper A", "abstract": "Abstract A", "year": 2025}
    import copy
    original = copy.deepcopy(paper)
    scorer.score(paper, "test")
    assert paper == original, "Scorer mutated input paper"


def test_weighter_no_side_effects():
    """Weighter doesn't mutate input lists."""
    weighter = EvidenceWeighter()
    import copy
    creds = [CredibilityScore(0.5, 0.5, 0.5, 0.5, 0.5, 0.5)]
    papers = [{"year": 2025, "citation_count": 10}]
    orig_creds = copy.deepcopy(creds)
    orig_papers = copy.deepcopy(papers)
    weighter.weight("claim", papers, creds)
    assert papers == orig_papers
    assert creds[0].composite == orig_creds[0].composite


# ---------------------------------------------------------------------------
# Data flow: output of one stage feeds correctly into next
# ---------------------------------------------------------------------------

def test_triage_to_scorer_flow():
    """Triage results can feed into scorer without adaptation."""
    lt = LayeredTriage(runtime=None)
    scorer = SourceScorer(runtime=None)
    papers = [
        {"title": "Dense retrieval methods", "abstract": "We study dense retrieval.",
         "year": 2025, "citation_count": 50, "source": "arxiv_cs_ai"},
        {"title": "Cooking tips", "abstract": "How to cook.", "year": 2020, "source": "reddit"},
    ]
    results = lt.triage_batch(papers, "retrieval")
    # Scorer should work on the same paper dicts
    scores = scorer.score_batch(papers, "retrieval")
    assert len(scores) == 2
    assert all(isinstance(s, CredibilityScore) for s in scores)


def test_scorer_to_weighter_flow():
    """Scorer output feeds directly into weighter."""
    scorer = SourceScorer(runtime=None)
    weighter = EvidenceWeighter()
    papers = [
        {"title": "Paper A", "abstract": "Research on X.", "year": 2025, "citation_count": 100},
        {"title": "Paper B", "abstract": "Research on Y.", "year": 2023, "citation_count": 10},
    ]
    creds = scorer.score_batch(papers, "research")
    result = weighter.weight("research claim", papers, creds)
    assert len(result.weights) == 2
    assert abs(sum(result.weights) - 1.0) < 0.001


def test_full_pipeline_no_llm():
    """Full pipeline without LLM: triage -> score -> weight -> synthesis."""
    papers = [
        {"title": f"Paper {i}: retrieval methods", "abstract": f"Study {i} on dense retrieval.",
         "year": 2020 + i, "citation_count": i * 20, "source": "arxiv_cs_ai",
         "key_claims": [f"Claim {i}: method improves accuracy"]}
        for i in range(5)
    ]
    # Stage 1: Triage
    lt = LayeredTriage(runtime=None)
    triage_results = lt.triage_batch(papers, "retrieval", satellite_cutoff=0.1)
    assert len(triage_results) == 5

    # Stage 2: Score
    scorer = SourceScorer(runtime=None)
    creds = scorer.score_batch(papers, "retrieval")
    assert len(creds) == 5

    # Stage 3: Weight
    weighter = EvidenceWeighter()
    evidence = weighter.weight("retrieval improves QA", papers, creds)
    assert len(evidence.weights) == 5

    # Stage 4: Synthesize
    synth = SynthesisMatrix(runtime=None)
    report = synth.build(papers, "retrieval methods")
    assert report.total_sources == 5


# ---------------------------------------------------------------------------
# PRISMA integration with pipeline stages
# ---------------------------------------------------------------------------

def test_prisma_tracks_full_pipeline(tmp_path):
    """PRISMA tracker correctly tracks papers through all stages."""
    tracker = PrismaTracker(db_path=str(tmp_path / "pipeline.db"))

    papers = [
        {"title": f"Paper {i}", "abstract": f"About topic {i}.", "year": 2025}
        for i in range(10)
    ]

    # Identify all
    for p in papers:
        h = _content_hash(p)
        tracker.identify(p["title"], "arxiv", h)

    # Screen: pass 7, fail 3
    for i, p in enumerate(papers):
        h = _content_hash(p)
        tracker.screen(h, passed=(i < 7), reason="relevance")

    # Eligible: pass 4 of the 7
    for i, p in enumerate(papers[:7]):
        h = _content_hash(p)
        tracker.eligible(h, passed=(i < 4), reason="quality")

    # Include: 2 of the 4
    for p in papers[:2]:
        h = _content_hash(p)
        tracker.include(h, reason="final selection")

    counts = tracker.flow_counts()
    assert counts["identified"] == 10
    assert counts["screened"] == 7
    assert counts["eligible"] == 4
    assert counts["included"] == 2
    # excluded = 3 (screening) + 3 (eligibility) = 6
    assert counts["excluded"] == 6
    tracker.close()


def test_prisma_flow_diagram_text(tmp_path):
    """Flow diagram text includes all stages and counts."""
    tracker = PrismaTracker(db_path=str(tmp_path / "flow.db"))
    h = "abc123"
    tracker.identify("Paper 1", "source1", h)
    tracker.screen(h, passed=True, reason="ok")
    tracker.eligible(h, passed=True, reason="good")
    tracker.include(h, reason="final")

    diagram = tracker.flow_diagram_text()
    assert "Identified" in diagram
    assert "Screened" in diagram
    assert "Eligible" in diagram
    assert "Included" in diagram
    tracker.close()


# ---------------------------------------------------------------------------
# Cross-module data integrity
# ---------------------------------------------------------------------------

def test_content_hash_consistent_across_modules():
    """Same paper produces same hash regardless of which module uses it."""
    paper = {"title": "Consistent Paper", "abstract": "Testing hash consistency."}
    h1 = _content_hash(paper)
    h2 = _content_hash(paper)
    h3 = _content_hash(paper)
    assert h1 == h2 == h3


def test_empty_pipeline_graceful():
    """Empty paper list flows through entire pipeline without errors."""
    lt = LayeredTriage(runtime=None)
    scorer = SourceScorer(runtime=None)
    weighter = EvidenceWeighter()
    synth = SynthesisMatrix(runtime=None)

    papers = []
    results = lt.triage_batch(papers, "test")
    assert results == []

    scores = scorer.score_batch(papers, "test")
    assert scores == []

    evidence = weighter.weight("claim", papers, scores)
    assert evidence.weights == []
    assert evidence.weighted_confidence == 0.0


def test_mismatched_creds_papers_weighter():
    """Weighter handles mismatched list lengths gracefully."""
    weighter = EvidenceWeighter()
    creds = [CredibilityScore(0.5, 0.5, 0.5, 0.5, 0.5, 0.5)]
    papers = [
        {"year": 2025, "citation_count": 10},
        {"year": 2024, "citation_count": 5},
    ]
    # More papers than creds - should handle gracefully or error cleanly
    try:
        result = weighter.weight("claim", papers, creds)
        # If it succeeds, weights should be reasonable
        assert len(result.weights) >= 1
    except (IndexError, ValueError):
        pass  # Clean error is acceptable


def test_advocate_with_fetch_fn():
    """Devil's advocate with a mock fetch function produces valid report."""
    def mock_fetch(query):
        return [{"title": f"Counter: {query}", "abstract": "Opposing view."}]

    advocate = DevilsAdvocate(runtime=None, fetch_fn=mock_fetch)
    supporting = [{"title": "Supporting paper", "abstract": "Agrees with claim."}]
    report = advocate.challenge("AI improves productivity", supporting)
    assert report.supporting_count == 1
    assert report.opposing_count >= 1
    assert 0.0 <= report.balance_ratio <= 1.0
    assert len(report.counter_queries) >= 1


def test_verifier_without_fetch():
    """Verifier without fetch_fn returns unverified result."""
    verifier = ClaimVerifier(runtime=None)
    result = verifier.verify("Test claim")
    assert result.verified is False
    assert result.confidence == 0.0
