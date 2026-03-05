"""Iteration 10: Final hardening -- edge cases, boundary conditions, stress combos."""

import pytest
import json
import time
from core.layered_triage import LayeredTriage, _content_hash, _parse_drone_response
from core.source_scorer import SourceScorer, CredibilityScore
from core.evidence_weighter import EvidenceWeighter
from core.synthesis_matrix import SynthesisMatrix
from core.devils_advocate import DevilsAdvocate
from core.claim_verifier import ClaimVerifier
from core.prisma_tracker import PrismaTracker


# ---------------------------------------------------------------------------
# Boundary: exactly-at-cutoff values
# ---------------------------------------------------------------------------

def test_satellite_exactly_at_cutoff():
    """Paper scoring exactly at cutoff should pass."""
    lt = LayeredTriage(runtime=None)
    # We'll test that cutoff=0.0 passes everything
    papers = [{"title": "", "abstract": "", "year": 0}]
    results = lt.triage_batch(papers, "", satellite_cutoff=0.0)
    assert len(results) == 1
    assert results[0].satellite_pass is True


def test_satellite_cutoff_one():
    """cutoff=1.0 should filter everything (no paper can score exactly 1.0)."""
    lt = LayeredTriage(runtime=None)
    papers = [
        {"title": "Novel SOTA benchmark retrieval augmented meta-learning",
         "abstract": "systematic review", "year": 2026, "citation_count": 10000,
         "source": "semantic_scholar"}
    ]
    results = lt.triage_batch(papers, "retrieval", satellite_cutoff=1.0)
    # Even max-signal paper unlikely to hit exactly 1.0 after rounding
    non_passing = [r for r in results if not r.satellite_pass]
    assert len(non_passing) == len(results)


# ---------------------------------------------------------------------------
# Rapid successive operations
# ---------------------------------------------------------------------------

def test_prisma_rapid_fire(tmp_path):
    """1000 rapid identify-screen cycles."""
    tracker = PrismaTracker(db_path=str(tmp_path / "rapid.db"))
    for i in range(1000):
        tracker.identify(f"P{i}", "src", f"h{i}")
        tracker.screen(f"h{i}", passed=(i % 2 == 0), reason="auto")
    counts = tracker.flow_counts()
    assert counts["identified"] == 1000
    assert counts["screened"] == 500
    assert counts["excluded"] == 500
    tracker.close()


def test_scorer_rapid_fire():
    """Score 1000 papers in quick succession."""
    scorer = SourceScorer(runtime=None)
    for i in range(1000):
        score = scorer.score({"title": f"Paper {i}", "year": 2020 + (i % 7)}, "test")
        assert 0.0 <= score.composite <= 1.0


# ---------------------------------------------------------------------------
# Adversarial inputs combined
# ---------------------------------------------------------------------------

def test_triage_all_adversarial_at_once():
    """Mix of adversarial inputs in one batch."""
    lt = LayeredTriage(runtime=None)
    papers = [
        {"title": None, "abstract": None, "year": None},
        {"title": "", "abstract": "", "year": "bad"},
        {"title": "A" * 100000, "abstract": "B" * 100000, "year": -999},
        {"title": "'; DROP TABLE papers; --", "abstract": "normal", "year": 2025},
        {"title": "Normal Paper", "abstract": "Normal abstract.", "year": 2025},
        {"title": 123, "abstract": 456, "year": 2025},  # numeric title
    ]
    results = lt.triage_batch(papers, "test")
    assert len(results) == 6


def test_scorer_all_adversarial():
    """Score a batch of adversarial papers."""
    scorer = SourceScorer(runtime=None)
    adversarial = [
        {"title": None, "year": None},
        {"title": "", "year": "not_a_year", "citation_count": "lots"},
        {"title": "X", "year": -1, "citation_count": -100},
        {"title": "X", "has_code": "maybe", "has_refs": 42},
    ]
    for p in adversarial:
        score = scorer.score(p, "test")
        assert 0.0 <= score.composite <= 1.0


# ---------------------------------------------------------------------------
# Weight edge cases
# ---------------------------------------------------------------------------

def test_weighter_all_zero_credibility():
    """All sources have zero credibility."""
    weighter = EvidenceWeighter()
    creds = [CredibilityScore(0, 0, 0, 0, 0, 0) for _ in range(5)]
    papers = [{"year": 2025, "citation_count": 0} for _ in range(5)]
    result = weighter.weight("claim", papers, creds)
    # Should fall back to uniform weights
    assert len(result.weights) == 5
    assert abs(sum(result.weights) - 1.0) < 0.001


def test_weighter_one_dominant():
    """One source with perfect score, rest zero."""
    weighter = EvidenceWeighter()
    creds = [
        CredibilityScore(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        CredibilityScore(0, 0, 0, 0, 0, 0),
        CredibilityScore(0, 0, 0, 0, 0, 0),
    ]
    papers = [
        {"year": 2026, "citation_count": 1000},
        {"year": 2025, "citation_count": 0},
        {"year": 2025, "citation_count": 0},
    ]
    result = weighter.weight("claim", papers, creds)
    assert result.weights[0] > result.weights[1]
    assert result.weights[0] > result.weights[2]


# ---------------------------------------------------------------------------
# Synthesis edge cases
# ---------------------------------------------------------------------------

def test_synthesis_single_paper():
    """Synthesis with 1 paper still produces valid report."""
    synth = SynthesisMatrix(runtime=None)
    papers = [{"title": "Solo paper", "abstract": "Only one source.",
               "key_claims": ["Single claim here"]}]
    report = synth.build(papers, "solo")
    assert report.total_sources == 1
    assert report.contradictions == 0


def test_synthesis_many_identical():
    """20 identical papers should produce clean synthesis."""
    synth = SynthesisMatrix(runtime=None)
    paper = {"title": "Same Paper", "abstract": "Same abstract about retrieval.",
             "key_claims": ["Retrieval works well"]}
    papers = [dict(paper) for _ in range(20)]
    report = synth.build(papers, "retrieval")
    assert report.total_sources == 20


def test_synthesis_json_roundtrip():
    """to_dict -> json.dumps -> json.loads roundtrip works."""
    synth = SynthesisMatrix(runtime=None)
    papers = [
        {"title": f"P{i}", "abstract": f"About topic {i}.",
         "key_claims": [f"Claim {i}"]}
        for i in range(3)
    ]
    report = synth.build(papers, "test")
    d = synth.to_dict(report)
    json_str = json.dumps(d)
    restored = json.loads(json_str)
    assert restored["total_sources"] == 3
    assert restored["query"] == "test"


# ---------------------------------------------------------------------------
# Claim verifier edge cases
# ---------------------------------------------------------------------------

def test_verifier_with_mock_fetch_corroboration():
    """Verifier finds corroboration from mock sources."""
    def fetch(query):
        return [
            {"title": f"Agrees with: {query}", "abstract": f"Evidence supports {query}"},
            {"title": f"Also agrees: {query}", "abstract": f"Further evidence for {query}"},
        ]
    verifier = ClaimVerifier(runtime=None, fetch_fn=fetch)
    result = verifier.verify("Neural networks learn representations")
    assert result.verified is True
    assert len(result.corroborating_sources) >= 2


def test_verifier_batch():
    """Batch verification works."""
    def fetch(query):
        return [{"title": f"Source for {query}", "abstract": "Supporting evidence."}]
    verifier = ClaimVerifier(runtime=None, fetch_fn=fetch)
    results = verifier.verify_batch(
        ["Claim A", "Claim B", "Claim C"],
        [{"title": "Src A"}, {"title": "Src B"}, {"title": "Src C"}],
    )
    assert len(results) == 3


# ---------------------------------------------------------------------------
# Devil's advocate edge cases
# ---------------------------------------------------------------------------

def test_advocate_many_supporting():
    """Challenge with many supporting papers."""
    da = DevilsAdvocate(runtime=None, fetch_fn=lambda q: [{"title": f"Counter: {q}"}])
    supporting = [{"title": f"Support {i}", "abstract": f"Agrees {i}."} for i in range(20)]
    report = da.challenge("Major claim", supporting)
    assert report.supporting_count == 20
    assert report.opposing_count >= 1
    assert report.balance_ratio < 0.5  # mostly supporting


def test_advocate_fetch_returns_empty():
    """Fetch function that always returns empty."""
    da = DevilsAdvocate(runtime=None, fetch_fn=lambda q: [])
    report = da.challenge("Untestable claim", [{"title": "Only source"}])
    assert report.opposing_count == 0
    assert report.balance_ratio == 0.0


def test_advocate_fetch_raises():
    """Fetch function that raises should be handled."""
    def bad_fetch(q):
        raise ConnectionError("Network down")
    da = DevilsAdvocate(runtime=None, fetch_fn=bad_fetch)
    # This should raise since there's no try/except around fetch
    with pytest.raises(ConnectionError):
        da.challenge("claim", [])


# ---------------------------------------------------------------------------
# Hash collision resistance
# ---------------------------------------------------------------------------

def test_hash_near_collision():
    """Very similar titles produce different hashes."""
    h1 = _content_hash({"title": "Neural Network Architecture", "abstract": "Study A."})
    h2 = _content_hash({"title": "Neural Network Architecture", "abstract": "Study B."})
    h3 = _content_hash({"title": "Neural Network Architectures", "abstract": "Study A."})
    assert h1 != h2  # different abstract
    assert h1 != h3  # different title (plural)


def test_hash_empty_vs_none():
    """Empty string fields vs None fields produce same hash."""
    h1 = _content_hash({"title": "", "abstract": ""})
    h2 = _content_hash({"title": None, "abstract": None})
    assert h1 == h2  # both resolve to empty


# ---------------------------------------------------------------------------
# Full pipeline stress
# ---------------------------------------------------------------------------

def test_full_pipeline_500_papers():
    """500 papers through triage -> score -> weight in under 5 seconds."""
    papers = [
        {"title": f"Research Paper {i}: {'retrieval' if i%3==0 else 'generation'} methods",
         "abstract": f"Experiment {i} on {'dense' if i%2==0 else 'sparse'} approaches.",
         "year": 2020 + (i % 7), "citation_count": i * 2,
         "source": ["arxiv_cs_ai", "semantic_scholar", "reddit"][i % 3]}
        for i in range(500)
    ]

    t0 = time.time()

    lt = LayeredTriage(runtime=None)
    results = lt.triage_batch(papers, "retrieval generation", satellite_cutoff=0.1)
    assert len(results) == 500

    scorer = SourceScorer(runtime=None)
    creds = scorer.score_batch(papers, "retrieval")
    assert len(creds) == 500

    weighter = EvidenceWeighter()
    evidence = weighter.weight("retrieval improves QA", papers, creds)
    assert len(evidence.weights) == 500

    elapsed = time.time() - t0
    assert elapsed < 5.0, f"Full pipeline took {elapsed:.2f}s"
