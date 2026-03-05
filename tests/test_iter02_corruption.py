"""Iteration 2: Data quality, corruption, and malicious input tests."""

import pytest
from core.layered_triage import LayeredTriage, _content_hash
from core.source_scorer import SourceScorer
from core.evidence_weighter import EvidenceWeighter
from core.synthesis_matrix import SynthesisMatrix
from core.devils_advocate import DevilsAdvocate
from core.claim_verifier import ClaimVerifier
from core.prisma_tracker import PrismaTracker


# ---------------------------------------------------------------------------
# Huge strings (memory safety)
# ---------------------------------------------------------------------------

def test_triage_huge_title():
    """Paper with 100KB title doesn't OOM or hang."""
    lt = LayeredTriage(runtime=None)
    papers = [{"title": "A" * 100_000, "abstract": "B" * 100_000, "year": 2025}]
    results = lt.triage_batch(papers, "test")
    assert len(results) == 1


def test_scorer_huge_abstract():
    """Scorer handles 100KB abstract."""
    scorer = SourceScorer(runtime=None)
    paper = {"title": "X", "abstract": "research " * 50_000, "year": 2025}
    score = scorer.score(paper, "research")
    assert 0.0 <= score.composite <= 1.0


def test_hash_huge_abstract():
    """Content hash doesn't process full 100KB -- truncates at 200 chars."""
    import time
    p = {"title": "T", "abstract": "X" * 100_000}
    t0 = time.time()
    for _ in range(1000):
        _content_hash(p)
    elapsed = time.time() - t0
    assert elapsed < 1.0, f"1000 hashes of 100KB paper took {elapsed:.2f}s"


# ---------------------------------------------------------------------------
# Special characters and encoding
# ---------------------------------------------------------------------------

def test_triage_sql_injection_title():
    """SQL-like strings in title don't break anything."""
    lt = LayeredTriage(runtime=None)
    papers = [
        {"title": "'; DROP TABLE papers; --", "abstract": "Test.", "year": 2025},
        {"title": 'Robert"); DELETE FROM prisma_log;', "abstract": "Test.", "year": 2025},
    ]
    results = lt.triage_batch(papers, "test")
    assert len(results) == 2


def test_prisma_sql_injection(tmp_path):
    """PRISMA tracker resists SQL injection in title/reason/source."""
    tracker = PrismaTracker(db_path=str(tmp_path / "p.db"))
    tracker.identify("'; DROP TABLE prisma_log; --", "evil_source", "hash1")
    tracker.screen("hash1", passed=True, reason="'; DELETE FROM prisma_log; --")
    counts = tracker.flow_counts()
    assert counts["identified"] == 1
    assert counts["screened"] == 1
    tracker.close()


def test_triage_newlines_tabs_in_title():
    """Newlines and tabs in paper fields don't break parsing."""
    lt = LayeredTriage(runtime=None)
    papers = [{"title": "Paper\nWith\nNewlines\tAnd\tTabs",
               "abstract": "Abstract\r\nWith\r\nCRLF.", "year": 2025}]
    results = lt.triage_batch(papers, "test")
    assert len(results) == 1


def test_triage_zero_length_strings():
    """All fields are empty strings."""
    lt = LayeredTriage(runtime=None)
    papers = [{"title": "", "abstract": "", "year": 0, "citation_count": 0, "source": ""}]
    results = lt.triage_batch(papers, "")
    assert len(results) == 1


# ---------------------------------------------------------------------------
# Extreme numeric values
# ---------------------------------------------------------------------------

def test_scorer_extreme_year():
    """Year 0 and year 9999 don't break scoring."""
    scorer = SourceScorer(runtime=None)
    for year in [0, -100, 9999, 2026]:
        score = scorer.score({"title": "X", "year": year}, "test")
        assert 0.0 <= score.currency <= 1.0, f"year={year} broke currency"


def test_scorer_extreme_citations():
    """Citations from 0 to 1 million don't break scoring."""
    scorer = SourceScorer(runtime=None)
    for cites in [0, 1, 100, 10000, 1000000]:
        score = scorer.score({"title": "X", "citation_count": cites}, "test")
        assert 0.0 <= score.authority <= 1.0, f"cites={cites} broke authority"


def test_weighter_extreme_weights():
    """One source with 1M citations doesn't make all others zero."""
    weighter = EvidenceWeighter()
    from core.source_scorer import CredibilityScore
    creds = [
        CredibilityScore(0.8, 0.9, 0.9, 0.9, 0.8, 0.86),
        CredibilityScore(0.5, 0.5, 0.3, 0.3, 0.5, 0.42),
    ]
    papers = [
        {"year": 2026, "citation_count": 1000000},
        {"year": 2024, "citation_count": 1},
    ]
    result = weighter.weight("claim", papers, creds)
    # Small source should still have SOME weight (not exactly 0)
    assert result.weights[1] > 0.0, "Small source got zero weight"
    assert abs(sum(result.weights) - 1.0) < 0.001


# ---------------------------------------------------------------------------
# Duplicate and identical papers
# ---------------------------------------------------------------------------

def test_triage_all_identical_papers():
    """10 identical papers should all get same score."""
    lt = LayeredTriage(runtime=None)
    paper = {"title": "Same Paper", "abstract": "Same abstract.", "year": 2025, "source": "arxiv_cs_ai"}
    papers = [dict(paper) for _ in range(10)]
    results = lt.triage_batch(papers, "test")
    scores = [r.satellite_score for r in results]
    assert len(set(scores)) == 1, "Identical papers got different scores"


def test_hash_identical_papers():
    """Identical papers produce identical hashes."""
    p = {"title": "Same", "abstract": "Same."}
    hashes = [_content_hash(p) for _ in range(100)]
    assert len(set(hashes)) == 1


# ---------------------------------------------------------------------------
# Mixed-type fields
# ---------------------------------------------------------------------------

def test_triage_year_as_string():
    """Year provided as string instead of int doesn't crash."""
    lt = LayeredTriage(runtime=None)
    papers = [{"title": "Paper", "abstract": "Test.", "year": "2025"}]
    # May not score correctly but should not crash
    try:
        results = lt.triage_batch(papers, "test")
        assert len(results) == 1
    except TypeError:
        pytest.fail("String year caused TypeError")


def test_triage_citations_as_float():
    """Citation count as float doesn't crash."""
    lt = LayeredTriage(runtime=None)
    papers = [{"title": "Paper", "abstract": "Test.", "citation_count": 50.7}]
    results = lt.triage_batch(papers, "test")
    assert len(results) == 1


def test_scorer_boolean_has_code():
    """has_code as various truthy/falsy values."""
    scorer = SourceScorer(runtime=None)
    for val in [True, False, 1, 0, "yes", "", None]:
        paper = {"title": "X", "has_code": val, "has_refs": val}
        score = scorer.score(paper, "test")
        assert 0.0 <= score.accuracy <= 1.0


# ---------------------------------------------------------------------------
# Empty collections
# ---------------------------------------------------------------------------

def test_weighter_empty_sources():
    """Empty source list doesn't crash."""
    weighter = EvidenceWeighter()
    result = weighter.weight("claim", [], [])
    assert result.weighted_confidence == 0.0
    assert result.weights == []


def test_synthesis_empty_claims():
    """Papers with no key_claims still produce matrix."""
    synth = SynthesisMatrix(runtime=None)
    papers = [
        {"title": "P1", "abstract": "about retrieval"},
        {"title": "P2", "abstract": "about generation"},
    ]
    report = synth.build(papers, "methods")
    assert report.total_sources == 2


def test_advocate_empty_claim():
    """Empty string claim doesn't crash."""
    advocate = DevilsAdvocate(runtime=None)
    report = advocate.challenge("", [])
    assert report.balance_ratio == 0.0


def test_verifier_empty_claim():
    """Empty string claim doesn't crash."""
    verifier = ClaimVerifier(runtime=None)
    result = verifier.verify("")
    assert result.verified is False
