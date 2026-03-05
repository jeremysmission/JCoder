"""Hard iteration 2: Chaotic pipeline and state corruption simulation.

Tests:
- Out-of-order pipeline calls (screen before identify, include before eligible)
- Corrupted intermediate state passed between modules
- Module replacement/monkey-patching mid-pipeline
- PRISMA database corruption recovery
- Mismatched list lengths across modules
- Duplicate/recycled content hashes
- State leakage between separate pipeline runs
"""

import pytest
import os
import sqlite3
import time
import copy

from core.layered_triage import LayeredTriage, _content_hash
from core.source_scorer import SourceScorer, CredibilityScore
from core.evidence_weighter import EvidenceWeighter
from core.synthesis_matrix import SynthesisMatrix
from core.devils_advocate import DevilsAdvocate
from core.claim_verifier import ClaimVerifier
from core.prisma_tracker import PrismaTracker


# ---------------------------------------------------------------------------
# Out-of-order PRISMA calls
# ---------------------------------------------------------------------------

def test_prisma_screen_before_identify(tmp_path):
    """Screening a hash that was never identified."""
    tracker = PrismaTracker(db_path=str(tmp_path / "ooo.db"))
    # Screen something never identified
    tracker.screen("phantom_hash", passed=True, reason="mystery")
    counts = tracker.flow_counts()
    assert counts["screened"] == 1
    assert counts["identified"] == 0
    tracker.close()


def test_prisma_include_before_screen(tmp_path):
    """Include a paper that skipped screening and eligibility."""
    tracker = PrismaTracker(db_path=str(tmp_path / "skip.db"))
    tracker.identify("Paper A", "src", "h1")
    tracker.include("h1", reason="VIP paper")
    counts = tracker.flow_counts()
    assert counts["identified"] == 1
    assert counts["included"] == 1
    assert counts["screened"] == 0
    tracker.close()


def test_prisma_exclude_then_include(tmp_path):
    """Exclude a paper, then try to include it (both should log)."""
    tracker = PrismaTracker(db_path=str(tmp_path / "revive.db"))
    tracker.identify("Paper", "src", "h1")
    tracker.screen("h1", passed=False, reason="bad quality")
    tracker.include("h1", reason="reconsidered")
    counts = tracker.flow_counts()
    assert counts["excluded"] == 1
    assert counts["included"] == 1
    tracker.close()


def test_prisma_same_hash_multiple_identifies(tmp_path):
    """Identify the same hash 100 times."""
    tracker = PrismaTracker(db_path=str(tmp_path / "dup.db"))
    for i in range(100):
        tracker.identify(f"Title variant {i}", "src", "same_hash")
    counts = tracker.flow_counts()
    assert counts["identified"] == 100
    tracker.close()


# ---------------------------------------------------------------------------
# Corrupted intermediate data between modules
# ---------------------------------------------------------------------------

def test_corrupted_triage_results_to_scorer():
    """Pass corrupted TriageResult-like dicts to scorer."""
    scorer = SourceScorer(runtime=None)
    # Simulate papers that came from triage but got mangled
    mangled = [
        {"title": None, "abstract": None, "year": None, "satellite_score": 0.5},
        {"title": "", "year": "not_a_year", "citation_count": "many"},
        {"title": "Valid", "year": 2025, "citation_count": 10},
    ]
    scores = scorer.score_batch(mangled, "test")
    assert len(scores) == 3
    for s in scores:
        assert 0.0 <= s.composite <= 1.0


def test_mismatched_creds_length():
    """Weighter receives different-length sources and creds."""
    weighter = EvidenceWeighter()
    creds = [CredibilityScore(0.5, 0.5, 0.5, 0.5, 0.5, 0.5)]
    papers = [
        {"year": 2025, "citation_count": 10},
        {"year": 2024, "citation_count": 5},
        {"year": 2023, "citation_count": 1},
    ]
    with pytest.raises(ValueError):
        weighter.weight("claim", papers, creds)


def test_weighter_with_nan_credibility():
    """CredibilityScore with NaN values."""
    weighter = EvidenceWeighter()
    nan = float('nan')
    creds = [
        CredibilityScore(nan, nan, nan, nan, nan, nan),
        CredibilityScore(0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
    ]
    papers = [{"year": 2025}, {"year": 2024}]
    result = weighter.weight("claim", papers, creds)
    assert len(result.weights) == 2
    # NaN composite should result in fallback or zero weight
    assert abs(sum(result.weights) - 1.0) < 0.01 or all(w >= 0 for w in result.weights)


# ---------------------------------------------------------------------------
# PRISMA database corruption recovery
# ---------------------------------------------------------------------------

def test_prisma_corrupt_db_file(tmp_path):
    """Write garbage to the db file, then try to open a tracker."""
    db_path = str(tmp_path / "corrupt.db")
    with open(db_path, "wb") as f:
        f.write(b"THIS IS NOT A SQLITE DATABASE" * 100)
    with pytest.raises(Exception):
        tracker = PrismaTracker(db_path=db_path)


def test_prisma_missing_table(tmp_path):
    """Database exists but table is missing."""
    db_path = str(tmp_path / "notable.db")
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE other_table (id INTEGER)")
    conn.commit()
    conn.close()
    # PrismaTracker should create its table
    tracker = PrismaTracker(db_path=db_path)
    tracker.identify("Paper", "src", "h1")
    assert tracker.flow_counts()["identified"] == 1
    tracker.close()


def test_prisma_readonly_after_close(tmp_path):
    """After close, the tracker should fail on writes."""
    tracker = PrismaTracker(db_path=str(tmp_path / "ro.db"))
    tracker.identify("Paper", "src", "h1")
    tracker.close()
    with pytest.raises(Exception):
        tracker.identify("Paper 2", "src", "h2")


# ---------------------------------------------------------------------------
# State leakage between pipeline runs
# ---------------------------------------------------------------------------

def test_triage_no_state_leakage():
    """Two separate triage runs on different data don't cross-contaminate."""
    lt = LayeredTriage(runtime=None)
    papers_a = [{"title": f"Alpha {i}", "abstract": f"About alpha {i}.", "year": 2025} for i in range(50)]
    papers_b = [{"title": f"Beta {i}", "abstract": f"About beta {i}.", "year": 2020} for i in range(30)]

    results_a = lt.triage_batch(papers_a, "alpha research")
    results_b = lt.triage_batch(papers_b, "beta research")

    assert len(results_a) == 50
    assert len(results_b) == 30
    # Titles should not cross-contaminate
    titles_a = {r.title for r in results_a}
    titles_b = {r.title for r in results_b}
    assert titles_a.isdisjoint(titles_b)


def test_scorer_no_state_leakage():
    """Sequential scoring runs produce independent results."""
    scorer = SourceScorer(runtime=None)
    p1 = {"title": "Old Low Paper", "year": 2005, "citation_count": 0}
    p2 = {"title": "New High Paper", "year": 2026, "citation_count": 1000, "source": "arxiv_cs_ai",
          "has_code": True, "has_refs": True}
    s1 = scorer.score(p1, "test")
    s2 = scorer.score(p2, "test")
    # Scores should reflect the individual papers, not accumulate
    assert s2.composite > s1.composite


# ---------------------------------------------------------------------------
# Recycled content hashes
# ---------------------------------------------------------------------------

def test_triage_recycled_hashes():
    """Papers with same hash (identical title+abstract[:200]) but different metadata."""
    lt = LayeredTriage(runtime=None)
    papers = [
        {"title": "Same Title", "abstract": "Same abstract.", "year": 2026, "citation_count": 1000, "source": "arxiv_cs_ai"},
        {"title": "Same Title", "abstract": "Same abstract.", "year": 2005, "citation_count": 0, "source": "reddit"},
    ]
    results = lt.triage_batch(papers, "test")
    # Both papers should appear in results even with same hash
    assert len(results) == 2


# ---------------------------------------------------------------------------
# Extreme pipeline chaining
# ---------------------------------------------------------------------------

def test_pipeline_10_sequential_runs():
    """Run full pipeline 10 times sequentially, verify no degradation."""
    for run in range(10):
        papers = [
            {"title": f"Run{run} Paper {i}", "abstract": f"Study {i}.",
             "year": 2020 + (i % 7), "citation_count": i * 5, "source": "arxiv_cs_ai"}
            for i in range(100)
        ]
        lt = LayeredTriage(runtime=None)
        results = lt.triage_batch(papers, "study", satellite_cutoff=0.1)
        assert len(results) == 100

        scorer = SourceScorer(runtime=None)
        creds = scorer.score_batch(papers, "study")
        assert len(creds) == 100

        weighter = EvidenceWeighter()
        evidence = weighter.weight(f"claim {run}", papers, creds)
        assert abs(sum(evidence.weights) - 1.0) < 0.01


def test_synthesis_after_filter():
    """Synthesize only papers that passed triage filter."""
    lt = LayeredTriage(runtime=None)
    papers = [
        {"title": f"Paper {i}: retrieval methods", "abstract": f"Dense retrieval study {i}.",
         "year": 2020 + i, "citation_count": i * 20, "source": "arxiv_cs_ai",
         "key_claims": [f"Claim {i}: works well"]}
        for i in range(50)
    ]
    results = lt.triage_batch(papers, "retrieval", satellite_cutoff=0.4)
    passing_hashes = {r.content_hash for r in results if r.satellite_pass}
    filtered_papers = [p for p in papers if _content_hash(p) in passing_hashes]

    if filtered_papers:
        synth = SynthesisMatrix(runtime=None)
        report = synth.build(filtered_papers, "retrieval")
        assert report.total_sources == len(filtered_papers)
        assert report.total_sources <= 50


# ---------------------------------------------------------------------------
# Advocate and verifier with corrupted inputs
# ---------------------------------------------------------------------------

def test_advocate_with_non_dict_papers():
    """Supporting papers list contains non-dict items."""
    da = DevilsAdvocate(runtime=None)
    # Mix of valid dicts and garbage
    supporting = [
        {"title": "Valid Paper"},
        "not a dict",
        42,
        None,
        {"title": "Another Valid"},
    ]
    # Should handle non-dict items without crash
    report = da.challenge("test claim", supporting)
    assert report.supporting_count == 5  # counts all items


def test_verifier_with_none_original():
    """Verifier with None as original source."""
    verifier = ClaimVerifier(runtime=None)
    result = verifier.verify("Some claim")
    assert result.verified is False


# ---------------------------------------------------------------------------
# Database size growth
# ---------------------------------------------------------------------------

def test_prisma_db_size_reasonable(tmp_path):
    """10000 operations produce a database < 5MB."""
    db_path = str(tmp_path / "size.db")
    tracker = PrismaTracker(db_path=db_path)
    for i in range(5000):
        h = f"h{i}"
        tracker.identify(f"Paper {i} with a somewhat long title for size testing", "source_name", h)
        tracker.screen(h, passed=(i % 2 == 0), reason="auto-screen reason text here")
    tracker.close()
    file_size = os.path.getsize(db_path)
    assert file_size < 5_000_000, f"Database is {file_size / 1_000_000:.1f}MB (expected < 5MB)"
