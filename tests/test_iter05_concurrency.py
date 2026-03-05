"""Iteration 5: Concurrency, state management, and resource cleanup."""

import pytest
import threading
import time
from core.layered_triage import LayeredTriage, _content_hash
from core.source_scorer import SourceScorer
from core.evidence_weighter import EvidenceWeighter
from core.prisma_tracker import PrismaTracker


# ---------------------------------------------------------------------------
# Thread safety of stateless modules
# ---------------------------------------------------------------------------

def test_triage_concurrent_batches():
    """Two threads triaging different batches concurrently."""
    lt = LayeredTriage(runtime=None)
    results = {}
    errors = []

    def triage_batch(name, papers, query):
        try:
            r = lt.triage_batch(papers, query, satellite_cutoff=0.1)
            results[name] = r
        except Exception as e:
            errors.append((name, e))

    papers_a = [{"title": f"Alpha {i}", "abstract": f"Research {i}.", "year": 2025} for i in range(50)]
    papers_b = [{"title": f"Beta {i}", "abstract": f"Study {i}.", "year": 2024} for i in range(50)]

    t1 = threading.Thread(target=triage_batch, args=("a", papers_a, "research"))
    t2 = threading.Thread(target=triage_batch, args=("b", papers_b, "study"))
    t1.start()
    t2.start()
    t1.join(timeout=10)
    t2.join(timeout=10)

    assert not errors, f"Errors in concurrent triage: {errors}"
    assert len(results["a"]) == 50
    assert len(results["b"]) == 50


def test_scorer_concurrent():
    """Concurrent scoring doesn't corrupt results."""
    scorer = SourceScorer(runtime=None)
    results = {}
    errors = []

    def score_batch(name, papers):
        try:
            r = scorer.score_batch(papers, "test")
            results[name] = r
        except Exception as e:
            errors.append((name, e))

    papers_a = [{"title": f"Paper A{i}", "year": 2025, "citation_count": i * 10} for i in range(50)]
    papers_b = [{"title": f"Paper B{i}", "year": 2020, "citation_count": i} for i in range(50)]

    t1 = threading.Thread(target=score_batch, args=("a", papers_a))
    t2 = threading.Thread(target=score_batch, args=("b", papers_b))
    t1.start()
    t2.start()
    t1.join(timeout=10)
    t2.join(timeout=10)

    assert not errors, f"Errors in concurrent scoring: {errors}"
    assert len(results["a"]) == 50
    assert len(results["b"]) == 50


def test_hash_concurrent():
    """Content hash is thread-safe (pure function)."""
    papers = [{"title": f"Paper {i}", "abstract": f"Abstract {i}."} for i in range(100)]
    expected = [_content_hash(p) for p in papers]
    results = [None] * 100
    errors = []

    def hash_range(start, end):
        try:
            for i in range(start, end):
                results[i] = _content_hash(papers[i])
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=hash_range, args=(i * 25, (i + 1) * 25)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert not errors
    assert results == expected


# ---------------------------------------------------------------------------
# PRISMA state management
# ---------------------------------------------------------------------------

def test_prisma_reopen_preserves_data(tmp_path):
    """Closing and reopening PRISMA tracker preserves all data."""
    db_path = str(tmp_path / "reopen.db")

    tracker = PrismaTracker(db_path=db_path)
    tracker.identify("Paper 1", "arxiv", "hash1")
    tracker.identify("Paper 2", "arxiv", "hash2")
    tracker.screen("hash1", passed=True, reason="good")
    tracker.screen("hash2", passed=False, reason="irrelevant")
    counts_before = tracker.flow_counts()
    tracker.close()

    # Reopen
    tracker2 = PrismaTracker(db_path=db_path)
    counts_after = tracker2.flow_counts()
    assert counts_after == counts_before
    tracker2.close()


def test_prisma_double_close(tmp_path):
    """Double close doesn't crash."""
    tracker = PrismaTracker(db_path=str(tmp_path / "double.db"))
    tracker.identify("Paper", "src", "h1")
    tracker.close()
    # Second close should not raise
    try:
        tracker.close()
    except Exception as e:
        pytest.fail(f"Double close raised: {e}")


def test_prisma_operations_after_close(tmp_path):
    """Operations after close raise clean error, not corruption."""
    tracker = PrismaTracker(db_path=str(tmp_path / "closed.db"))
    tracker.identify("Paper", "src", "h1")
    tracker.close()

    with pytest.raises(Exception):
        tracker.identify("Paper 2", "src", "h2")


# ---------------------------------------------------------------------------
# Repeated operations (idempotency-like)
# ---------------------------------------------------------------------------

def test_triage_same_batch_twice():
    """Running same batch twice produces identical results."""
    lt = LayeredTriage(runtime=None)
    papers = [
        {"title": f"Paper {i}", "abstract": f"Study on topic {i}.",
         "year": 2020 + i, "citation_count": i * 5, "source": "arxiv_cs_ai"}
        for i in range(20)
    ]
    r1 = lt.triage_batch(papers, "topic study", satellite_cutoff=0.2)
    r2 = lt.triage_batch(papers, "topic study", satellite_cutoff=0.2)

    scores1 = [(r.content_hash, r.satellite_score) for r in r1]
    scores2 = [(r.content_hash, r.satellite_score) for r in r2]
    assert scores1 == scores2


def test_scorer_idempotent():
    """Scoring same paper twice gives identical results."""
    scorer = SourceScorer(runtime=None)
    paper = {"title": "Test", "abstract": "Research.", "year": 2025,
             "citation_count": 100, "source": "semantic_scholar"}
    s1 = scorer.score(paper, "research")
    s2 = scorer.score(paper, "research")
    assert s1.composite == s2.composite
    assert s1.currency == s2.currency
    assert s1.authority == s2.authority


def test_weighter_idempotent():
    """Weighting same inputs twice gives identical results."""
    from core.source_scorer import CredibilityScore
    weighter = EvidenceWeighter()
    creds = [
        CredibilityScore(0.8, 0.8, 0.8, 0.8, 0.8, 0.8),
        CredibilityScore(0.4, 0.4, 0.4, 0.4, 0.4, 0.4),
    ]
    papers = [{"year": 2025, "citation_count": 50}, {"year": 2020, "citation_count": 5}]
    r1 = weighter.weight("claim", papers, creds)
    r2 = weighter.weight("claim", papers, creds)
    assert r1.weights == r2.weights
    assert r1.weighted_confidence == r2.weighted_confidence


# ---------------------------------------------------------------------------
# Memory / resource cleanup
# ---------------------------------------------------------------------------

def test_prisma_many_open_close_cycles(tmp_path):
    """Opening and closing tracker 50 times doesn't leak resources."""
    db_path = str(tmp_path / "cycles.db")
    for i in range(50):
        tracker = PrismaTracker(db_path=db_path)
        tracker.identify(f"Paper {i}", "src", f"hash{i}")
        tracker.close()

    # Final check: all 50 papers should be there
    tracker = PrismaTracker(db_path=db_path)
    counts = tracker.flow_counts()
    assert counts["identified"] == 50
    tracker.close()


def test_large_batch_memory():
    """Processing 2000 papers doesn't accumulate excessive memory."""
    import sys
    lt = LayeredTriage(runtime=None)
    papers = [{"title": f"P{i}", "abstract": f"A{i}", "year": 2025} for i in range(2000)]
    results = lt.triage_batch(papers, "test", satellite_cutoff=0.0)
    assert len(results) == 2000
    # Results should be reasonable size (not holding huge refs)
    total_size = sum(sys.getsizeof(r.title) for r in results)
    assert total_size < 1_000_000, f"Results hold {total_size} bytes of titles"
