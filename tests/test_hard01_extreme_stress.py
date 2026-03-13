"""Hard iteration 1: Extreme adversarial stress simulation.

10x harder than previous rounds. Combines:
- 5000+ paper batches
- Deeply malicious/corrupt inputs mixed with valid data
- Concurrent abuse from multiple threads
- Cascading failures across modules
- Memory pressure from huge payloads
- Timing attacks and rapid-fire abuse
"""

import pytest
import threading
import time
import hashlib
import random
import sys

from core.layered_triage import LayeredTriage, _content_hash, _parse_drone_response
from core.source_scorer import SourceScorer, CredibilityScore
from core.evidence_weighter import EvidenceWeighter
from core.synthesis_matrix import SynthesisMatrix
from core.devils_advocate import DevilsAdvocate
from core.claim_verifier import ClaimVerifier
from core.prisma_tracker import PrismaTracker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _poison_papers(n):
    """Generate n papers with progressively more adversarial content."""
    papers = []
    for i in range(n):
        kind = i % 20
        if kind == 0:
            papers.append({"title": None, "abstract": None, "year": None})
        elif kind == 1:
            papers.append({"title": 12345, "abstract": 67890, "year": "bad"})
        elif kind == 2:
            papers.append({"title": "A" * 500_000, "abstract": "B" * 500_000, "year": -9999})
        elif kind == 3:
            papers.append({"title": "'; DROP TABLE papers; --", "abstract": "Robert'); DELETE *;--", "year": 2025})
        elif kind == 4:
            papers.append({"title": "\x00\x01\x02\x03\x04", "abstract": "\xff\xfe\xfd", "year": 2025})
        elif kind == 5:
            papers.append({"title": "Normal Paper", "abstract": "Normal abstract.", "year": 2025,
                           "citation_count": 50, "source": "arxiv_cs_ai"})
        elif kind == 6:
            papers.append({"title": "", "abstract": "", "year": 0, "citation_count": 0, "source": ""})
        elif kind == 7:
            papers.append({"title": "Paper\nWith\nNewlines\tAnd\tTabs\rAnd\rCR",
                           "abstract": "Abstract\r\n\r\nWith\r\nCRLF.", "year": 2025})
        elif kind == 8:
            papers.append({"title": True, "abstract": False, "year": [2025], "citation_count": {10: 20}})
        elif kind == 9:
            papers.append({"title": "X", "year": 99999, "citation_count": -99999999})
        elif kind == 10:
            papers.append({"title": "X", "year": 2025, "citation_count": float('inf')})
        elif kind == 11:
            papers.append({"title": "X", "year": 2025, "citation_count": float('nan')})
        elif kind == 12:
            papers.append({"title": "X" * 50, "abstract": "Y" * 50, "year": 2025,
                           "has_code": "maybe", "has_refs": 42, "peer_reviewed": "yes"})
        elif kind == 13:
            papers.append({"title": "<script>alert('xss')</script>", "abstract": "{{template injection}}", "year": 2025})
        elif kind == 14:
            papers.append({"title": "Paper", "abstract": "a " * 100_000, "year": 2025})
        elif kind == 15:
            papers.append({"title": f"Valid Paper {i}", "abstract": f"Study {i} on retrieval methods.",
                           "year": 2020 + (i % 7), "citation_count": i, "source": "semantic_scholar"})
        elif kind == 16:
            papers.append({})  # completely empty dict
        elif kind == 17:
            papers.append({"title": "X", "year": 2025, "extra_field_1": object(), "extra_field_2": lambda x: x})
        elif kind == 18:
            papers.append({"title": "  \t  \n  ", "abstract": "  \r\n  ", "year": "  2025  "})
        elif kind == 19:
            papers.append({"title": "Paper", "abstract": "Test.", "year": 2025,
                           "citation_count": 10**18})  # quintillion citations
    return papers


# ---------------------------------------------------------------------------
# Test 1: 5000 poison papers through triage
# ---------------------------------------------------------------------------

def test_triage_5000_poison_papers():
    """5000 papers with 25% adversarial content through triage in under 10s."""
    lt = LayeredTriage(runtime=None)
    papers = _poison_papers(5000)
    t0 = time.time()
    results = lt.triage_batch(papers, "retrieval augmented generation", satellite_cutoff=0.1)
    elapsed = time.time() - t0
    assert len(results) == 5000, f"Expected 5000, got {len(results)}"
    assert elapsed < 10.0, f"5000 poison papers took {elapsed:.2f}s"
    for r in results:
        assert 0.0 <= r.satellite_score <= 1.0


def test_scorer_5000_poison_papers():
    """Score 5000 adversarial papers without any crash."""
    scorer = SourceScorer(runtime=None)
    papers = _poison_papers(5000)
    t0 = time.time()
    scores = scorer.score_batch(papers, "test query")
    elapsed = time.time() - t0
    assert len(scores) == 5000
    assert elapsed < 10.0, f"Scoring took {elapsed:.2f}s"
    for s in scores:
        assert 0.0 <= s.composite <= 1.0


# ---------------------------------------------------------------------------
# Test 2: PRISMA under fire - 5000 papers with SQL injection
# ---------------------------------------------------------------------------

def test_prisma_5000_adversarial(tmp_path):
    """5000 papers with SQL injection attempts through PRISMA."""
    tracker = PrismaTracker(db_path=str(tmp_path / "stress.db"))
    papers = _poison_papers(5000)
    t0 = time.time()
    for i, p in enumerate(papers):
        title = str(p.get("title") or f"paper_{i}")
        h = f"hash_{i}"
        tracker.identify(title, str(p.get("source", "")), h)
        tracker.screen(h, passed=(i % 3 != 0), reason=str(p.get("abstract", ""))[:200])
    elapsed = time.time() - t0
    counts = tracker.flow_counts()
    assert counts["identified"] == 5000
    assert counts["screened"] + counts["excluded"] >= 5000
    assert elapsed < 15.0, f"PRISMA stress took {elapsed:.2f}s"
    tracker.close()


# ---------------------------------------------------------------------------
# Test 3: Concurrent triage from 10 threads
# ---------------------------------------------------------------------------

def test_10_thread_concurrent_triage():
    """10 threads each triaging 500 papers simultaneously."""
    lt = LayeredTriage(runtime=None)
    errors = []
    results = {}

    def worker(thread_id):
        try:
            papers = [
                {"title": f"T{thread_id}_Paper_{i}", "abstract": f"Abstract {i}.",
                 "year": 2020 + (i % 7), "source": "arxiv_cs_ai"}
                for i in range(500)
            ]
            r = lt.triage_batch(papers, f"query_{thread_id}", satellite_cutoff=0.1)
            results[thread_id] = len(r)
        except Exception as e:
            errors.append((thread_id, str(e)))

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(10)]
    t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    elapsed = time.time() - t0

    assert not errors, f"Thread errors: {errors}"
    assert len(results) == 10
    assert all(v == 500 for v in results.values())
    assert elapsed < 15.0, f"10-thread triage took {elapsed:.2f}s"


# ---------------------------------------------------------------------------
# Test 4: Concurrent PRISMA writes from 5 threads
# ---------------------------------------------------------------------------

def test_concurrent_prisma_writes(tmp_path):
    """5 threads writing to same PRISMA database."""
    db_path = str(tmp_path / "concurrent.db")
    errors = []

    def writer(thread_id):
        try:
            tracker = PrismaTracker(db_path=db_path)
            for i in range(200):
                h = f"t{thread_id}_h{i}"
                tracker.identify(f"Paper {thread_id}-{i}", f"src_{thread_id}", h)
                tracker.screen(h, passed=(i % 2 == 0), reason="auto")
            tracker.close()
        except Exception as e:
            errors.append((thread_id, str(e)))

    threads = [threading.Thread(target=writer, args=(t,)) for t in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert not errors, f"Concurrent PRISMA errors: {errors}"
    # Verify final state
    tracker = PrismaTracker(db_path=db_path)
    counts = tracker.flow_counts()
    assert counts["identified"] == 1000  # 5 threads * 200
    tracker.close()


# ---------------------------------------------------------------------------
# Test 5: Cascading pipeline with poison data
# ---------------------------------------------------------------------------

def test_cascading_pipeline_poison():
    """Full pipeline: triage -> score -> weight -> synthesis with 1000 poison papers."""
    papers = _poison_papers(1000)

    # Stage 1: Triage
    lt = LayeredTriage(runtime=None)
    triage_results = lt.triage_batch(papers, "retrieval", satellite_cutoff=0.0)
    assert len(triage_results) == 1000

    # Stage 2: Score all
    scorer = SourceScorer(runtime=None)
    creds = scorer.score_batch(papers, "retrieval")
    assert len(creds) == 1000
    for c in creds:
        assert 0.0 <= c.composite <= 1.0

    # Stage 3: Weight
    weighter = EvidenceWeighter()
    evidence = weighter.weight("retrieval improves QA", papers, creds)
    assert len(evidence.weights) == 1000
    assert abs(sum(evidence.weights) - 1.0) < 0.01

    # Stage 4: Synthesize (only use papers with string titles)
    safe_papers = [p for p in papers if isinstance(p.get("title"), str) and p.get("title")][:50]
    if safe_papers:
        synth = SynthesisMatrix(runtime=None)
        report = synth.build(safe_papers, "retrieval")
        assert report.total_sources == len(safe_papers)


# ---------------------------------------------------------------------------
# Test 6: Memory bomb - 1MB titles and abstracts
# ---------------------------------------------------------------------------

def test_triage_1mb_papers():
    """10 papers each with 1MB title + 1MB abstract."""
    lt = LayeredTriage(runtime=None)
    papers = [
        {"title": f"T{'X' * 1_000_000}", "abstract": f"A{'Y' * 1_000_000}", "year": 2025}
        for _ in range(10)
    ]
    t0 = time.time()
    results = lt.triage_batch(papers, "test")
    elapsed = time.time() - t0
    assert len(results) == 10
    assert elapsed < 5.0, f"1MB papers took {elapsed:.2f}s"


# ---------------------------------------------------------------------------
# Test 7: Rapid create-close cycles under pressure
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_prisma_rapid_create_close(tmp_path):
    """200 rapid open-write-close cycles."""
    db_path = str(tmp_path / "rapid.db")
    t0 = time.time()
    for i in range(200):
        tracker = PrismaTracker(db_path=db_path)
        tracker.identify(f"Paper {i}", "src", f"h{i}")
        tracker.screen(f"h{i}", passed=True, reason="ok")
        tracker.close()
    elapsed = time.time() - t0
    assert elapsed < 30.0, f"200 cycles took {elapsed:.2f}s"

    tracker = PrismaTracker(db_path=db_path)
    assert tracker.flow_counts()["identified"] == 200
    tracker.close()


# ---------------------------------------------------------------------------
# Test 8: Drone parser with 10000-line LLM response
# ---------------------------------------------------------------------------

def test_parse_drone_10000_lines():
    """Parse a 10000-line drone response."""
    lines = []
    for i in range(10000):
        if i % 100 == 0:
            lines.append("This is garbage preamble text")
        elif i % 50 == 0:
            lines.append(f"{i}|2.5|Invalid score out of range")
        elif i % 30 == 0:
            lines.append(f"bad|0.5|Non-integer index")
        else:
            lines.append(f"{i % 5000}|{(i % 100) / 100:.2f}|Summary for paper {i}")
    text = "\n".join(lines)
    t0 = time.time()
    result = _parse_drone_response(text, 5000)
    elapsed = time.time() - t0
    assert elapsed < 2.0, f"Parsing 10K lines took {elapsed:.2f}s"
    assert len(result) > 0


# ---------------------------------------------------------------------------
# Test 9: Devil's advocate with adversarial fetch
# ---------------------------------------------------------------------------

def test_advocate_adversarial_fetch():
    """Fetch function returns huge, malformed, and duplicate data."""
    call_count = [0]
    def adversarial_fetch(query):
        call_count[0] += 1
        if call_count[0] == 1:
            return [{"title": f"Paper {i}", "abstract": "X" * 100_000} for i in range(100)]
        elif call_count[0] == 2:
            return [{"title": "Same Title"} for _ in range(500)]  # all duplicates
        else:
            return [{}]  # no title

    da = DevilsAdvocate(runtime=None, fetch_fn=adversarial_fetch)
    supporting = [{"title": f"Support {i}"} for i in range(50)]
    report = da.challenge("Test claim", supporting, max_counter_queries=3)
    assert report.supporting_count == 50
    assert report.balance_ratio >= 0.0


# ---------------------------------------------------------------------------
# Test 10: Evidence weighter with 5000 sources
# ---------------------------------------------------------------------------

def test_weighter_5000_sources():
    """Weight 5000 sources - verify normalization holds."""
    weighter = EvidenceWeighter()
    random.seed(42)
    creds = [
        CredibilityScore(
            random.random(), random.random(), random.random(),
            random.random(), random.random(), random.random()
        )
        for _ in range(5000)
    ]
    papers = [
        {"year": random.randint(2000, 2026), "citation_count": random.randint(0, 10000)}
        for _ in range(5000)
    ]
    t0 = time.time()
    result = weighter.weight("large claim", papers, creds)
    elapsed = time.time() - t0
    assert len(result.weights) == 5000
    assert abs(sum(result.weights) - 1.0) < 0.01, f"Weights sum to {sum(result.weights)}"
    assert all(w >= 0 for w in result.weights), "Negative weight found"
    assert elapsed < 2.0, f"5000 source weighting took {elapsed:.2f}s"


# ---------------------------------------------------------------------------
# Test 11: Synthesis with 100 papers
# ---------------------------------------------------------------------------

def test_synthesis_100_papers():
    """Synthesis matrix with 100 papers, each with multiple claims."""
    synth = SynthesisMatrix(runtime=None)
    papers = [
        {"title": f"Paper {i}: {'retrieval' if i%3==0 else 'generation' if i%3==1 else 'hybrid'} study",
         "abstract": f"We study {'dense' if i%2==0 else 'sparse'} methods for {'QA' if i%4==0 else 'summarization'}.",
         "key_claims": [f"Claim {i}a: method A works", f"Claim {i}b: outperforms baseline",
                        f"Claim {i}c: generalizes to new domains"]}
        for i in range(100)
    ]
    t0 = time.time()
    report = synth.build(papers, "retrieval generation methods")
    elapsed = time.time() - t0
    assert report.total_sources == 100
    assert len(report.themes) >= 1
    assert elapsed < 10.0, f"100 paper synthesis took {elapsed:.2f}s"


# ---------------------------------------------------------------------------
# Test 12: Hash collision brute force
# ---------------------------------------------------------------------------

def test_hash_collision_resistance():
    """10000 distinct papers should have < 0.1% collision rate."""
    papers = [
        {"title": f"Unique Title {i} with extra words {i*7}",
         "abstract": f"Distinct abstract content number {i} discussing topic {i*13}."}
        for i in range(10000)
    ]
    hashes = [_content_hash(p) for p in papers]
    unique = len(set(hashes))
    collision_rate = 1 - unique / len(hashes)
    assert collision_rate < 0.001, f"Collision rate {collision_rate:.4f} (expected < 0.001)"
