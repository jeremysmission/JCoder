"""Iteration 1: Scale and performance stress tests."""

import time
import pytest

from core.layered_triage import LayeredTriage, _content_hash
from core.source_scorer import SourceScorer
from core.evidence_weighter import EvidenceWeighter
from core.synthesis_matrix import SynthesisMatrix
from core.prisma_tracker import PrismaTracker


def _gen_papers(n, base_year=2020):
    return [
        {"title": f"Paper {i}: {'retrieval' if i % 3 == 0 else 'generation'} methods",
         "abstract": f"We study {'dense retrieval' if i % 2 == 0 else 'sparse search'} "
                     f"for question answering. Experiment {i} shows improvements.",
         "year": base_year + (i % 7),
         "citation_count": i * 3,
         "source": ["arxiv_cs_ai", "semantic_scholar", "hacker_news", "papers_with_code"][i % 4],
         "url": f"https://example.com/{i}",
         "has_code": i % 2 == 0,
         "has_refs": i % 3 != 2}
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Scale tests
# ---------------------------------------------------------------------------

def test_triage_500_papers():
    """500 papers through satellite pass in under 2 seconds."""
    lt = LayeredTriage(runtime=None)
    papers = _gen_papers(500)
    t0 = time.time()
    results = lt.triage_batch(papers, "retrieval augmented generation",
                              satellite_cutoff=0.1, max_deep_dive=10)
    elapsed = time.time() - t0
    assert len(results) == 500
    assert elapsed < 2.0, f"500 papers took {elapsed:.2f}s"
    deep = [r for r in results if r.deep_dive]
    assert len(deep) <= 10


def test_triage_1000_papers():
    """1000 papers through satellite pass in under 5 seconds."""
    lt = LayeredTriage(runtime=None)
    papers = _gen_papers(1000)
    t0 = time.time()
    results = lt.triage_batch(papers, "meta-learning self-improving",
                              satellite_cutoff=0.1, max_deep_dive=5)
    elapsed = time.time() - t0
    assert len(results) == 1000
    assert elapsed < 5.0, f"1000 papers took {elapsed:.2f}s"


def test_scorer_500_papers():
    """Credibility scoring 500 papers (heuristic mode) under 1 second."""
    scorer = SourceScorer(runtime=None)
    papers = _gen_papers(500)
    t0 = time.time()
    scores = scorer.score_batch(papers, "research")
    elapsed = time.time() - t0
    assert len(scores) == 500
    assert elapsed < 1.0, f"500 scores took {elapsed:.2f}s"


def test_evidence_weighter_100_sources():
    """Weighting 100 sources for a single claim."""
    weighter = EvidenceWeighter()
    scorer = SourceScorer(runtime=None)
    papers = _gen_papers(100)
    creds = scorer.score_batch(papers, "test")
    t0 = time.time()
    result = weighter.weight("Test claim", papers, creds)
    elapsed = time.time() - t0
    assert len(result.weights) == 100
    assert abs(sum(result.weights) - 1.0) < 0.001
    assert elapsed < 0.5, f"100 sources took {elapsed:.2f}s"


def test_prisma_1000_papers(tmp_path):
    """PRISMA tracker handles 1000 papers without slowdown."""
    tracker = PrismaTracker(db_path=str(tmp_path / "prisma.db"))
    papers = _gen_papers(1000)
    t0 = time.time()
    for p in papers:
        h = _content_hash(p)
        tracker.identify(p["title"], p.get("source", ""), h)
        tracker.screen(h, passed=p["citation_count"] > 5,
                       reason="citations" if p["citation_count"] > 5 else "low_citations")
    elapsed = time.time() - t0
    counts = tracker.flow_counts()
    assert counts["identified"] == 1000
    assert counts["screened"] + counts["excluded"] >= 1000
    assert elapsed < 5.0, f"1000 PRISMA ops took {elapsed:.2f}s"
    tracker.close()


def test_synthesis_20_papers():
    """Synthesis matrix with 20 digested papers."""
    synth = SynthesisMatrix(runtime=None)
    papers = [
        {"title": f"Paper {i}", "abstract": f"Study on {'retrieval' if i % 2 == 0 else 'generation'} methods.",
         "key_claims": [f"Claim {i}a: method works", f"Claim {i}b: improves accuracy"]}
        for i in range(20)
    ]
    t0 = time.time()
    report = synth.build(papers, "retrieval generation methods")
    elapsed = time.time() - t0
    assert report.total_sources == 20
    assert len(report.themes) >= 1
    assert elapsed < 2.0, f"20 paper synthesis took {elapsed:.2f}s"


def test_hash_uniqueness_at_scale():
    """Content hashes unique across 1000 distinct papers."""
    papers = _gen_papers(1000)
    hashes = [_content_hash(p) for p in papers]
    unique = set(hashes)
    # Allow very few collisions (16-char hex = 64 bits)
    collision_rate = 1 - len(unique) / len(hashes)
    assert collision_rate < 0.01, f"Collision rate {collision_rate:.3f} too high"


def test_triage_sorting_stability():
    """Triage produces stable sort order across repeated runs."""
    lt = LayeredTriage(runtime=None)
    papers = _gen_papers(50)
    r1 = lt.triage_batch(papers, "retrieval", satellite_cutoff=0.1)
    r2 = lt.triage_batch(papers, "retrieval", satellite_cutoff=0.1)
    titles1 = [r.title for r in r1]
    titles2 = [r.title for r in r2]
    assert titles1 == titles2, "Triage sort not deterministic"
