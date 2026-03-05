"""Hard iteration 4: Mathematical invariant and property-based testing.

Verifies mathematical properties hold under randomized inputs:
- Weight normalization invariant (sum=1.0) across 10K random inputs
- Score monotonicity (better inputs -> better scores)
- Hash distribution uniformity
- Idempotency (f(f(x)) == f(x) for scoring)
- Ranking transitivity (A > B and B > C implies A > C)
- Scoring commutativity (order independence)
- Boundary value completeness
"""

import pytest
import random
import math
import hashlib
from collections import Counter

from core.layered_triage import LayeredTriage, _content_hash
from core.source_scorer import SourceScorer, CredibilityScore
from core.evidence_weighter import EvidenceWeighter
from core.synthesis_matrix import SynthesisMatrix


# Seed for reproducibility
SEED = 20260303
random.seed(SEED)


# ---------------------------------------------------------------------------
# Weight normalization invariant
# ---------------------------------------------------------------------------

def test_weight_normalization_10k_random():
    """Weight normalization holds across 10,000 random input sets."""
    weighter = EvidenceWeighter()
    failures = []

    for trial in range(10000):
        n = random.randint(1, 20)
        creds = [
            CredibilityScore(
                random.random(), random.random(), random.random(),
                random.random(), random.random(), random.random()
            )
            for _ in range(n)
        ]
        papers = [
            {"year": random.randint(1990, 2026), "citation_count": random.randint(0, 5000)}
            for _ in range(n)
        ]
        result = weighter.weight(f"claim_{trial}", papers, creds)
        weight_sum = sum(result.weights)
        if abs(weight_sum - 1.0) > 0.01:
            failures.append((trial, n, weight_sum))

    assert not failures, f"{len(failures)} normalization failures out of 10K: first={failures[0]}"


def test_all_weights_non_negative_10k():
    """No negative weights across 10K random trials."""
    weighter = EvidenceWeighter()
    for trial in range(10000):
        n = random.randint(1, 15)
        creds = [
            CredibilityScore(
                random.random(), random.random(), random.random(),
                random.random(), random.random(), random.random()
            )
            for _ in range(n)
        ]
        papers = [
            {"year": random.randint(1990, 2026), "citation_count": random.randint(0, 5000)}
            for _ in range(n)
        ]
        result = weighter.weight(f"claim_{trial}", papers, creds)
        for i, w in enumerate(result.weights):
            assert w >= 0.0, f"Trial {trial}, source {i}: weight={w}"


# ---------------------------------------------------------------------------
# Score monotonicity
# ---------------------------------------------------------------------------

def test_recency_monotonicity():
    """More recent year -> higher currency score (monotonically non-decreasing)."""
    scorer = SourceScorer(runtime=None)
    prev_currency = -1.0
    for year in range(2000, 2027):
        s = scorer.score({"title": "X", "year": year}, "test")
        assert s.currency >= prev_currency or abs(s.currency - prev_currency) < 0.001, \
            f"year={year}: currency={s.currency} < prev={prev_currency}"
        prev_currency = s.currency


def test_citation_monotonicity():
    """More citations -> higher authority score (monotonically non-decreasing)."""
    scorer = SourceScorer(runtime=None)
    prev_authority = -1.0
    for cites in [0, 1, 5, 10, 50, 100, 500, 1000, 5000, 10000]:
        s = scorer.score({"title": "X", "citation_count": cites, "source": "arxiv_cs_ai"}, "test")
        assert s.authority >= prev_authority - 0.001, \
            f"cites={cites}: authority={s.authority} < prev={prev_authority}"
        prev_authority = s.authority


def test_satellite_recency_monotonicity():
    """In satellite pass, newer papers score >= older papers, all else equal."""
    lt = LayeredTriage(runtime=None)
    prev_score = -1.0
    for year in range(2010, 2027):
        papers = [{"title": "Same Title", "abstract": f"Abstract for {year}.", "year": year,
                   "source": "arxiv_cs_ai", "citation_count": 50}]
        results = lt.triage_batch(papers, "test", satellite_cutoff=0.0)
        score = results[0].satellite_score
        assert score >= prev_score - 0.001, \
            f"year={year}: score={score} < prev={prev_score}"
        prev_score = score


# ---------------------------------------------------------------------------
# Hash distribution uniformity
# ---------------------------------------------------------------------------

def test_hash_first_char_distribution():
    """Content hashes should have roughly uniform first-character distribution."""
    papers = [
        {"title": f"Unique Paper Title {i}", "abstract": f"Unique abstract content {i}."}
        for i in range(10000)
    ]
    hashes = [_content_hash(p) for p in papers]
    first_chars = Counter(h[0] for h in hashes)

    # 16 hex chars, 10000 papers -> expect ~625 per char
    for char, count in first_chars.items():
        assert 300 < count < 1000, \
            f"Char '{char}' has {count} occurrences (expected ~625, range 300-1000)"


def test_hash_uniqueness_10k():
    """10,000 distinct papers produce 10,000 distinct hashes."""
    papers = [
        {"title": f"Distinct Title {i}", "abstract": f"Distinct abstract {i}."}
        for i in range(10000)
    ]
    hashes = [_content_hash(p) for p in papers]
    unique = len(set(hashes))
    collision_rate = 1 - unique / len(hashes)
    assert collision_rate < 0.001, f"Collision rate {collision_rate:.5f}"


def test_hash_avalanche():
    """Single character change produces completely different hash (avalanche effect)."""
    base = {"title": "Test Paper About Machine Learning", "abstract": "A comprehensive study."}
    base_hash = _content_hash(base)

    # Change each character position in the title
    title = base["title"]
    different_count = 0
    for i in range(len(title)):
        modified = title[:i] + chr((ord(title[i]) + 1) % 128) + title[i+1:]
        mod_hash = _content_hash({"title": modified, "abstract": base["abstract"]})
        if mod_hash != base_hash:
            different_count += 1

    # At least 90% of single-char changes should produce different hashes
    assert different_count >= len(title) * 0.9


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------

def test_scorer_idempotency_1k():
    """Scoring the same paper 1000 times produces identical results."""
    scorer = SourceScorer(runtime=None)
    paper = {"title": "Idempotency Test", "abstract": "Testing.", "year": 2025,
             "citation_count": 42, "source": "arxiv_cs_ai"}
    first = scorer.score(paper, "test")
    for _ in range(999):
        s = scorer.score(paper, "test")
        assert s.composite == first.composite
        assert s.currency == first.currency
        assert s.authority == first.authority


def test_triage_idempotency_1k():
    """Triaging the same batch 100 times produces identical results."""
    lt = LayeredTriage(runtime=None)
    papers = [
        {"title": f"Paper {i}", "abstract": f"Abstract {i}.", "year": 2020 + i, "source": "arxiv_cs_ai"}
        for i in range(10)
    ]
    first_results = lt.triage_batch(papers, "test", satellite_cutoff=0.1)
    first_scores = [(r.content_hash, r.satellite_score) for r in first_results]

    for _ in range(99):
        results = lt.triage_batch(papers, "test", satellite_cutoff=0.1)
        scores = [(r.content_hash, r.satellite_score) for r in results]
        assert scores == first_scores


# ---------------------------------------------------------------------------
# Ranking transitivity
# ---------------------------------------------------------------------------

def test_ranking_transitivity():
    """If paper A > B and B > C, then A > C in satellite scoring."""
    lt = LayeredTriage(runtime=None)
    # Paper A: recent, high-signal, good source, cited
    paper_a = {"title": "Novel SOTA benchmark retrieval augmented", "abstract": "Systematic review.",
               "year": 2026, "citation_count": 500, "source": "semantic_scholar"}
    # Paper B: decent but older
    paper_b = {"title": "Study on retrieval methods", "abstract": "Research.",
               "year": 2022, "citation_count": 50, "source": "arxiv_cs_ai"}
    # Paper C: low quality
    paper_c = {"title": "Random note", "abstract": "Brief.",
               "year": 2015, "citation_count": 0, "source": "reddit"}

    all_papers = [paper_a, paper_b, paper_c]
    results = lt.triage_batch(all_papers, "retrieval augmented generation", satellite_cutoff=0.0)

    scores = {r.content_hash: r.satellite_score for r in results}
    ha, hb, hc = _content_hash(paper_a), _content_hash(paper_b), _content_hash(paper_c)

    assert scores[ha] > scores[hb], f"A ({scores[ha]}) should beat B ({scores[hb]})"
    assert scores[hb] > scores[hc], f"B ({scores[hb]}) should beat C ({scores[hc]})"
    # Transitivity: A > C
    assert scores[ha] > scores[hc], f"Transitivity: A ({scores[ha]}) should beat C ({scores[hc]})"


# ---------------------------------------------------------------------------
# Order independence (commutativity)
# ---------------------------------------------------------------------------

def test_triage_order_independence():
    """Satellite scores don't depend on paper order in the batch."""
    lt = LayeredTriage(runtime=None)
    papers = [
        {"title": f"Paper {i}", "abstract": f"About topic {i}.",
         "year": 2020 + (i % 7), "citation_count": i * 10, "source": "arxiv_cs_ai"}
        for i in range(50)
    ]
    shuffled = list(papers)
    random.shuffle(shuffled)

    r1 = lt.triage_batch(papers, "topic", satellite_cutoff=0.0)
    r2 = lt.triage_batch(shuffled, "topic", satellite_cutoff=0.0)

    scores1 = {r.content_hash: r.satellite_score for r in r1}
    scores2 = {r.content_hash: r.satellite_score for r in r2}

    for h in scores1:
        assert h in scores2, f"Hash {h} missing from shuffled results"
        assert abs(scores1[h] - scores2[h]) < 0.0001, \
            f"Hash {h}: ordered={scores1[h]} vs shuffled={scores2[h]}"


def test_weighter_order_independence():
    """Weight values don't change based on input order (for same papers)."""
    weighter = EvidenceWeighter()
    creds = [
        CredibilityScore(0.9, 0.9, 0.9, 0.9, 0.9, 0.9),
        CredibilityScore(0.3, 0.3, 0.3, 0.3, 0.3, 0.3),
        CredibilityScore(0.6, 0.6, 0.6, 0.6, 0.6, 0.6),
    ]
    papers = [
        {"year": 2026, "citation_count": 500},
        {"year": 2020, "citation_count": 5},
        {"year": 2023, "citation_count": 50},
    ]
    r1 = weighter.weight("claim", papers, creds)

    # Reverse order
    r2 = weighter.weight("claim", papers[::-1], creds[::-1])

    # r2.weights should be r1.weights reversed
    assert abs(r1.weights[0] - r2.weights[2]) < 0.001
    assert abs(r1.weights[1] - r2.weights[1]) < 0.001
    assert abs(r1.weights[2] - r2.weights[0]) < 0.001


# ---------------------------------------------------------------------------
# Boundary completeness
# ---------------------------------------------------------------------------

def test_all_score_dimensions_bounded_random():
    """1000 random papers: all CRAAP dimensions in [0, 1]."""
    scorer = SourceScorer(runtime=None)
    for _ in range(1000):
        paper = {
            "title": f"Random Paper {random.random()}",
            "abstract": f"Random abstract {random.random()}",
            "year": random.randint(-100, 9999),
            "citation_count": random.randint(-1000, 1000000),
            "source": random.choice(["arxiv_cs_ai", "reddit", "", "unknown", "semantic_scholar"]),
            "has_code": random.choice([True, False, 0, 1, "yes", None]),
            "has_refs": random.choice([True, False, 0, 1, "yes", None]),
        }
        s = scorer.score(paper, "random query")
        assert 0.0 <= s.currency <= 1.0, f"currency={s.currency}"
        assert 0.0 <= s.relevance <= 1.0, f"relevance={s.relevance}"
        assert 0.0 <= s.authority <= 1.0, f"authority={s.authority}"
        assert 0.0 <= s.accuracy <= 1.0, f"accuracy={s.accuracy}"
        assert 0.0 <= s.purpose <= 1.0, f"purpose={s.purpose}"
        assert 0.0 <= s.composite <= 1.0, f"composite={s.composite}"


def test_satellite_bounded_random():
    """1000 random papers: satellite scores in [0, 1]."""
    lt = LayeredTriage(runtime=None)
    papers = [
        {"title": f"Paper {random.random()}", "abstract": f"Abs {random.random()}",
         "year": random.randint(-100, 9999), "citation_count": random.randint(-1000, 1000000),
         "source": random.choice(["arxiv_cs_ai", "reddit", "", "unknown"])}
        for _ in range(1000)
    ]
    results = lt.triage_batch(papers, "random query words here", satellite_cutoff=0.0)
    for r in results:
        assert 0.0 <= r.satellite_score <= 1.0, f"satellite={r.satellite_score}"
