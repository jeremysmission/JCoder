"""Iteration 3: Scoring accuracy, ranking quality, and weight distribution."""

import pytest
from core.layered_triage import LayeredTriage, _content_hash
from core.source_scorer import SourceScorer
from core.evidence_weighter import EvidenceWeighter


# ---------------------------------------------------------------------------
# Satellite scoring correctness
# ---------------------------------------------------------------------------

def test_recent_paper_beats_old():
    """2026 paper should score higher than 2010 paper, all else equal."""
    lt = LayeredTriage(runtime=None)
    papers = [
        {"title": "Dense retrieval methods", "abstract": "Study on retrieval.", "year": 2026, "source": "arxiv_cs_ai"},
        {"title": "Dense retrieval methods", "abstract": "Study on retrieval.", "year": 2010, "source": "arxiv_cs_ai"},
    ]
    results = lt.triage_batch(papers, "dense retrieval")
    assert results[0].satellite_score > results[1].satellite_score, \
        f"2026 ({results[0].satellite_score}) should beat 2010 ({results[1].satellite_score})"


def test_high_tier_beats_low_tier():
    """arxiv_cs_ai paper should score higher than reddit post, all else equal."""
    lt = LayeredTriage(runtime=None)
    # Different abstracts to ensure different content hashes
    papers = [
        {"title": "Study on RAG", "abstract": "Research from arxiv.", "year": 2025, "source": "arxiv_cs_ai"},
        {"title": "Study on RAG", "abstract": "Research from reddit.", "year": 2025, "source": "reddit"},
    ]
    results = lt.triage_batch(papers, "RAG")
    arxiv = next(r for r in results if "arxiv" in _get_source(papers, r.content_hash))
    reddit = next(r for r in results if "reddit" in _get_source(papers, r.content_hash))
    assert arxiv.satellite_score > reddit.satellite_score


def test_cited_paper_beats_uncited():
    """Paper with 500 citations should score higher than 0 citations."""
    lt = LayeredTriage(runtime=None)
    papers = [
        {"title": "Method X", "abstract": "Test.", "year": 2024, "citation_count": 500, "source": "semantic_scholar"},
        {"title": "Method X", "abstract": "Test.", "year": 2024, "citation_count": 0, "source": "semantic_scholar"},
    ]
    results = lt.triage_batch(papers, "method")
    cited = [r for r in results if r.satellite_score == max(r.satellite_score for r in results)][0]
    uncited = [r for r in results if r.satellite_score == min(r.satellite_score for r in results)][0]
    assert cited.satellite_score > uncited.satellite_score


def test_keyword_rich_title_boosts_score():
    """Title with high-signal keywords should score higher."""
    lt = LayeredTriage(runtime=None)
    papers = [
        {"title": "Novel state-of-the-art breakthrough in retrieval augmented generation",
         "abstract": "We surpass benchmarks.", "year": 2025},
        {"title": "A note on something",
         "abstract": "Brief comment.", "year": 2025},
    ]
    results = lt.triage_batch(papers, "retrieval")
    rich = results[0]  # sorted desc
    plain = results[-1]
    assert rich.satellite_score > plain.satellite_score


def test_query_overlap_boosts():
    """Paper whose title matches query keywords scores higher."""
    lt = LayeredTriage(runtime=None)
    papers = [
        {"title": "Retrieval augmented generation for QA", "abstract": "RAG pipeline.", "year": 2025},
        {"title": "Cooking recipes simplified", "abstract": "Food blog.", "year": 2025},
    ]
    results = lt.triage_batch(papers, "retrieval augmented generation")
    assert results[0].title.startswith("Retrieval")


# ---------------------------------------------------------------------------
# Score bounds
# ---------------------------------------------------------------------------

def test_satellite_scores_bounded():
    """All satellite scores are in [0, 1]."""
    lt = LayeredTriage(runtime=None)
    papers = [
        {"title": "Novel SOTA benchmark empirical retrieval augmented meta-learning",
         "abstract": "systematic review meta-analysis survey framework methodology pipeline architecture ablation reproducible open source replication",
         "year": 2026, "citation_count": 10000, "source": "semantic_scholar"},
        {"title": "", "abstract": "", "year": 0, "citation_count": 0, "source": ""},
    ]
    results = lt.triage_batch(papers, "retrieval augmented generation meta-learning survey")
    for r in results:
        assert 0.0 <= r.satellite_score <= 1.0, f"Score {r.satellite_score} out of bounds"


def test_credibility_scores_bounded():
    """All CRAAP dimensions are in [0, 1]."""
    scorer = SourceScorer(runtime=None)
    papers = [
        {"title": "X", "year": 2026, "citation_count": 10000, "has_code": True, "has_refs": True,
         "peer_reviewed": True, "source": "arxiv_cs_ai", "url": "https://arxiv.org/abs/1234"},
        {"title": "", "year": 0, "citation_count": 0, "source": "", "url": ""},
    ]
    for p in papers:
        score = scorer.score(p, "test query")
        assert 0.0 <= score.currency <= 1.0, f"currency={score.currency}"
        assert 0.0 <= score.relevance <= 1.0, f"relevance={score.relevance}"
        assert 0.0 <= score.authority <= 1.0, f"authority={score.authority}"
        assert 0.0 <= score.accuracy <= 1.0, f"accuracy={score.accuracy}"
        assert 0.0 <= score.purpose <= 1.0, f"purpose={score.purpose}"
        assert 0.0 <= score.composite <= 1.0, f"composite={score.composite}"


# ---------------------------------------------------------------------------
# CRAAP dimension correctness
# ---------------------------------------------------------------------------

def test_currency_recent_beats_old():
    """2026 paper should have higher currency than 2005."""
    scorer = SourceScorer(runtime=None)
    s1 = scorer.score({"title": "X", "year": 2026}, "test")
    s2 = scorer.score({"title": "X", "year": 2005}, "test")
    assert s1.currency > s2.currency


def test_authority_high_citations_beats_low():
    """1000 citations should yield higher authority than 0."""
    scorer = SourceScorer(runtime=None)
    s1 = scorer.score({"title": "X", "citation_count": 1000, "source": "semantic_scholar"}, "test")
    s2 = scorer.score({"title": "X", "citation_count": 0, "source": "semantic_scholar"}, "test")
    assert s1.authority > s2.authority


def test_accuracy_code_refs_beats_none():
    """Paper with code and refs should have higher accuracy."""
    scorer = SourceScorer(runtime=None)
    s1 = scorer.score({"title": "X", "has_code": True, "has_refs": True, "peer_reviewed": True}, "test")
    s2 = scorer.score({"title": "X", "has_code": False, "has_refs": False, "peer_reviewed": False}, "test")
    assert s1.accuracy > s2.accuracy


# ---------------------------------------------------------------------------
# Evidence weighting distribution
# ---------------------------------------------------------------------------

def test_weights_sum_to_one():
    """Weights from weighter always sum to 1.0."""
    from core.source_scorer import CredibilityScore
    weighter = EvidenceWeighter()
    creds = [
        CredibilityScore(0.9, 0.9, 0.9, 0.9, 0.9, 0.9),
        CredibilityScore(0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
        CredibilityScore(0.1, 0.1, 0.1, 0.1, 0.1, 0.1),
    ]
    papers = [
        {"year": 2026, "citation_count": 100},
        {"year": 2022, "citation_count": 10},
        {"year": 2018, "citation_count": 1},
    ]
    result = weighter.weight("test claim", papers, creds)
    assert abs(sum(result.weights) - 1.0) < 0.001


def test_high_quality_gets_most_weight():
    """Highest credibility source gets the most weight."""
    from core.source_scorer import CredibilityScore
    weighter = EvidenceWeighter()
    creds = [
        CredibilityScore(0.9, 0.9, 0.9, 0.9, 0.9, 0.9),
        CredibilityScore(0.3, 0.3, 0.3, 0.3, 0.3, 0.3),
    ]
    papers = [
        {"year": 2026, "citation_count": 100},
        {"year": 2026, "citation_count": 100},
    ]
    result = weighter.weight("claim", papers, creds)
    assert result.weights[0] > result.weights[1], \
        f"High quality ({result.weights[0]}) should outweigh low ({result.weights[1]})"


def test_single_source_gets_full_weight():
    """Single source should get weight 1.0."""
    from core.source_scorer import CredibilityScore
    weighter = EvidenceWeighter()
    creds = [CredibilityScore(0.5, 0.5, 0.5, 0.5, 0.5, 0.5)]
    papers = [{"year": 2024, "citation_count": 10}]
    result = weighter.weight("claim", papers, creds)
    assert len(result.weights) == 1
    assert abs(result.weights[0] - 1.0) < 0.001


def test_equal_sources_get_equal_weight():
    """Identical sources should get equal weights."""
    from core.source_scorer import CredibilityScore
    weighter = EvidenceWeighter()
    cred = CredibilityScore(0.7, 0.7, 0.7, 0.7, 0.7, 0.7)
    creds = [cred, cred, cred]
    paper = {"year": 2025, "citation_count": 50}
    papers = [paper, paper, paper]
    result = weighter.weight("claim", papers, creds)
    assert abs(result.weights[0] - result.weights[1]) < 0.001
    assert abs(result.weights[1] - result.weights[2]) < 0.001


# ---------------------------------------------------------------------------
# Ranking stability
# ---------------------------------------------------------------------------

def test_ranking_order_deterministic():
    """Same input produces same ranking every time."""
    lt = LayeredTriage(runtime=None)
    papers = [
        {"title": f"Paper {i}", "abstract": f"About {'retrieval' if i%2==0 else 'generation'}.",
         "year": 2020 + (i % 6), "citation_count": i * 10,
         "source": ["arxiv_cs_ai", "semantic_scholar", "reddit"][i % 3]}
        for i in range(30)
    ]
    r1 = lt.triage_batch(papers, "retrieval generation")
    r2 = lt.triage_batch(papers, "retrieval generation")
    scores1 = [(r.content_hash, r.satellite_score) for r in r1]
    scores2 = [(r.content_hash, r.satellite_score) for r in r2]
    assert scores1 == scores2


def test_cutoff_filters_low_scorers():
    """Papers below satellite cutoff should not be satellite_pass."""
    lt = LayeredTriage(runtime=None)
    papers = [
        {"title": "", "abstract": "", "year": 0, "source": ""},  # minimal
        {"title": "SOTA benchmark retrieval augmented", "abstract": "systematic review",
         "year": 2026, "citation_count": 500, "source": "arxiv_cs_ai"},
    ]
    results = lt.triage_batch(papers, "retrieval", satellite_cutoff=0.3)
    low = [r for r in results if not r.satellite_pass]
    high = [r for r in results if r.satellite_pass]
    # The minimal paper should fail satellite cutoff
    assert len(low) >= 1, "No papers filtered by satellite cutoff"
    for r in low:
        assert r.satellite_score < 0.3


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _get_source(papers, content_hash):
    """Reverse lookup source from content hash."""
    for p in papers:
        if _content_hash(p) == content_hash:
            return p.get("source", "")
    return ""
