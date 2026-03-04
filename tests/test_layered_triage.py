"""Tests for layered triage engine (Satellite / Drone / Deep Dive)."""

from core.layered_triage import LayeredTriage, TriageResult, _content_hash, _parse_drone_response


def _make_paper(title="Test Paper", abstract="An abstract.", year=2025,
                citations=10, source="arxiv_cs_ai", url="https://arxiv.org/abs/1"):
    return {
        "title": title,
        "abstract": abstract,
        "year": year,
        "citation_count": citations,
        "source": source,
        "url": url,
    }


# ---------------------------------------------------------------------------
# Satellite pass tests
# ---------------------------------------------------------------------------

def test_satellite_recent_high_signal():
    """Recent paper with high-signal keywords scores well."""
    lt = LayeredTriage()
    paper = _make_paper(
        title="A Novel Self-Improving RAG Benchmark",
        year=2026, citations=50, source="semantic_scholar",
    )
    results = lt._satellite_pass([paper], "self-improving RAG")
    assert len(results) == 1
    assert results[0].satellite_score > 0.5
    assert results[0].satellite_pass is True


def test_satellite_old_low_signal():
    """Old paper from low-tier source with no keyword overlap scores low."""
    lt = LayeredTriage()
    paper = _make_paper(
        title="Random Thoughts on Cooking",
        year=2018, citations=0, source="hacker_news",
    )
    results = lt._satellite_pass([paper], "retrieval augmented generation")
    assert len(results) == 1
    assert results[0].satellite_score < 0.3
    assert results[0].satellite_pass is False


def test_satellite_query_overlap_boosts():
    """Query keyword overlap increases score."""
    lt = LayeredTriage()
    paper = _make_paper(
        title="Retrieval Augmented Generation Survey",
        abstract="This paper surveys retrieval augmented generation methods.",
        year=2025,
    )
    score_with = lt._satellite_pass([paper], "retrieval augmented generation")[0].satellite_score
    score_without = lt._satellite_pass([paper], "quantum computing algorithms")[0].satellite_score
    assert score_with > score_without


def test_satellite_filters_batch():
    """Satellite pass filters a batch, keeping only above-cutoff papers."""
    lt = LayeredTriage()
    papers = [
        _make_paper(title="Novel Meta-Learning Framework", year=2026, citations=100),
        _make_paper(title="Old Unrelated Work", year=2015, citations=0, source="reddit"),
        _make_paper(title="Self-Improving Code Generation", year=2025, citations=20),
    ]
    results = lt._satellite_pass(papers, "meta-learning", cutoff=0.3)
    passed = [r for r in results if r.satellite_pass]
    failed = [r for r in results if not r.satellite_pass]
    assert len(passed) >= 1
    assert len(failed) >= 1


# ---------------------------------------------------------------------------
# Drone pass parsing tests
# ---------------------------------------------------------------------------

def test_parse_drone_response_valid():
    """Parse well-formed drone response lines."""
    text = "0|0.85|Great paper on RAG.\n1|0.30|Weak relevance.\n2|0.95|Breakthrough work."
    parsed = _parse_drone_response(text, 3)
    assert len(parsed) == 3
    assert parsed[0]["score"] == 0.85
    assert parsed[2]["summary"] == "Breakthrough work."


def test_parse_drone_response_malformed():
    """Malformed lines are skipped gracefully."""
    text = "garbage\n0|0.85|Valid line.\nbad|bad|bad\n"
    parsed = _parse_drone_response(text, 2)
    assert len(parsed) == 1
    assert 0 in parsed


def test_parse_drone_response_out_of_range():
    """Scores outside 0-1 or indices outside range are rejected."""
    text = "0|1.5|Too high.\n5|0.5|Index too big.\n1|0.7|Valid."
    parsed = _parse_drone_response(text, 3)
    assert 0 not in parsed  # score > 1.0
    assert 5 not in parsed  # index >= expected
    assert 1 in parsed


# ---------------------------------------------------------------------------
# Full triage pipeline tests
# ---------------------------------------------------------------------------

def test_triage_batch_no_runtime():
    """Full triage without runtime uses satellite scores for drone pass."""
    lt = LayeredTriage(runtime=None)
    papers = [
        _make_paper(title="Novel RAG Framework", year=2026, citations=50),
        _make_paper(title="Cooking Recipes", year=2015, source="reddit"),
    ]
    results = lt.triage_batch(papers, "RAG framework", satellite_cutoff=0.2)
    assert len(results) == 2
    # Best paper should be first (sorted by score)
    assert "RAG" in results[0].title or results[0].satellite_score >= results[1].satellite_score


def test_deep_dive_selection():
    """Only max_deep_dive papers get deep_dive=True."""
    lt = LayeredTriage(runtime=None)
    papers = [
        _make_paper(title=f"Paper {i} on novel retrieval", year=2025, citations=i * 10)
        for i in range(10)
    ]
    results = lt.triage_batch(
        papers, "retrieval",
        satellite_cutoff=0.1, drone_cutoff=0.1, max_deep_dive=3,
    )
    deep = [r for r in results if r.deep_dive]
    assert len(deep) <= 3


def test_empty_papers():
    """Empty input returns empty results."""
    lt = LayeredTriage(runtime=None)
    results = lt.triage_batch([], "anything")
    assert results == []


def test_content_hash_deterministic():
    """Same paper produces same hash."""
    paper = _make_paper(title="Consistent Paper", abstract="Same abstract.")
    h1 = _content_hash(paper)
    h2 = _content_hash(paper)
    assert h1 == h2
    assert len(h1) == 16


def test_content_hash_different_papers():
    """Different papers produce different hashes."""
    p1 = _make_paper(title="Paper A")
    p2 = _make_paper(title="Paper B")
    assert _content_hash(p1) != _content_hash(p2)
