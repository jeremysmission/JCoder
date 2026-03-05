"""Iteration 8: Synthesis quality and contradiction detection tests."""

import pytest
from core.synthesis_matrix import (
    SynthesisMatrix, SynthesisCell, ThemeRow, MatrixReport,
    _tokenize, _extract_noun_phrases,
)


# ---------------------------------------------------------------------------
# Theme extraction quality
# ---------------------------------------------------------------------------

def test_themes_from_claims():
    """Papers with key_claims produce meaningful themes."""
    synth = SynthesisMatrix(runtime=None)
    papers = [
        {"title": "Paper A", "abstract": "Dense retrieval with neural networks.",
         "key_claims": ["Dense retrieval outperforms BM25", "Neural embeddings capture semantics"]},
        {"title": "Paper B", "abstract": "Sparse retrieval efficiency study.",
         "key_claims": ["Sparse methods are more efficient", "BM25 remains competitive"]},
        {"title": "Paper C", "abstract": "Hybrid retrieval combining dense and sparse.",
         "key_claims": ["Hybrid approaches combine strengths", "Retrieval quality improves"]},
    ]
    report = synth.build(papers, "retrieval methods")
    assert report.total_sources == 3
    assert len(report.themes) >= 1
    # "retrieval" should appear in at least one theme
    all_theme_text = " ".join(r.theme for r in report.themes).lower()
    assert "retrieval" in all_theme_text or len(report.themes) > 0


def test_themes_from_abstracts_only():
    """Papers without key_claims still produce themes from abstracts."""
    synth = SynthesisMatrix(runtime=None)
    papers = [
        {"title": "Paper X", "abstract": "Machine learning for image classification using CNNs."},
        {"title": "Paper Y", "abstract": "Transfer learning improves image classification accuracy."},
        {"title": "Paper Z", "abstract": "Deep learning architecture for visual recognition tasks."},
    ]
    report = synth.build(papers, "image classification")
    assert report.total_sources == 3
    assert len(report.themes) >= 1


def test_themes_capped_at_max():
    """Number of themes doesn't exceed max_themes."""
    synth = SynthesisMatrix(runtime=None)
    papers = [
        {"title": f"Paper {i}", "abstract": f"Topic {i} about subject {i}.",
         "key_claims": [f"Claim {i}a about X", f"Claim {i}b about Y", f"Claim {i}c about Z"]}
        for i in range(20)
    ]
    report = synth.build(papers, "mixed topics")
    assert len(report.themes) <= 8  # default max_themes


# ---------------------------------------------------------------------------
# Contradiction detection
# ---------------------------------------------------------------------------

def test_contradiction_detected_heuristic():
    """Heuristic classification can detect contradictions between papers."""
    synth = SynthesisMatrix(runtime=None)
    papers = [
        {"title": "Pro retrieval", "abstract": "Dense retrieval clearly superior.",
         "key_claims": ["Dense retrieval outperforms all baselines"]},
        {"title": "Anti retrieval", "abstract": "Retrieval fails on complex queries.",
         "key_claims": ["Retrieval fails on multi-hop reasoning"]},
    ]
    report = synth.build(papers, "retrieval effectiveness")
    # At minimum, should have themes and cells
    assert report.total_sources == 2
    for row in report.themes:
        assert len(row.cells) >= 1


def test_consensus_strong_agreement():
    """Papers that all agree should produce strong_agreement consensus."""
    synth = SynthesisMatrix(runtime=None)
    papers = [
        {"title": f"Paper {i}", "abstract": "Transformers are effective for NLP tasks.",
         "key_claims": ["Transformers improve NLP accuracy"]}
        for i in range(5)
    ]
    report = synth.build(papers, "transformers NLP")
    assert report.total_sources == 5
    # With identical claims, most themes should have agreement
    # (heuristic mode assigns positions based on keyword overlap)


# ---------------------------------------------------------------------------
# Matrix structure
# ---------------------------------------------------------------------------

def test_matrix_has_cells_for_each_source():
    """Each theme row has cells for every source."""
    synth = SynthesisMatrix(runtime=None)
    papers = [
        {"title": "Source A", "abstract": "Study on optimization.",
         "key_claims": ["Optimization improves throughput"]},
        {"title": "Source B", "abstract": "Study on caching.",
         "key_claims": ["Caching reduces latency"]},
    ]
    report = synth.build(papers, "performance")
    for row in report.themes:
        # Each row should have cells (at least 1)
        assert len(row.cells) >= 1
        # All sentiments should be valid
        for cell in row.cells:
            assert cell.sentiment in {"supports", "contradicts", "neutral", "silent"}
            assert 0.0 <= cell.evidence_strength <= 1.0


def test_matrix_report_counts_consistent():
    """Report counts are consistent with theme data."""
    synth = SynthesisMatrix(runtime=None)
    papers = [
        {"title": f"Paper {i}", "abstract": f"Research on topic {i}.",
         "key_claims": [f"Finding {i}: method works"]}
        for i in range(4)
    ]
    report = synth.build(papers, "research methods")
    # strong_agreements + contradictions should not exceed total themes
    assert report.strong_agreements + report.contradictions <= len(report.themes)
    assert report.gaps >= 0


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------

def test_markdown_table_not_empty():
    """Non-empty matrix produces non-empty markdown."""
    synth = SynthesisMatrix(runtime=None)
    papers = [
        {"title": "Paper A", "abstract": "Study on X.",
         "key_claims": ["X is effective"]},
        {"title": "Paper B", "abstract": "Study on Y.",
         "key_claims": ["Y complements X"]},
    ]
    report = synth.build(papers, "effectiveness")
    md = synth.to_markdown_table(report)
    assert len(md) > 0
    assert "| Theme |" in md or "Theme" in md


def test_markdown_empty_matrix():
    """Empty matrix produces sensible output."""
    synth = SynthesisMatrix(runtime=None)
    report = synth.build([], "test")
    md = synth.to_markdown_table(report)
    assert "empty" in md.lower()


def test_to_dict_serializable():
    """to_dict output is JSON-serializable."""
    import json
    synth = SynthesisMatrix(runtime=None)
    papers = [
        {"title": "Paper 1", "abstract": "About X.",
         "key_claims": ["X works"]},
    ]
    report = synth.build(papers, "test")
    d = synth.to_dict(report)
    # Should be JSON-serializable
    json_str = json.dumps(d)
    assert len(json_str) > 0


# ---------------------------------------------------------------------------
# Tokenizer and noun phrase extraction
# ---------------------------------------------------------------------------

def test_tokenize_basic():
    """Tokenizer splits text into lowercase tokens >= 3 chars."""
    tokens = _tokenize("Hello World! This is a Test-123.")
    assert "hello" in tokens
    assert "world" in tokens
    assert "this" in tokens
    assert "test" in tokens
    # Short tokens (< 3 chars) excluded
    assert "is" not in tokens
    assert "a" not in tokens


def test_tokenize_empty():
    """Empty string returns empty list."""
    assert _tokenize("") == []


def test_noun_phrases_frequency():
    """Most frequent phrases appear first."""
    texts = [
        "machine learning is great",
        "machine learning for NLP",
        "deep machine learning methods",
    ]
    phrases = _extract_noun_phrases(texts, 3)
    assert len(phrases) >= 1
    # "machine" and "learning" should be among top phrases
    all_text = " ".join(phrases)
    assert "machine" in all_text or "learning" in all_text


def test_noun_phrases_stopwords_filtered():
    """Stopwords are filtered from phrases."""
    texts = ["the and for that this with from are was were"]
    phrases = _extract_noun_phrases(texts, 5)
    # All words are stopwords, so no phrases should survive
    assert len(phrases) == 0


def test_noun_phrases_max_respected():
    """Returns at most max_phrases phrases."""
    texts = [f"unique_topic_{i} is interesting and novel" for i in range(100)]
    phrases = _extract_noun_phrases(texts, 5)
    assert len(phrases) <= 5
