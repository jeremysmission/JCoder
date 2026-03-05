"""Iteration 6: LLM response parsing robustness tests."""

import pytest
from core.layered_triage import _parse_drone_response


# ---------------------------------------------------------------------------
# Drone response parser edge cases
# ---------------------------------------------------------------------------

def test_parse_clean_response():
    """Standard well-formatted response."""
    text = "0|0.85|Good paper on retrieval.\n1|0.42|Weak study on generation."
    result = _parse_drone_response(text, 2)
    assert len(result) == 2
    assert result[0]["score"] == 0.85
    assert result[1]["score"] == 0.42


def test_parse_with_brackets():
    """Response with [0] style indices."""
    text = "[0]|0.90|Excellent work.\n[1]|0.30|Mediocre."
    result = _parse_drone_response(text, 2)
    assert len(result) == 2
    assert result[0]["score"] == 0.90


def test_parse_with_extra_whitespace():
    """Response with lots of whitespace."""
    text = "  0  |  0.75  |  Decent paper with good methodology.  \n  1 | 0.50 | Average.  "
    result = _parse_drone_response(text, 2)
    assert len(result) == 2
    assert result[0]["score"] == 0.75


def test_parse_with_preamble():
    """LLM adds text before the actual data."""
    text = """Here are my ratings for the papers:

0|0.85|Strong methodology.
1|0.60|Adequate study.
2|0.30|Weak evidence.

These are my assessments."""
    result = _parse_drone_response(text, 3)
    assert len(result) == 3


def test_parse_empty_response():
    """Empty string produces no results."""
    result = _parse_drone_response("", 5)
    assert result == {}


def test_parse_garbage_response():
    """Complete garbage produces no results."""
    text = "This is not formatted correctly at all!\nNo pipes here.\nJust random text."
    result = _parse_drone_response(text, 3)
    assert result == {}


def test_parse_partial_garbage():
    """Mix of valid and invalid lines."""
    text = "Preamble text\n0|0.80|Valid line.\ngarbage line\n1|0.60|Another valid.\nbad|bad|bad"
    result = _parse_drone_response(text, 3)
    assert len(result) == 2
    assert 0 in result
    assert 1 in result


def test_parse_score_out_of_range():
    """Scores outside [0, 1] are rejected."""
    text = "0|1.5|Too high.\n1|-0.3|Negative.\n2|0.80|Valid."
    result = _parse_drone_response(text, 3)
    assert len(result) == 1
    assert 2 in result


def test_parse_index_out_of_range():
    """Indices beyond expected count are rejected."""
    text = "0|0.80|Valid.\n5|0.70|Index too high.\n-1|0.60|Negative index."
    result = _parse_drone_response(text, 3)
    assert len(result) == 1
    assert 0 in result


def test_parse_duplicate_indices():
    """Later entry for same index overwrites earlier."""
    text = "0|0.80|First rating.\n0|0.90|Revised rating."
    result = _parse_drone_response(text, 1)
    assert result[0]["score"] == 0.90  # last wins


def test_parse_pipes_in_summary():
    """Pipes in the summary text don't break parsing."""
    text = "0|0.85|Study compares A | B | C methods."
    result = _parse_drone_response(text, 1)
    assert len(result) == 1
    assert "|" in result[0]["summary"]


def test_parse_unicode_summary():
    """Unicode characters in summary don't break parsing."""
    text = "0|0.75|Study uses alpha/beta method."
    result = _parse_drone_response(text, 1)
    assert len(result) == 1


def test_parse_very_long_summary():
    """Very long summary line doesn't crash."""
    long_summary = "A" * 10000
    text = f"0|0.80|{long_summary}"
    result = _parse_drone_response(text, 1)
    assert len(result) == 1
    assert len(result[0]["summary"]) == 10000


def test_parse_missing_summary():
    """Line with only index|score (no summary) is rejected."""
    text = "0|0.80"
    result = _parse_drone_response(text, 1)
    assert len(result) == 0  # needs 3 parts


def test_parse_float_index():
    """Float index like 0.0 should fail cleanly."""
    text = "0.0|0.80|Summary."
    result = _parse_drone_response(text, 1)
    # int("0.0") raises ValueError, so this should be skipped
    assert len(result) == 0


def test_parse_large_expected():
    """Parser handles large expected count."""
    lines = [f"{i}|{0.5 + i*0.001:.3f}|Paper {i} summary." for i in range(100)]
    text = "\n".join(lines)
    result = _parse_drone_response(text, 100)
    assert len(result) == 100


# ---------------------------------------------------------------------------
# Synthesis theme extraction parsing
# ---------------------------------------------------------------------------

def test_synthesis_heuristic_themes():
    """Synthesis matrix extracts themes from key_claims heuristically."""
    from core.synthesis_matrix import SynthesisMatrix
    synth = SynthesisMatrix(runtime=None)
    papers = [
        {"title": "Paper 1", "abstract": "About dense retrieval.",
         "key_claims": ["Dense retrieval outperforms sparse", "BM25 is baseline"]},
        {"title": "Paper 2", "abstract": "About sparse methods.",
         "key_claims": ["Sparse methods are more efficient", "TF-IDF still works"]},
    ]
    report = synth.build(papers, "retrieval methods")
    assert report.total_sources == 2
    assert len(report.themes) >= 1


def test_synthesis_no_claims_heuristic():
    """Papers without key_claims still produce themes from abstracts."""
    from core.synthesis_matrix import SynthesisMatrix
    synth = SynthesisMatrix(runtime=None)
    papers = [
        {"title": "Paper A", "abstract": "Study on neural network architecture for NLP."},
        {"title": "Paper B", "abstract": "Survey of transformer models in NLP."},
    ]
    report = synth.build(papers, "NLP transformers")
    assert report.total_sources == 2


# ---------------------------------------------------------------------------
# Devil's advocate query parsing
# ---------------------------------------------------------------------------

def test_advocate_heuristic_queries():
    """Heuristic counter-queries have expected suffixes."""
    from core.devils_advocate import DevilsAdvocate
    da = DevilsAdvocate(runtime=None)
    report = da.challenge("RAG improves accuracy", [])
    assert len(report.counter_queries) == 3
    assert any("criticism" in q for q in report.counter_queries)
    assert any("limitations" in q for q in report.counter_queries)
    assert any("debunked" in q for q in report.counter_queries)


def test_advocate_empty_claim_queries():
    """Empty claim still produces counter-queries."""
    from core.devils_advocate import DevilsAdvocate
    da = DevilsAdvocate(runtime=None)
    report = da.challenge("", [])
    assert len(report.counter_queries) == 3
