"""
Tests for core.rapid_digest -- 3-pass research digestion engine.
All LLM calls are mocked; no live runtime needed.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest

from core.rapid_digest import (
    PaperDigest,
    PrototypeStub,
    RapidDigester,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_runtime(responses=None):
    rt = MagicMock()
    if responses:
        rt.generate.side_effect = responses
    else:
        rt.generate.return_value = (
            "CATEGORY: breakthrough\n"
            "RELEVANCE: 0.8\n"
            "SUMMARY: A new method for retrieval augmented generation."
        )
    return rt


TRIAGE_RESPONSE = (
    "CATEGORY: breakthrough\n"
    "RELEVANCE: 0.8\n"
    "SUMMARY: Novel retrieval method improves RAG accuracy by 20%."
)

EXTRACT_RESPONSE = (
    "CLAIMS:\n"
    "- Claim one about improvement\n"
    "- Claim two about efficiency\n"
    "METHOD: Uses graph-based retrieval with PPR scoring for better recall.\n"
    "RESULTS: 20% improvement on HotpotQA benchmark\n"
    "LIMITS:\n"
    "- Only tested on English datasets\n"
    "CODE: yes\n"
    "CODE_URL: https://github.com/example/repo"
)

SYNTHESIZE_RESPONSE = (
    "IDEAS:\n"
    "- Implement PPR scoring in our retrieval engine\n"
    "- Use graph structure for chunk linking\n"
    "SKETCH: def ppr_retrieve(query, graph):\n    seeds = find_seeds(query)\n    return pagerank(seeds)\n"
    "CONNECTS: HippoRAG, GraphRAG, knowledge graphs"
)

PROTOTYPE_RESPONSE = (
    "class PPRRetriever:\n"
    '    """PPR-based retrieval."""\n'
    "    def __init__(self, db_path):\n"
    "        self.db_path = db_path\n"
    "    def retrieve(self, query):\n"
    "        return []\n"
)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

class TestDataclasses:

    def test_paper_digest_defaults(self):
        d = PaperDigest(
            digest_id="d1", title="Test Paper", source_url="",
            category="incremental", relevance=0.5,
            triage_summary="test", key_claims=[], novel_method="",
            results_summary="", limitations=[], code_available=False,
            code_url="", actionable_ideas=[], implementation_sketch="",
            connections=[],
        )
        assert d.digested_at == 0.0
        assert d.total_seconds == 0.0

    def test_prototype_stub(self):
        s = PrototypeStub(
            paper_title="Test", class_name="TestClass",
            description="desc", code="pass", estimated_effort="trivial",
        )
        assert s.dependencies == []


# ---------------------------------------------------------------------------
# Parser tests
# ---------------------------------------------------------------------------

class TestParsers:

    def test_parse_triage(self):
        cat, rel, summary = RapidDigester._parse_triage(TRIAGE_RESPONSE)
        assert cat == "breakthrough"
        assert rel == 0.8
        assert "retrieval" in summary.lower()

    def test_parse_triage_skip(self):
        cat, rel, summary = RapidDigester._parse_triage(
            "CATEGORY: skip\nRELEVANCE: 0.1\nSUMMARY: Not relevant"
        )
        assert cat == "skip"
        assert rel == 0.1

    def test_parse_extraction(self):
        result = RapidDigester._parse_extraction(EXTRACT_RESPONSE)
        assert len(result["claims"]) == 2
        assert "graph" in result["method"].lower()
        assert result["code"] is True
        assert "github" in result["code_url"]
        assert len(result["limits"]) == 1

    def test_parse_synthesis(self):
        result = RapidDigester._parse_synthesis(SYNTHESIZE_RESPONSE)
        assert len(result["ideas"]) == 2
        assert "ppr" in result["sketch"].lower() or "pagerank" in result["sketch"].lower()

    def test_parse_triage_bad_format(self):
        cat, rel, summary = RapidDigester._parse_triage("garbage input")
        assert cat == "skip"
        assert rel == 0.0


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:

    def test_creates_db(self, tmp_path):
        db = str(tmp_path / "digest.db")
        rd = RapidDigester(runtime=_mock_runtime(), db_path=db)
        stats = rd.stats()
        assert stats["total_digested"] == 0


# ---------------------------------------------------------------------------
# digest (full pipeline)
# ---------------------------------------------------------------------------

class TestDigest:

    def test_full_digest(self, tmp_path):
        db = str(tmp_path / "digest.db")
        rt = _mock_runtime([
            TRIAGE_RESPONSE,    # triage
            EXTRACT_RESPONSE,   # extract
            SYNTHESIZE_RESPONSE,  # synthesize
            PROTOTYPE_RESPONSE,  # prototype
        ])
        rd = RapidDigester(runtime=rt, db_path=db)
        result = rd.digest(
            title="Graph RAG Paper",
            abstract="We propose a graph-based approach to RAG.",
            source_url="https://arxiv.org/abs/1234",
        )
        assert isinstance(result, PaperDigest)
        assert result.category == "breakthrough"
        assert result.relevance == 0.8
        assert len(result.key_claims) >= 1
        assert result.total_seconds >= 0

    def test_skip_paper(self, tmp_path):
        db = str(tmp_path / "digest.db")
        rt = _mock_runtime([
            "CATEGORY: skip\nRELEVANCE: 0.1\nSUMMARY: Not relevant",
        ])
        rd = RapidDigester(runtime=rt, db_path=db)
        result = rd.digest(title="Irrelevant Paper", abstract="About cooking.")
        assert result.category == "skip"
        assert result.key_claims == []
        assert result.novel_method == ""

    def test_low_relevance_skips_prototype(self, tmp_path):
        db = str(tmp_path / "digest.db")
        rt = _mock_runtime([
            "CATEGORY: incremental\nRELEVANCE: 0.3\nSUMMARY: Minor improvement",
            EXTRACT_RESPONSE,
            SYNTHESIZE_RESPONSE,
            # No prototype call expected (relevance < 0.5)
        ])
        rd = RapidDigester(runtime=rt, db_path=db)
        result = rd.digest(title="Minor Paper", abstract="Small improvement.")
        assert result.category == "incremental"

    def test_persistence(self, tmp_path):
        db = str(tmp_path / "digest.db")
        rt = _mock_runtime([
            TRIAGE_RESPONSE, EXTRACT_RESPONSE,
            SYNTHESIZE_RESPONSE, PROTOTYPE_RESPONSE,
        ])
        rd = RapidDigester(runtime=rt, db_path=db)
        rd.digest(title="Test Paper", abstract="Test abstract.")
        stats = rd.stats()
        assert stats["total_digested"] == 1


# ---------------------------------------------------------------------------
# batch_triage
# ---------------------------------------------------------------------------

class TestBatchTriage:

    def test_filters_by_relevance(self, tmp_path):
        db = str(tmp_path / "digest.db")
        rt = _mock_runtime([
            "CATEGORY: breakthrough\nRELEVANCE: 0.9\nSUMMARY: Great paper",
            "CATEGORY: skip\nRELEVANCE: 0.1\nSUMMARY: Not relevant",
            "CATEGORY: incremental\nRELEVANCE: 0.5\nSUMMARY: OK paper",
        ])
        rd = RapidDigester(runtime=rt, db_path=db)
        papers = [
            {"title": "Paper A", "abstract": "Abstract A"},
            {"title": "Paper B", "abstract": "Abstract B"},
            {"title": "Paper C", "abstract": "Abstract C"},
        ]
        results = rd.batch_triage(papers, min_relevance=0.3)
        assert len(results) == 2  # Paper B filtered out
        assert results[0]["relevance"] >= results[1]["relevance"]  # sorted


# ---------------------------------------------------------------------------
# generate_prototype
# ---------------------------------------------------------------------------

class TestGeneratePrototype:

    def test_generates_stub(self, tmp_path):
        db = str(tmp_path / "digest.db")
        rt = _mock_runtime()
        rt.generate.return_value = PROTOTYPE_RESPONSE
        rd = RapidDigester(runtime=rt, db_path=db)
        stub = rd.generate_prototype("Test", "method desc", "sketch")
        assert isinstance(stub, PrototypeStub)
        assert stub.class_name == "PPRRetriever"
        assert stub.estimated_effort == "trivial"

    def test_empty_method_returns_none(self, tmp_path):
        db = str(tmp_path / "digest.db")
        rd = RapidDigester(runtime=_mock_runtime(), db_path=db)
        assert rd.generate_prototype("Test", "", "sketch") is None

    def test_extract_failure_logs_warning(self, tmp_path, caplog):
        db = str(tmp_path / "digest.db")
        rt = _mock_runtime()
        rt.generate.side_effect = RuntimeError("llm down")
        rd = RapidDigester(runtime=rt, db_path=db)

        with caplog.at_level(logging.WARNING, logger="core.rapid_digest"):
            result = rd._extract("Broken Paper", "content")

        assert result == {}
        assert any("Failed to extract paper" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# top_actionable
# ---------------------------------------------------------------------------

class TestTopActionable:

    def test_empty_db(self, tmp_path):
        db = str(tmp_path / "digest.db")
        rd = RapidDigester(runtime=_mock_runtime(), db_path=db)
        assert rd.top_actionable() == []
