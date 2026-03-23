"""Tests for R13: Open Knowledge Fallback + Retrieval Tiering.

Verifies that SmartOrchestrator falls back to LLM parametric knowledge
when retrieval returns no results or very low-scoring chunks.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from core.smart_orchestrator import SmartOrchestrator, SmartAnswerResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_orchestrator(open_knowledge=True, chunks=None, corrective=None):
    """Build a SmartOrchestrator with mocked retriever/runtime."""
    retriever = MagicMock()
    retriever.retrieve.return_value = chunks or []

    runtime = MagicMock()
    runtime.generate.return_value = "The border collie is the smartest dog breed."

    return SmartOrchestrator(
        retriever=retriever,
        runtime=runtime,
        open_knowledge_fallback=open_knowledge,
        corrective=corrective,
    )


# ---------------------------------------------------------------------------
# Open Knowledge Fallback: No Chunks
# ---------------------------------------------------------------------------

class TestOpenKnowledgeNoChunks:

    def test_fallback_when_no_chunks_and_enabled(self):
        orch = _make_orchestrator(open_knowledge=True, chunks=[])
        result = orch.answer("What is the smartest dog breed?")
        assert "Open Knowledge Mode" in result.answer
        assert "border collie" in result.answer.lower()
        assert result.retrieval_strategy == "open_knowledge_fallback"
        assert result.confidence == 0.1

    def test_no_fallback_when_disabled(self):
        orch = _make_orchestrator(open_knowledge=False, chunks=[])
        result = orch.answer("What is the smartest dog breed?")
        assert "No relevant code found" in result.answer
        assert result.confidence == 0.0

    def test_fallback_still_logs_telemetry(self):
        telemetry = MagicMock()
        orch = _make_orchestrator(open_knowledge=True, chunks=[])
        orch.telemetry = telemetry
        orch.answer("What is 2+2?")
        telemetry.log.assert_called_once()

    def test_fallback_sources_empty(self):
        orch = _make_orchestrator(open_knowledge=True, chunks=[])
        result = orch.answer("General knowledge question")
        assert result.sources == []
        assert result.chunk_count == 0


# ---------------------------------------------------------------------------
# Open Knowledge Fallback: Low-Scoring Chunks
# ---------------------------------------------------------------------------

class TestOpenKnowledgeLowScoring:

    def test_fallback_on_very_low_scores(self):
        low_chunks = [
            {"content": "irrelevant noise", "federated_score": 0.05, "source_path": "x.py"},
            {"content": "more noise", "federated_score": 0.08, "source_path": "y.py"},
        ]
        orch = _make_orchestrator(open_knowledge=True, chunks=low_chunks)
        result = orch.answer("What color is the sky?")
        assert "Open Knowledge Mode" in result.answer
        assert result.retrieval_strategy == "open_knowledge_fallback"

    def test_no_fallback_on_decent_scores(self):
        good_chunks = [
            {"content": "def sort(items): return sorted(items)", "federated_score": 0.6, "source_path": "sort.py"},
        ]
        orch = _make_orchestrator(open_knowledge=True, chunks=good_chunks)
        result = orch.answer("How to sort a list?")
        assert "Open Knowledge Mode" not in result.answer
        assert result.retrieval_strategy != "open_knowledge_fallback"

    def test_threshold_boundary_at_015(self):
        """Chunks scoring exactly 0.15 should NOT trigger fallback (>= threshold)."""
        boundary_chunks = [
            {"content": "some code", "federated_score": 0.15, "source_path": "x.py"},
        ]
        orch = _make_orchestrator(open_knowledge=True, chunks=boundary_chunks)
        result = orch.answer("Some query")
        assert result.retrieval_strategy != "open_knowledge_fallback"

    def test_uses_score_key_fallback(self):
        """Chunks without federated_score should use 'score' key."""
        chunks = [
            {"content": "noise", "score": 0.03, "source_path": "a.py"},
        ]
        orch = _make_orchestrator(open_knowledge=True, chunks=chunks)
        result = orch.answer("Random question")
        assert "Open Knowledge Mode" in result.answer

    def test_chunks_without_score_default_to_05(self):
        """Chunks with no score key at all default to 0.5 (no fallback)."""
        chunks = [
            {"content": "def foo(): pass", "source_path": "b.py"},
        ]
        orch = _make_orchestrator(open_knowledge=True, chunks=chunks)
        result = orch.answer("What does foo do?")
        assert result.retrieval_strategy != "open_knowledge_fallback"


# ---------------------------------------------------------------------------
# Corrective Retrieval Integration
# ---------------------------------------------------------------------------

class TestOpenKnowledgeWithCorrective:

    def test_fallback_after_corrective_returns_empty(self):
        corrective = MagicMock()
        corrective.retrieve.return_value = ([], {"strategy": "corrective", "confidence": 0.0, "attempts": 2})
        orch = _make_orchestrator(open_knowledge=True, corrective=corrective)
        result = orch.answer("What is quantum computing?")
        assert "Open Knowledge Mode" in result.answer


# ---------------------------------------------------------------------------
# SmartAnswerResult fields
# ---------------------------------------------------------------------------

class TestSmartAnswerResultFields:

    def test_result_has_all_expected_fields(self):
        orch = _make_orchestrator(open_knowledge=True, chunks=[])
        result = orch.answer("Test query")
        assert hasattr(result, "answer")
        assert hasattr(result, "sources")
        assert hasattr(result, "chunk_count")
        assert hasattr(result, "confidence")
        assert hasattr(result, "retrieval_strategy")
        assert hasattr(result, "retrieval_attempts")

    def test_default_confidence_gate(self):
        orch = _make_orchestrator(open_knowledge=False)
        assert orch.confidence_gate == 0.2

    def test_open_knowledge_flag_stored(self):
        orch = _make_orchestrator(open_knowledge=True)
        assert orch.open_knowledge_fallback is True


# ---------------------------------------------------------------------------
# Quality tiering (retrieval-level)
# ---------------------------------------------------------------------------

class TestQualityTiering:

    def test_quality_score_filter_in_index_engine(self):
        """Verify search_fts5_direct accepts min_quality parameter."""
        from core.index_engine import IndexEngine
        # Just verify the method signature accepts the param
        import inspect
        sig = inspect.signature(IndexEngine.search_fts5_direct)
        assert "min_quality" in sig.parameters
