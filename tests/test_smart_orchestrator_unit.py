"""Unit tests for core/smart_orchestrator.py (R18).

Tests the SmartOrchestrator's answer pipeline, confidence gating,
telemetry integration, and corrective retrieval routing.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from core.smart_orchestrator import SmartOrchestrator, SmartAnswerResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _mock_retriever(chunks=None):
    r = MagicMock()
    r.retrieve.return_value = chunks or []
    return r


def _mock_runtime(response="def foo(): pass"):
    r = MagicMock()
    r.generate.return_value = response
    return r


def _make_orch(**kwargs):
    defaults = {
        "retriever": _mock_retriever(kwargs.pop("chunks", None)),
        "runtime": _mock_runtime(kwargs.pop("response", "def foo(): pass")),
    }
    defaults.update(kwargs)
    return SmartOrchestrator(**defaults)


# ---------------------------------------------------------------------------
# Basic answer flow
# ---------------------------------------------------------------------------

class TestAnswerFlow:

    def test_returns_smart_answer_result(self):
        chunks = [{"content": "def foo(): pass", "source_path": "foo.py"}]
        orch = _make_orch(chunks=chunks)
        result = orch.answer("What does foo do?")
        assert isinstance(result, SmartAnswerResult)

    def test_answer_includes_sources(self):
        chunks = [
            {"content": "code1", "source_path": "a.py"},
            {"content": "code2", "source_path": "b.py"},
        ]
        orch = _make_orch(chunks=chunks)
        result = orch.answer("test")
        assert "a.py" in result.sources
        assert "b.py" in result.sources

    def test_no_chunks_returns_no_relevant_code(self):
        orch = _make_orch(chunks=[])
        result = orch.answer("test")
        assert "No relevant code" in result.answer
        assert result.confidence == 0.0

    def test_chunk_count_tracked(self):
        chunks = [{"content": "x", "source_path": "x.py"}] * 5
        orch = _make_orch(chunks=chunks)
        result = orch.answer("test")
        assert result.chunk_count == 5


# ---------------------------------------------------------------------------
# Confidence gating
# ---------------------------------------------------------------------------

class TestConfidenceGating:

    def test_low_confidence_adds_disclaimer(self):
        chunks = [{"content": "code", "source_path": "x.py"}]
        reflection = MagicMock()
        reflection.full_reflection.return_value = {"confidence": 0.1}
        orch = _make_orch(chunks=chunks, reflection=reflection)
        result = orch.answer("test")
        assert "not confident" in result.answer.lower()

    def test_high_confidence_no_disclaimer(self):
        chunks = [{"content": "code", "source_path": "x.py"}]
        reflection = MagicMock()
        reflection.full_reflection.return_value = {"confidence": 0.9}
        orch = _make_orch(chunks=chunks, reflection=reflection)
        result = orch.answer("test")
        assert "not confident" not in result.answer.lower()

    def test_custom_confidence_gate(self):
        chunks = [{"content": "code", "source_path": "x.py"}]
        reflection = MagicMock()
        reflection.full_reflection.return_value = {"confidence": 0.4}
        orch = _make_orch(chunks=chunks, reflection=reflection, confidence_gate=0.5)
        result = orch.answer("test")
        assert "not confident" in result.answer.lower()


# ---------------------------------------------------------------------------
# Corrective retrieval
# ---------------------------------------------------------------------------

class TestCorrectiveRetrieval:

    def test_corrective_used_when_available(self):
        corrective = MagicMock()
        corrective.retrieve.return_value = (
            [{"content": "corrected", "source_path": "c.py"}],
            {"strategy": "corrective", "confidence": 0.8, "attempts": 2},
        )
        orch = _make_orch(corrective=corrective)
        result = orch.answer("test")
        corrective.retrieve.assert_called_once()
        assert result.retrieval_strategy == "corrective"

    def test_standard_retrieval_without_corrective(self):
        chunks = [{"content": "code", "source_path": "x.py"}]
        orch = _make_orch(chunks=chunks)
        result = orch.answer("test")
        assert result.retrieval_strategy == "standard"


# ---------------------------------------------------------------------------
# Telemetry
# ---------------------------------------------------------------------------

class TestTelemetry:

    def test_telemetry_logged_on_answer(self):
        telemetry = MagicMock()
        chunks = [{"content": "code", "source_path": "x.py", "id": "c1"}]
        orch = _make_orch(chunks=chunks, telemetry=telemetry)
        orch.answer("test")
        telemetry.log.assert_called_once()

    def test_telemetry_failure_does_not_crash(self):
        telemetry = MagicMock()
        telemetry.log.side_effect = RuntimeError("DB error")
        chunks = [{"content": "code", "source_path": "x.py"}]
        orch = _make_orch(chunks=chunks, telemetry=telemetry)
        result = orch.answer("test")
        assert result.answer  # Still returns an answer


# ---------------------------------------------------------------------------
# Reflection
# ---------------------------------------------------------------------------

class TestReflection:

    def test_reflection_scores_captured(self):
        chunks = [{"content": "code", "source_path": "x.py"}]
        reflection = MagicMock()
        reflection.full_reflection.return_value = {
            "relevant": 0.9, "supported": 0.8, "useful": 0.7, "confidence": 0.85
        }
        orch = _make_orch(chunks=chunks, reflection=reflection)
        result = orch.answer("test")
        assert result.reflection["relevant"] == 0.9
        assert result.confidence == 0.85

    def test_reflection_failure_graceful(self):
        chunks = [{"content": "code", "source_path": "x.py"}]
        reflection = MagicMock()
        reflection.full_reflection.side_effect = RuntimeError("Model timeout")
        orch = _make_orch(chunks=chunks, reflection=reflection)
        result = orch.answer("test")
        assert result.answer  # Still works


# ---------------------------------------------------------------------------
# SmartAnswerResult
# ---------------------------------------------------------------------------

class TestSmartAnswerResult:

    def test_default_fields(self):
        r = SmartAnswerResult(answer="test", sources=["x.py"], chunk_count=1)
        assert r.confidence == 0.0
        assert r.reflection == {}
        assert r.retrieval_strategy == "standard"
        assert r.retrieval_attempts == 1
