"""
Tests for core.smart_orchestrator -- SmartOrchestrator.
All dependencies are mocked; no live runtime needed.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest

from core.smart_orchestrator import SmartAnswerResult, SmartOrchestrator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_retriever(chunks=None):
    rt = MagicMock()
    rt.retrieve.return_value = chunks or []
    return rt


def _mock_runtime(response="def solve(): return 42"):
    rt = MagicMock()
    rt.generate.return_value = response
    return rt


def _mock_reflection(scores=None):
    ref = MagicMock()
    if scores is None:
        scores = {"relevant": 0.8, "supported": 0.7, "useful": 0.9, "confidence": 0.8}
    ref.full_reflection.return_value = scores
    return ref


def _mock_corrective(chunks=None, meta=None):
    cr = MagicMock()
    cr.retrieve.return_value = (
        chunks or _sample_chunks(),
        meta or {"strategy": "standard_confident", "confidence": 0.9, "attempts": 1},
    )
    return cr


def _mock_telemetry():
    t = MagicMock()
    return t


def _sample_chunks(n=3):
    return [
        {"id": f"chunk_{i}", "content": f"def func_{i}(): pass", "source_path": f"src/f{i}.py"}
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# SmartAnswerResult
# ---------------------------------------------------------------------------

class TestSmartAnswerResult:

    def test_defaults(self):
        r = SmartAnswerResult(answer="test", sources=[], chunk_count=0)
        assert r.confidence == 0.0
        assert r.reflection == {}
        assert r.retrieval_strategy == "standard"
        assert r.retrieval_attempts == 1

    def test_custom_fields(self):
        r = SmartAnswerResult(
            answer="hello", sources=["a.py"], chunk_count=2,
            confidence=0.9, retrieval_strategy="corrective_merged",
            retrieval_attempts=3,
        )
        assert r.confidence == 0.9
        assert r.retrieval_attempts == 3


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:

    def test_minimal(self):
        so = SmartOrchestrator(
            retriever=_mock_retriever(),
            runtime=_mock_runtime(),
        )
        assert so.confidence_gate == 0.2
        assert so.telemetry is None
        assert so.reflection is None
        assert so.corrective is None

    def test_with_all_modules(self):
        so = SmartOrchestrator(
            retriever=_mock_retriever(),
            runtime=_mock_runtime(),
            telemetry=_mock_telemetry(),
            reflection=_mock_reflection(),
            corrective=_mock_corrective(),
            confidence_gate=0.3,
        )
        assert so.confidence_gate == 0.3
        assert so.telemetry is not None
        assert so.reflection is not None
        assert so.corrective is not None


# ---------------------------------------------------------------------------
# Answer pipeline
# ---------------------------------------------------------------------------

class TestAnswer:

    def test_basic_answer(self):
        chunks = _sample_chunks()
        so = SmartOrchestrator(
            retriever=_mock_retriever(chunks),
            runtime=_mock_runtime("the answer"),
        )
        result = so.answer("test question")
        assert isinstance(result, SmartAnswerResult)
        assert result.answer == "the answer"
        assert result.chunk_count == 3
        assert len(result.sources) > 0

    def test_empty_retrieval(self):
        so = SmartOrchestrator(
            retriever=_mock_retriever([]),
            runtime=_mock_runtime(),
        )
        result = so.answer("test")
        assert "No relevant code" in result.answer
        assert result.confidence == 0.0
        assert result.chunk_count == 0

    def test_corrective_retrieval_used(self):
        chunks = _sample_chunks()
        cr = _mock_corrective(chunks, {"strategy": "corrective_merged", "confidence": 0.7, "attempts": 2})
        so = SmartOrchestrator(
            retriever=_mock_retriever(),
            runtime=_mock_runtime("answer"),
            corrective=cr,
        )
        result = so.answer("test")
        cr.retrieve.assert_called_once_with("test")
        assert result.retrieval_strategy == "corrective_merged"
        assert result.retrieval_attempts == 2

    def test_reflection_scores_captured(self):
        chunks = _sample_chunks()
        ref_scores = {"relevant": 0.9, "supported": 0.8, "useful": 0.7, "confidence": 0.85}
        so = SmartOrchestrator(
            retriever=_mock_retriever(chunks),
            runtime=_mock_runtime("answer"),
            reflection=_mock_reflection(ref_scores),
        )
        result = so.answer("test")
        assert result.reflection == ref_scores
        assert result.confidence == 0.85

    def test_reflection_failure_isolated(self):
        chunks = _sample_chunks()
        ref = MagicMock()
        ref.full_reflection.side_effect = RuntimeError("crash")
        so = SmartOrchestrator(
            retriever=_mock_retriever(chunks),
            runtime=_mock_runtime("answer"),
            reflection=ref,
        )
        result = so.answer("test")
        # Should still return result despite reflection crash
        assert isinstance(result, SmartAnswerResult)
        assert result.answer == "answer"


# ---------------------------------------------------------------------------
# Confidence gating
# ---------------------------------------------------------------------------

class TestConfidenceGating:

    def test_low_confidence_adds_warning(self):
        chunks = _sample_chunks()
        ref_scores = {"relevant": 0.1, "supported": 0.1, "useful": 0.1, "confidence": 0.1}
        so = SmartOrchestrator(
            retriever=_mock_retriever(chunks),
            runtime=_mock_runtime("raw answer"),
            reflection=_mock_reflection(ref_scores),
            confidence_gate=0.2,
        )
        result = so.answer("test")
        assert "not confident" in result.answer.lower()
        assert "raw answer" in result.answer

    def test_high_confidence_no_warning(self):
        chunks = _sample_chunks()
        ref_scores = {"relevant": 0.9, "supported": 0.9, "useful": 0.9, "confidence": 0.9}
        so = SmartOrchestrator(
            retriever=_mock_retriever(chunks),
            runtime=_mock_runtime("clean answer"),
            reflection=_mock_reflection(ref_scores),
            confidence_gate=0.2,
        )
        result = so.answer("test")
        assert result.answer == "clean answer"
        assert "not confident" not in result.answer.lower()


# ---------------------------------------------------------------------------
# Telemetry
# ---------------------------------------------------------------------------

class TestTelemetry:

    def test_telemetry_logged_on_success(self):
        chunks = _sample_chunks()
        tel = _mock_telemetry()
        so = SmartOrchestrator(
            retriever=_mock_retriever(chunks),
            runtime=_mock_runtime("answer"),
            telemetry=tel,
        )
        so.answer("test")
        tel.log.assert_called_once()

    def test_telemetry_logged_on_empty_retrieval(self):
        tel = _mock_telemetry()
        so = SmartOrchestrator(
            retriever=_mock_retriever([]),
            runtime=_mock_runtime(),
            telemetry=tel,
        )
        so.answer("test")
        tel.log.assert_called_once()

    def test_telemetry_failure_isolated(self, caplog):
        chunks = _sample_chunks()
        tel = MagicMock()
        tel.log.side_effect = RuntimeError("db error")
        so = SmartOrchestrator(
            retriever=_mock_retriever(chunks),
            runtime=_mock_runtime("answer"),
            telemetry=tel,
        )
        # Should not raise despite telemetry crash
        with caplog.at_level(logging.WARNING, logger="core.smart_orchestrator"):
            result = so.answer("test")
        assert isinstance(result, SmartAnswerResult)
        assert any("Telemetry logging failed" in rec.message for rec in caplog.records)

    def test_no_telemetry_still_works(self):
        chunks = _sample_chunks()
        so = SmartOrchestrator(
            retriever=_mock_retriever(chunks),
            runtime=_mock_runtime("answer"),
        )
        result = so.answer("test")
        assert result.answer == "answer"
