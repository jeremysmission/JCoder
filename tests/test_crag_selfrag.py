"""
Integration tests for CRAG and Self-RAG features.

Tests corrective retrieval (CorrectiveRetriever), reflection scoring
(ReflectionEngine), and the SmartOrchestrator that combines both.
All mocked -- no Ollama or network required.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from core.corrective_retrieval import CorrectiveRetriever
from core.reflection import ReflectionEngine, _extract_score
from core.smart_orchestrator import SmartAnswerResult, SmartOrchestrator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_chunk(cid: str, content: str, source: str = "src/app.py"):
    return {"id": cid, "content": content, "source_path": source}


GOOD_CHUNKS = [
    _make_chunk("c1", "def foo(): return 42", "core/foo.py"),
    _make_chunk("c2", "class Bar: pass", "core/bar.py"),
]

ALT_CHUNKS = [
    _make_chunk("c3", "def baz(): return 99", "core/baz.py"),
]


def _mock_retriever(chunks=None):
    r = MagicMock()
    r.retrieve.return_value = chunks if chunks is not None else GOOD_CHUNKS
    return r


def _mock_runtime(response="Here is the answer."):
    rt = MagicMock()
    rt.generate.return_value = response
    return rt


def _mock_reflection(rel=0.9, sup=0.9, use=0.9):
    """Create a ReflectionEngine mock with controllable scores."""
    engine = MagicMock(spec=ReflectionEngine)
    engine.score_relevance.return_value = rel
    engine.score_support.return_value = sup
    engine.score_usefulness.return_value = use
    confidence = 0.3 * rel + 0.4 * sup + 0.3 * use
    engine.full_reflection.return_value = {
        "relevant": rel,
        "supported": sup,
        "useful": use,
        "confidence": round(confidence, 3),
    }
    return engine


# ===================================================================
# CRAG tests -- CorrectiveRetriever
# ===================================================================

class TestCorrectiveRetrieverHighConfidence:
    """When retrieval confidence is high, corrective actions are skipped."""

    def test_high_confidence_returns_original_chunks(self):
        retriever = _mock_retriever(GOOD_CHUNKS)
        reflection = _mock_reflection(rel=0.8)
        crag = CorrectiveRetriever(
            retriever, reflection, confidence_threshold=0.5)

        chunks, meta = crag.retrieve("How does foo work?")

        assert chunks == GOOD_CHUNKS
        assert meta["strategy"] == "standard_confident"
        assert meta["confidence"] == 0.8
        assert meta["attempts"] == 1
        assert retriever.retrieve.call_count == 1

    def test_no_reflection_engine_returns_standard(self):
        retriever = _mock_retriever(GOOD_CHUNKS)
        crag = CorrectiveRetriever(retriever, reflection=None)

        chunks, meta = crag.retrieve("anything")

        assert chunks == GOOD_CHUNKS
        assert meta["strategy"] == "standard"
        assert meta["confidence"] == 1.0


class TestCorrectiveRetrieverLowConfidence:
    """When retrieval confidence is low, corrective strategies fire."""

    def test_low_confidence_triggers_reformulation(self):
        retriever = MagicMock()
        # First call: original query returns low-relevance chunks.
        # Subsequent calls from keyword + decompose strategies return
        # additional chunks (provide enough for all possible calls).
        retriever.retrieve.side_effect = [
            GOOD_CHUNKS, ALT_CHUNKS, [], [], [],
        ]
        reflection = _mock_reflection(rel=0.2)
        # After corrective merge, re-score returns higher
        reflection.score_relevance.side_effect = [0.2, 0.6]

        crag = CorrectiveRetriever(
            retriever, reflection,
            confidence_threshold=0.5, max_reformulations=3)

        chunks, meta = crag.retrieve(
            "How does foo work and where is bar defined?")

        assert meta["strategy"] == "corrective_merged"
        assert meta["attempts"] > 1
        # Should have merged original + alt, deduped
        ids = [c["id"] for c in chunks]
        assert "c1" in ids
        assert "c3" in ids

    def test_max_reformulations_respected(self):
        retriever = MagicMock()
        retriever.retrieve.return_value = GOOD_CHUNKS
        reflection = _mock_reflection(rel=0.1)
        # Always returns low score
        reflection.score_relevance.return_value = 0.1

        crag = CorrectiveRetriever(
            retriever, reflection,
            confidence_threshold=0.5, max_reformulations=2)

        chunks, meta = crag.retrieve(
            "complex query and another part and third part")

        # max_reformulations=2, so attempts capped at max_reformulations+2
        assert meta["attempts"] <= crag.max_reformulations + 2


class TestCorrectiveRetrieverEmptyResults:
    """Edge case: initial retrieval returns nothing."""

    def test_empty_initial_retrieval_tries_corrective(self):
        retriever = MagicMock()
        retriever.retrieve.side_effect = [[], ALT_CHUNKS, []]

        crag = CorrectiveRetriever(
            retriever, reflection=None, max_reformulations=2)

        chunks, meta = crag.retrieve("What is baz?")

        assert len(chunks) > 0
        assert meta["attempts"] > 1

    def test_fully_empty_retrieval(self):
        retriever = MagicMock()
        retriever.retrieve.return_value = []

        crag = CorrectiveRetriever(
            retriever, reflection=None, max_reformulations=2)

        chunks, meta = crag.retrieve("What is baz?")

        assert chunks == []
        assert meta["strategy"] == "corrective_no_improvement"


# ===================================================================
# Self-RAG tests -- ReflectionEngine
# ===================================================================

class TestReflectionScoring:
    """Unit tests for reflection score extraction and computation."""

    @pytest.mark.parametrize("raw,expected", [
        ("8", 0.8),
        ("  3  ", 0.3),
        ("Score: 7", 0.7),
        ("10", 1.0),
        ("0", 0.0),
        ("no number here", 0.5),  # default fallback
    ])
    def test_extract_score(self, raw, expected):
        assert _extract_score(raw) == expected

    def test_full_reflection_computes_weighted_confidence(self):
        runtime = _mock_runtime("8")
        engine = ReflectionEngine(runtime)

        # All calls return "8" -> 0.8
        result = engine.full_reflection("q?", GOOD_CHUNKS, "answer")

        assert result["relevant"] == 0.8
        assert result["supported"] == 0.8
        assert result["useful"] == 0.8
        expected_conf = 0.3 * 0.8 + 0.4 * 0.8 + 0.3 * 0.8
        assert abs(result["confidence"] - expected_conf) < 0.01

    def test_full_reflection_low_support_drags_confidence(self):
        runtime = MagicMock()
        # Relevance=7, Support=1, Usefulness=5
        runtime.generate.side_effect = ["7", "1", "5"]
        engine = ReflectionEngine(runtime)

        result = engine.full_reflection("q?", GOOD_CHUNKS, "hallucinated")

        assert result["supported"] == 0.1
        expected_conf = 0.3 * 0.7 + 0.4 * 0.1 + 0.3 * 0.5
        assert abs(result["confidence"] - expected_conf) < 0.01
        assert result["confidence"] < 0.5

    def test_empty_llm_response_returns_default(self):
        runtime = _mock_runtime("")
        engine = ReflectionEngine(runtime)

        score = engine.score_relevance("q?", GOOD_CHUNKS)

        # Empty string -> _extract_score returns 0.5
        assert score == 0.5


# ===================================================================
# SmartOrchestrator integration tests
# ===================================================================

class TestSmartOrchestratorCRAGIntegration:
    """SmartOrchestrator with corrective retrieval enabled."""

    def test_high_confidence_retrieval_skips_correction(self):
        retriever = _mock_retriever(GOOD_CHUNKS)
        runtime = _mock_runtime("Great answer.")
        reflection = _mock_reflection(rel=0.9, sup=0.9, use=0.9)
        corrective = CorrectiveRetriever(
            retriever, reflection, confidence_threshold=0.5)

        orch = SmartOrchestrator(
            retriever=retriever, runtime=runtime,
            reflection=reflection, corrective=corrective)

        result = orch.answer("How does foo work?")

        assert isinstance(result, SmartAnswerResult)
        assert "Great answer." in result.answer
        assert result.retrieval_strategy == "standard_confident"
        assert result.retrieval_attempts == 1

    def test_low_confidence_retrieval_triggers_corrective(self):
        retriever = MagicMock()
        retriever.retrieve.side_effect = [
            GOOD_CHUNKS, ALT_CHUNKS, [], [], [],
        ]
        runtime = _mock_runtime("Corrected answer.")
        reflection = _mock_reflection(rel=0.2, sup=0.8, use=0.8)
        # Low initial relevance, higher after merge
        reflection.score_relevance.side_effect = [0.2, 0.7]

        corrective = CorrectiveRetriever(
            retriever, reflection,
            confidence_threshold=0.5, max_reformulations=3)

        orch = SmartOrchestrator(
            retriever=retriever, runtime=runtime,
            reflection=reflection, corrective=corrective)

        result = orch.answer(
            "How does foo work and where is bar defined?")

        assert result.retrieval_strategy == "corrective_merged"
        assert result.retrieval_attempts > 1


class TestSmartOrchestratorSelfRAG:
    """SmartOrchestrator with Self-RAG reflection enabled."""

    def test_good_answer_accepted(self):
        retriever = _mock_retriever(GOOD_CHUNKS)
        runtime = _mock_runtime("Good answer.")
        reflection = _mock_reflection(rel=0.9, sup=0.9, use=0.9)

        orch = SmartOrchestrator(
            retriever=retriever, runtime=runtime,
            reflection=reflection, confidence_gate=0.2)

        result = orch.answer("How does foo work?")

        assert "Good answer." in result.answer
        assert result.confidence > 0.8
        # No "not confident" disclaimer
        assert "not confident" not in result.answer.lower()

    def test_hallucinated_answer_gets_disclaimer(self):
        retriever = _mock_retriever(GOOD_CHUNKS)
        runtime = _mock_runtime("Possibly wrong answer.")
        reflection = _mock_reflection(rel=0.3, sup=0.1, use=0.2)

        orch = SmartOrchestrator(
            retriever=retriever, runtime=runtime,
            reflection=reflection, confidence_gate=0.3)

        result = orch.answer("Explain quantum computing in Python")

        # Low confidence triggers disclaimer
        assert result.confidence < 0.3
        assert "not confident" in result.answer.lower()

    def test_reflection_failure_proceeds_gracefully(self):
        retriever = _mock_retriever(GOOD_CHUNKS)
        runtime = _mock_runtime("Fallback answer.")
        reflection = MagicMock(spec=ReflectionEngine)
        reflection.full_reflection.side_effect = RuntimeError("LLM down")

        orch = SmartOrchestrator(
            retriever=retriever, runtime=runtime,
            reflection=reflection, confidence_gate=0.2)

        result = orch.answer("What is bar?")

        # Should still return an answer despite reflection failure
        assert "Fallback answer." in result.answer
        assert result.reflection == {}


class TestSmartOrchestratorEdgeCases:
    """Edge cases for the full pipeline."""

    def test_empty_retrieval_returns_no_relevant_code(self):
        retriever = _mock_retriever([])
        runtime = _mock_runtime("Should not be called.")
        corrective = MagicMock()
        corrective.retrieve.return_value = ([], {
            "strategy": "corrective_no_improvement",
            "confidence": 0.0,
            "attempts": 3,
        })

        orch = SmartOrchestrator(
            retriever=retriever, runtime=runtime,
            corrective=corrective, confidence_gate=0.2)

        result = orch.answer("Nonexistent module?")

        assert "No relevant code found" in result.answer
        assert result.confidence == 0.0
        assert result.chunk_count == 0
        # Runtime.generate should NOT have been called
        runtime.generate.assert_not_called()

    def test_llm_returns_empty_string(self):
        retriever = _mock_retriever(GOOD_CHUNKS)
        runtime = _mock_runtime("")
        reflection = _mock_reflection(rel=0.5, sup=0.5, use=0.1)

        orch = SmartOrchestrator(
            retriever=retriever, runtime=runtime,
            reflection=reflection, confidence_gate=0.5)

        result = orch.answer("What does foo do?")

        # Low confidence (0.36) -> disclaimer prepended to empty answer
        assert isinstance(result.answer, str)

    def test_conflicting_evidence_lowers_support(self):
        """Chunks disagree with generated answer -> low ISSUP."""
        retriever = _mock_retriever(GOOD_CHUNKS)
        runtime = _mock_runtime("foo returns 99")
        # Chunks say 42, answer says 99 -> support score is low
        reflection = _mock_reflection(rel=0.8, sup=0.1, use=0.5)

        orch = SmartOrchestrator(
            retriever=retriever, runtime=runtime,
            reflection=reflection, confidence_gate=0.45)

        result = orch.answer("What does foo return?")

        assert result.reflection["supported"] == 0.1
        # Weighted: 0.3*0.8 + 0.4*0.1 + 0.3*0.5 = 0.43
        assert result.confidence < 0.45
        assert "not confident" in result.answer.lower()

    def test_telemetry_logged_on_success(self):
        retriever = _mock_retriever(GOOD_CHUNKS)
        runtime = _mock_runtime("Logged answer.")
        telemetry = MagicMock()

        orch = SmartOrchestrator(
            retriever=retriever, runtime=runtime,
            telemetry=telemetry, confidence_gate=0.0)

        orch.answer("Test query?")

        telemetry.log.assert_called_once()
        event = telemetry.log.call_args[0][0]
        assert event.query_text == "Test query?"
        assert event.confidence >= 0.0

    def test_telemetry_failure_does_not_crash(self):
        retriever = _mock_retriever(GOOD_CHUNKS)
        runtime = _mock_runtime("Answer.")
        telemetry = MagicMock()
        telemetry.log.side_effect = RuntimeError("DB locked")

        orch = SmartOrchestrator(
            retriever=retriever, runtime=runtime,
            telemetry=telemetry, confidence_gate=0.0)

        # Should not raise
        result = orch.answer("Test query?")
        assert "Answer." in result.answer
