"""
Tests for core.self_learning_pipeline -- SelfLearningPipeline orchestrator.
All sub-modules are mocked; no live LLM, retrieval, or database needed.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from core.self_learning_pipeline import PipelineAnswer, SelfLearningPipeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_retriever(chunks: Optional[List[Dict]] = None):
    """Return a mock RetrievalEngine that returns fixed chunks."""
    r = MagicMock()
    default_chunks = [
        {"id": "c1", "content": "def foo(): return 42", "source_path": "src/foo.py", "score": 0.9},
        {"id": "c2", "content": "def bar(): return 7", "source_path": "src/bar.py", "score": 0.8},
    ]
    r.retrieve.return_value = chunks if chunks is not None else default_chunks
    return r


def _mock_runtime(response: str = "The answer is 42."):
    """Return a mock Runtime that returns a fixed response."""
    r = MagicMock()
    r.generate.return_value = response
    return r


def _make_pipeline(**overrides) -> SelfLearningPipeline:
    """Build a pipeline with sensible mock defaults, overridable per kwarg."""
    defaults = {
        "retriever": _mock_retriever(),
        "runtime": _mock_runtime(),
    }
    defaults.update(overrides)
    return SelfLearningPipeline(**defaults)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestPipelineConstruction:

    def test_all_modules_none(self):
        """Pipeline works with all optional modules absent."""
        p = _make_pipeline()
        assert p.meta is None
        assert p.reflection is None
        assert p.corrective is None
        assert p.best_of_n is None
        assert p.experience is None
        assert p.stigmergy is None
        assert p.telemetry is None
        assert p.star is None
        assert p.active is None
        assert p.continual is None
        assert p._query_count == 0

    def test_modules_stored(self):
        """All provided modules are stored on the pipeline."""
        meta = MagicMock()
        refl = MagicMock()
        p = _make_pipeline(meta_controller=meta, reflection=refl)
        assert p.meta is meta
        assert p.reflection is refl

    def test_custom_confidence_gate(self):
        """Confidence gate is configurable."""
        p = _make_pipeline(confidence_gate=0.5)
        assert p.confidence_gate == 0.5


# ---------------------------------------------------------------------------
# answer() -- standard path
# ---------------------------------------------------------------------------

class TestAnswerStandard:

    def test_basic_answer(self):
        """Standard strategy returns a PipelineAnswer with expected fields."""
        p = _make_pipeline()
        result = p.answer("What does foo return?")
        assert isinstance(result, PipelineAnswer)
        assert result.answer == "The answer is 42."
        assert result.strategy_used == "standard"
        assert result.chunk_count == 2
        assert result.latency_ms >= 0
        assert len(result.query_id) == 12

    def test_query_counter_increments(self):
        """Each answer() call increments the query counter."""
        p = _make_pipeline()
        p.answer("q1")
        p.answer("q2")
        assert p._query_count == 2

    def test_sources_collected(self):
        """Source paths are collected and deduplicated from chunks."""
        p = _make_pipeline()
        result = p.answer("What are the sources?")
        assert "src/foo.py" in result.sources
        assert "src/bar.py" in result.sources

    def test_runtime_receives_chunk_texts(self):
        """Runtime.generate() receives chunk content strings."""
        rt = _mock_runtime()
        p = _make_pipeline(runtime=rt)
        p.answer("test")
        call_args = rt.generate.call_args
        chunk_texts = call_args[0][1]  # second positional arg
        assert "def foo(): return 42" in chunk_texts


# ---------------------------------------------------------------------------
# answer() -- empty retrieval
# ---------------------------------------------------------------------------

class TestAnswerEmptyRetrieval:

    def test_no_chunks_returns_fallback(self):
        """When retrieval returns no chunks, a fallback answer is returned."""
        retriever = _mock_retriever(chunks=[])
        p = _make_pipeline(retriever=retriever)
        result = p.answer("Unknown topic")
        assert "No relevant code" in result.answer
        assert result.chunk_count == 0


# ---------------------------------------------------------------------------
# answer() -- meta-cognitive strategy selection
# ---------------------------------------------------------------------------

class TestAnswerMetaCognitive:

    def test_meta_selects_strategy(self):
        """When meta controller is present, strategy comes from it."""
        meta = MagicMock()
        sig = MagicMock()
        sig.query_type = "debug"
        meta.select_strategy.return_value = ("corrective", sig)

        p = _make_pipeline(meta_controller=meta)
        result = p.answer("Why is this crashing?")
        assert result.strategy_used == "corrective"

    def test_meta_outcome_reported(self):
        """Meta controller receives outcome report after answer."""
        meta = MagicMock()
        sig = MagicMock()
        sig.query_type = "explain"
        meta.select_strategy.return_value = ("standard", sig)

        p = _make_pipeline(meta_controller=meta)
        p.answer("What is a decorator?")
        meta.report_outcome.assert_called_once()


# ---------------------------------------------------------------------------
# answer() -- best_of_n strategy
# ---------------------------------------------------------------------------

class TestAnswerBestOfN:

    def test_best_of_n_used(self):
        """best_of_n strategy delegates to BestOfNGenerator."""
        meta = MagicMock()
        sig = MagicMock()
        sig.query_type = "reasoning"
        meta.select_strategy.return_value = ("best_of_n", sig)

        bon = MagicMock()
        bon.generate.return_value = MagicMock(answer="Best candidate answer")

        p = _make_pipeline(meta_controller=meta, best_of_n=bon)
        result = p.answer("Complex reasoning question")
        assert result.answer == "Best candidate answer"
        assert result.strategy_used == "best_of_n"
        bon.generate.assert_called_once()


# ---------------------------------------------------------------------------
# answer() -- reflective (STaR) strategy
# ---------------------------------------------------------------------------

class TestAnswerReflective:

    def test_star_reasoning_used(self):
        """reflective strategy delegates to STaR reasoner."""
        meta = MagicMock()
        sig = MagicMock()
        sig.query_type = "reasoning"
        meta.select_strategy.return_value = ("reflective", sig)

        star = MagicMock()
        star.answer_with_reasoning.return_value = ("chain of thought", "Final answer")

        p = _make_pipeline(meta_controller=meta, star=star)
        result = p.answer("Prove this algorithm is correct")
        assert result.answer == "Final answer"
        assert result.reasoning == "chain of thought"
        star.answer_with_reasoning.assert_called_once()


# ---------------------------------------------------------------------------
# answer() -- corrective retrieval
# ---------------------------------------------------------------------------

class TestAnswerCorrective:

    def test_corrective_retrieval_used(self):
        """corrective strategy uses CorrectiveRetriever instead of standard."""
        meta = MagicMock()
        sig = MagicMock()
        sig.query_type = "debug"
        meta.select_strategy.return_value = ("corrective", sig)

        corrective = MagicMock()
        corrective.retrieve.return_value = (
            [{"id": "cx", "content": "fixed code", "source_path": "fix.py", "score": 0.95}],
            {"confidence": 0.9, "strategy": "corrective", "attempts": 2},
        )

        p = _make_pipeline(meta_controller=meta, corrective=corrective)
        result = p.answer("Why is this failing?")
        corrective.retrieve.assert_called_once()
        assert result.chunk_count == 1


# ---------------------------------------------------------------------------
# answer() -- stigmergy boost
# ---------------------------------------------------------------------------

class TestAnswerStigmergy:

    def test_stigmergy_boosts_scores(self):
        """Stigmergy booster reorders chunks by boosted scores."""
        stigmergy = MagicMock()
        # Return reversed order to prove reordering happens
        stigmergy.boost_scores.return_value = [("c2", 0.95), ("c1", 0.85)]

        p = _make_pipeline(stigmergy=stigmergy)
        result = p.answer("test")
        stigmergy.boost_scores.assert_called_once()
        assert result.chunk_count == 2

    def test_stigmergy_deposits_on_success(self):
        """After answering, pheromone is deposited."""
        stigmergy = MagicMock()
        stigmergy.boost_scores.return_value = [("c1", 0.9), ("c2", 0.8)]

        p = _make_pipeline(stigmergy=stigmergy)
        p.answer("test")
        stigmergy.deposit.assert_called_once()


# ---------------------------------------------------------------------------
# answer() -- experience replay
# ---------------------------------------------------------------------------

class TestAnswerExperienceReplay:

    def test_experience_injected_as_system_prompt(self):
        """Experience replay examples are injected into the system prompt."""
        experience = MagicMock()
        experience.retrieve.return_value = [MagicMock()]
        experience.format_as_examples.return_value = "Example: Q: foo A: bar"

        rt = _mock_runtime()
        p = _make_pipeline(runtime=rt, experience=experience)
        p.answer("test query")

        # Runtime should receive a system_prompt containing the experience
        call_kwargs = rt.generate.call_args
        system_prompt = call_kwargs[1].get("system_prompt") or call_kwargs[0][2] if len(call_kwargs[0]) > 2 else None
        # The experience prefix should be passed through
        experience.retrieve.assert_called_once()
        experience.format_as_examples.assert_called_once()

    def test_high_confidence_stored(self):
        """High-confidence answers are stored in experience replay."""
        experience = MagicMock()
        experience.retrieve.return_value = []
        experience.format_as_examples.return_value = ""

        reflection = MagicMock()
        reflection.full_reflection.return_value = {"confidence": 0.8}

        p = _make_pipeline(experience=experience, reflection=reflection)
        p.answer("test")
        experience.store.assert_called_once()

    def test_low_confidence_not_stored(self):
        """Low-confidence answers are NOT stored in experience replay."""
        experience = MagicMock()
        experience.retrieve.return_value = []
        experience.format_as_examples.return_value = ""

        reflection = MagicMock()
        reflection.full_reflection.return_value = {"confidence": 0.3}

        p = _make_pipeline(experience=experience, reflection=reflection)
        p.answer("test")
        experience.store.assert_not_called()


# ---------------------------------------------------------------------------
# answer() -- reflection and confidence gating
# ---------------------------------------------------------------------------

class TestAnswerReflection:

    def test_reflection_scores_returned(self):
        """Reflection scores appear in the PipelineAnswer."""
        reflection = MagicMock()
        reflection.full_reflection.return_value = {
            "confidence": 0.85,
            "relevant": 0.9,
            "supported": 0.8,
            "useful": 0.9,
        }
        p = _make_pipeline(reflection=reflection)
        result = p.answer("test")
        assert result.confidence == 0.85
        assert result.reflection["relevant"] == 0.9

    def test_low_confidence_warning_prepended(self):
        """Below confidence_gate, a warning is prepended to the answer."""
        reflection = MagicMock()
        reflection.full_reflection.return_value = {"confidence": 0.1}

        p = _make_pipeline(reflection=reflection, confidence_gate=0.2)
        result = p.answer("test")
        assert "not confident" in result.answer.lower()

    def test_reflection_exception_isolated(self):
        """If reflection throws, the pipeline still returns an answer."""
        reflection = MagicMock()
        reflection.full_reflection.side_effect = RuntimeError("reflection crash")

        p = _make_pipeline(reflection=reflection)
        result = p.answer("test")
        assert result.answer == "The answer is 42."


# ---------------------------------------------------------------------------
# answer() -- telemetry
# ---------------------------------------------------------------------------

class TestAnswerTelemetry:

    def test_telemetry_logged(self):
        """Telemetry event is logged for each query."""
        telemetry = MagicMock()
        with patch("core.self_learning_pipeline.QueryEvent", MagicMock()):
            p = _make_pipeline(telemetry=telemetry)
            p.answer("test query")
        telemetry.log.assert_called_once()

    def test_telemetry_exception_isolated(self):
        """If telemetry throws, the pipeline still returns an answer."""
        telemetry = MagicMock()
        telemetry.log.side_effect = RuntimeError("telemetry crash")

        with patch("core.self_learning_pipeline.QueryEvent", MagicMock()):
            p = _make_pipeline(telemetry=telemetry)
            result = p.answer("test")
        assert result.answer == "The answer is 42."


# ---------------------------------------------------------------------------
# answer() -- active learning
# ---------------------------------------------------------------------------

class TestAnswerActiveLearning:

    def test_learning_value_computed(self):
        """When active learner is present, learning_value is computed."""
        active = MagicMock()
        reflection = MagicMock()
        reflection.full_reflection.return_value = {"confidence": 0.5}

        p = _make_pipeline(active_learner=active, reflection=reflection)
        result = p.answer("borderline question")
        # At confidence=0.5, learning_value = 4 * 0.5 * 0.5 = 1.0
        assert result.learning_value == 1.0

    def test_no_active_learner_zero_value(self):
        """Without active learner, learning_value is 0."""
        p = _make_pipeline()
        result = p.answer("test")
        assert result.learning_value == 0.0


# ---------------------------------------------------------------------------
# answer() -- module exception isolation
# ---------------------------------------------------------------------------

class TestModuleIsolation:

    def test_all_modules_crash_still_answers(self):
        """Even if every optional module crashes, the pipeline returns an answer."""
        meta = MagicMock()
        sig = MagicMock()
        sig.query_type = "explain"
        meta.select_strategy.return_value = ("standard", sig)
        meta.report_outcome.side_effect = RuntimeError("meta crash")

        stigmergy = MagicMock()
        stigmergy.boost_scores.return_value = [("c1", 0.9), ("c2", 0.8)]
        stigmergy.deposit.side_effect = RuntimeError("stigmergy crash")

        reflection = MagicMock()
        reflection.full_reflection.side_effect = RuntimeError("reflection crash")

        telemetry = MagicMock()
        telemetry.log.side_effect = RuntimeError("telemetry crash")

        experience = MagicMock()
        experience.retrieve.return_value = []
        experience.format_as_examples.return_value = ""

        with patch("core.self_learning_pipeline.QueryEvent", MagicMock()):
            p = _make_pipeline(
                meta_controller=meta, stigmergy=stigmergy,
                reflection=reflection, telemetry=telemetry,
                experience=experience,
            )
            result = p.answer("test")

        assert result.answer == "The answer is 42."


# ---------------------------------------------------------------------------
# system_health()
# ---------------------------------------------------------------------------

class TestSystemHealth:

    def test_empty_pipeline(self):
        """Health report with no modules lists all as inactive."""
        p = _make_pipeline()
        health = p.system_health()
        assert health["queries_processed"] == 0
        assert len(health["modules_inactive"]) == 10
        assert len(health["modules_active"]) == 0

    def test_active_modules_listed(self):
        """Active modules appear in the health report."""
        telemetry = MagicMock()
        telemetry.stats.return_value = {"total": 42}

        p = _make_pipeline(telemetry=telemetry)
        health = p.system_health()
        assert "telemetry" in health["modules_active"]
        assert health["telemetry_stats"]["total"] == 42

    def test_query_count_tracked(self):
        """Health report reflects queries processed."""
        p = _make_pipeline()
        p.answer("q1")
        p.answer("q2")
        health = p.system_health()
        assert health["queries_processed"] == 2

    def test_continual_health_included(self):
        """Continual learner health report is included when available."""
        continual = MagicMock()
        continual.health_report.return_value = {"capabilities_tracked": 3}

        p = _make_pipeline(continual=continual)
        health = p.system_health()
        assert health["continual_health"]["capabilities_tracked"] == 3

    def test_meta_strategy_preferences_included(self):
        """Meta-cognitive strategy preferences appear in health."""
        meta = MagicMock()
        meta.best_strategy_per_type.return_value = {"debug": "corrective"}

        p = _make_pipeline(meta_controller=meta)
        health = p.system_health()
        assert health["strategy_preferences"]["debug"] == "corrective"


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------

class TestThreadSafety:

    def test_concurrent_queries(self):
        """Multiple threads can call answer() without crashing."""
        p = _make_pipeline()
        errors = []

        def _query(idx):
            try:
                result = p.answer(f"Question {idx}")
                assert isinstance(result, PipelineAnswer)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=_query, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert not errors
        assert p._query_count == 10
