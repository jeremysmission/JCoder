"""Tests for agent/bridge.py -- AgentBridge integration point.

Verifies that the bridge correctly routes data to self-learning modules
and degrades gracefully when modules are unavailable.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from agent.bridge import AgentBridge
from agent.core import Agent, AgentResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _mock_agent():
    return MagicMock(spec=Agent)


def _mock_result(success=True, iterations=3, summary="Done"):
    r = MagicMock(spec=AgentResult)
    r.success = success
    r.iterations = iterations
    r.summary = summary
    r.total_elapsed_s = 1.5
    r.steps = []
    r.total_input_tokens = 100
    r.total_output_tokens = 200
    return r


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

class TestBridgeInit:

    def test_init_with_no_modules(self):
        bridge = AgentBridge(agent=_mock_agent())
        assert bridge.telemetry is None
        assert bridge.experience is None
        assert bridge.cascade is None

    def test_init_with_all_modules(self):
        bridge = AgentBridge(
            agent=_mock_agent(),
            telemetry=MagicMock(),
            experience_store=MagicMock(),
            active_learner=MagicMock(),
            meta_cognitive=MagicMock(),
            memory=MagicMock(),
            smart_orchestrator=MagicMock(),
            cascade=MagicMock(),
        )
        assert bridge.telemetry is not None
        assert bridge.experience is not None
        assert bridge.cascade is not None


# ---------------------------------------------------------------------------
# on_task_complete
# ---------------------------------------------------------------------------

class TestOnTaskComplete:

    def test_telemetry_logged_on_success(self):
        telemetry = MagicMock()
        bridge = AgentBridge(agent=_mock_agent(), telemetry=telemetry)
        bridge.on_task_complete("test task", _mock_result())
        telemetry.log.assert_called_once()

    def test_experience_stored_on_success(self):
        exp = MagicMock()
        exp.store.return_value = True
        bridge = AgentBridge(agent=_mock_agent(), experience_store=exp)
        bridge.on_task_complete("test task", _mock_result())
        exp.store.assert_called_once()

    def test_experience_low_confidence_on_many_iterations(self):
        exp = MagicMock()
        exp.store.return_value = True
        bridge = AgentBridge(agent=_mock_agent(), experience_store=exp)
        bridge.on_task_complete("hard task", _mock_result(iterations=30))
        call_args = exp.store.call_args
        # confidence is passed positionally or by keyword
        if call_args.kwargs.get("confidence") is not None:
            confidence = call_args.kwargs["confidence"]
        else:
            # positional: store(exp_id, query, answer, source_files, confidence)
            confidence = call_args[0][4] if len(call_args[0]) > 4 else 1.0
        assert confidence < 1.0

    def test_failure_still_stores_experience(self):
        exp = MagicMock()
        exp.store.return_value = True
        bridge = AgentBridge(agent=_mock_agent(), experience_store=exp)
        bridge.on_task_complete("failed task", _mock_result(success=False))
        exp.store.assert_called_once()

    def test_meta_cognitive_notified(self):
        meta = MagicMock()
        meta.report_outcome = MagicMock()
        bridge = AgentBridge(agent=_mock_agent(), meta_cognitive=meta)
        bridge.on_task_complete("task", _mock_result())
        meta.report_outcome.assert_called_once()

    def test_module_exception_does_not_propagate(self):
        telemetry = MagicMock()
        telemetry.log.side_effect = RuntimeError("DB locked")
        bridge = AgentBridge(agent=_mock_agent(), telemetry=telemetry)
        # Should not raise
        bridge.on_task_complete("task", _mock_result())

    def test_no_modules_no_error(self):
        bridge = AgentBridge(agent=_mock_agent())
        bridge.on_task_complete("task", _mock_result())


# ---------------------------------------------------------------------------
# suggest_next_study
# ---------------------------------------------------------------------------

class TestSuggestNextStudy:

    def test_returns_none_without_active_learner(self):
        bridge = AgentBridge(agent=_mock_agent())
        assert bridge.suggest_next_study() is None

    def test_returns_suggestion_with_active_learner(self):
        active = MagicMock()
        active.top_learning_opportunities.return_value = [
            {"query": "async Python", "learning_value": 0.9, "uncertainty": 0.7}
        ]
        bridge = AgentBridge(agent=_mock_agent(), active_learner=active)
        suggestion = bridge.suggest_next_study()
        assert suggestion is not None
        assert "async Python" in suggestion

    def test_skips_known_topics(self):
        active = MagicMock()
        active.top_learning_opportunities.return_value = [
            {"query": "known topic", "learning_value": 0.5, "uncertainty": 0.3}
        ]
        memory = MagicMock()
        memory.search.return_value = [{"score": 0.95}]
        bridge = AgentBridge(agent=_mock_agent(), active_learner=active, memory=memory)
        assert bridge.suggest_next_study() is None


# ---------------------------------------------------------------------------
# get_memory_stats / pipeline_health
# ---------------------------------------------------------------------------

class TestUtilities:

    def test_memory_stats_returns_empty_without_memory(self):
        bridge = AgentBridge(agent=_mock_agent())
        assert bridge.get_memory_stats() == {}

    def test_memory_stats_returns_data(self):
        memory = MagicMock()
        memory.stats.return_value = {"total": 100, "avg_score": 0.8}
        bridge = AgentBridge(agent=_mock_agent(), memory=memory)
        stats = bridge.get_memory_stats()
        assert stats["total"] == 100

    def test_pipeline_health_returns_empty_without_pipeline(self):
        bridge = AgentBridge(agent=_mock_agent())
        assert bridge.pipeline_health() == {}

    def test_check_regression_returns_none_without_learner(self):
        bridge = AgentBridge(agent=_mock_agent())
        assert bridge.check_regression() is None
