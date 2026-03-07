"""
Tests for agent.bridge -- AgentBridge self-learning pipeline integration.
All self-learning modules are mocked; no live LLM or database needed.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from agent.bridge import AgentBridge, create_wired_agent, _summarize_args
from agent.core import AgentResult, AgentStep


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_result(
    success: bool = True,
    summary: str = "All done.",
    steps: Optional[List[AgentStep]] = None,
    iterations: int = 3,
    elapsed_s: float = 2.5,
    input_tokens: int = 500,
    output_tokens: int = 200,
) -> AgentResult:
    return AgentResult(
        success=success,
        summary=summary,
        steps=steps or [],
        total_input_tokens=input_tokens,
        total_output_tokens=output_tokens,
        total_elapsed_s=elapsed_s,
        iterations=iterations,
    )


def _make_steps() -> List[AgentStep]:
    return [
        AgentStep(iteration=1, tool_name="read_file",
                  tool_args={"path": "src/main.py"},
                  tool_result="def main():\n    pass\n",
                  tool_success=True, elapsed_s=0.1),
        AgentStep(iteration=2, tool_name="edit_file",
                  tool_args={"path": "src/main.py",
                             "old_text": "pass", "new_text": "print('hello')"},
                  tool_result="OK, 1 replacement made",
                  tool_success=True, elapsed_s=0.05),
        AgentStep(iteration=3, tool_name="run_command",
                  tool_args={"command": "python src/main.py"},
                  tool_result="hello",
                  tool_success=True, elapsed_s=0.3),
    ]


@pytest.fixture()
def mock_agent():
    return MagicMock()


# ---------------------------------------------------------------------------
# AgentBridge -- basic construction
# ---------------------------------------------------------------------------

class TestBridgeConstruction:

    def test_all_modules_none(self, mock_agent):
        """Bridge works with all modules absent (full graceful degradation)."""
        bridge = AgentBridge(agent=mock_agent)
        assert bridge.telemetry is None
        assert bridge.experience is None
        assert bridge.active is None
        assert bridge.meta is None
        assert bridge.memory is None

    def test_modules_tracked(self, mock_agent):
        tel = MagicMock()
        exp = MagicMock()
        bridge = AgentBridge(agent=mock_agent, telemetry=tel,
                             experience_store=exp)
        assert bridge.telemetry is tel
        assert bridge.experience is exp


# ---------------------------------------------------------------------------
# on_task_complete
# ---------------------------------------------------------------------------

class TestOnTaskComplete:

    def test_no_modules_no_crash(self, mock_agent):
        """Calling on_task_complete with no modules does nothing (no crash)."""
        bridge = AgentBridge(agent=mock_agent)
        result = _make_result()
        bridge.on_task_complete("Fix bug in parser", result)

    def test_telemetry_logged(self, mock_agent):
        """Telemetry log is called when telemetry + QueryEvent available."""
        tel = MagicMock()
        bridge = AgentBridge(agent=mock_agent, telemetry=tel)

        # Patch QueryEvent to exist
        with patch("agent.bridge.QueryEvent", MagicMock()):
            result = _make_result(steps=_make_steps())
            bridge.on_task_complete("Fix bug", result)

        tel.log.assert_called_once()

    def test_experience_stored_on_success(self, mock_agent):
        """Experience store receives data only on successful tasks."""
        exp = MagicMock()
        bridge = AgentBridge(agent=mock_agent, experience_store=exp)

        result_ok = _make_result(success=True)
        bridge.on_task_complete("Task A", result_ok)
        exp.store.assert_called_once()

    def test_experience_skipped_on_failure(self, mock_agent):
        """Failed tasks are NOT stored in experience replay."""
        exp = MagicMock()
        bridge = AgentBridge(agent=mock_agent, experience_store=exp)

        result_fail = _make_result(success=False)
        bridge.on_task_complete("Task B", result_fail)
        exp.store.assert_not_called()

    def test_meta_cognitive_outcome_reported(self, mock_agent):
        """Meta-cognitive controller receives outcome report."""
        meta = MagicMock()
        meta.report_outcome = MagicMock()
        bridge = AgentBridge(agent=mock_agent, meta_cognitive=meta)

        result = _make_result(success=True)
        bridge.on_task_complete("Refactor module", result)
        meta.report_outcome.assert_called_once()

    def test_memory_ingest_on_success(self, mock_agent):
        """Agent memory ingests successful task results."""
        mem = MagicMock()
        mem.ingest_task_result = MagicMock()
        bridge = AgentBridge(agent=mock_agent, memory=mem)

        result = _make_result(success=True)
        bridge.on_task_complete("Implement feature", result)
        mem.ingest_task_result.assert_called_once()

    def test_module_exception_isolated(self, mock_agent):
        """If one module throws, others still execute."""
        tel = MagicMock()
        tel.log.side_effect = RuntimeError("telemetry crash")
        exp = MagicMock()
        bridge = AgentBridge(agent=mock_agent, telemetry=tel,
                             experience_store=exp)

        with patch("agent.bridge.QueryEvent", MagicMock()):
            result = _make_result(success=True)
            bridge.on_task_complete("Task", result)

        # Telemetry crashed but experience should still be called
        exp.store.assert_called_once()


# ---------------------------------------------------------------------------
# suggest_next_study
# ---------------------------------------------------------------------------

class TestSuggestNextStudy:

    def test_no_active_learner(self, mock_agent):
        bridge = AgentBridge(agent=mock_agent)
        assert bridge.suggest_next_study() is None

    def test_returns_suggestion(self, mock_agent):
        active = MagicMock()
        active.top_learning_opportunities.return_value = [
            {"query": "How to implement B-trees",
             "learning_value": 0.8, "uncertainty": 0.7},
        ]
        bridge = AgentBridge(agent=mock_agent, active_learner=active)
        suggestion = bridge.suggest_next_study()
        assert suggestion is not None
        assert "B-trees" in suggestion

    def test_skips_known_topics(self, mock_agent):
        """Topics already well-covered in memory are skipped."""
        active = MagicMock()
        active.top_learning_opportunities.return_value = [
            {"query": "well-known topic", "learning_value": 0.3, "uncertainty": 0.2},
        ]
        mem = MagicMock()
        mem.search.return_value = [{"score": 0.95}]  # strong match
        bridge = AgentBridge(agent=mock_agent, active_learner=active, memory=mem)
        assert bridge.suggest_next_study() is None

    def test_empty_opportunities(self, mock_agent):
        active = MagicMock()
        active.top_learning_opportunities.return_value = []
        bridge = AgentBridge(agent=mock_agent, active_learner=active)
        assert bridge.suggest_next_study() is None


# ---------------------------------------------------------------------------
# select_strategy
# ---------------------------------------------------------------------------

class TestSelectStrategy:

    def test_no_meta_returns_defaults(self, mock_agent):
        bridge = AgentBridge(agent=mock_agent)
        s = bridge.select_strategy("Fix a bug")
        assert s["strategy"] == "standard"
        assert s["temperature"] == 0.1
        assert s["max_iterations"] == 50

    def test_meta_returns_strategy(self, mock_agent):
        meta = MagicMock()
        sig = MagicMock()
        sig.complexity = 0.3
        sig.query_type = "lookup"
        sig.has_code = True
        meta.select_strategy.return_value = ("fast_lookup", sig)

        with patch("agent.bridge.classify_query", lambda q: sig):
            bridge = AgentBridge(agent=mock_agent, meta_cognitive=meta)
            s = bridge.select_strategy("What is os.path.join?")

        assert s["strategy"] == "fast_lookup"
        assert s["max_iterations"] == 30  # lookup = reduced iterations
        assert s["query_type"] == "lookup"


# ---------------------------------------------------------------------------
# wrap_rag_callback
# ---------------------------------------------------------------------------

class TestWrapRagCallback:

    def test_passthrough(self, mock_agent):
        """Without telemetry, wrapped callback is a passthrough."""
        bridge = AgentBridge(agent=mock_agent)

        def base_cb(q: str) -> str:
            return f"Answer: {q}"

        wrapped = bridge.wrap_rag_callback(base_cb)
        assert wrapped("test query") == "Answer: test query"

    def test_telemetry_logged(self, mock_agent):
        """With telemetry, wrapped callback logs a QueryEvent."""
        tel = MagicMock()
        bridge = AgentBridge(agent=mock_agent, telemetry=tel)

        with patch("agent.bridge.QueryEvent", MagicMock()):
            wrapped = bridge.wrap_rag_callback(lambda q: "result")
            wrapped("search query")

        tel.log.assert_called_once()


# ---------------------------------------------------------------------------
# get_agent_trajectory
# ---------------------------------------------------------------------------

class TestGetAgentTrajectory:

    def test_trajectory_structure(self, mock_agent):
        bridge = AgentBridge(agent=mock_agent)
        result = _make_result(success=True, steps=_make_steps())
        traj = bridge.get_agent_trajectory(result)

        assert traj["outcome"] == "success"
        assert len(traj["steps"]) == 3
        assert traj["steps"][0]["action"].startswith("read_file(")
        assert traj["tokens"]["total"] == 700
        assert traj["iterations"] == 3

    def test_failed_trajectory(self, mock_agent):
        bridge = AgentBridge(agent=mock_agent)
        result = _make_result(success=False, summary="Out of tokens")
        traj = bridge.get_agent_trajectory(result)
        assert traj["outcome"] == "failure"


# ---------------------------------------------------------------------------
# _extract_source_files
# ---------------------------------------------------------------------------

class TestExtractSourceFiles:

    def test_extracts_file_paths(self, mock_agent):
        bridge = AgentBridge(agent=mock_agent)
        result = _make_result(steps=_make_steps())
        files = bridge._extract_source_files(result)
        assert "src/main.py" in files

    def test_deduplicates(self, mock_agent):
        """Same file touched by read_file and edit_file appears once."""
        bridge = AgentBridge(agent=mock_agent)
        result = _make_result(steps=_make_steps())
        files = bridge._extract_source_files(result)
        assert files.count("src/main.py") == 1

    def test_skips_failed_tools(self, mock_agent):
        bridge = AgentBridge(agent=mock_agent)
        steps = [
            AgentStep(iteration=1, tool_name="read_file",
                      tool_args={"path": "missing.py"},
                      tool_result="ERROR", tool_success=False, elapsed_s=0.1),
        ]
        result = _make_result(steps=steps)
        files = bridge._extract_source_files(result)
        assert files == []


# ---------------------------------------------------------------------------
# get_memory_stats
# ---------------------------------------------------------------------------

class TestMemoryStats:

    def test_no_memory(self, mock_agent):
        bridge = AgentBridge(agent=mock_agent)
        assert bridge.get_memory_stats() == {}

    def test_with_memory(self, mock_agent):
        mem = MagicMock()
        mem.stats.return_value = {"total_chunks": 42, "index_count": 3}
        bridge = AgentBridge(agent=mock_agent, memory=mem)
        stats = bridge.get_memory_stats()
        assert stats["total_chunks"] == 42


# ---------------------------------------------------------------------------
# _summarize_args utility
# ---------------------------------------------------------------------------

class TestSummarizeArgs:

    def test_short_args(self):
        result = _summarize_args({"path": "main.py"})
        assert result == "path=main.py"

    def test_long_value_truncated(self):
        result = _summarize_args({"content": "x" * 100})
        assert "..." in result
        assert len(result) <= 80

    def test_empty_args(self):
        assert _summarize_args({}) == ""


# ---------------------------------------------------------------------------
# create_wired_agent -- factory override semantics
# ---------------------------------------------------------------------------


class TestCreateWiredAgent:
    """create_wired_agent delegates to build_agent_from_config and
    honours caller-provided config and backend overrides."""

    def test_returns_agent_and_bridge(self):
        """Factory returns a (Agent, AgentBridge) tuple."""
        mock_agent = MagicMock(name="MockAgent", _system_prompt="sys")
        mock_tools = MagicMock(name="MockTools")
        mock_tools._rag_callback = None
        mock_stack = {
            "agent": mock_agent,
            "backend": MagicMock(),
            "tools": mock_tools,
            "memory": None,
            "session_store": None,
            "logger": None,
        }

        with patch("agent.config_loader.build_agent_from_config", return_value=mock_stack):
            with patch("agent.config_loader.load_agent_config", return_value=MagicMock()):
                agent, bridge = create_wired_agent()

        assert agent is mock_agent
        assert isinstance(bridge, AgentBridge)
        assert bridge.agent is mock_agent

    def test_caller_config_forwarded(self):
        """Pre-built config is passed through to build_agent_from_config."""
        from agent.config_loader import AgentConfig
        custom_cfg = AgentConfig(backend="ollama", model="phi4:14b",
                                 max_iterations=10)

        mock_agent = MagicMock(name="MockAgent", _system_prompt="sys")
        mock_tools = MagicMock(name="MockTools")
        mock_tools._rag_callback = None
        mock_stack = {
            "agent": mock_agent,
            "backend": MagicMock(),
            "tools": mock_tools,
            "memory": None,
            "session_store": None,
            "logger": None,
        }

        with patch("agent.config_loader.build_agent_from_config",
                    return_value=mock_stack) as mock_build:
            with patch("agent.config_loader.load_agent_config"):
                create_wired_agent(config=custom_cfg)

        mock_build.assert_called_once_with(custom_cfg)

    def test_caller_backend_overrides_config_backend(self):
        """When backend= is provided, the agent is rebuilt with that backend."""
        from agent.config_loader import AgentConfig
        custom_backend = MagicMock(name="CallerBackend")
        cfg = AgentConfig(max_iterations=25, max_tokens_budget=100_000)

        mock_tools = MagicMock(name="MockTools")
        mock_tools._rag_callback = None
        config_agent = MagicMock(name="ConfigAgent", _system_prompt="sys prompt")
        mock_stack = {
            "agent": config_agent,
            "backend": MagicMock(name="ConfigBackend"),
            "tools": mock_tools,
            "memory": None,
            "session_store": MagicMock(),
            "logger": MagicMock(),
        }

        rebuilt_agent = MagicMock(name="RebuiltAgent")

        with patch("agent.config_loader.build_agent_from_config", return_value=mock_stack):
            with patch("agent.config_loader.load_agent_config"):
                with patch("agent.core.Agent", return_value=rebuilt_agent) as mock_cls:
                    agent, bridge = create_wired_agent(
                        config=cfg, backend=custom_backend)

        # Agent was rebuilt with caller's backend
        assert agent is rebuilt_agent
        call_kwargs = mock_cls.call_args.kwargs
        assert call_kwargs["backend"] is custom_backend
        assert call_kwargs["max_iterations"] == 25
        assert call_kwargs["max_tokens_budget"] == 100_000

    def test_working_dir_applied_to_config(self):
        """working_dir parameter is set on the config before building."""
        from agent.config_loader import AgentConfig
        cfg = AgentConfig()

        mock_agent = MagicMock(_system_prompt="sys")
        mock_tools = MagicMock()
        mock_tools._rag_callback = None
        mock_stack = {
            "agent": mock_agent,
            "backend": MagicMock(),
            "tools": mock_tools,
            "memory": None,
            "session_store": None,
            "logger": None,
        }

        with patch("agent.config_loader.build_agent_from_config", return_value=mock_stack):
            with patch("agent.config_loader.load_agent_config"):
                create_wired_agent(config=cfg, working_dir="/my/project")

        assert cfg.working_dir == "/my/project"

    def test_no_config_loads_from_yaml(self):
        """When config=None, load_agent_config() is called."""
        loaded_cfg = MagicMock(name="LoadedConfig")
        loaded_cfg.working_dir = "."
        loaded_cfg.max_iterations = 50
        loaded_cfg.max_tokens_budget = 500_000

        mock_agent = MagicMock(_system_prompt="sys")
        mock_tools = MagicMock()
        mock_tools._rag_callback = None
        mock_stack = {
            "agent": mock_agent,
            "backend": MagicMock(),
            "tools": mock_tools,
            "memory": None,
            "session_store": None,
            "logger": None,
        }

        with patch("agent.config_loader.build_agent_from_config",
                    return_value=mock_stack) as mock_build:
            with patch("agent.config_loader.load_agent_config",
                       return_value=loaded_cfg):
                create_wired_agent()

        mock_build.assert_called_once_with(loaded_cfg)
