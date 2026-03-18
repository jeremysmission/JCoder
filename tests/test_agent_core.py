"""
Unit tests for agent.core.Agent plan/execute/observe loop.

All LLM calls and tool executions are mocked -- no Ollama, no network,
no file system side effects.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.core import Agent, AgentResult, AgentStep, _truncate
from agent.llm_backend import ChatResponse, ToolCall
from agent.tools import ToolRegistry, ToolResult


# ---------------------------------------------------------------------------
# Mock factories
# ---------------------------------------------------------------------------

def _chat_response(
    content: str = "",
    tool_calls: Optional[List[ToolCall]] = None,
    input_tokens: int = 100,
    output_tokens: int = 50,
) -> ChatResponse:
    """Build a ChatResponse for testing."""
    return ChatResponse(
        content=content,
        tool_calls=tool_calls or [],
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


def _tool_call(name: str, arguments: Optional[Dict[str, Any]] = None, tc_id: str = "") -> ToolCall:
    """Build a ToolCall for testing."""
    return ToolCall(
        id=tc_id or f"call_{name}",
        name=name,
        arguments=arguments or {},
    )


def _make_backend(responses: List[ChatResponse]) -> MagicMock:
    """Mock LLMBackend that returns responses in sequence."""
    backend = MagicMock()
    backend.model = "test-model"
    backend.chat.side_effect = list(responses)
    return backend


def _make_tools(results: Optional[Dict[str, ToolResult]] = None) -> MagicMock:
    """Mock ToolRegistry with pre-configured execute results."""
    tools = MagicMock(spec=ToolRegistry)
    tools.schemas = []
    _results = results or {}

    def _execute(name: str, arguments: Dict[str, Any]) -> ToolResult:
        if name in _results:
            return _results[name]
        return ToolResult(success=True, output=f"executed {name}", elapsed_s=0.01)

    tools.execute.side_effect = _execute
    return tools


# ---------------------------------------------------------------------------
# Agent construction
# ---------------------------------------------------------------------------

class TestAgentConstruction:
    def test_defaults(self):
        backend = _make_backend([])
        tools = _make_tools()
        agent = Agent(backend=backend, tools=tools)
        assert agent._max_iterations == 50
        assert agent._max_tokens_budget == 500_000
        assert agent.session_id  # UUID string

    def test_custom_params(self):
        backend = _make_backend([])
        tools = _make_tools()
        agent = Agent(
            backend=backend, tools=tools,
            max_iterations=5, max_tokens_budget=1000,
        )
        assert agent._max_iterations == 5
        assert agent._max_tokens_budget == 1000


# ---------------------------------------------------------------------------
# Final-answer path (LLM returns text, no tool calls)
# ---------------------------------------------------------------------------

class TestFinalAnswer:
    def test_immediate_answer(self):
        resp = _chat_response(content="The answer is 42.")
        backend = _make_backend([resp])
        tools = _make_tools()
        agent = Agent(backend=backend, tools=tools)

        result = agent.run("What is the answer?")

        assert result.success is True
        assert result.summary == "The answer is 42."
        assert result.iterations == 1
        assert result.steps == []
        assert result.total_input_tokens == 100
        assert result.total_output_tokens == 50

    def test_history_initialized_correctly(self):
        resp = _chat_response(content="done")
        backend = _make_backend([resp])
        tools = _make_tools()
        agent = Agent(backend=backend, tools=tools, system_prompt="Be helpful.")

        agent.run("do something")

        call_args = backend.chat.call_args
        messages = call_args[0][0] if call_args[0] else call_args[1]["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "Be helpful."
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "do something"


# ---------------------------------------------------------------------------
# Tool dispatch loop
# ---------------------------------------------------------------------------

class TestToolDispatch:
    def test_single_tool_call_then_answer(self):
        tc = _tool_call("read_file", {"path": "main.py"})
        resp1 = _chat_response(content="Let me read it.", tool_calls=[tc])
        resp2 = _chat_response(content="File contains X.")
        backend = _make_backend([resp1, resp2])
        tools = _make_tools({
            "read_file": ToolResult(success=True, output="print('hello')", elapsed_s=0.01),
        })
        agent = Agent(backend=backend, tools=tools)

        result = agent.run("Read main.py")

        assert result.success is True
        assert result.iterations == 2
        assert len(result.steps) == 1
        assert result.steps[0].tool_name == "read_file"
        assert result.steps[0].tool_success is True
        tools.execute.assert_called_once_with("read_file", {"path": "main.py"})

    def test_multiple_tool_calls_in_one_response(self):
        tc1 = _tool_call("read_file", {"path": "a.py"}, tc_id="c1")
        tc2 = _tool_call("search_content", {"pattern": "TODO"}, tc_id="c2")
        resp1 = _chat_response(tool_calls=[tc1, tc2])
        resp2 = _chat_response(content="All done.")
        backend = _make_backend([resp1, resp2])
        tools = _make_tools()
        agent = Agent(backend=backend, tools=tools)

        result = agent.run("Find TODOs in a.py")

        assert result.success is True
        assert len(result.steps) == 2
        assert result.steps[0].tool_name == "read_file"
        assert result.steps[1].tool_name == "search_content"

    def test_tool_error_recorded(self):
        tc = _tool_call("read_file", {"path": "missing.py"})
        resp1 = _chat_response(tool_calls=[tc])
        resp2 = _chat_response(content="File not found, sorry.")
        backend = _make_backend([resp1, resp2])
        tools = _make_tools({
            "read_file": ToolResult(
                success=False, output="", error="File not found: missing.py",
                elapsed_s=0.01,
            ),
        })
        agent = Agent(backend=backend, tools=tools)

        result = agent.run("Read missing.py")

        assert result.success is True  # agent recovered and gave answer
        assert len(result.steps) == 1
        assert result.steps[0].tool_success is False
        assert "ERROR" in result.steps[0].tool_result


# ---------------------------------------------------------------------------
# Task completion signal
# ---------------------------------------------------------------------------

class TestTaskComplete:
    def test_task_complete_tool_ends_loop(self):
        tc = _tool_call("task_complete", {"summary": "All tests pass"})
        resp = _chat_response(tool_calls=[tc])
        backend = _make_backend([resp])
        tools = _make_tools({
            "task_complete": ToolResult(
                success=True, output="TASK_COMPLETE: All tests pass",
                elapsed_s=0.01,
            ),
        })
        agent = Agent(backend=backend, tools=tools)

        result = agent.run("Run the tests")

        assert result.success is True
        assert result.summary == "All tests pass"
        assert result.iterations == 1
        # LLM called only once -- loop exited on TASK_COMPLETE
        assert backend.chat.call_count == 1


# ---------------------------------------------------------------------------
# Iteration limit
# ---------------------------------------------------------------------------

class TestIterationLimit:
    def test_stops_at_max_iterations(self):
        # Every response triggers a tool call -- loop never ends naturally
        def _make_tool_response():
            tc = _tool_call("search_content", {"pattern": "x"})
            return _chat_response(tool_calls=[tc], input_tokens=10, output_tokens=5)

        responses = [_make_tool_response() for _ in range(5)]
        backend = _make_backend(responses)
        tools = _make_tools()
        agent = Agent(backend=backend, tools=tools, max_iterations=3)

        result = agent.run("infinite loop task")

        assert result.success is False
        assert result.timed_out is True
        assert result.iterations == 3
        assert "Iteration limit" in result.summary

    def test_max_iterations_one(self):
        tc = _tool_call("read_file", {"path": "x.py"})
        resp = _chat_response(tool_calls=[tc], input_tokens=10, output_tokens=5)
        backend = _make_backend([resp])
        tools = _make_tools()
        agent = Agent(backend=backend, tools=tools, max_iterations=1)

        result = agent.run("task")

        assert result.success is False
        assert result.timed_out is True
        assert result.iterations == 1


# ---------------------------------------------------------------------------
# Token budget
# ---------------------------------------------------------------------------

class TestTokenBudget:
    def test_stops_when_budget_exhausted(self):
        tc = _tool_call("read_file", {"path": "x.py"})
        # Each response uses 600 tokens total -> budget of 1000 exceeded after 2
        resp1 = _chat_response(tool_calls=[tc], input_tokens=400, output_tokens=200)
        resp2 = _chat_response(tool_calls=[tc], input_tokens=400, output_tokens=200)
        backend = _make_backend([resp1, resp2])
        tools = _make_tools()
        agent = Agent(
            backend=backend, tools=tools,
            max_iterations=10, max_tokens_budget=1000,
        )

        result = agent.run("big task")

        assert result.success is False
        assert result.timed_out is True
        assert "Token budget" in result.summary
        assert result.total_input_tokens == 800
        assert result.total_output_tokens == 400

    def test_under_budget_completes(self):
        resp = _chat_response(content="done", input_tokens=50, output_tokens=50)
        backend = _make_backend([resp])
        tools = _make_tools()
        agent = Agent(
            backend=backend, tools=tools, max_tokens_budget=1000,
        )

        result = agent.run("small task")
        assert result.success is True
        assert result.tokens == 100


# ---------------------------------------------------------------------------
# LLM error handling
# ---------------------------------------------------------------------------

class TestLLMErrors:
    def test_llm_exception_returns_failure(self):
        backend = MagicMock()
        backend.model = "test"
        backend.chat.side_effect = RuntimeError("Connection refused")
        tools = _make_tools()
        agent = Agent(backend=backend, tools=tools)

        result = agent.run("task")

        assert result.success is False
        assert "LLM error" in result.summary
        assert "Connection refused" in result.summary
        assert result.iterations == 1


# ---------------------------------------------------------------------------
# Concurrency guard
# ---------------------------------------------------------------------------

class TestConcurrencyGuard:
    def test_cannot_run_twice_concurrently(self):
        import threading

        barrier = threading.Barrier(2, timeout=5)

        def blocking_chat(*args, **kwargs):
            barrier.wait()
            return _chat_response(content="done")

        backend = MagicMock()
        backend.model = "test"
        backend.chat.side_effect = blocking_chat
        tools = _make_tools()
        agent = Agent(backend=backend, tools=tools)

        errors = []

        def run_agent():
            try:
                agent.run("task")
            except RuntimeError as e:
                errors.append(str(e))

        t1 = threading.Thread(target=run_agent)
        t2 = threading.Thread(target=run_agent)
        t1.start()
        t2.start()
        t1.join(timeout=10)
        t2.join(timeout=10)

        assert any("already running" in e for e in errors)


# ---------------------------------------------------------------------------
# Helper: _build_assistant_message
# ---------------------------------------------------------------------------

class TestBuildAssistantMessage:
    def test_structure(self):
        tc = ToolCall(id="tc_1", name="read_file", arguments={"path": "x.py"})
        msg = Agent._build_assistant_message("thinking...", [tc])

        assert msg["role"] == "assistant"
        assert msg["content"] == "thinking..."
        assert len(msg["tool_calls"]) == 1
        assert msg["tool_calls"][0]["id"] == "tc_1"
        assert msg["tool_calls"][0]["type"] == "function"
        assert msg["tool_calls"][0]["function"]["name"] == "read_file"

    def test_empty_content(self):
        msg = Agent._build_assistant_message("", [])
        assert msg["content"] == ""
        assert msg["tool_calls"] == []


# ---------------------------------------------------------------------------
# Helper: _truncate
# ---------------------------------------------------------------------------

class TestTruncate:
    def test_short_string_unchanged(self):
        assert _truncate("hello", 10) == "hello"

    def test_long_string_truncated(self):
        result = _truncate("a" * 100, 20)
        assert len(result) == 20
        assert result.endswith("...")

    def test_exact_length(self):
        assert _truncate("12345", 5) == "12345"


# ---------------------------------------------------------------------------
# AgentResult / AgentStep data classes
# ---------------------------------------------------------------------------

class TestDataClasses:
    def test_agent_result_tokens_property(self):
        r = AgentResult(
            success=True, summary="ok",
            total_input_tokens=200, total_output_tokens=100,
        )
        assert r.tokens == 300

    def test_agent_step_fields(self):
        s = AgentStep(
            iteration=1, tool_name="read_file",
            tool_args={"path": "x.py"}, tool_result="content",
            tool_success=True, elapsed_s=0.5,
        )
        assert s.iteration == 1
        assert s.tool_name == "read_file"
        assert s.elapsed_s == 0.5

    def test_step_log_returns_copy(self):
        backend = _make_backend([_chat_response(content="done")])
        tools = _make_tools()
        agent = Agent(backend=backend, tools=tools)
        agent.run("task")
        log1 = agent.step_log
        log2 = agent.step_log
        assert log1 is not log2

    def test_history_returns_copy(self):
        backend = _make_backend([_chat_response(content="done")])
        tools = _make_tools()
        agent = Agent(backend=backend, tools=tools)
        agent.run("task")
        h1 = agent.history
        h2 = agent.history
        assert h1 is not h2


# ---------------------------------------------------------------------------
# Session persistence
# ---------------------------------------------------------------------------

class TestSessionPersistence:
    def test_session_saved_on_completion(self):
        store = MagicMock()
        resp = _chat_response(content="done")
        backend = _make_backend([resp])
        tools = _make_tools()
        agent = Agent(
            backend=backend, tools=tools,
            session_store=store,
        )

        agent.run("task")

        store.save.assert_called()
        call_kwargs = store.save.call_args
        # Should have been saved with status "completed"
        assert any(
            kw.get("status") == "completed"
            for _, kw in [call_kwargs]
        )

    def test_no_session_store_is_fine(self):
        resp = _chat_response(content="done")
        backend = _make_backend([resp])
        tools = _make_tools()
        agent = Agent(backend=backend, tools=tools, session_store=None)

        result = agent.run("task")
        assert result.success is True

    def test_resume_without_store_raises(self):
        backend = _make_backend([])
        tools = _make_tools()
        agent = Agent(backend=backend, tools=tools, session_store=None)

        with pytest.raises(RuntimeError, match="no session_store"):
            agent.resume("some-session-id")
