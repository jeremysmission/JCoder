"""
Tests for agent error recovery features:
  A. Exponential backoff on LLM errors
  B. Tool circuit breaker
  C. Partial result recovery on max_iterations
  D. Graceful degradation on unreachable LLM
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.core import (
    Agent,
    AgentResult,
    CircuitBreaker,
    CIRCUIT_BREAKER_THRESHOLD,
    LLM_BACKOFF_BASE_S,
    LLM_MAX_RETRIES,
)
from agent.llm_backend import ChatResponse, ToolCall
from agent.tools import ToolRegistry, ToolResult


# ---------------------------------------------------------------------------
# Shared factories
# ---------------------------------------------------------------------------

def _chat_response(
    content: str = "",
    tool_calls: Optional[List[ToolCall]] = None,
    input_tokens: int = 100,
    output_tokens: int = 50,
) -> ChatResponse:
    return ChatResponse(
        content=content,
        tool_calls=tool_calls or [],
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


def _tool_call(
    name: str,
    arguments: Optional[Dict[str, Any]] = None,
    tc_id: str = "",
) -> ToolCall:
    return ToolCall(
        id=tc_id or f"call_{name}",
        name=name,
        arguments=arguments or {},
    )


def _make_backend(responses) -> MagicMock:
    backend = MagicMock()
    backend.model = "test-model"
    backend.chat.side_effect = list(responses)
    return backend


def _make_tools(results: Optional[Dict[str, ToolResult]] = None) -> MagicMock:
    tools = MagicMock(spec=ToolRegistry)
    tools.schemas = []
    _results = results or {}

    def _execute(name: str, arguments: Dict[str, Any]) -> ToolResult:
        if name in _results:
            r = _results[name]
            if callable(r):
                return r()
            return r
        return ToolResult(
            success=True, output=f"executed {name}", elapsed_s=0.01,
        )

    tools.execute.side_effect = _execute
    return tools


def _noop_sleep(_seconds: float) -> None:
    """Instant sleep for tests."""
    pass


def _make_agent(backend, tools, **kwargs):
    kwargs.setdefault("sleep_fn", _noop_sleep)
    return Agent(backend=backend, tools=tools, **kwargs)


# ---------------------------------------------------------------------------
# A. Exponential backoff on LLM errors
# ---------------------------------------------------------------------------

class TestRetrySuccess:
    """LLM fails then recovers within retry budget."""

    def test_succeeds_after_transient_errors(self):
        """Fail twice, succeed on third attempt."""
        responses = [
            RuntimeError("timeout"),
            RuntimeError("timeout"),
            _chat_response(content="recovered answer"),
        ]
        backend = MagicMock()
        backend.model = "test"
        backend.chat.side_effect = responses
        tools = _make_tools()
        agent = _make_agent(backend, tools)

        result = agent.run("task")

        assert result.success is True
        assert result.summary == "recovered answer"
        assert backend.chat.call_count == 3

    def test_backoff_delays_called(self):
        """Verify sleep_fn is called with correct backoff delays."""
        delays = []

        def track_sleep(seconds: float) -> None:
            delays.append(seconds)

        responses = [
            RuntimeError("err"),
            RuntimeError("err"),
            _chat_response(content="ok"),
        ]
        backend = MagicMock()
        backend.model = "test"
        backend.chat.side_effect = responses
        tools = _make_tools()
        agent = _make_agent(backend, tools, sleep_fn=track_sleep)

        agent.run("task")

        assert delays == [
            LLM_BACKOFF_BASE_S * 1,   # 2^0 = 1s
            LLM_BACKOFF_BASE_S * 2,   # 2^1 = 2s
        ]


class TestRetryExhaustion:
    """LLM fails on every attempt -- all retries exhausted."""

    def test_returns_failure_after_all_retries(self):
        backend = MagicMock()
        backend.model = "test"
        backend.chat.side_effect = RuntimeError("dead")
        tools = _make_tools()
        agent = _make_agent(backend, tools)

        result = agent.run("task")

        assert result.success is False
        assert "unreachable" in result.summary.lower()
        # initial + LLM_MAX_RETRIES retries = 4 total calls
        assert backend.chat.call_count == LLM_MAX_RETRIES + 1

    def test_exhaustion_includes_iteration_info(self):
        backend = MagicMock()
        backend.model = "test"
        backend.chat.side_effect = ConnectionError("refused")
        tools = _make_tools()
        agent = _make_agent(backend, tools)

        result = agent.run("task")

        assert result.iterations == 1
        assert "iteration" in result.summary.lower()

    def test_failure_midway_through_iterations(self):
        """Succeed first iteration, fail on second with retries."""
        tc = _tool_call("read_file", {"path": "a.py"})
        r1 = _chat_response(tool_calls=[tc])
        # Second LLM call: all retries fail
        responses = [r1] + [RuntimeError("gone")] * (LLM_MAX_RETRIES + 1)
        backend = MagicMock()
        backend.model = "test"
        backend.chat.side_effect = responses
        tools = _make_tools()
        agent = _make_agent(backend, tools)

        result = agent.run("task")

        assert result.success is False
        assert result.iterations == 2
        assert len(result.steps) == 1  # one tool call completed


# ---------------------------------------------------------------------------
# B. Tool circuit breaker
# ---------------------------------------------------------------------------

class TestCircuitBreakerUnit:
    """Unit tests for CircuitBreaker class."""

    def test_not_disabled_initially(self):
        cb = CircuitBreaker()
        assert not cb.is_disabled("read_file")

    def test_disabled_after_threshold_failures(self):
        cb = CircuitBreaker(threshold=3)
        for _ in range(3):
            cb.record_failure("read_file")
        assert cb.is_disabled("read_file")

    def test_success_resets_counter(self):
        cb = CircuitBreaker(threshold=3)
        cb.record_failure("read_file")
        cb.record_failure("read_file")
        cb.record_success("read_file")
        cb.record_failure("read_file")
        assert not cb.is_disabled("read_file")

    def test_different_tools_independent(self):
        cb = CircuitBreaker(threshold=2)
        cb.record_failure("read_file")
        cb.record_failure("write_file")
        cb.record_failure("read_file")
        assert cb.is_disabled("read_file")
        assert not cb.is_disabled("write_file")

    def test_reset_clears_all(self):
        cb = CircuitBreaker(threshold=1)
        cb.record_failure("read_file")
        assert cb.is_disabled("read_file")
        cb.reset()
        assert not cb.is_disabled("read_file")

    def test_disabled_tools_property(self):
        cb = CircuitBreaker(threshold=1)
        cb.record_failure("a")
        cb.record_failure("b")
        assert cb.disabled_tools == {"a", "b"}


class TestCircuitBreakerIntegration:
    """Circuit breaker wired into agent loop."""

    def test_tool_disabled_after_consecutive_failures(self):
        fail_result = ToolResult(
            success=False, output="", error="crash", elapsed_s=0.01,
        )
        call_count = 0

        def make_fail():
            nonlocal call_count
            call_count += 1
            return fail_result

        tc = _tool_call("bad_tool", {})
        # 4 iterations of trying bad_tool, then final answer
        responses = [_chat_response(tool_calls=[tc]) for _ in range(4)]
        responses.append(_chat_response(content="gave up"))
        backend = _make_backend(responses)
        tools = _make_tools({"bad_tool": make_fail})
        agent = _make_agent(backend, tools, max_iterations=5)

        result = agent.run("task")

        assert result.success is True
        # First 3 calls execute, 4th is skipped by circuit breaker
        assert call_count == CIRCUIT_BREAKER_THRESHOLD
        disabled_steps = [
            s for s in result.steps
            if "circuit breaker" in s.tool_result.lower()
        ]
        assert len(disabled_steps) >= 1

    def test_disabled_tools_in_result(self):
        fail_result = ToolResult(
            success=False, output="", error="err", elapsed_s=0.01,
        )
        tc = _tool_call("flaky", {})
        responses = [_chat_response(tool_calls=[tc]) for _ in range(4)]
        backend = _make_backend(responses)
        tools = _make_tools({"flaky": fail_result})
        agent = _make_agent(backend, tools, max_iterations=4)

        result = agent.run("task")

        assert "flaky" in result.disabled_tools


# ---------------------------------------------------------------------------
# C. Partial result recovery on max_iterations
# ---------------------------------------------------------------------------

class TestPartialResults:
    """When iteration limit is hit, successful tool outputs are captured."""

    def test_partial_results_collected(self):
        tc = _tool_call("read_file", {"path": "a.py"})
        responses = [_chat_response(tool_calls=[tc]) for _ in range(3)]
        backend = _make_backend(responses)
        tools = _make_tools({
            "read_file": ToolResult(
                success=True, output="file content here",
                elapsed_s=0.01,
            ),
        })
        agent = _make_agent(backend, tools, max_iterations=3)

        result = agent.run("read all files")

        assert result.success is False
        assert result.timed_out is True
        assert len(result.partial_results) == 3
        assert all("read_file" in p for p in result.partial_results)
        assert "partial result" in result.summary.lower()

    def test_no_partial_results_when_all_fail(self):
        tc = _tool_call("bad", {})
        responses = [_chat_response(tool_calls=[tc]) for _ in range(2)]
        backend = _make_backend(responses)
        tools = _make_tools({
            "bad": ToolResult(
                success=False, output="", error="nope", elapsed_s=0.01,
            ),
        })
        agent = _make_agent(backend, tools, max_iterations=2)

        result = agent.run("task")

        assert result.partial_results == []

    def test_mixed_success_and_failure_partial(self):
        tc_ok = _tool_call("read_file", {"path": "a.py"}, tc_id="c1")
        tc_bad = _tool_call("write_file", {"path": "b.py"}, tc_id="c2")
        resp = _chat_response(tool_calls=[tc_ok, tc_bad])
        backend = _make_backend([resp])
        tools = _make_tools({
            "read_file": ToolResult(
                success=True, output="good data", elapsed_s=0.01,
            ),
            "write_file": ToolResult(
                success=False, output="", error="denied", elapsed_s=0.01,
            ),
        })
        agent = _make_agent(backend, tools, max_iterations=1)

        result = agent.run("task")

        assert len(result.partial_results) == 1
        assert "read_file" in result.partial_results[0]


# ---------------------------------------------------------------------------
# D. Graceful degradation on unreachable LLM
# ---------------------------------------------------------------------------

class TestGracefulDegradation:
    """Agent returns partial work when LLM is completely unreachable."""

    def test_immediate_unreachable(self):
        backend = MagicMock()
        backend.model = "test"
        backend.chat.side_effect = ConnectionError("refused")
        tools = _make_tools()
        agent = _make_agent(backend, tools)

        result = agent.run("task")

        assert result.success is False
        assert "unreachable" in result.summary.lower()
        assert result.partial_results == []

    def test_unreachable_after_some_work(self):
        """One successful iteration, then LLM goes down."""
        tc = _tool_call("read_file", {"path": "a.py"})
        good_resp = _chat_response(tool_calls=[tc])
        side_effects = [good_resp] + [
            OSError("network down")
        ] * (LLM_MAX_RETRIES + 1)
        backend = MagicMock()
        backend.model = "test"
        backend.chat.side_effect = side_effects
        tools = _make_tools({
            "read_file": ToolResult(
                success=True, output="important data", elapsed_s=0.01,
            ),
        })
        agent = _make_agent(backend, tools)

        result = agent.run("task")

        assert result.success is False
        assert len(result.partial_results) == 1
        assert "important data" in result.partial_results[0]

    def test_graceful_summary_with_partials(self):
        summary = Agent._graceful_degradation_summary(3, ["a", "b"])
        assert "unreachable" in summary.lower()
        assert "2 partial result" in summary

    def test_graceful_summary_without_partials(self):
        summary = Agent._graceful_degradation_summary(1, [])
        assert "no partial results" in summary.lower()
