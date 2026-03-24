"""
Agent Core Loop
----------------
Plan/execute/observe loop that drives JCoder autonomously.

The agent receives a task, then iterates:
  1. Send conversation history + tool schemas to the LLM
  2. If the LLM requests tool calls, execute them and feed results back
  3. If the LLM returns a final text answer (no tool calls), return it
  4. If any tool returns TASK_COMPLETE, return with success
  5. Stop if iteration or token budget is exhausted

Error recovery:
  - Exponential backoff on LLM errors (3 retries: 1s, 2s, 4s)
  - Tool circuit breaker (3 consecutive fails disables tool for session)
  - Partial result recovery on max_iterations
  - Graceful degradation on unreachable LLM

Uses OpenAI message format throughout. The AnthropicBackend converts
internally, so this module never needs to care which backend is active.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Set

from agent.llm_backend import ChatResponse, LLMBackend, ToolCall
from agent.logger import AgentLogger
from agent.session import SessionStore
from agent.tools import ToolRegistry, ToolResult

from agent.core_recovery import (
    CIRCUIT_BREAKER_THRESHOLD,
    LLM_BACKOFF_BASE_S,
    LLM_MAX_RETRIES,
    AgentResult,
    AgentStep,
    call_llm_with_retry,
    collect_partial_results,
    graceful_degradation_summary,
    restore_token_totals,
    save_session,
)
from agent.prompts import AGENT_SYSTEM_PROMPT as DEFAULT_SYSTEM_PROMPT

log = logging.getLogger(__name__)

_TASK_COMPLETE_PREFIX = "TASK_COMPLETE:"


# ---------------------------------------------------------------------------
# Circuit breaker
# ---------------------------------------------------------------------------

class CircuitBreaker:
    """Disable tools after consecutive failures."""

    def __init__(self, threshold: int = CIRCUIT_BREAKER_THRESHOLD):
        self._threshold = threshold
        self._consecutive_failures: Dict[str, int] = {}
        self._disabled: Set[str] = set()

    def record_success(self, tool_name: str) -> None:
        self._consecutive_failures[tool_name] = 0

    def record_failure(self, tool_name: str) -> None:
        count = self._consecutive_failures.get(tool_name, 0) + 1
        self._consecutive_failures[tool_name] = count
        if count >= self._threshold:
            self._disabled.add(tool_name)
            log.warning("Circuit breaker tripped for tool: %s", tool_name)

    def is_disabled(self, tool_name: str) -> bool:
        return tool_name in self._disabled

    @property
    def disabled_tools(self) -> Set[str]:
        return set(self._disabled)

    def reset(self) -> None:
        self._consecutive_failures.clear()
        self._disabled.clear()


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class Agent:
    """
    Autonomous coding agent that loops: plan -> act -> observe.

    Parameters
    ----------
    backend : LLMBackend
        The LLM to use for planning and reasoning.
    tools : ToolRegistry
        Available tools the agent can invoke.
    system_prompt : str
        System-level instructions for the LLM.
    max_iterations : int
        Hard cap on loop iterations (safety net).
    max_tokens_budget : int
        Combined input+output token budget. Stops when exceeded.
    """

    def __init__(
        self,
        backend: LLMBackend,
        tools: ToolRegistry,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_iterations: int = 50,
        max_tokens_budget: int = 500_000,
        session_store: Optional[SessionStore] = None,
        logger: Optional[AgentLogger] = None,
        sleep_fn: Optional[Callable[[float], None]] = None,
    ):
        self._backend = backend
        self._tools = tools
        self._system_prompt = system_prompt
        self._max_iterations = max_iterations
        self._max_tokens_budget = max_tokens_budget
        self._session_store = session_store
        self._logger = logger
        self._session_id: str = str(uuid.uuid4())
        self._sleep_fn = sleep_fn or time.sleep

        self._history: List[Dict[str, Any]] = []
        self._steps: List[AgentStep] = []
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._run_lock = threading.Lock()
        self._running = False
        self._circuit_breaker = CircuitBreaker()

    # -- Public API --------------------------------------------------------

    def run(self, task: str) -> AgentResult:
        """
        Execute a task autonomously. Returns when the agent finishes,
        signals task_complete, or exhausts its budget.
        """
        self._claim_run_slot()
        try:
            # Initialise conversation
            self._history = [
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": task},
            ]
            self._steps = []
            self._total_input_tokens = 0
            self._total_output_tokens = 0
            self._circuit_breaker.reset()

            if self._logger:
                self._logger.log_task_start(self._session_id, task)

            return self._iterate(task, start_iteration=1)
        finally:
            self._release_run_slot()

    def resume(self, session_id: str) -> AgentResult:
        """Resume a previously saved session."""
        if not self._session_store:
            raise RuntimeError("Cannot resume: no session_store configured")

        self._claim_run_slot()
        try:
            data = self._session_store.load(session_id)
            self._session_id = session_id
            try:
                self._history = data["history"]
                task = data["task"]
            except KeyError as e:
                raise RuntimeError(
                    f"Corrupt session {session_id}: missing key {e}"
                ) from e
            self._steps = []
            inp, outp = restore_token_totals(data)
            self._total_input_tokens = inp
            self._total_output_tokens = outp

            prior_iterations = data.get("iterations", 0)
            log.info(
                "Resuming session %s (%d prior iterations, %d messages)",
                session_id, prior_iterations, len(self._history),
            )
            return self._iterate(task, start_iteration=prior_iterations + 1)
        finally:
            self._release_run_slot()

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def step_log(self) -> List[AgentStep]:
        return list(self._steps)

    @property
    def history(self) -> List[Dict[str, Any]]:
        return list(self._history)

    # -- Core iteration loop -----------------------------------------------

    def _iterate(self, task: str, start_iteration: int = 1) -> AgentResult:
        """Shared iteration loop used by both run() and resume()."""
        t_start = time.monotonic()
        end_iteration = start_iteration + (self._max_iterations - (start_iteration - 1))

        for iteration in range(start_iteration, end_iteration):
            log.info("Iteration %d / %d", iteration, self._max_iterations)

            # --- Call LLM with retry ---
            response = call_llm_with_retry(
                self._backend, self._history, self._tools.schemas,
                iteration, self._logger, self._session_id, self._sleep_fn,
            )
            if response is None:
                partial = collect_partial_results(self._steps)
                self._save(task, "failed", iteration)
                return self._result(
                    False,
                    graceful_degradation_summary(iteration, partial),
                    t_start, iteration,
                    partial_results=partial,
                )

            self._total_input_tokens += response.input_tokens
            self._total_output_tokens += response.output_tokens

            if self._logger:
                self._logger.log_llm_call(
                    self._session_id,
                    model=getattr(self._backend, "model", "unknown"),
                    input_tokens=response.input_tokens,
                    output_tokens=response.output_tokens,
                    elapsed_s=response.elapsed_s,
                )

            # --- No tool calls: final answer ---
            if not response.has_tool_calls:
                log.info("Agent returned final answer (no tool calls)")
                self._log_complete(True, response.content, iteration)
                self._save(task, "completed", iteration)
                return self._result(True, response.content, t_start, iteration)

            # --- Execute tool calls ---
            assistant_msg = _build_assistant_message(
                response.content, response.tool_calls,
            )
            self._history.append(assistant_msg)

            completion = self._execute_tool_calls(
                response.tool_calls, task, iteration, t_start,
            )
            if completion is not None:
                return completion

            # --- Checkpoint ---
            self._save(task, "active", iteration)

            # --- Budget check ---
            total_tokens = self._total_input_tokens + self._total_output_tokens
            if total_tokens >= self._max_tokens_budget:
                log.warning(
                    "Token budget exhausted: %d >= %d",
                    total_tokens, self._max_tokens_budget,
                )
                msg = (
                    f"Token budget exhausted after {iteration} iterations "
                    f"({total_tokens:,} tokens used)"
                )
                self._log_complete(False, msg, iteration)
                self._save(task, "failed", iteration)
                return self._result(False, msg, t_start, iteration, timed_out=True)

        # --- Loop exhausted ---
        final_iter = end_iteration - 1
        log.warning("Iteration limit reached: %d", final_iter)
        partial = collect_partial_results(self._steps)
        limit_msg = (
            f"Iteration limit ({self._max_iterations}) reached. "
            f"Last response: "
            f"{_truncate(self._last_assistant_content(), 500)}"
        )
        if partial:
            limit_msg += f" | {len(partial)} partial result(s) recovered"
        self._log_complete(False, limit_msg, final_iter)
        self._save(task, "failed", final_iter)
        return self._result(
            False, limit_msg, t_start, final_iter,
            timed_out=True, partial_results=partial,
        )

    # -- Tool execution ----------------------------------------------------

    def _execute_tool_calls(
        self,
        tool_calls: List[ToolCall],
        task: str,
        iteration: int,
        t_start: float,
    ) -> Optional[AgentResult]:
        """Execute tool calls, returning AgentResult if task completes."""
        for tc in tool_calls:
            log.info(
                "  Tool: %s(%s)", tc.name,
                _truncate(json.dumps(tc.arguments), 120),
            )

            if self._logger:
                self._logger.log_tool_call(
                    self._session_id, tc.name, tc.arguments, iteration,
                )

            # Circuit breaker: skip disabled tools
            if self._circuit_breaker.is_disabled(tc.name):
                result_text = (
                    f"ERROR: Tool '{tc.name}' disabled by circuit "
                    f"breaker ({CIRCUIT_BREAKER_THRESHOLD} consecutive "
                    f"failures)"
                )
                log.warning("  Skipped disabled tool: %s", tc.name)
                self._record_tool_result(tc, result_text, False, 0.0, iteration)
                continue

            try:
                result = self._tools.execute(tc.name, tc.arguments)
            except Exception as exc:
                log.error(
                    "Tool execution crashed for %s: %s",
                    tc.name, exc, exc_info=True,
                )
                result = ToolResult(
                    success=False, output="",
                    error=f"Tool crashed: {exc}",
                )

            tool_output = result.output or ""

            # Update circuit breaker
            if result.success:
                self._circuit_breaker.record_success(tc.name)
            else:
                self._circuit_breaker.record_failure(tc.name)

            # Compose result text
            if result.success:
                result_text = tool_output
            else:
                result_text = f"ERROR: {result.error}"
                if tool_output:
                    result_text = f"{tool_output}\nERROR: {result.error}"

            log.info(
                "  Result: success=%s  (%.1fs)  %s",
                result.success, result.elapsed_s,
                _truncate(result_text, 200),
            )

            if self._logger:
                self._logger.log_tool_result(
                    self._session_id, tc.name,
                    result.success, result_text, result.elapsed_s,
                )

            self._record_tool_result(
                tc, result_text, result.success, result.elapsed_s, iteration,
            )

            # Check for task completion signal
            if result.success and tool_output.startswith(_TASK_COMPLETE_PREFIX):
                summary = tool_output[len(_TASK_COMPLETE_PREFIX):].strip()
                log.info("Task complete: %s", summary)
                self._log_complete(True, summary, iteration)
                self._save(task, "completed", iteration)
                return self._result(True, summary, t_start, iteration)

        return None

    # -- Helpers -----------------------------------------------------------

    def _record_tool_result(
        self, tc: ToolCall, result_text: str,
        success: bool, elapsed_s: float, iteration: int,
    ) -> None:
        """Append tool result to history and step log."""
        self._history.append({
            "role": "tool",
            "tool_call_id": tc.id,
            "content": result_text,
        })
        self._steps.append(AgentStep(
            iteration=iteration,
            tool_name=tc.name,
            tool_args=tc.arguments,
            tool_result=result_text,
            tool_success=success,
            elapsed_s=elapsed_s,
        ))

    def _result(
        self, success: bool, summary: str, t_start: float,
        iterations: int, timed_out: bool = False,
        partial_results: Optional[List[str]] = None,
    ) -> AgentResult:
        """Build an AgentResult with current state."""
        return AgentResult(
            success=success,
            summary=summary,
            steps=list(self._steps),
            total_input_tokens=self._total_input_tokens,
            total_output_tokens=self._total_output_tokens,
            total_elapsed_s=time.monotonic() - t_start,
            iterations=iterations,
            timed_out=timed_out,
            partial_results=partial_results or [],
            disabled_tools=sorted(self._circuit_breaker.disabled_tools),
        )

    def _save(self, task: str, status: str, iteration: int) -> None:
        if status in ("completed", "failed"):
            self._release_run_slot()
        save_session(
            self._session_store, self._session_id, task,
            self._history, status, iteration,
            self._total_input_tokens, self._total_output_tokens,
        )

    def _log_complete(
        self, success: bool, summary: str, iteration: int,
    ) -> None:
        if self._logger:
            self._logger.log_task_complete(
                self._session_id, success=success,
                summary=summary,
                total_tokens=(
                    self._total_input_tokens + self._total_output_tokens
                ),
                iterations=iteration,
            )

    def _last_assistant_content(self) -> str:
        for msg in reversed(self._history):
            if msg.get("role") == "assistant" and msg.get("content"):
                return msg["content"]
        return "(no assistant response recorded)"

    @staticmethod
    def _build_assistant_message(
        content: str, tool_calls: List[ToolCall],
    ) -> Dict[str, Any]:
        """Build an OpenAI-format assistant message (delegates to module fn)."""
        return _build_assistant_message(content, tool_calls)

    @staticmethod
    def _graceful_degradation_summary(
        iteration: int, partial_results: List[str],
    ) -> str:
        """Backwards-compat delegate to core_recovery."""
        return graceful_degradation_summary(iteration, partial_results)

    def _claim_run_slot(self) -> None:
        with self._run_lock:
            if self._running:
                raise RuntimeError("Agent is already running a task")
            self._running = True

    def _release_run_slot(self) -> None:
        with self._run_lock:
            self._running = False


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _build_assistant_message(
    content: str, tool_calls: List[ToolCall],
) -> Dict[str, Any]:
    """Build an OpenAI-format assistant message with tool_calls."""
    return {
        "role": "assistant",
        "content": content or "",
        "tool_calls": [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.name,
                    "arguments": json.dumps(tc.arguments),
                },
            }
            for tc in tool_calls
        ],
    }


def _truncate(text: str, max_len: int) -> str:
    """Truncate text for logging, preserving the start."""
    if len(text) <= max_len:
        return text
    return text[:max_len - 3] + "..."
