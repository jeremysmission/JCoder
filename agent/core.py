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
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

from agent.llm_backend import ChatResponse, LLMBackend, ToolCall
from agent.logger import AgentLogger
from agent.session import SessionStore
from agent.tools import ToolRegistry, ToolResult

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default system prompt
# ---------------------------------------------------------------------------

DEFAULT_SYSTEM_PROMPT = """\
You are JCoder, an expert autonomous coding agent. You solve programming \
tasks by reading, writing, and executing code.

Follow these rules strictly:

1. PLAN FIRST. Before writing any code, outline the steps you will take. \
If the task is complex, break it into small, verifiable sub-tasks.

2. READ BEFORE WRITING. Always read the relevant files before modifying \
them. Never assume file contents -- verify first.

3. MAKE MINIMAL CHANGES. Change only what is necessary. Do not refactor \
unrelated code or add features that were not requested.

4. TEST AFTER CHANGES. After modifying code, run the relevant tests or \
a quick verification command. If tests fail, diagnose and fix before \
moving on.

5. USE TOOLS. You have file I/O, shell execution, search, and a code \
knowledge base. Use them -- do not try to solve everything from memory.

6. HANDLE ERRORS. If a tool call fails, read the error, adjust, and \
retry. Do not repeat the exact same failing call.

7. STAY ON TASK. Do not perform actions unrelated to the current task. \
Do not create unnecessary files or install unnecessary packages.

8. SIGNAL COMPLETION. When the task is fully done and verified, call \
the task_complete tool with a brief summary of what you did.

9. BE CONCISE. Keep your reasoning brief. Focus on actions and results, \
not lengthy explanations.
"""

_TASK_COMPLETE_PREFIX = "TASK_COMPLETE:"

# ---------------------------------------------------------------------------
# Error recovery constants
# ---------------------------------------------------------------------------

LLM_MAX_RETRIES = 3
LLM_BACKOFF_BASE_S = 1.0  # 1s, 2s, 4s
CIRCUIT_BREAKER_THRESHOLD = 3  # consecutive fails to trip


class CircuitBreaker:
    """Track consecutive tool failures; disable after threshold."""

    def __init__(self, threshold: int = CIRCUIT_BREAKER_THRESHOLD):
        self._threshold = threshold
        self._consecutive_fails: Dict[str, int] = {}
        self._disabled: Set[str] = set()

    def record_success(self, tool_name: str) -> None:
        self._consecutive_fails[tool_name] = 0

    def record_failure(self, tool_name: str) -> None:
        count = self._consecutive_fails.get(tool_name, 0) + 1
        self._consecutive_fails[tool_name] = count
        if count >= self._threshold:
            self._disabled.add(tool_name)

    def is_disabled(self, tool_name: str) -> bool:
        return tool_name in self._disabled

    @property
    def disabled_tools(self) -> Set[str]:
        return set(self._disabled)

    def reset(self) -> None:
        self._consecutive_fails.clear()
        self._disabled.clear()


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class AgentStep:
    """Record of a single tool invocation."""
    iteration: int
    tool_name: str
    tool_args: Dict[str, Any]
    tool_result: str
    tool_success: bool
    elapsed_s: float


@dataclass
class AgentResult:
    """Final outcome of an agent run."""
    success: bool
    summary: str
    steps: List[AgentStep] = field(default_factory=list)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_elapsed_s: float = 0.0
    iterations: int = 0
    timed_out: bool = False
    partial_results: List[str] = field(default_factory=list)
    disabled_tools: List[str] = field(default_factory=list)

    @property
    def tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens


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
            return self._run_loop(task)
        finally:
            # Safety net: normal paths release via _save_session, but
            # unexpected exceptions could bypass those.  _release_run_slot
            # is idempotent so the double-call on normal exits is harmless.
            self._release_run_slot()

    def _run_loop(self, task: str) -> AgentResult:
        """Inner loop for run(), separated so run() can wrap in try/finally."""
        t_start = time.monotonic()

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

        for iteration in range(1, self._max_iterations + 1):
            log.info("Iteration %d / %d", iteration, self._max_iterations)

            # --- Call the LLM with exponential backoff ------------------------
            response = self._call_llm_with_retry(iteration)
            if response is None:
                # All retries exhausted -- graceful degradation
                partial = self._collect_partial_results()
                self._save_session(task, "failed", iteration)
                return AgentResult(
                    success=False,
                    summary=self._graceful_degradation_summary(
                        iteration, partial,
                    ),
                    steps=list(self._steps),
                    total_input_tokens=self._total_input_tokens,
                    total_output_tokens=self._total_output_tokens,
                    total_elapsed_s=time.monotonic() - t_start,
                    iterations=iteration,
                    partial_results=partial,
                    disabled_tools=sorted(
                        self._circuit_breaker.disabled_tools,
                    ),
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

            # --- No tool calls: final answer ----------------------------------
            if not response.has_tool_calls:
                log.info("Agent returned final answer (no tool calls)")
                total_tok = self._total_input_tokens + self._total_output_tokens
                if self._logger:
                    self._logger.log_task_complete(
                        self._session_id, success=True,
                        summary=response.content,
                        total_tokens=total_tok, iterations=iteration,
                    )
                self._save_session(task, "completed", iteration)
                return AgentResult(
                    success=True,
                    summary=response.content,
                    steps=list(self._steps),
                    total_input_tokens=self._total_input_tokens,
                    total_output_tokens=self._total_output_tokens,
                    total_elapsed_s=time.monotonic() - t_start,
                    iterations=iteration,
                )

            # --- Execute tool calls -------------------------------------------
            assistant_msg = self._build_assistant_message(
                response.content, response.tool_calls,
            )
            self._history.append(assistant_msg)

            for tc in response.tool_calls:
                log.info(
                    "  Tool: %s(%s)",
                    tc.name,
                    _truncate(json.dumps(tc.arguments), 120),
                )

                if self._logger:
                    self._logger.log_tool_call(
                        self._session_id, tc.name,
                        tc.arguments, iteration,
                    )

                # Circuit breaker: skip disabled tools
                if self._circuit_breaker.is_disabled(tc.name):
                    result_text = (
                        f"ERROR: Tool '{tc.name}' disabled by circuit "
                        f"breaker ({CIRCUIT_BREAKER_THRESHOLD} consecutive "
                        f"failures)"
                    )
                    log.warning("  Skipped disabled tool: %s", tc.name)
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
                        tool_success=False,
                        elapsed_s=0.0,
                    ))
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

                # Compose result text for the LLM
                if result.success:
                    result_text = tool_output
                else:
                    result_text = f"ERROR: {result.error}"
                    if tool_output:
                        result_text = f"{tool_output}\nERROR: {result.error}"

                log.info(
                    "  Result: success=%s  (%.1fs)  %s",
                    result.success,
                    result.elapsed_s,
                    _truncate(result_text, 200),
                )

                if self._logger:
                    self._logger.log_tool_result(
                        self._session_id, tc.name,
                        result.success, result_text, result.elapsed_s,
                    )

                # Append tool result to history
                self._history.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result_text,
                })

                # Record step
                self._steps.append(AgentStep(
                    iteration=iteration,
                    tool_name=tc.name,
                    tool_args=tc.arguments,
                    tool_result=result_text,
                    tool_success=result.success,
                    elapsed_s=result.elapsed_s,
                ))

                # Check for task completion signal
                if result.success and tool_output.startswith(
                    _TASK_COMPLETE_PREFIX,
                ):
                    summary = tool_output[len(_TASK_COMPLETE_PREFIX):].strip()
                    log.info("Task complete: %s", summary)
                    total_tok = (
                        self._total_input_tokens
                        + self._total_output_tokens
                    )
                    if self._logger:
                        self._logger.log_task_complete(
                            self._session_id, success=True,
                            summary=summary,
                            total_tokens=total_tok, iterations=iteration,
                        )
                    self._save_session(task, "completed", iteration)
                    return AgentResult(
                        success=True,
                        summary=summary,
                        steps=list(self._steps),
                        total_input_tokens=self._total_input_tokens,
                        total_output_tokens=self._total_output_tokens,
                        total_elapsed_s=time.monotonic() - t_start,
                        iterations=iteration,
                    )

            # --- Checkpoint session after each iteration ----------------------
            self._save_session(task, "active", iteration)

            # --- Budget check -------------------------------------------------
            total_tokens = self._total_input_tokens + self._total_output_tokens
            if total_tokens >= self._max_tokens_budget:
                log.warning(
                    "Token budget exhausted: %d >= %d",
                    total_tokens,
                    self._max_tokens_budget,
                )
                budget_msg = (
                    f"Token budget exhausted after {iteration} iterations "
                    f"({total_tokens:,} tokens used)"
                )
                if self._logger:
                    self._logger.log_task_complete(
                        self._session_id, success=False,
                        summary=budget_msg,
                        total_tokens=total_tokens, iterations=iteration,
                    )
                self._save_session(task, "failed", iteration)
                return AgentResult(
                    success=False,
                    summary=budget_msg,
                    steps=list(self._steps),
                    total_input_tokens=self._total_input_tokens,
                    total_output_tokens=self._total_output_tokens,
                    total_elapsed_s=time.monotonic() - t_start,
                    iterations=iteration,
                    timed_out=True,
                )

        # --- Loop exhausted: partial result recovery --------------------------
        log.warning("Iteration limit reached: %d", self._max_iterations)
        total_tok = self._total_input_tokens + self._total_output_tokens
        partial = self._collect_partial_results()
        limit_msg = (
            f"Iteration limit ({self._max_iterations}) reached. "
            f"Last response: "
            f"{_truncate(self._last_assistant_content(), 500)}"
        )
        if partial:
            limit_msg += f" | {len(partial)} partial result(s) recovered"
        if self._logger:
            self._logger.log_task_complete(
                self._session_id, success=False,
                summary=limit_msg,
                total_tokens=total_tok, iterations=self._max_iterations,
            )
        self._save_session(task, "failed", self._max_iterations)
        return AgentResult(
            success=False,
            summary=limit_msg,
            steps=list(self._steps),
            total_input_tokens=self._total_input_tokens,
            total_output_tokens=self._total_output_tokens,
            total_elapsed_s=time.monotonic() - t_start,
            iterations=self._max_iterations,
            timed_out=True,
            partial_results=partial,
            disabled_tools=sorted(self._circuit_breaker.disabled_tools),
        )

    @property
    def session_id(self) -> str:
        """Return the current session ID."""
        return self._session_id

    def resume(self, session_id: str) -> AgentResult:
        """Resume a previously saved session.

        Loads the conversation history and continues the agent loop
        from where it left off.  Requires a ``session_store`` to have
        been provided at init time.

        Raises
        ------
        RuntimeError
            If no session store is configured.
        FileNotFoundError
            If the session ID does not exist.
        """
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
            self._restore_token_totals(data)

            prior_iterations = data.get("iterations", 0)
            remaining = max(1, self._max_iterations - prior_iterations)

            log.info(
                "Resuming session %s (%d prior iterations, %d messages)",
                session_id, prior_iterations, len(self._history),
            )

            t_start = time.monotonic()

            for iteration in range(prior_iterations + 1,
                                   prior_iterations + remaining + 1):
                log.info("Iteration %d / %d", iteration, self._max_iterations)

                try:
                    response = self._backend.chat(
                        self._history, tools=self._tools.schemas,
                    )
                except Exception as exc:
                    log.error("LLM call failed: %s", exc)
                    self._save_session(task, "failed", iteration)
                    return AgentResult(
                        success=False,
                        summary=f"LLM error on iteration {iteration}: {exc}",
                        steps=list(self._steps),
                        total_input_tokens=self._total_input_tokens,
                        total_output_tokens=self._total_output_tokens,
                        total_elapsed_s=time.monotonic() - t_start,
                        iterations=iteration,
                    )

                self._total_input_tokens += response.input_tokens
                self._total_output_tokens += response.output_tokens

                if not response.has_tool_calls:
                    self._save_session(task, "completed", iteration)
                    return AgentResult(
                        success=True,
                        summary=response.content,
                        steps=list(self._steps),
                        total_input_tokens=self._total_input_tokens,
                        total_output_tokens=self._total_output_tokens,
                        total_elapsed_s=time.monotonic() - t_start,
                        iterations=iteration,
                    )

                assistant_msg = self._build_assistant_message(
                    response.content, response.tool_calls,
                )
                self._history.append(assistant_msg)

                for tc in response.tool_calls:
                    try:
                        result = self._tools.execute(tc.name, tc.arguments)
                    except Exception as exc:
                        log.error("Tool execution crashed for %s: %s", tc.name, exc, exc_info=True)
                        from agent.tools import ToolResult
                        result = ToolResult(success=False, output="", error=f"Tool crashed: {exc}")
                    tool_output = result.output or ""
                    result_text = (
                        tool_output if result.success
                        else f"ERROR: {result.error}"
                    )
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
                        tool_success=result.success,
                        elapsed_s=result.elapsed_s,
                    ))

                    if result.success and tool_output.startswith(_TASK_COMPLETE_PREFIX):
                        summary = tool_output[len(_TASK_COMPLETE_PREFIX):].strip()
                        self._save_session(task, "completed", iteration)
                        return AgentResult(
                            success=True, summary=summary,
                            steps=list(self._steps),
                            total_input_tokens=self._total_input_tokens,
                            total_output_tokens=self._total_output_tokens,
                            total_elapsed_s=time.monotonic() - t_start,
                            iterations=iteration,
                        )

                self._save_session(task, "active", iteration)

                total_tokens = self._total_input_tokens + self._total_output_tokens
                if total_tokens >= self._max_tokens_budget:
                    self._save_session(task, "failed", iteration)
                    return AgentResult(
                        success=False,
                        summary=f"Token budget exhausted ({total_tokens:,} tokens)",
                        steps=list(self._steps),
                        total_input_tokens=self._total_input_tokens,
                        total_output_tokens=self._total_output_tokens,
                        total_elapsed_s=time.monotonic() - t_start,
                        iterations=iteration, timed_out=True,
                    )

            final_iter = prior_iterations + remaining
            self._save_session(task, "failed", final_iter)
            return AgentResult(
                success=False,
                summary=f"Iteration limit reached after resume ({final_iter})",
                steps=list(self._steps),
                total_input_tokens=self._total_input_tokens,
                total_output_tokens=self._total_output_tokens,
                total_elapsed_s=time.monotonic() - t_start,
                iterations=final_iter, timed_out=True,
            )
        finally:
            self._release_run_slot()

    @property
    def step_log(self) -> List[AgentStep]:
        """Return the log of all steps taken so far."""
        return list(self._steps)

    @property
    def history(self) -> List[Dict[str, Any]]:
        """Return the full conversation history (read-only copy)."""
        return list(self._history)

    # -- Error recovery helpers --------------------------------------------

    def _call_llm_with_retry(
        self, iteration: int,
    ) -> Optional[ChatResponse]:
        """Call the LLM with exponential backoff on failure.

        Returns the ChatResponse on success, or None if all retries are
        exhausted (graceful degradation).
        """
        last_exc: Optional[Exception] = None
        for attempt in range(LLM_MAX_RETRIES + 1):
            try:
                response = self._backend.chat(
                    self._history,
                    tools=self._tools.schemas,
                )
                return response
            except Exception as exc:
                last_exc = exc
                if attempt < LLM_MAX_RETRIES:
                    delay = LLM_BACKOFF_BASE_S * (2 ** attempt)
                    log.warning(
                        "LLM call failed (attempt %d/%d), retrying in "
                        "%.1fs: %s",
                        attempt + 1, LLM_MAX_RETRIES + 1, delay, exc,
                    )
                    self._sleep_fn(delay)
                else:
                    log.error(
                        "LLM call failed after %d retries: %s",
                        LLM_MAX_RETRIES + 1, exc,
                    )

        if self._logger and last_exc:
            self._logger.log_error(
                self._session_id, str(last_exc),
                context=f"LLM call, iteration {iteration} "
                        f"(all {LLM_MAX_RETRIES + 1} attempts failed)",
            )
        return None

    def _collect_partial_results(self) -> List[str]:
        """Gather successful tool outputs as partial results."""
        partials: List[str] = []
        for step in self._steps:
            if step.tool_success and step.tool_result:
                partials.append(
                    f"[{step.tool_name}] {_truncate(step.tool_result, 300)}"
                )
        return partials

    @staticmethod
    def _graceful_degradation_summary(
        iteration: int,
        partial_results: List[str],
    ) -> str:
        """Build summary when LLM is completely unreachable."""
        msg = (
            f"LLM unreachable after exhausting retries on iteration "
            f"{iteration}."
        )
        if partial_results:
            msg += (
                f" {len(partial_results)} partial result(s) recovered "
                f"from prior tool executions."
            )
        else:
            msg += " No partial results available."
        return msg

    # -- Internal helpers --------------------------------------------------

    @staticmethod
    def _build_assistant_message(
        content: str, tool_calls: List[ToolCall],
    ) -> Dict[str, Any]:
        """Build an OpenAI-format assistant message with tool_calls."""
        msg: Dict[str, Any] = {
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
        return msg

    def _last_assistant_content(self) -> str:
        """Extract the last assistant text from history."""
        for msg in reversed(self._history):
            if msg.get("role") == "assistant" and msg.get("content"):
                return msg["content"]
        return "(no assistant response recorded)"

    def _save_session(
        self, task: str, status: str, iteration: int,
    ) -> None:
        """Persist current session state if a store is configured."""
        if status in ("completed", "failed"):
            self._release_run_slot()
        if not self._session_store:
            return
        total = self._total_input_tokens + self._total_output_tokens
        try:
            self._session_store.save(
                session_id=self._session_id,
                task=task,
                history=self._history,
                status=status,
                iterations=iteration,
                tokens=total,
                input_tokens=self._total_input_tokens,
                output_tokens=self._total_output_tokens,
            )
        except OSError:
            log.warning("Session save failed for %s", self._session_id)

    def _claim_run_slot(self) -> None:
        with self._run_lock:
            if self._running:
                raise RuntimeError("Agent is already running a task")
            self._running = True

    def _release_run_slot(self) -> None:
        with self._run_lock:
            self._running = False

    def _restore_token_totals(self, data: Dict[str, Any]) -> None:
        input_tokens = data.get("input_tokens")
        output_tokens = data.get("output_tokens")
        if input_tokens is None and output_tokens is None:
            self._total_input_tokens = int(data.get("total_tokens", 0))
            self._total_output_tokens = 0
            return

        self._total_input_tokens = int(input_tokens or 0)
        self._total_output_tokens = int(output_tokens or 0)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _truncate(text: str, max_len: int) -> str:
    """Truncate text for logging, preserving the start."""
    if len(text) <= max_len:
        return text
    return text[:max_len - 3] + "..."
