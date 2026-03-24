"""
Agent error-recovery, session-management helpers, and shared data classes.
--------------------------------------------------------------------------
Extracted from core.py to stay within the 500-line module limit.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

from agent.llm_backend import ChatResponse
from agent.session import SessionStore


# ---------------------------------------------------------------------------
# Data classes (shared by core.py and consumers like memory.py, tests)
# ---------------------------------------------------------------------------

@dataclass
class AgentStep:
    """Record of a single tool invocation."""
    iteration: int
    tool_name: str
    tool_args: Dict[str, Any]
    tool_result: str
    tool_success: bool
    elapsed_s: float = 0.0


@dataclass
class AgentResult:
    """Final outcome of an Agent.run() call."""
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
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens

    @property
    def tokens(self) -> int:
        """Alias for total_tokens (backwards compat)."""
        return self.total_tokens

log = logging.getLogger(__name__)

# Constants (also importable by core.py)
LLM_MAX_RETRIES = 3
LLM_BACKOFF_BASE_S = 1.0
CIRCUIT_BREAKER_THRESHOLD = 3


def call_llm_with_retry(
    backend,
    history: List[Dict[str, Any]],
    tool_schemas: List[Dict[str, Any]],
    iteration: int,
    logger=None,
    session_id: str = "",
    sleep_fn: Callable[[float], None] = time.sleep,
) -> Optional[ChatResponse]:
    """Call the LLM with exponential backoff on failure.

    Returns the ChatResponse on success, or None if all retries are
    exhausted (graceful degradation).
    """
    last_exc: Optional[Exception] = None
    for attempt in range(LLM_MAX_RETRIES + 1):
        try:
            return backend.chat(history, tools=tool_schemas)
        except Exception as exc:
            last_exc = exc
            if attempt < LLM_MAX_RETRIES:
                delay = LLM_BACKOFF_BASE_S * (2 ** attempt)
                log.warning(
                    "LLM call failed (attempt %d/%d), retrying in "
                    "%.1fs: %s",
                    attempt + 1, LLM_MAX_RETRIES + 1, delay, exc,
                )
                sleep_fn(delay)
            else:
                log.error(
                    "LLM call failed after %d retries: %s",
                    LLM_MAX_RETRIES + 1, exc,
                )

    if logger and last_exc:
        logger.log_error(
            session_id, str(last_exc),
            context=f"LLM call, iteration {iteration} "
                    f"(all {LLM_MAX_RETRIES + 1} attempts failed)",
        )
    return None


def collect_partial_results(steps) -> List[str]:
    """Gather successful tool outputs as partial results."""
    partials: List[str] = []
    for step in steps:
        if step.tool_success and step.tool_result:
            partials.append(
                f"[{step.tool_name}] {_truncate(step.tool_result, 300)}"
            )
    return partials


def graceful_degradation_summary(
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


def restore_token_totals(data: Dict[str, Any]):
    """Extract token totals from saved session data.

    Returns (input_tokens, output_tokens).
    """
    input_tokens = data.get("input_tokens")
    output_tokens = data.get("output_tokens")
    if input_tokens is None and output_tokens is None:
        return int(data.get("total_tokens", 0)), 0
    return int(input_tokens or 0), int(output_tokens or 0)


def save_session(
    session_store: Optional[SessionStore],
    session_id: str,
    task: str,
    history: List[Dict[str, Any]],
    status: str,
    iteration: int,
    input_tokens: int,
    output_tokens: int,
) -> None:
    """Persist current session state if a store is configured."""
    if not session_store:
        return
    total = input_tokens + output_tokens
    try:
        session_store.save(
            session_id=session_id,
            task=task,
            history=history,
            status=status,
            iterations=iteration,
            tokens=total,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
    except OSError:
        log.warning("Session save failed for %s", session_id)


def _truncate(text: str, max_len: int) -> str:
    """Truncate text for logging, preserving the start."""
    if len(text) <= max_len:
        return text
    return text[:max_len - 3] + "..."
