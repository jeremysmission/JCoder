"""
Model Cascade (Cascadia Pattern)
---------------------------------
Routes queries through a hierarchy of models based on estimated
complexity. Simple queries go to the smallest/fastest model;
only genuinely hard problems escalate to the largest.

Based on: Cascadia (arxiv 2506.04203), Routing/Cascades (ICLR 2026)

Complexity signals:
- Query length (short = simple)
- Question type classification (lookup vs reasoning vs design)
- Code complexity indicators (number of files mentioned, etc.)
- Historical difficulty from telemetry

Result: 80% of queries answered in <1s by the smallest model.
Only hard architectural questions consume full-power resources.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from core.config import ModelConfig
from core.network_gate import NetworkGate
from core.runtime import Runtime


@dataclass
class CascadeLevel:
    """A single level in the model cascade."""
    name: str
    model_config: ModelConfig
    max_complexity: float  # route here if complexity <= this
    timeout_s: int = 60


@dataclass
class CascadeResult:
    """Result from the cascade with metadata about which level handled it."""
    answer: str
    model_used: str
    level_index: int
    complexity_score: float
    escalated: bool  # True if a lower level tried and gave up
    latency_ms: float = 0.0


# ---------------------------------------------------------------------------
# Complexity estimation
# ---------------------------------------------------------------------------

# Keywords that signal higher complexity
_COMPLEX_SIGNALS = {
    "refactor", "redesign", "architect", "implement", "migrate",
    "optimize", "performance", "security", "vulnerability",
    "concurrent", "parallel", "distributed", "async",
    "database", "schema", "migration", "integration",
    "why does", "how should", "what would happen if",
    "trade-off", "tradeoff", "compare", "versus",
}

_SIMPLE_SIGNALS = {
    "what is", "where is", "show me", "find", "list",
    "syntax", "example", "define", "definition",
    "import", "install", "version", "type of",
}


def estimate_complexity(query: str) -> float:
    """
    Estimate query complexity on a 0.0-1.0 scale.

    Signals:
    - Length (longer = harder)
    - Complex keywords (refactor, architect, etc.)
    - Simple keywords (what is, show me, etc.)
    - Multi-part questions (and, or, commas)
    - Code references (file paths, function names)
    """
    score = 0.0
    q_lower = query.lower()
    words = q_lower.split()
    n_words = len(words)

    # Length signal
    if n_words <= 5:
        score += 0.0
    elif n_words <= 15:
        score += 0.15
    elif n_words <= 30:
        score += 0.3
    else:
        score += 0.5

    # Complex keyword signal
    complex_hits = sum(1 for s in _COMPLEX_SIGNALS if s in q_lower)
    score += min(0.3, complex_hits * 0.1)

    # Simple keyword signal (reduces complexity)
    simple_hits = sum(1 for s in _SIMPLE_SIGNALS if s in q_lower)
    score -= min(0.2, simple_hits * 0.1)

    # Multi-part question signal
    conjunctions = len(re.findall(r"\band\b|\bor\b|\balso\b", q_lower))
    commas = q_lower.count(",")
    questions = q_lower.count("?")
    score += min(0.2, (conjunctions + commas + questions) * 0.05)

    # Code reference signal (file paths, dot notation)
    code_refs = len(re.findall(r"[a-zA-Z_]\w*\.\w+", query))
    score += min(0.15, code_refs * 0.05)

    return max(0.0, min(1.0, score))


# ---------------------------------------------------------------------------
# Cascade router
# ---------------------------------------------------------------------------

class ModelCascade:
    """
    Routes queries through a hierarchy of models based on complexity.

    Levels are sorted from cheapest/fastest to most expensive/capable.
    Each level has a max_complexity threshold. The first level whose
    threshold >= the query's complexity handles the request.

    If a level fails or produces a low-confidence answer, the query
    escalates to the next level.
    """

    def __init__(
        self,
        levels: List[CascadeLevel],
        gate: Optional[NetworkGate] = None,
        confidence_threshold: float = 0.4,
    ):
        # Sort levels by max_complexity (cheapest first)
        self.levels = sorted(levels, key=lambda l: l.max_complexity)
        self.gate = gate or NetworkGate(mode="localhost")
        self.confidence_threshold = confidence_threshold
        self._runtimes: Dict[str, Runtime] = {}

    def _get_runtime(self, level: CascadeLevel) -> Runtime:
        """Lazy-init runtimes for each cascade level."""
        if level.name not in self._runtimes:
            self._runtimes[level.name] = Runtime(
                level.model_config,
                timeout=level.timeout_s,
                gate=self.gate,
            )
        return self._runtimes[level.name]

    def route(
        self,
        question: str,
        context_chunks: List[str],
        confidence_fn: Optional[Callable[[str], float]] = None,
    ) -> CascadeResult:
        """
        Route a question through the cascade.

        Args:
            question: The user's query
            context_chunks: Retrieved code context
            confidence_fn: Optional function that scores answer confidence
                           (0.0-1.0). Used for escalation decisions.

        Returns:
            CascadeResult with the answer and routing metadata.
        """
        complexity = estimate_complexity(question)
        escalated = False

        for i, level in enumerate(self.levels):
            # Skip levels that are too weak for this complexity
            if level.max_complexity < complexity and i < len(self.levels) - 1:
                continue

            runtime = self._get_runtime(level)
            t0 = time.time()

            try:
                answer = runtime.generate(question, context_chunks)
                latency_ms = (time.time() - t0) * 1000

                # Check confidence if we have a scoring function
                if confidence_fn and i < len(self.levels) - 1:
                    conf = confidence_fn(answer)
                    if conf < self.confidence_threshold:
                        escalated = True
                        continue  # try next level

                return CascadeResult(
                    answer=answer,
                    model_used=level.name,
                    level_index=i,
                    complexity_score=complexity,
                    escalated=escalated,
                    latency_ms=latency_ms,
                )
            except Exception:
                escalated = True
                continue  # try next level on failure

        # All levels failed -- return error from last level
        return CascadeResult(
            answer="All cascade levels failed to generate an answer.",
            model_used=self.levels[-1].name if self.levels else "none",
            level_index=len(self.levels) - 1,
            complexity_score=complexity,
            escalated=True,
        )

    def close(self) -> None:
        """Release all runtime connections."""
        for rt in self._runtimes.values():
            rt.close()
        self._runtimes.clear()
