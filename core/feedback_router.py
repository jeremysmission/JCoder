"""
Feedback Router — Closes the GVU Triangle
-------------------------------------------
Routes outcomes from ANY subsystem to ALL connected subsystems
simultaneously. This is the critical missing connection that
turns isolated systems into a compounding feedback loop.

The GVU Triangle:
    Generator (Self-Play) → creates challenges
    Verifier (Experience Replay) → records what worked
    Updater (Prompt Evolution) → improves based on outcomes
    → back to Generator (now with better prompts = harder challenges)

Without the router: each system runs in isolation.
With the router: each outcome feeds THREE systems at once,
creating compound improvement.

The key insight from research: near-misses (P2Value ≈ 0.5)
are where compounding happens fastest. The router prioritizes
these boundary cases for all downstream consumers.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

log = logging.getLogger(__name__)


@dataclass
class Outcome:
    """A structured outcome from any subsystem interaction."""
    query: str
    answer: str = ""
    score: float = 0.0
    source: str = ""  # which subsystem generated this
    category: str = ""
    context_chunks: int = 0
    error: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    @property
    def is_success(self) -> bool:
        return self.score >= 0.5

    @property
    def is_near_miss(self) -> bool:
        """Near-misses (0.3-0.7) are where the most learning happens."""
        return 0.3 <= self.score <= 0.7

    @property
    def learnability(self) -> float:
        """How much the system can learn from this outcome.

        Maximum at score=0.5 (maximum uncertainty), zero at 0 and 1.
        This is the Absolute Zero insight: the frontier of capability
        is where improvement happens fastest.
        """
        return 1.0 - abs(2.0 * self.score - 1.0)


class FeedbackRouter:
    """Routes outcomes to all connected subsystems simultaneously.

    Consumers register via on_outcome(). When route() is called,
    ALL registered consumers receive the outcome. This creates
    the compound effect: one interaction improves multiple systems.
    """

    def __init__(self, log_dir: str = "logs/feedback_router"):
        self._consumers: Dict[str, Callable[[Outcome], None]] = {}
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._outcome_count = 0
        self._high_learnability_count = 0

    def register(self, name: str, consumer: Callable[[Outcome], None]) -> None:
        """Register a consumer that receives all outcomes."""
        self._consumers[name] = consumer
        log.info("FeedbackRouter: registered consumer '%s'", name)

    def route(self, outcome: Outcome) -> Dict[str, bool]:
        """Route an outcome to ALL registered consumers.

        Returns dict of {consumer_name: success} for each consumer.
        """
        self._outcome_count += 1
        if outcome.is_near_miss:
            self._high_learnability_count += 1

        results = {}
        for name, consumer in self._consumers.items():
            try:
                consumer(outcome)
                results[name] = True
            except Exception as exc:
                log.warning(
                    "FeedbackRouter: consumer '%s' failed: %s", name, exc,
                )
                results[name] = False

        # Log high-learnability outcomes for analysis
        if outcome.learnability > 0.6:
            self._log_high_learnability(outcome)

        return results

    def _log_high_learnability(self, outcome: Outcome) -> None:
        """Log near-miss outcomes that are most valuable for learning."""
        log_file = self._log_dir / "high_learnability.jsonl"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "query": outcome.query[:200],
                "score": outcome.score,
                "learnability": outcome.learnability,
                "source": outcome.source,
                "category": outcome.category,
                "timestamp": outcome.timestamp,
            }) + "\n")

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "consumers": list(self._consumers.keys()),
            "outcomes_routed": self._outcome_count,
            "high_learnability": self._high_learnability_count,
            "learnability_rate": (
                self._high_learnability_count / max(self._outcome_count, 1)
            ),
        }


def create_default_router() -> FeedbackRouter:
    """Create a FeedbackRouter with standard JCoder consumers wired in.

    Connects:
    - Experience Replay (store successful trajectories)
    - Strategy Evolver (record strategy outcomes)
    - Prompt candidate scoring (feed eval traces to prompt evolver)
    """
    router = FeedbackRouter()

    # Consumer 1: Experience Replay
    try:
        from core.experience_replay import ExperienceStore
        store = ExperienceStore("_experience/agent_replay.db")

        def _feed_experience(outcome: Outcome) -> None:
            if outcome.is_success:
                store.add(
                    query=outcome.query,
                    answer=outcome.answer,
                    source_files=outcome.metadata.get("source_files", []),
                    confidence=outcome.score,
                )

        router.register("experience_replay", _feed_experience)
    except ImportError:
        log.debug("Experience replay not available for router")

    # Consumer 2: Strategy Evolver
    try:
        from core.strategy_evolver import StrategyEvolver
        evolver = StrategyEvolver()

        def _feed_strategy(outcome: Outcome) -> None:
            strategy_id = outcome.metadata.get("strategy_id")
            if strategy_id:
                evolver.record_outcome(strategy_id, outcome.query, outcome.score)

        router.register("strategy_evolver", _feed_strategy)
    except ImportError:
        log.debug("Strategy evolver not available for router")

    return router
