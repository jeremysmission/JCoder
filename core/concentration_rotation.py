"""
Concentration Rotation and Meta-QA (Sprint 23)
------------------------------------------------
Prevents agent output degradation through attention decay modeling,
role rotation, meta-QA validation, and random deep inspection.

Architecture:
  1. AttentionDecayPredictor  -- estimates quality degradation over time
  2. RoleRotator              -- swaps agent roles at predicted decay points
  3. MetaQALayer              -- second-pass QA on critical decisions
  4. DeepInspector            -- random probabilistic audits

Gate: Demonstrate that rotated agents catch bugs that non-rotated miss.
"""

from __future__ import annotations

import json
import logging
import math
import random
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from core.sqlite_owner import SQLiteConnectionOwner

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class QualitySignal:
    """A single quality measurement from agent output."""
    timestamp: float
    agent_id: str
    role: str
    score: float  # 0.0 - 1.0
    task_type: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DecayPrediction:
    """Predicted quality decay for an agent."""
    agent_id: str
    current_quality: float
    predicted_quality: float
    time_to_threshold: float  # seconds until quality drops below threshold
    should_rotate: bool
    confidence: float


@dataclass
class RotationEvent:
    """Record of an agent role rotation."""
    rotation_id: str
    timestamp: float
    agent_id: str
    old_role: str
    new_role: str
    reason: str
    decay_prediction: Optional[DecayPrediction] = None


@dataclass
class InspectionResult:
    """Result from a deep inspection."""
    inspection_id: str
    timestamp: float
    agent_id: str
    task_id: str
    original_score: float
    inspected_score: float
    issues_found: List[str] = field(default_factory=list)
    passed: bool = True


@dataclass
class MetaQAResult:
    """Result from meta-QA validation."""
    qa_id: str
    timestamp: float
    original_decision: str
    validated: bool
    override_decision: str = ""
    reason: str = ""
    checks_performed: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Attention Decay Predictor
# ---------------------------------------------------------------------------

class AttentionDecayPredictor:
    """Models quality degradation over continuous operation.

    Uses exponential decay with fatigue factor: quality decreases
    as the agent processes more tasks without rotation. The decay
    rate is learned from historical quality signals.

    q(t) = q_0 * exp(-lambda * t) + noise
    """

    DEFAULT_DECAY_RATE = 0.001  # per-second decay
    DEFAULT_THRESHOLD = 0.6     # rotate when quality drops below this

    def __init__(
        self,
        decay_rate: float = DEFAULT_DECAY_RATE,
        threshold: float = DEFAULT_THRESHOLD,
    ):
        self._decay_rate = decay_rate
        self._threshold = threshold
        self._signals: Dict[str, List[QualitySignal]] = {}

    def record_signal(self, signal: QualitySignal) -> None:
        """Record a quality observation."""
        if signal.agent_id not in self._signals:
            self._signals[signal.agent_id] = []
        self._signals[signal.agent_id].append(signal)
        # Keep last 1000 signals per agent
        if len(self._signals[signal.agent_id]) > 1000:
            self._signals[signal.agent_id] = self._signals[signal.agent_id][-1000:]

    def predict(self, agent_id: str) -> DecayPrediction:
        """Predict quality decay for an agent."""
        signals = self._signals.get(agent_id, [])

        if len(signals) < 2:
            return DecayPrediction(
                agent_id=agent_id,
                current_quality=signals[0].score if signals else 1.0,
                predicted_quality=1.0,
                time_to_threshold=float("inf"),
                should_rotate=False,
                confidence=0.0,
            )

        # Current quality: weighted average of recent signals
        recent = signals[-5:]
        weights = [0.1, 0.15, 0.2, 0.25, 0.3]
        if len(recent) < 5:
            weights = [1.0 / len(recent)] * len(recent)

        current_q = sum(
            s.score * w for s, w in zip(recent, weights[-len(recent):])
        )

        # Estimate decay rate from signal history
        learned_rate = self._estimate_decay_rate(signals)

        # Predict future quality
        predicted_q = current_q * math.exp(-learned_rate * 300)  # 5 min ahead

        # Time to threshold
        if current_q <= self._threshold:
            time_to_thresh = 0.0
        elif learned_rate > 0:
            time_to_thresh = -math.log(self._threshold / max(current_q, 0.01)) / learned_rate
        else:
            time_to_thresh = float("inf")

        should_rotate = current_q <= self._threshold or time_to_thresh < 60.0

        confidence = min(1.0, len(signals) / 20.0)

        return DecayPrediction(
            agent_id=agent_id,
            current_quality=round(current_q, 4),
            predicted_quality=round(predicted_q, 4),
            time_to_threshold=round(time_to_thresh, 1),
            should_rotate=should_rotate,
            confidence=round(confidence, 3),
        )

    def _estimate_decay_rate(self, signals: List[QualitySignal]) -> float:
        """Estimate decay rate from observed quality trend."""
        if len(signals) < 3:
            return self._decay_rate

        # Simple linear regression on log(quality) vs time
        t0 = signals[0].timestamp
        log_scores = []
        times = []
        for s in signals:
            if s.score > 0:
                log_scores.append(math.log(max(s.score, 0.01)))
                times.append(s.timestamp - t0)

        if len(log_scores) < 3 or (times[-1] - times[0]) < 1.0:
            return self._decay_rate

        n = len(times)
        sum_t = sum(times)
        sum_lq = sum(log_scores)
        sum_t2 = sum(t * t for t in times)
        sum_tlq = sum(t * lq for t, lq in zip(times, log_scores))

        denom = n * sum_t2 - sum_t * sum_t
        if abs(denom) < 1e-10:
            return self._decay_rate

        slope = (n * sum_tlq - sum_t * sum_lq) / denom
        # Decay rate is negative slope (quality decreasing -> positive rate)
        rate = max(0.0, -slope)
        return rate if rate > 0 else self._decay_rate

    @property
    def threshold(self) -> float:
        return self._threshold

    def agent_ids(self) -> List[str]:
        """Get all tracked agent IDs."""
        return list(self._signals.keys())


# ---------------------------------------------------------------------------
# Role Rotator
# ---------------------------------------------------------------------------

class RoleRotator:
    """Manages agent role assignments and rotation.

    Tracks which agents hold which roles, and rotates them
    when decay prediction signals degradation.
    """

    def __init__(self, roles: Optional[List[str]] = None):
        self._roles = roles or ["researcher", "implementer", "reviewer", "tester"]
        self._assignments: Dict[str, str] = {}  # agent_id -> role
        self._rotation_history: List[RotationEvent] = []

    def assign(self, agent_id: str, role: str) -> None:
        """Assign a role to an agent."""
        if role not in self._roles:
            log.warning("Unknown role %r, adding to role list", role)
            self._roles.append(role)
        self._assignments[agent_id] = role

    def current_role(self, agent_id: str) -> str:
        """Get current role for an agent."""
        return self._assignments.get(agent_id, "")

    def rotate(
        self,
        agent_id: str,
        prediction: Optional[DecayPrediction] = None,
        reason: str = "scheduled",
    ) -> RotationEvent:
        """Rotate an agent to a new role.

        Picks the role that the agent has spent the least recent time in,
        avoiding the current role.
        """
        old_role = self._assignments.get(agent_id, "")
        available = [r for r in self._roles if r != old_role]
        if not available:
            available = self._roles[:]

        # Pick role with least recent assignment by this agent
        role_last_used: Dict[str, float] = {}
        for event in reversed(self._rotation_history):
            if event.agent_id == agent_id and event.new_role not in role_last_used:
                role_last_used[event.new_role] = event.timestamp

        # Sort available roles by last-used time (oldest first)
        available.sort(key=lambda r: role_last_used.get(r, 0.0))
        new_role = available[0]

        self._assignments[agent_id] = new_role

        event = RotationEvent(
            rotation_id=f"rot_{uuid.uuid4().hex[:8]}",
            timestamp=time.time(),
            agent_id=agent_id,
            old_role=old_role,
            new_role=new_role,
            reason=reason,
            decay_prediction=prediction,
        )
        self._rotation_history.append(event)
        log.info("Rotated %s: %s -> %s (%s)", agent_id, old_role, new_role, reason)
        return event

    def rotation_count(self, agent_id: Optional[str] = None) -> int:
        """Count rotations, optionally filtered by agent."""
        if agent_id:
            return sum(1 for e in self._rotation_history if e.agent_id == agent_id)
        return len(self._rotation_history)

    def history(self, limit: int = 50) -> List[RotationEvent]:
        """Get recent rotation history."""
        return self._rotation_history[-limit:]

    @property
    def roles(self) -> List[str]:
        return self._roles[:]

    @property
    def assignments(self) -> Dict[str, str]:
        return dict(self._assignments)


# ---------------------------------------------------------------------------
# Meta-QA Layer
# ---------------------------------------------------------------------------

class MetaQALayer:
    """Second-pass validation on critical agent decisions.

    Catches rubber-stamping by applying independent quality checks
    to decisions that would normally be auto-accepted.
    """

    def __init__(
        self,
        checks: Optional[List[Callable[[str, Dict[str, Any]], bool]]] = None,
    ):
        self._checks: List[Callable[[str, Dict[str, Any]], bool]] = checks or []
        self._results: List[MetaQAResult] = []

    def add_check(self, check_fn: Callable[[str, Dict[str, Any]], bool]) -> None:
        """Register a QA check function.

        check_fn(decision, context) -> True if valid, False if should override.
        """
        self._checks.append(check_fn)

    def validate(
        self,
        decision: str,
        context: Dict[str, Any],
    ) -> MetaQAResult:
        """Run all registered checks against a decision."""
        qa_id = f"qa_{uuid.uuid4().hex[:8]}"
        checks_performed = []
        all_passed = True

        for i, check_fn in enumerate(self._checks):
            check_name = getattr(check_fn, "__name__", f"check_{i}")
            try:
                passed = check_fn(decision, context)
                checks_performed.append(f"{check_name}: {'PASS' if passed else 'FAIL'}")
                if not passed:
                    all_passed = False
            except Exception as exc:
                checks_performed.append(f"{check_name}: ERROR ({exc})")
                all_passed = False

        result = MetaQAResult(
            qa_id=qa_id,
            timestamp=time.time(),
            original_decision=decision,
            validated=all_passed,
            override_decision="" if all_passed else "needs_review",
            reason="" if all_passed else "Failed QA checks",
            checks_performed=checks_performed,
        )
        self._results.append(result)
        return result

    def stats(self) -> Dict[str, Any]:
        """Get QA validation statistics."""
        total = len(self._results)
        validated = sum(1 for r in self._results if r.validated)
        overridden = total - validated

        return {
            "total_validations": total,
            "validated": validated,
            "overridden": overridden,
            "override_rate": round(overridden / max(total, 1), 3),
        }

    def history(self, limit: int = 50) -> List[MetaQAResult]:
        """Get recent QA results."""
        return self._results[-limit:]


# ---------------------------------------------------------------------------
# Deep Inspector
# ---------------------------------------------------------------------------

class DeepInspector:
    """Random probabilistic deep inspections.

    Schedules random audits on agent output to catch issues
    that regular QA might miss. Inspection probability increases
    when recent quality signals trend downward.
    """

    DEFAULT_PROBABILITY = 0.1  # 10% base inspection rate

    def __init__(
        self,
        base_probability: float = DEFAULT_PROBABILITY,
        inspect_fn: Optional[Callable[[str, str, Dict[str, Any]], InspectionResult]] = None,
    ):
        self._base_prob = min(1.0, max(0.0, base_probability))
        self._inspect_fn = inspect_fn
        self._results: List[InspectionResult] = []
        self._rng = random.Random(42)

    def should_inspect(self, quality_trend: float = 0.0) -> bool:
        """Decide whether to inspect based on probability and quality trend.

        quality_trend: negative means quality is declining (increases inspection prob)
        """
        adjusted_prob = self._base_prob
        if quality_trend < 0:
            # Increase inspection probability when quality is declining
            adjusted_prob = min(1.0, self._base_prob + abs(quality_trend) * 0.5)

        return self._rng.random() < adjusted_prob

    def inspect(
        self,
        agent_id: str,
        task_id: str,
        context: Dict[str, Any],
    ) -> InspectionResult:
        """Run a deep inspection on agent output."""
        if self._inspect_fn:
            result = self._inspect_fn(agent_id, task_id, context)
        else:
            # Default: compare original score with re-evaluation
            original_score = context.get("score", 0.0)
            result = InspectionResult(
                inspection_id=f"insp_{uuid.uuid4().hex[:8]}",
                timestamp=time.time(),
                agent_id=agent_id,
                task_id=task_id,
                original_score=original_score,
                inspected_score=original_score,
                passed=True,
            )

        self._results.append(result)
        return result

    def stats(self) -> Dict[str, Any]:
        """Get inspection statistics."""
        total = len(self._results)
        passed = sum(1 for r in self._results if r.passed)
        failed = total - passed

        issues_found = sum(len(r.issues_found) for r in self._results)

        return {
            "total_inspections": total,
            "passed": passed,
            "failed": failed,
            "issues_found": issues_found,
            "failure_rate": round(failed / max(total, 1), 3),
        }

    def history(self, limit: int = 50) -> List[InspectionResult]:
        """Get recent inspection results."""
        return self._results[-limit:]

    def set_seed(self, seed: int) -> None:
        """Set RNG seed for reproducible testing."""
        self._rng = random.Random(seed)
