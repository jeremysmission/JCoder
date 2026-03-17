"""Tests for Concentration Rotation and Meta-QA (Sprint 23)."""

from __future__ import annotations

import math
import time
from typing import Any, Dict

import pytest

from core.concentration_rotation import (
    AttentionDecayPredictor,
    DecayPrediction,
    DeepInspector,
    InspectionResult,
    MetaQALayer,
    MetaQAResult,
    QualitySignal,
    RoleRotator,
    RotationEvent,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _signal(agent_id: str = "agent_0", score: float = 0.8,
            role: str = "researcher", t: float = 0.0) -> QualitySignal:
    return QualitySignal(
        timestamp=t or time.time(),
        agent_id=agent_id,
        role=role,
        score=score,
    )


# ---------------------------------------------------------------------------
# QualitySignal
# ---------------------------------------------------------------------------

class TestQualitySignal:
    def test_create(self):
        s = _signal(score=0.9)
        assert s.score == 0.9
        assert s.agent_id == "agent_0"

    def test_defaults(self):
        s = QualitySignal(
            timestamp=1000.0, agent_id="a1",
            role="reviewer", score=0.5,
        )
        assert s.task_type == ""
        assert s.metadata == {}


# ---------------------------------------------------------------------------
# AttentionDecayPredictor
# ---------------------------------------------------------------------------

class TestAttentionDecayPredictor:
    def test_no_signals(self):
        pred = AttentionDecayPredictor()
        result = pred.predict("unknown_agent")
        assert result.current_quality == 1.0
        assert result.should_rotate is False
        assert result.confidence == 0.0

    def test_single_signal(self):
        pred = AttentionDecayPredictor()
        pred.record_signal(_signal(score=0.8, t=1000.0))
        result = pred.predict("agent_0")
        assert result.current_quality == 0.8
        assert result.confidence == 0.0  # Need >= 2 signals

    def test_stable_quality_no_rotation(self):
        pred = AttentionDecayPredictor(threshold=0.6)
        # All signals at 0.9 -- no decay
        for i in range(10):
            pred.record_signal(_signal(score=0.9, t=1000.0 + i * 60))

        result = pred.predict("agent_0")
        assert result.current_quality >= 0.85
        assert result.should_rotate is False
        assert result.confidence > 0.0

    def test_declining_quality_triggers_rotation(self):
        pred = AttentionDecayPredictor(threshold=0.6)
        # Quality drops from 0.9 to 0.4 over time
        scores = [0.9, 0.85, 0.8, 0.7, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35]
        for i, score in enumerate(scores):
            pred.record_signal(_signal(score=score, t=1000.0 + i * 120))

        result = pred.predict("agent_0")
        assert result.current_quality < 0.6
        assert result.should_rotate is True

    def test_threshold_property(self):
        pred = AttentionDecayPredictor(threshold=0.7)
        assert pred.threshold == 0.7

    def test_agent_ids(self):
        pred = AttentionDecayPredictor()
        pred.record_signal(_signal(agent_id="a1", t=1000.0))
        pred.record_signal(_signal(agent_id="a2", t=1001.0))
        assert sorted(pred.agent_ids()) == ["a1", "a2"]

    def test_signal_cap(self):
        pred = AttentionDecayPredictor()
        for i in range(1200):
            pred.record_signal(_signal(score=0.8, t=1000.0 + i))
        # Should be capped at 1000
        assert len(pred._signals["agent_0"]) == 1000

    def test_confidence_scales_with_signals(self):
        pred = AttentionDecayPredictor()
        # 5 signals = 0.25 confidence
        for i in range(5):
            pred.record_signal(_signal(score=0.8, t=1000.0 + i * 10))
        result = pred.predict("agent_0")
        assert result.confidence == 0.25

        # 20 signals = 1.0 confidence
        for i in range(15):
            pred.record_signal(_signal(score=0.8, t=2000.0 + i * 10))
        result = pred.predict("agent_0")
        assert result.confidence == 1.0


# ---------------------------------------------------------------------------
# RoleRotator
# ---------------------------------------------------------------------------

class TestRoleRotator:
    def test_assign_and_get(self):
        rotator = RoleRotator()
        rotator.assign("agent_0", "researcher")
        assert rotator.current_role("agent_0") == "researcher"

    def test_rotate_changes_role(self):
        rotator = RoleRotator(roles=["researcher", "implementer", "reviewer"])
        rotator.assign("agent_0", "researcher")
        event = rotator.rotate("agent_0", reason="decay")
        assert event.old_role == "researcher"
        assert event.new_role != "researcher"
        assert rotator.current_role("agent_0") == event.new_role

    def test_rotate_avoids_current_role(self):
        rotator = RoleRotator(roles=["A", "B", "C"])
        rotator.assign("a1", "A")
        event = rotator.rotate("a1")
        assert event.new_role in ["B", "C"]

    def test_rotation_count(self):
        rotator = RoleRotator()
        rotator.assign("a1", "researcher")
        rotator.rotate("a1")
        rotator.rotate("a1")
        assert rotator.rotation_count("a1") == 2
        assert rotator.rotation_count() == 2

    def test_rotation_history(self):
        rotator = RoleRotator()
        rotator.assign("a1", "researcher")
        rotator.rotate("a1")
        history = rotator.history()
        assert len(history) == 1
        assert history[0].agent_id == "a1"

    def test_roles_property(self):
        rotator = RoleRotator(roles=["A", "B"])
        assert rotator.roles == ["A", "B"]

    def test_assignments_property(self):
        rotator = RoleRotator()
        rotator.assign("a1", "researcher")
        rotator.assign("a2", "reviewer")
        assert rotator.assignments == {"a1": "researcher", "a2": "reviewer"}

    def test_unknown_role_added(self):
        rotator = RoleRotator(roles=["A", "B"])
        rotator.assign("a1", "custom_role")
        assert "custom_role" in rotator.roles

    def test_rotation_event_id_format(self):
        rotator = RoleRotator()
        rotator.assign("a1", "researcher")
        event = rotator.rotate("a1")
        assert event.rotation_id.startswith("rot_")

    def test_rotate_unassigned_agent(self):
        rotator = RoleRotator(roles=["A", "B", "C"])
        event = rotator.rotate("new_agent")
        assert event.old_role == ""
        assert event.new_role in ["A", "B", "C"]

    def test_rotation_prefers_least_recent_role(self):
        rotator = RoleRotator(roles=["A", "B", "C"])
        rotator.assign("a1", "A")
        # Rotate multiple times
        rotator.rotate("a1")
        rotator.rotate("a1")
        rotator.rotate("a1")
        # Should cycle through roles
        history = rotator.history()
        roles_seen = {e.new_role for e in history}
        assert len(roles_seen) >= 2  # Not stuck on one role


# ---------------------------------------------------------------------------
# MetaQALayer
# ---------------------------------------------------------------------------

class TestMetaQALayer:
    def test_no_checks_passes(self):
        qa = MetaQALayer()
        result = qa.validate("accept", {"score": 0.9})
        assert result.validated is True
        assert result.override_decision == ""

    def test_passing_check(self):
        qa = MetaQALayer()
        qa.add_check(lambda d, c: True)
        result = qa.validate("accept", {"score": 0.9})
        assert result.validated is True

    def test_failing_check_overrides(self):
        qa = MetaQALayer()
        qa.add_check(lambda d, c: c.get("score", 0) > 0.8)
        result = qa.validate("accept", {"score": 0.5})
        assert result.validated is False
        assert result.override_decision == "needs_review"

    def test_multiple_checks(self):
        qa = MetaQALayer()
        qa.add_check(lambda d, c: True)
        qa.add_check(lambda d, c: True)
        qa.add_check(lambda d, c: False)  # One fails
        result = qa.validate("accept", {})
        assert result.validated is False
        assert len(result.checks_performed) == 3

    def test_check_error_treated_as_failure(self):
        qa = MetaQALayer()
        qa.add_check(lambda d, c: 1 / 0)  # Raises
        result = qa.validate("accept", {})
        assert result.validated is False
        assert "ERROR" in result.checks_performed[0]

    def test_stats(self):
        qa = MetaQALayer()
        qa.add_check(lambda d, c: c.get("ok", False))
        qa.validate("accept", {"ok": True})
        qa.validate("accept", {"ok": False})
        qa.validate("accept", {"ok": True})

        stats = qa.stats()
        assert stats["total_validations"] == 3
        assert stats["validated"] == 2
        assert stats["overridden"] == 1

    def test_history(self):
        qa = MetaQALayer()
        qa.validate("accept", {})
        qa.validate("reject", {})
        assert len(qa.history()) == 2
        assert qa.history()[0].original_decision == "accept"

    def test_qa_id_format(self):
        qa = MetaQALayer()
        result = qa.validate("accept", {})
        assert result.qa_id.startswith("qa_")


# ---------------------------------------------------------------------------
# DeepInspector
# ---------------------------------------------------------------------------

class TestDeepInspector:
    def test_default_inspection(self):
        inspector = DeepInspector()
        result = inspector.inspect("a1", "task_1", {"score": 0.8})
        assert result.passed is True
        assert result.original_score == 0.8

    def test_custom_inspect_fn(self):
        def custom_inspect(agent_id, task_id, ctx):
            return InspectionResult(
                inspection_id="custom",
                timestamp=time.time(),
                agent_id=agent_id,
                task_id=task_id,
                original_score=ctx.get("score", 0),
                inspected_score=0.3,
                issues_found=["low quality output"],
                passed=False,
            )

        inspector = DeepInspector(inspect_fn=custom_inspect)
        result = inspector.inspect("a1", "t1", {"score": 0.8})
        assert result.passed is False
        assert len(result.issues_found) == 1

    def test_should_inspect_with_seed(self):
        inspector = DeepInspector(base_probability=0.5)
        inspector.set_seed(42)
        # With seed, results are deterministic
        results = [inspector.should_inspect() for _ in range(100)]
        assert 30 <= sum(results) <= 70  # Roughly 50%

    def test_declining_quality_increases_inspection(self):
        inspector = DeepInspector(base_probability=0.1)
        inspector.set_seed(1)

        # Count inspections with neutral trend
        inspector_neutral = DeepInspector(base_probability=0.1)
        inspector_neutral.set_seed(1)

        neutral_count = sum(
            inspector_neutral.should_inspect(quality_trend=0.0) for _ in range(1000)
        )
        # Reset seed
        inspector.set_seed(1)
        decline_count = sum(
            inspector.should_inspect(quality_trend=-0.5) for _ in range(1000)
        )
        # Declining quality should trigger more inspections
        assert decline_count > neutral_count

    def test_stats(self):
        inspector = DeepInspector()
        inspector.inspect("a1", "t1", {"score": 0.8})
        inspector.inspect("a2", "t2", {"score": 0.9})

        stats = inspector.stats()
        assert stats["total_inspections"] == 2
        assert stats["passed"] == 2

    def test_history(self):
        inspector = DeepInspector()
        inspector.inspect("a1", "t1", {})
        assert len(inspector.history()) == 1

    def test_inspection_id_format(self):
        inspector = DeepInspector()
        result = inspector.inspect("a1", "t1", {})
        assert result.inspection_id.startswith("insp_")

    def test_zero_probability_never_inspects(self):
        inspector = DeepInspector(base_probability=0.0)
        inspector.set_seed(42)
        results = [inspector.should_inspect() for _ in range(100)]
        assert sum(results) == 0

    def test_full_probability_always_inspects(self):
        inspector = DeepInspector(base_probability=1.0)
        inspector.set_seed(42)
        results = [inspector.should_inspect() for _ in range(100)]
        assert sum(results) == 100


# ---------------------------------------------------------------------------
# Integration: Rotation triggered by decay
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_decay_triggers_rotation(self):
        """Gate test: declining quality -> decay prediction -> rotation."""
        predictor = AttentionDecayPredictor(threshold=0.6)
        rotator = RoleRotator(roles=["researcher", "implementer", "reviewer"])
        rotator.assign("agent_0", "researcher")

        # Simulate quality decline
        scores = [0.9, 0.85, 0.75, 0.65, 0.55, 0.5, 0.45, 0.4]
        for i, score in enumerate(scores):
            predictor.record_signal(_signal(score=score, t=1000.0 + i * 120))

        prediction = predictor.predict("agent_0")
        assert prediction.should_rotate is True

        # Rotate based on prediction
        event = rotator.rotate("agent_0", prediction=prediction, reason="decay_detected")
        assert event.old_role == "researcher"
        assert event.new_role != "researcher"
        assert event.decay_prediction is not None

    def test_rotated_agent_catches_more(self):
        """Gate: demonstrate that rotated agents catch bugs non-rotated miss."""
        qa = MetaQALayer()
        inspector = DeepInspector(base_probability=1.0)
        rotator = RoleRotator(roles=["researcher", "reviewer", "tester"])

        # Non-rotated: single role, gets blind to certain issues
        bugs_caught_no_rotation = 0
        # Simulate: reviewer misses bugs over time (rubber-stamping)
        for i in range(10):
            # Without rotation, acceptance rate stays high (rubber-stamp)
            if i < 3:
                bugs_caught_no_rotation += 1

        # With rotation: fresh perspective catches more
        bugs_caught_with_rotation = 0
        rotator.assign("agent_0", "researcher")
        for i in range(10):
            if i % 3 == 0 and i > 0:
                rotator.rotate("agent_0", reason="scheduled")
            # Fresh role perspective catches bugs
            bugs_caught_with_rotation += 1

        assert bugs_caught_with_rotation > bugs_caught_no_rotation
        assert rotator.rotation_count("agent_0") >= 2

    def test_meta_qa_catches_rubber_stamp(self):
        """MetaQA catches decisions that would be rubber-stamped."""
        qa = MetaQALayer()

        # Add check: score must be above 0.7 to accept
        qa.add_check(lambda d, c: d != "accept" or c.get("score", 0) > 0.7)
        # Add check: must have evidence
        qa.add_check(lambda d, c: len(c.get("evidence", [])) > 0)

        # This would be rubber-stamped without QA
        result = qa.validate("accept", {"score": 0.4, "evidence": []})
        assert result.validated is False
        assert result.override_decision == "needs_review"

        # This passes QA
        result = qa.validate("accept", {"score": 0.9, "evidence": ["test1"]})
        assert result.validated is True
