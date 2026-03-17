"""Tests for Autonomous Research Lab (Sprint 26)."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List

import pytest

from core.autonomous_research_lab import (
    AutonomousLab,
    GapDetector,
    Hypothesis,
    HypothesisEngine,
    PrototypeResult,
    PrototypeRunner,
    ResearchCycle,
    ResearchGap,
    ResearchLedger,
    ResearchStatus,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ledger(tmp_path):
    led = ResearchLedger(tmp_path / "research.db")
    yield led
    led.close()


@pytest.fixture
def lab(tmp_path):
    lab = AutonomousLab(ledger_path=tmp_path / "lab.db")
    yield lab
    lab.close()


def _failures(n: int = 5, category: str = "accuracy") -> List[Dict[str, Any]]:
    return [
        {"id": f"fail_{i}", "category": category, "score": 0.3}
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# ResearchStatus enum
# ---------------------------------------------------------------------------

class TestResearchStatus:
    def test_values(self):
        assert ResearchStatus.PROPOSED == "proposed"
        assert ResearchStatus.ACCEPTED == "accepted"
        assert ResearchStatus.REJECTED == "rejected"
        assert ResearchStatus.PROTOTYPING == "prototyping"


# ---------------------------------------------------------------------------
# ResearchGap
# ---------------------------------------------------------------------------

class TestResearchGap:
    def test_create(self):
        gap = ResearchGap(
            gap_id="gap_1", category="accuracy",
            description="Low precision on factual queries",
        )
        assert gap.severity == 0.0
        assert gap.evidence == []

    def test_full(self):
        gap = ResearchGap(
            gap_id="gap_2", category="robustness",
            description="Crashes on adversarial input",
            evidence=["fail_1", "fail_2"],
            severity=0.9, source="eval_failure",
        )
        assert len(gap.evidence) == 2


# ---------------------------------------------------------------------------
# Hypothesis
# ---------------------------------------------------------------------------

class TestHypothesis:
    def test_create(self):
        hyp = Hypothesis(
            hypothesis_id="hyp_1", gap_id="gap_1",
            title="Add re-ranking", description="test",
        )
        assert hyp.status == ResearchStatus.PROPOSED
        assert hyp.confidence == 0.0

    def test_with_inspiration(self):
        hyp = Hypothesis(
            hypothesis_id="hyp_2", gap_id="gap_1",
            title="Ensemble scoring", description="test",
            inspiration_sources=["paper_1", "pattern_analysis"],
        )
        assert len(hyp.inspiration_sources) == 2


# ---------------------------------------------------------------------------
# ResearchLedger
# ---------------------------------------------------------------------------

class TestResearchLedger:
    def test_record_gap(self, ledger):
        gap = ResearchGap(
            gap_id="gap_t1", category="accuracy",
            description="test gap", severity=0.7,
        )
        ledger.record_gap(gap)
        gaps = ledger.get_gaps()
        assert len(gaps) == 1
        assert gaps[0]["gap_id"] == "gap_t1"

    def test_record_hypothesis(self, ledger):
        hyp = Hypothesis(
            hypothesis_id="hyp_t1", gap_id="gap_1",
            title="Test hypothesis", description="desc",
            confidence=0.6,
        )
        ledger.record_hypothesis(hyp)
        hyps = ledger.get_hypotheses()
        assert len(hyps) == 1
        assert hyps[0]["title"] == "Test hypothesis"

    def test_get_hypotheses_by_status(self, ledger):
        for i, status in enumerate(["proposed", "accepted", "rejected"]):
            hyp = Hypothesis(
                hypothesis_id=f"hyp_{i}", gap_id="gap_1",
                title=f"H{i}", description="d", status=status,
            )
            ledger.record_hypothesis(hyp)

        accepted = ledger.get_hypotheses(status="accepted")
        assert len(accepted) == 1

    def test_record_prototype(self, ledger):
        proto = PrototypeResult(
            prototype_id="proto_t1", hypothesis_id="hyp_1",
            baseline_score=70.0, prototype_score=78.0,
            improvement=8.0, passed_gate=True,
        )
        ledger.record_prototype(proto)
        discoveries = ledger.get_discoveries()
        assert len(discoveries) == 0  # No matching hypothesis in DB

    def test_discoveries_with_hypothesis(self, ledger):
        hyp = Hypothesis(
            hypothesis_id="hyp_d1", gap_id="gap_1",
            title="Discovery technique", description="novel",
        )
        ledger.record_hypothesis(hyp)

        proto = PrototypeResult(
            prototype_id="proto_d1", hypothesis_id="hyp_d1",
            baseline_score=70.0, prototype_score=80.0,
            improvement=10.0, passed_gate=True,
        )
        ledger.record_prototype(proto)

        discoveries = ledger.get_discoveries()
        assert len(discoveries) == 1
        assert discoveries[0]["improvement"] == 10.0

    def test_stats_empty(self, ledger):
        s = ledger.stats()
        assert s["total_gaps"] == 0
        assert s["discoveries"] == 0

    def test_stats_populated(self, ledger):
        gap = ResearchGap(gap_id="g1", category="accuracy", description="test")
        ledger.record_gap(gap)

        hyp = Hypothesis(
            hypothesis_id="h1", gap_id="g1",
            title="test", description="d",
        )
        ledger.record_hypothesis(hyp)

        proto = PrototypeResult(
            prototype_id="p1", hypothesis_id="h1",
            baseline_score=70, prototype_score=75,
            improvement=5, passed_gate=True,
        )
        ledger.record_prototype(proto)

        s = ledger.stats()
        assert s["total_gaps"] == 1
        assert s["total_hypotheses"] == 1
        assert s["total_prototypes"] == 1
        assert s["discoveries"] == 1

    def test_record_cycle(self, ledger):
        cycle = ResearchCycle(
            cycle_id="rc_1", started_at=1000.0,
            completed_at=1010.0,
            gaps_detected=3, hypotheses_proposed=5,
            prototypes_run=2, discoveries=1,
            status="accepted",
        )
        ledger.record_cycle(cycle)
        s = ledger.stats()
        assert s["research_cycles"] == 1


# ---------------------------------------------------------------------------
# GapDetector
# ---------------------------------------------------------------------------

class TestGapDetector:
    def test_no_failures(self):
        detector = GapDetector()
        gaps = detector.detect([])
        assert gaps == []

    def test_detect_from_pattern(self):
        detector = GapDetector()
        failures = _failures(5, "accuracy")
        gaps = detector.detect(failures)
        assert len(gaps) >= 1
        assert gaps[0].category == "accuracy"

    def test_multiple_categories(self):
        detector = GapDetector()
        failures = _failures(4, "accuracy") + _failures(3, "robustness")
        gaps = detector.detect(failures)
        categories = {g.category for g in gaps}
        assert "accuracy" in categories
        assert "robustness" in categories

    def test_severity_scales_with_count(self):
        detector = GapDetector()
        small = detector.detect(_failures(3, "accuracy"))
        large = detector.detect(_failures(10, "accuracy"))
        assert large[0].severity >= small[0].severity

    def test_below_threshold_no_gap(self):
        detector = GapDetector()
        gaps = detector.detect(_failures(2, "accuracy"))  # < 3
        assert len(gaps) == 0

    def test_custom_analyzer(self):
        detector = GapDetector()

        def custom(failures):
            return [ResearchGap(
                gap_id="custom_gap", category="novel",
                description="Custom detection", severity=0.8,
            )]

        detector.add_analyzer(custom)
        gaps = detector.detect([{"id": "f1"}])
        custom_gaps = [g for g in gaps if g.gap_id == "custom_gap"]
        assert len(custom_gaps) == 1

    def test_analyzer_error_handled(self):
        detector = GapDetector()
        detector.add_analyzer(lambda f: 1 / 0)  # Raises
        gaps = detector.detect(_failures(5))  # Should not crash
        assert isinstance(gaps, list)


# ---------------------------------------------------------------------------
# HypothesisEngine
# ---------------------------------------------------------------------------

class TestHypothesisEngine:
    def test_template_proposal(self):
        engine = HypothesisEngine()
        gap = ResearchGap(
            gap_id="gap_1", category="accuracy",
            description="Low precision", severity=0.7,
        )
        hyps = engine.propose(gap)
        assert len(hyps) >= 1
        assert all(h.gap_id == "gap_1" for h in hyps)

    def test_novel_category_fallback(self):
        engine = HypothesisEngine()
        gap = ResearchGap(
            gap_id="gap_x", category="novel",
            description="Unknown gap", severity=0.5,
        )
        hyps = engine.propose(gap)
        assert len(hyps) >= 1

    def test_custom_propose_fn(self):
        def custom_propose(gap):
            return [Hypothesis(
                hypothesis_id="custom_hyp", gap_id=gap.gap_id,
                title="Custom technique", description="test",
            )]

        engine = HypothesisEngine(propose_fn=custom_propose)
        gap = ResearchGap(gap_id="g1", category="speed", description="slow")
        hyps = engine.propose(gap)
        assert len(hyps) == 1
        assert hyps[0].hypothesis_id == "custom_hyp"

    def test_expected_improvement_scales_with_severity(self):
        engine = HypothesisEngine()
        low = ResearchGap(gap_id="g1", category="accuracy", description="d", severity=0.2)
        high = ResearchGap(gap_id="g2", category="accuracy", description="d", severity=0.9)

        low_hyps = engine.propose(low)
        high_hyps = engine.propose(high)

        assert high_hyps[0].expected_improvement > low_hyps[0].expected_improvement


# ---------------------------------------------------------------------------
# PrototypeRunner
# ---------------------------------------------------------------------------

class TestPrototypeRunner:
    def test_default_no_improvement(self):
        runner = PrototypeRunner()
        hyp = Hypothesis(
            hypothesis_id="h1", gap_id="g1",
            title="test", description="d",
        )
        result = runner.run(hyp, baseline_score=70.0)
        assert result.improvement == 0.0
        assert result.passed_gate is False

    def test_custom_eval(self):
        def custom_eval(hyp, ctx):
            return PrototypeResult(
                prototype_id="proto_c",
                hypothesis_id=hyp.hypothesis_id,
                baseline_score=0,
                prototype_score=78.0,
            )

        runner = PrototypeRunner(eval_fn=custom_eval, min_improvement=0.5)
        hyp = Hypothesis(hypothesis_id="h1", gap_id="g1", title="t", description="d")
        result = runner.run(hyp, baseline_score=70.0)
        assert result.improvement == 8.0
        assert result.passed_gate is True

    def test_min_improvement_property(self):
        runner = PrototypeRunner(min_improvement=2.0)
        assert runner.min_improvement == 2.0

    def test_gate_threshold(self):
        def eval_fn(hyp, ctx):
            return PrototypeResult(
                prototype_id="p1", hypothesis_id=hyp.hypothesis_id,
                baseline_score=0, prototype_score=70.3,
            )

        runner = PrototypeRunner(eval_fn=eval_fn, min_improvement=0.5)
        hyp = Hypothesis(hypothesis_id="h1", gap_id="g1", title="t", description="d")
        result = runner.run(hyp, baseline_score=70.0)
        assert result.improvement == pytest.approx(0.3, abs=0.01)
        assert result.passed_gate is False  # 0.3 < 0.5


# ---------------------------------------------------------------------------
# AutonomousLab (integration)
# ---------------------------------------------------------------------------

class TestAutonomousLab:
    def test_cycle_no_failures(self, lab):
        cycle = lab.run_cycle([], baseline_score=70.0)
        assert cycle.gaps_detected == 0
        assert cycle.status == ResearchStatus.REJECTED

    def test_cycle_with_failures(self, lab):
        cycle = lab.run_cycle(_failures(5), baseline_score=70.0)
        assert cycle.gaps_detected >= 1
        assert cycle.hypotheses_proposed >= 1
        assert cycle.prototypes_run >= 1

    def test_cycle_with_discovery(self, tmp_path):
        """Gate: autonomous discovery of novel technique."""
        def eval_fn(hyp, ctx):
            return PrototypeResult(
                prototype_id=f"proto_{hyp.hypothesis_id}",
                hypothesis_id=hyp.hypothesis_id,
                baseline_score=0,
                prototype_score=78.0,
            )

        runner = PrototypeRunner(eval_fn=eval_fn, min_improvement=0.5)
        lab = AutonomousLab(
            ledger_path=tmp_path / "gate.db",
            prototype_runner=runner,
        )

        cycle = lab.run_cycle(_failures(5), baseline_score=70.0)
        lab.close()

        assert cycle.discoveries >= 1
        assert cycle.best_improvement >= 8.0
        assert cycle.status == ResearchStatus.ACCEPTED

    def test_get_gaps(self, lab):
        lab.run_cycle(_failures(4), baseline_score=70.0)
        gaps = lab.get_gaps()
        assert len(gaps) >= 1

    def test_get_hypotheses(self, lab):
        lab.run_cycle(_failures(4), baseline_score=70.0)
        hyps = lab.get_hypotheses()
        assert len(hyps) >= 1

    def test_stats(self, lab):
        lab.run_cycle(_failures(4), baseline_score=70.0)
        stats = lab.stats()
        assert stats["research_cycles"] == 1
        assert stats["total_gaps"] >= 1

    def test_multiple_cycles(self, lab):
        for i in range(3):
            lab.run_cycle(_failures(4 + i), baseline_score=70.0)
        stats = lab.stats()
        assert stats["research_cycles"] == 3

    def test_cycle_id_format(self, lab):
        cycle = lab.run_cycle(_failures(5), baseline_score=70.0)
        assert cycle.cycle_id.startswith("research_")

    def test_multi_category_gaps(self, lab):
        failures = _failures(4, "accuracy") + _failures(3, "robustness")
        cycle = lab.run_cycle(failures, baseline_score=70.0)
        assert cycle.gaps_detected >= 2
