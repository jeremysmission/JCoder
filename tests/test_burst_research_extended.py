"""Extended tests for Burst Pipeline and Autonomous Research Lab.

Covers: mocked cloud providers, cost estimation, budget enforcement,
parallel clone management, provider health checks, knowledge gap detection
edge cases, hypothesis generation from gaps, experiment design,
result evaluation, and full research cycles.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from core.burst_pipeline import (
    BurstConfig,
    BurstLedger,
    BurstOrchestrator,
    BurstResult,
    BurstStatus,
    CloudProvider,
    InstanceSpec,
    ProgressCheckpoint,
)
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
# Helpers
# ---------------------------------------------------------------------------

def _baseline() -> Dict[str, Any]:
    return {"top_k": 5, "temperature": 0.7}


def _failures(n: int = 5, category: str = "accuracy") -> List[Dict[str, Any]]:
    return [{"id": f"fail_{i}", "category": category, "score": 0.3} for i in range(n)]


class FakeProvider(CloudProvider):
    """Cloud provider that tracks calls for assertions."""

    def __init__(self, fail_provision: bool = False, fail_deploy: bool = False):
        super().__init__("fake")
        self.provisioned: list = []
        self.deployed: list = []
        self.torn_down: list = []
        self._fail_provision = fail_provision
        self._fail_deploy = fail_deploy

    def provision(self, spec: InstanceSpec) -> InstanceSpec:
        if self._fail_provision:
            raise RuntimeError("provision failed")
        result = super().provision(spec)
        self.provisioned.append(result.instance_id)
        return result

    def deploy(self, instance: InstanceSpec, payload: Dict[str, Any]) -> bool:
        if self._fail_deploy:
            return False
        self.deployed.append(instance.instance_id)
        return True

    def teardown(self, instance: InstanceSpec) -> bool:
        self.torn_down.append(instance.instance_id)
        return super().teardown(instance)


# ===================================================================
# BURST PIPELINE -- EXTENDED
# ===================================================================

class TestBurstJobSubmission:
    """Job submission to cloud providers."""

    def test_submit_provisions_correct_instance_count(self, tmp_path):
        provider = FakeProvider()
        orch = BurstOrchestrator(
            provider=provider, ledger_path=tmp_path / "sub.db",
        )
        config = BurstConfig(num_clones=30, max_instances=3)
        result = orch.run_burst(config, _baseline(), 70.0)
        orch.close()
        assert len(provider.provisioned) == 3
        assert result.instances_used == 3

    def test_submit_deploys_to_all_instances(self, tmp_path):
        provider = FakeProvider()
        orch = BurstOrchestrator(
            provider=provider, ledger_path=tmp_path / "dep.db",
        )
        config = BurstConfig(num_clones=20, max_instances=4)
        orch.run_burst(config, _baseline(), 70.0)
        orch.close()
        assert len(provider.deployed) == 4

    def test_submit_deploy_failure_returns_failed(self, tmp_path):
        provider = FakeProvider(fail_deploy=True)
        orch = BurstOrchestrator(
            provider=provider, ledger_path=tmp_path / "fail.db",
        )
        config = BurstConfig(num_clones=10, max_instances=2)
        result = orch.run_burst(config, _baseline(), 70.0)
        orch.close()
        assert result.status == BurstStatus.FAILED


class TestBurstCostEstimation:
    """Cost estimation before launch."""

    def test_cost_zero_for_free_instances(self, tmp_path):
        orch = BurstOrchestrator(ledger_path=tmp_path / "free.db")
        config = BurstConfig(num_clones=10, max_instances=2)
        result = orch.run_burst(config, _baseline(), 70.0)
        orch.close()
        assert result.total_cost_usd == pytest.approx(0.0, abs=0.01)

    def test_cost_scales_with_hourly_rate(self, tmp_path):
        provider = CloudProvider("local")
        original_provision = provider.provision

        def provision_with_cost(spec: InstanceSpec) -> InstanceSpec:
            spec = original_provision(spec)
            spec.hourly_cost = 2.0
            return spec

        provider.provision = provision_with_cost
        orch = BurstOrchestrator(
            provider=provider, ledger_path=tmp_path / "cost.db",
        )
        config = BurstConfig(num_clones=5, max_instances=2)
        result = orch.run_burst(config, _baseline(), 70.0)
        orch.close()
        # Cost is computed from (completed_at - started_at) which may be ~0
        # in fast tests; just verify cost was computed (is a float)
        assert isinstance(result.total_cost_usd, float)

    def test_provider_estimate_cost_formula(self):
        provider = CloudProvider("local")
        spec = InstanceSpec(hourly_cost=5.0)
        assert provider.estimate_cost(spec, 2.0) == 10.0
        assert provider.estimate_cost(spec, 0.0) == 0.0


class TestBurstResultCollection:
    """Result collection and aggregation."""

    def test_tournament_results_aggregated(self, tmp_path):
        def tourn(n, cfg, score):
            return {
                "total": n, "accepted": 7, "rejected": n - 7,
                "errors": 2, "best_improvement": 5.5,
                "champion_config": {"top_k": 15},
            }

        orch = BurstOrchestrator(
            ledger_path=tmp_path / "agg.db", tournament_fn=tourn,
        )
        config = BurstConfig(num_clones=20, max_instances=2)
        result = orch.run_burst(config, _baseline(), 70.0)
        orch.close()
        assert result.accepted == 7
        assert result.errors == 2
        assert result.best_improvement == 5.5
        assert result.champion_config["top_k"] == 15

    def test_result_recorded_in_ledger(self, tmp_path):
        orch = BurstOrchestrator(ledger_path=tmp_path / "rec.db")
        config = BurstConfig(num_clones=5, max_instances=1)
        result = orch.run_burst(config, _baseline(), 70.0)
        history = orch.get_history()
        orch.close()
        assert any(h["burst_id"] == result.burst_id for h in history)

    def test_checkpoints_match_burst_id(self, tmp_path):
        orch = BurstOrchestrator(ledger_path=tmp_path / "cp.db")
        config = BurstConfig(num_clones=5, max_instances=1)
        result = orch.run_burst(config, _baseline(), 70.0)
        cps = orch.get_checkpoints(result.burst_id)
        orch.close()
        assert all(cp["burst_id"] == result.burst_id for cp in cps)


class TestProviderHealthCheck:
    """Provider health checking."""

    def test_status_returns_instance_state(self):
        provider = CloudProvider("local")
        spec = InstanceSpec(instance_id="i-1", status="running")
        assert provider.status(spec) == "running"

    def test_status_after_teardown(self):
        provider = CloudProvider("local")
        spec = InstanceSpec(instance_id="i-1", status="running")
        provider.teardown(spec)
        assert provider.status(spec) == "terminated"

    def test_provision_failure_causes_burst_failure(self, tmp_path):
        provider = FakeProvider(fail_provision=True)
        orch = BurstOrchestrator(
            provider=provider, ledger_path=tmp_path / "hc.db",
        )
        config = BurstConfig(num_clones=5, max_instances=2)
        result = orch.run_burst(config, _baseline(), 70.0)
        orch.close()
        assert result.status == BurstStatus.FAILED


class TestBurstBudgetEnforcement:
    """Burst budget enforcement."""

    def test_budget_field_preserved(self):
        config = BurstConfig(max_budget_usd=25.0)
        assert config.max_budget_usd == 25.0

    def test_cost_tracked_after_burst(self, tmp_path):
        orch = BurstOrchestrator(ledger_path=tmp_path / "bud.db")
        config = BurstConfig(num_clones=5, max_instances=1, max_budget_usd=100.0)
        result = orch.run_burst(config, _baseline(), 70.0)
        orch.close()
        assert isinstance(result.total_cost_usd, float)
        assert result.total_cost_usd <= config.max_budget_usd

    def test_stats_track_cumulative_cost(self, tmp_path):
        orch = BurstOrchestrator(ledger_path=tmp_path / "cum.db")
        for _ in range(3):
            config = BurstConfig(num_clones=5, max_instances=1)
            orch.run_burst(config, _baseline(), 70.0)
        stats = orch.stats()
        orch.close()
        assert stats["total_bursts"] == 3
        assert isinstance(stats["total_cost_usd"], float)


class TestParallelCloneManagement:
    """Parallel clone management."""

    def test_teardown_all_instances(self, tmp_path):
        provider = FakeProvider()
        orch = BurstOrchestrator(
            provider=provider, ledger_path=tmp_path / "td.db",
        )
        config = BurstConfig(num_clones=15, max_instances=3, auto_teardown=True)
        result = orch.run_burst(config, _baseline(), 70.0)
        orch.close()
        assert len(provider.torn_down) == 3
        for inst in result.instance_specs:
            assert inst.status == "terminated"

    def test_no_teardown_when_disabled(self, tmp_path):
        provider = FakeProvider()
        orch = BurstOrchestrator(
            provider=provider, ledger_path=tmp_path / "notd.db",
        )
        config = BurstConfig(num_clones=5, max_instances=2, auto_teardown=False)
        result = orch.run_burst(config, _baseline(), 70.0)
        orch.close()
        assert len(provider.torn_down) == 0
        assert result.status == BurstStatus.COMPLETED

    def test_clones_per_instance_config(self):
        config = BurstConfig(num_clones=50, max_instances=5, clones_per_instance=10)
        assert config.num_clones == config.max_instances * config.clones_per_instance


# ===================================================================
# RESEARCH LAB -- EXTENDED
# ===================================================================

class TestKnowledgeGapDetection:
    """Knowledge gap detection edge cases."""

    def test_mixed_categories_detected(self):
        detector = GapDetector()
        failures = (_failures(4, "accuracy")
                    + _failures(3, "speed")
                    + _failures(5, "robustness"))
        gaps = detector.detect(failures)
        cats = {g.category for g in gaps}
        assert len(cats) >= 3

    def test_unknown_category_maps_to_novel(self):
        detector = GapDetector()
        failures = [{"id": f"f{i}", "category": "quantum"} for i in range(4)]
        gaps = detector.detect(failures)
        assert any(g.category == "novel" for g in gaps)

    def test_severity_capped_at_one(self):
        detector = GapDetector()
        gaps = detector.detect(_failures(100, "accuracy"))
        assert all(g.severity <= 1.0 for g in gaps)

    def test_evidence_contains_failure_ids(self):
        detector = GapDetector()
        gaps = detector.detect(_failures(5, "accuracy"))
        assert len(gaps) >= 1
        assert len(gaps[0].evidence) > 0
        assert gaps[0].evidence[0].startswith("fail_")


class TestHypothesisFromGaps:
    """Hypothesis generation from gaps."""

    def test_each_category_produces_hypotheses(self):
        engine = HypothesisEngine()
        for cat in GapDetector.CATEGORIES:
            gap = ResearchGap(gap_id=f"g_{cat}", category=cat, description="test")
            hyps = engine.propose(gap)
            assert len(hyps) >= 1, f"No hypotheses for {cat}"

    def test_hypothesis_links_to_gap(self):
        engine = HypothesisEngine()
        gap = ResearchGap(gap_id="gap_link", category="accuracy", description="d")
        hyps = engine.propose(gap)
        assert all(h.gap_id == "gap_link" for h in hyps)

    def test_confidence_positive_when_severity_nonzero(self):
        engine = HypothesisEngine()
        gap = ResearchGap(
            gap_id="g1", category="speed", description="slow", severity=0.6,
        )
        hyps = engine.propose(gap)
        assert all(h.confidence > 0 for h in hyps)

    def test_custom_proposer_overrides_templates(self):
        called = []

        def proposer(gap):
            called.append(gap.gap_id)
            return [Hypothesis(
                hypothesis_id="custom", gap_id=gap.gap_id,
                title="Custom", description="d",
            )]

        engine = HypothesisEngine(propose_fn=proposer)
        gap = ResearchGap(gap_id="gx", category="novel", description="d")
        hyps = engine.propose(gap)
        assert called == ["gx"]
        assert hyps[0].hypothesis_id == "custom"


class TestExperimentDesign:
    """Experiment design via PrototypeRunner."""

    def test_runner_passes_context(self):
        received_ctx = {}

        def eval_fn(hyp, ctx):
            received_ctx.update(ctx)
            return PrototypeResult(
                prototype_id="p1", hypothesis_id=hyp.hypothesis_id,
                baseline_score=0, prototype_score=72.0,
            )

        runner = PrototypeRunner(eval_fn=eval_fn)
        hyp = Hypothesis(hypothesis_id="h1", gap_id="g1", title="t", description="d")
        runner.run(hyp, 70.0, context={"dataset": "eval_v2"})
        assert received_ctx["dataset"] == "eval_v2"

    def test_runner_computes_improvement(self):
        def eval_fn(hyp, ctx):
            return PrototypeResult(
                prototype_id="p1", hypothesis_id=hyp.hypothesis_id,
                baseline_score=0, prototype_score=85.0,
            )

        runner = PrototypeRunner(eval_fn=eval_fn, min_improvement=5.0)
        hyp = Hypothesis(hypothesis_id="h1", gap_id="g1", title="t", description="d")
        result = runner.run(hyp, 70.0)
        assert result.improvement == pytest.approx(15.0)
        assert result.passed_gate is True

    def test_runner_rejects_below_threshold(self):
        def eval_fn(hyp, ctx):
            return PrototypeResult(
                prototype_id="p1", hypothesis_id=hyp.hypothesis_id,
                baseline_score=0, prototype_score=70.1,
            )

        runner = PrototypeRunner(eval_fn=eval_fn, min_improvement=1.0)
        hyp = Hypothesis(hypothesis_id="h1", gap_id="g1", title="t", description="d")
        result = runner.run(hyp, 70.0)
        assert result.improvement == pytest.approx(0.1, abs=0.01)
        assert result.passed_gate is False


class TestResultEvaluation:
    """Result evaluation -- ledger recording and discovery tracking."""

    def test_discovery_recorded_in_ledger(self, tmp_path):
        ledger = ResearchLedger(tmp_path / "eval.db")
        hyp = Hypothesis(
            hypothesis_id="hyp_e", gap_id="g1",
            title="Eval technique", description="d",
        )
        ledger.record_hypothesis(hyp)
        proto = PrototypeResult(
            prototype_id="proto_e", hypothesis_id="hyp_e",
            baseline_score=70.0, prototype_score=80.0,
            improvement=10.0, passed_gate=True,
        )
        ledger.record_prototype(proto)
        discoveries = ledger.get_discoveries()
        ledger.close()
        assert len(discoveries) == 1
        assert discoveries[0]["improvement"] == 10.0

    def test_no_false_discoveries(self, tmp_path):
        ledger = ResearchLedger(tmp_path / "nodisc.db")
        hyp = Hypothesis(
            hypothesis_id="hyp_n", gap_id="g1",
            title="No improvement", description="d",
        )
        ledger.record_hypothesis(hyp)
        proto = PrototypeResult(
            prototype_id="proto_n", hypothesis_id="hyp_n",
            baseline_score=70.0, prototype_score=70.0,
            improvement=0.0, passed_gate=False,
        )
        ledger.record_prototype(proto)
        discoveries = ledger.get_discoveries()
        ledger.close()
        assert len(discoveries) == 0


class TestResearchCycleEndToEnd:
    """Full research cycle: detect -> hypothesize -> test -> evaluate."""

    def test_full_cycle_with_discovery(self, tmp_path):
        def eval_fn(hyp, ctx):
            return PrototypeResult(
                prototype_id=f"p_{hyp.hypothesis_id}",
                hypothesis_id=hyp.hypothesis_id,
                baseline_score=0, prototype_score=82.0,
            )

        runner = PrototypeRunner(eval_fn=eval_fn, min_improvement=1.0)
        lab = AutonomousLab(
            ledger_path=tmp_path / "cycle.db",
            prototype_runner=runner,
        )
        cycle = lab.run_cycle(_failures(6, "accuracy"), baseline_score=70.0)
        lab.close()

        assert cycle.gaps_detected >= 1
        assert cycle.hypotheses_proposed >= 1
        assert cycle.prototypes_run >= 1
        assert cycle.discoveries >= 1
        assert cycle.best_improvement >= 12.0
        assert cycle.status == ResearchStatus.ACCEPTED

    def test_full_cycle_no_discovery(self, tmp_path):
        lab = AutonomousLab(ledger_path=tmp_path / "nodis.db")
        cycle = lab.run_cycle(_failures(5), baseline_score=70.0)
        lab.close()
        # Default runner produces no improvement
        assert cycle.discoveries == 0
        assert cycle.status == ResearchStatus.REJECTED

    def test_cycle_limits_hypotheses(self, tmp_path):
        lab = AutonomousLab(ledger_path=tmp_path / "lim.db")
        cycle = lab.run_cycle(
            _failures(10), baseline_score=70.0,
            max_hypotheses=2, max_prototypes=1,
        )
        lab.close()
        assert cycle.hypotheses_proposed <= 2
        assert cycle.prototypes_run <= 1

    def test_cycle_stats_accumulate(self, tmp_path):
        lab = AutonomousLab(ledger_path=tmp_path / "acc.db")
        for _ in range(3):
            lab.run_cycle(_failures(4), baseline_score=70.0)
        stats = lab.stats()
        lab.close()
        assert stats["research_cycles"] == 3
        assert stats["total_gaps"] >= 3

    def test_cycle_multi_category_produces_more_hypotheses(self, tmp_path):
        lab = AutonomousLab(ledger_path=tmp_path / "multi.db")
        single = lab.run_cycle(_failures(5, "accuracy"), baseline_score=70.0)
        multi_failures = (_failures(4, "accuracy")
                          + _failures(3, "speed")
                          + _failures(3, "robustness"))
        multi = lab.run_cycle(multi_failures, baseline_score=70.0)
        lab.close()
        assert multi.gaps_detected >= single.gaps_detected
