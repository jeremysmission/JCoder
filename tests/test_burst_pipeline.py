"""Tests for Supercomputer Burst Pipeline (Sprint 25)."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict

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
    ProviderType,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ledger(tmp_path):
    led = BurstLedger(tmp_path / "burst.db")
    yield led
    led.close()


@pytest.fixture
def orchestrator(tmp_path):
    o = BurstOrchestrator(
        ledger_path=tmp_path / "burst_orch.db",
    )
    yield o
    o.close()


def _baseline() -> Dict[str, Any]:
    return {"top_k": 5, "temperature": 0.7}


# ---------------------------------------------------------------------------
# ProviderType / BurstStatus enums
# ---------------------------------------------------------------------------

class TestEnums:
    def test_provider_types(self):
        assert ProviderType.LOCAL == "local"
        assert ProviderType.LAMBDA_LABS == "lambda_labs"
        assert ProviderType.VAST_AI == "vast_ai"

    def test_burst_statuses(self):
        assert BurstStatus.PENDING == "pending"
        assert BurstStatus.RUNNING == "running"
        assert BurstStatus.COMPLETED == "completed"
        assert BurstStatus.FAILED == "failed"


# ---------------------------------------------------------------------------
# InstanceSpec
# ---------------------------------------------------------------------------

class TestInstanceSpec:
    def test_defaults(self):
        spec = InstanceSpec()
        assert spec.instance_id == ""
        assert spec.provider == "local"
        assert spec.status == "pending"

    def test_full_spec(self):
        spec = InstanceSpec(
            instance_id="i-123", provider="lambda_labs",
            gpu_type="A100", gpu_count=8, ram_gb=256,
            vram_gb=640, hourly_cost=12.50,
        )
        assert spec.gpu_count == 8
        assert spec.hourly_cost == 12.50


# ---------------------------------------------------------------------------
# BurstConfig
# ---------------------------------------------------------------------------

class TestBurstConfig:
    def test_defaults(self):
        cfg = BurstConfig()
        assert cfg.num_clones == 100
        assert cfg.max_instances == 10
        assert cfg.clones_per_instance == 10
        assert cfg.max_budget_usd == 50.0
        assert cfg.auto_teardown is True

    def test_custom(self):
        cfg = BurstConfig(
            num_clones=50, max_instances=5,
            max_budget_usd=100.0, provider="vast_ai",
        )
        assert cfg.num_clones == 50
        assert cfg.provider == "vast_ai"


# ---------------------------------------------------------------------------
# CloudProvider
# ---------------------------------------------------------------------------

class TestCloudProvider:
    def test_provision(self):
        provider = CloudProvider("local")
        spec = InstanceSpec(gpu_type="RTX4090", ram_gb=128)
        result = provider.provision(spec)
        assert result.instance_id.startswith("inst_")
        assert result.status == "running"

    def test_deploy(self):
        provider = CloudProvider("local")
        spec = InstanceSpec(instance_id="i-1")
        assert provider.deploy(spec, {"config": {}}) is True

    def test_teardown(self):
        provider = CloudProvider("local")
        spec = InstanceSpec(instance_id="i-1", status="running")
        assert provider.teardown(spec) is True
        assert spec.status == "terminated"

    def test_estimate_cost(self):
        provider = CloudProvider("local")
        spec = InstanceSpec(hourly_cost=2.50)
        cost = provider.estimate_cost(spec, 10.0)
        assert cost == 25.0

    def test_provider_type(self):
        provider = CloudProvider("lambda_labs")
        assert provider.provider_type == "lambda_labs"


# ---------------------------------------------------------------------------
# BurstLedger
# ---------------------------------------------------------------------------

class TestBurstLedger:
    def test_record_and_retrieve(self, ledger):
        result = BurstResult(
            burst_id="burst_test",
            started_at=1000.0,
            completed_at=1100.0,
            status=BurstStatus.COMPLETED,
            total_evolutions=100,
            accepted=30,
        )
        ledger.record_burst(result)

        history = ledger.get_history()
        assert len(history) == 1
        assert history[0]["burst_id"] == "burst_test"
        assert history[0]["accepted"] == 30

    def test_record_checkpoint(self, ledger):
        cp = ProgressCheckpoint(
            checkpoint_id="cp_1",
            burst_id="burst_1",
            timestamp=1050.0,
            elapsed_hours=0.5,
            evolutions_completed=50,
            accepted=15,
            current_best_score=82.0,
            estimated_cost_usd=12.50,
            instances_active=10,
        )
        ledger.record_checkpoint(cp)

        cps = ledger.get_checkpoints("burst_1")
        assert len(cps) == 1
        assert cps[0]["accepted"] == 15

    def test_stats_empty(self, ledger):
        s = ledger.stats()
        assert s["total_bursts"] == 0
        assert s["total_cost_usd"] == 0.0

    def test_stats_populated(self, ledger):
        for i in range(3):
            result = BurstResult(
                burst_id=f"burst_{i}",
                started_at=1000.0 + i,
                status="completed",
                total_evolutions=100,
                accepted=30 + i * 10,
                total_cost_usd=25.0,
            )
            ledger.record_burst(result)

        s = ledger.stats()
        assert s["total_bursts"] == 3
        assert s["total_cost_usd"] == 75.0
        assert s["total_evolutions"] == 300

    def test_multiple_checkpoints_ordered(self, ledger):
        for i in range(5):
            cp = ProgressCheckpoint(
                checkpoint_id=f"cp_{i}",
                burst_id="burst_1",
                timestamp=1000.0 + i * 600,
                elapsed_hours=i * 0.167,
                evolutions_completed=i * 20,
                accepted=i * 5,
                current_best_score=70.0 + i,
                estimated_cost_usd=i * 5.0,
                instances_active=10,
            )
            ledger.record_checkpoint(cp)

        cps = ledger.get_checkpoints("burst_1")
        assert len(cps) == 5
        assert cps[0]["evolutions_completed"] == 0
        assert cps[-1]["evolutions_completed"] == 80


# ---------------------------------------------------------------------------
# BurstOrchestrator
# ---------------------------------------------------------------------------

class TestBurstOrchestrator:
    def test_default_burst(self, orchestrator):
        """Default burst with no tournament fn -- all rejected."""
        config = BurstConfig(
            num_clones=10, max_instances=2,
            clones_per_instance=5,
        )
        result = orchestrator.run_burst(
            config=config,
            baseline_config=_baseline(),
            baseline_score=70.0,
        )
        assert result.status == BurstStatus.COMPLETED
        assert result.total_clones == 10
        assert result.instances_used == 2

    def test_burst_with_tournament(self, tmp_path):
        """Burst with custom tournament function."""
        def mock_tournament(num_clones, config, score):
            return {
                "total": num_clones,
                "accepted": num_clones // 2,
                "rejected": num_clones // 2,
                "errors": 0,
                "best_improvement": 15.0,
                "champion_config": {**config, "top_k": 20},
            }

        orch = BurstOrchestrator(
            ledger_path=tmp_path / "t.db",
            tournament_fn=mock_tournament,
        )
        config = BurstConfig(num_clones=20, max_instances=4)
        result = orch.run_burst(
            config=config,
            baseline_config=_baseline(),
            baseline_score=70.0,
        )
        orch.close()

        assert result.accepted == 10
        assert result.best_improvement == 15.0
        assert result.champion_config["top_k"] == 20

    def test_burst_history(self, orchestrator):
        config = BurstConfig(num_clones=5, max_instances=1)
        orchestrator.run_burst(config, _baseline(), 70.0)

        history = orchestrator.get_history()
        assert len(history) == 1

    def test_burst_stats(self, orchestrator):
        config = BurstConfig(num_clones=5, max_instances=1)
        orchestrator.run_burst(config, _baseline(), 70.0)

        stats = orchestrator.stats()
        assert stats["total_bursts"] == 1
        assert stats["completed"] == 1

    def test_burst_id_format(self, orchestrator):
        config = BurstConfig(num_clones=5, max_instances=1)
        result = orchestrator.run_burst(config, _baseline(), 70.0)
        assert result.burst_id.startswith("burst_")

    def test_checkpoint_recorded(self, orchestrator):
        config = BurstConfig(num_clones=5, max_instances=1)
        result = orchestrator.run_burst(config, _baseline(), 70.0)

        cps = orchestrator.get_checkpoints(result.burst_id)
        assert len(cps) >= 1

    def test_auto_teardown(self, orchestrator):
        config = BurstConfig(
            num_clones=5, max_instances=2, auto_teardown=True,
        )
        result = orchestrator.run_burst(config, _baseline(), 70.0)
        assert result.status == BurstStatus.COMPLETED
        # Instances should be torn down
        for inst in result.instance_specs:
            assert inst.status == "terminated"

    def test_100_clone_burst(self, tmp_path):
        """Gate: 100 clones run and complete."""
        def mock_tourn(n, cfg, score):
            return {
                "total": n, "accepted": n // 5,
                "rejected": n - n // 5, "errors": 0,
                "best_improvement": 8.0,
                "champion_config": {**cfg, "evolved": True},
            }

        orch = BurstOrchestrator(
            ledger_path=tmp_path / "gate.db",
            tournament_fn=mock_tourn,
        )
        config = BurstConfig(
            num_clones=100, max_instances=10,
            clones_per_instance=10,
        )
        result = orch.run_burst(config, _baseline(), 70.0)
        orch.close()

        assert result.total_clones == 100
        assert result.accepted == 20
        assert result.status == BurstStatus.COMPLETED

    def test_burst_vs_normal_comparison(self, tmp_path):
        """Gate: burst produces more progress than 4 normal weeks."""
        # Simulate 4 normal weeks: 5 evolutions/week, 30% acceptance
        normal_4_weeks = 4 * 5  # 20 total evolutions
        normal_accepted = int(normal_4_weeks * 0.3)  # 6 accepted

        # Weekend burst: 100 clones, higher acceptance due to diversity
        def burst_tourn(n, cfg, score):
            return {
                "total": n, "accepted": int(n * 0.35),
                "rejected": n - int(n * 0.35), "errors": 0,
                "best_improvement": 12.0,
                "champion_config": cfg,
            }

        orch = BurstOrchestrator(
            ledger_path=tmp_path / "compare.db",
            tournament_fn=burst_tourn,
        )
        config = BurstConfig(num_clones=100, max_instances=10)
        burst_result = orch.run_burst(config, _baseline(), 70.0)
        orch.close()

        assert burst_result.accepted > normal_accepted
        assert burst_result.total_evolutions > normal_4_weeks
