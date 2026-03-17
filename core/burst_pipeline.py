"""
Supercomputer Burst Pipeline (Sprint 25)
------------------------------------------
Orchestrates large-scale evolution bursts on cloud infrastructure.
Scales the tournament runner from 10x (local) to 100x (cloud) parallel
clones for weekend burst sprints.

Architecture:
  1. CloudProvider     -- abstraction for Lambda Labs / Vast.ai / local
  2. BurstConfig       -- defines burst parameters (clones, budget, timeout)
  3. BurstOrchestrator -- manages spin-up, deploy, run, collect, teardown
  4. BurstLedger       -- audit trail for burst runs with cost tracking

Gate: Weekend burst produces more evolution progress than 4 normal weeks.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from core.sqlite_owner import SQLiteConnectionOwner

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

class ProviderType(str, Enum):
    LOCAL = "local"
    LAMBDA_LABS = "lambda_labs"
    VAST_AI = "vast_ai"
    CUSTOM = "custom"


class BurstStatus(str, Enum):
    PENDING = "pending"
    PROVISIONING = "provisioning"
    DEPLOYING = "deploying"
    RUNNING = "running"
    COLLECTING = "collecting"
    TEARING_DOWN = "tearing_down"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class InstanceSpec:
    """Specification for a cloud compute instance."""
    instance_id: str = ""
    provider: str = "local"
    gpu_type: str = ""
    gpu_count: int = 0
    ram_gb: int = 0
    vram_gb: int = 0
    vcpus: int = 0
    hourly_cost: float = 0.0
    region: str = ""
    status: str = "pending"


@dataclass
class BurstConfig:
    """Configuration for a burst evolution run."""
    num_clones: int = 100
    max_instances: int = 10
    clones_per_instance: int = 10
    max_budget_usd: float = 50.0
    max_duration_hours: float = 48.0
    provider: str = "local"
    gpu_type: str = ""
    min_ram_gb: int = 64
    min_vram_gb: int = 24
    auto_teardown: bool = True
    checkpoint_interval_min: int = 30


@dataclass
class BurstResult:
    """Result from a burst evolution run."""
    burst_id: str
    started_at: float
    completed_at: float = 0.0
    status: str = BurstStatus.PENDING
    config: BurstConfig = field(default_factory=BurstConfig)
    instances_used: int = 0
    total_clones: int = 0
    total_evolutions: int = 0
    accepted: int = 0
    rejected: int = 0
    errors: int = 0
    best_improvement: float = 0.0
    total_cost_usd: float = 0.0
    champion_config: Dict[str, Any] = field(default_factory=dict)
    checkpoints: List[Dict[str, Any]] = field(default_factory=list)
    instance_specs: List[InstanceSpec] = field(default_factory=list)


@dataclass
class ProgressCheckpoint:
    """Progress checkpoint during a burst run."""
    checkpoint_id: str
    burst_id: str
    timestamp: float
    elapsed_hours: float
    evolutions_completed: int
    accepted: int
    current_best_score: float
    estimated_cost_usd: float
    instances_active: int


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_BURST_SCHEMA = """
CREATE TABLE IF NOT EXISTS burst_runs (
    burst_id TEXT PRIMARY KEY,
    started_at REAL NOT NULL,
    completed_at REAL DEFAULT 0,
    status TEXT DEFAULT 'pending',
    config_json TEXT DEFAULT '{}',
    instances_used INTEGER DEFAULT 0,
    total_clones INTEGER DEFAULT 0,
    total_evolutions INTEGER DEFAULT 0,
    accepted INTEGER DEFAULT 0,
    rejected INTEGER DEFAULT 0,
    errors INTEGER DEFAULT 0,
    best_improvement REAL DEFAULT 0,
    total_cost_usd REAL DEFAULT 0,
    champion_config_json TEXT DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS burst_checkpoints (
    checkpoint_id TEXT PRIMARY KEY,
    burst_id TEXT NOT NULL,
    timestamp REAL NOT NULL,
    elapsed_hours REAL DEFAULT 0,
    evolutions_completed INTEGER DEFAULT 0,
    accepted INTEGER DEFAULT 0,
    current_best_score REAL DEFAULT 0,
    estimated_cost_usd REAL DEFAULT 0,
    instances_active INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS burst_instances (
    instance_id TEXT PRIMARY KEY,
    burst_id TEXT NOT NULL,
    provider TEXT DEFAULT 'local',
    gpu_type TEXT DEFAULT '',
    gpu_count INTEGER DEFAULT 0,
    ram_gb INTEGER DEFAULT 0,
    vram_gb INTEGER DEFAULT 0,
    hourly_cost REAL DEFAULT 0,
    status TEXT DEFAULT 'pending',
    started_at REAL DEFAULT 0,
    stopped_at REAL DEFAULT 0
);
"""


# ---------------------------------------------------------------------------
# Cloud Provider (abstraction layer)
# ---------------------------------------------------------------------------

class CloudProvider:
    """Abstract cloud provider for compute instance management.

    Default implementation runs locally (no actual cloud provisioning).
    Subclass for Lambda Labs, Vast.ai, etc.
    """

    def __init__(self, provider_type: str = "local"):
        self._type = provider_type

    @property
    def provider_type(self) -> str:
        return self._type

    def provision(self, spec: InstanceSpec) -> InstanceSpec:
        """Provision a compute instance. Returns updated spec with instance_id."""
        spec.instance_id = f"inst_{uuid.uuid4().hex[:8]}"
        spec.provider = self._type
        spec.status = "running"
        log.info("Provisioned %s instance: %s", self._type, spec.instance_id)
        return spec

    def deploy(self, instance: InstanceSpec, payload: Dict[str, Any]) -> bool:
        """Deploy JCoder + indexes to an instance."""
        log.info("Deployed to %s", instance.instance_id)
        return True

    def teardown(self, instance: InstanceSpec) -> bool:
        """Tear down an instance."""
        instance.status = "terminated"
        log.info("Torn down %s", instance.instance_id)
        return True

    def status(self, instance: InstanceSpec) -> str:
        """Check instance status."""
        return instance.status

    def estimate_cost(self, instance: InstanceSpec, hours: float) -> float:
        """Estimate cost for running an instance for given hours."""
        return instance.hourly_cost * hours


# ---------------------------------------------------------------------------
# Burst Ledger
# ---------------------------------------------------------------------------

class BurstLedger:
    """Audit trail for burst runs."""

    def __init__(self, db_path: str | Path):
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._owner = SQLiteConnectionOwner(self._db_path)
        conn = self._owner.connect()
        conn.executescript(_BURST_SCHEMA)
        conn.commit()

    @property
    def _conn(self):
        return self._owner.connect()

    def record_burst(self, result: BurstResult) -> None:
        """Record a burst run."""
        conn = self._conn
        conn.execute(
            "INSERT OR REPLACE INTO burst_runs "
            "(burst_id, started_at, completed_at, status, config_json, "
            "instances_used, total_clones, total_evolutions, accepted, "
            "rejected, errors, best_improvement, total_cost_usd, "
            "champion_config_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                result.burst_id, result.started_at, result.completed_at,
                result.status,
                json.dumps({
                    "num_clones": result.config.num_clones,
                    "max_instances": result.config.max_instances,
                    "max_budget_usd": result.config.max_budget_usd,
                    "provider": result.config.provider,
                }, default=str),
                result.instances_used, result.total_clones,
                result.total_evolutions, result.accepted,
                result.rejected, result.errors,
                result.best_improvement, result.total_cost_usd,
                json.dumps(result.champion_config, default=str),
            ),
        )
        conn.commit()

    def record_checkpoint(self, cp: ProgressCheckpoint) -> None:
        """Record a progress checkpoint."""
        conn = self._conn
        conn.execute(
            "INSERT OR REPLACE INTO burst_checkpoints "
            "(checkpoint_id, burst_id, timestamp, elapsed_hours, "
            "evolutions_completed, accepted, current_best_score, "
            "estimated_cost_usd, instances_active) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                cp.checkpoint_id, cp.burst_id, cp.timestamp,
                cp.elapsed_hours, cp.evolutions_completed,
                cp.accepted, cp.current_best_score,
                cp.estimated_cost_usd, cp.instances_active,
            ),
        )
        conn.commit()

    def get_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent burst runs."""
        rows = self._conn.execute(
            "SELECT burst_id, started_at, completed_at, status, "
            "total_evolutions, accepted, best_improvement, total_cost_usd "
            "FROM burst_runs ORDER BY started_at DESC LIMIT ?",
            (min(limit, 200),),
        ).fetchall()
        return [
            {
                "burst_id": r[0], "started_at": r[1], "completed_at": r[2],
                "status": r[3], "total_evolutions": r[4], "accepted": r[5],
                "best_improvement": r[6], "total_cost_usd": r[7],
            }
            for r in rows
        ]

    def get_checkpoints(self, burst_id: str) -> List[Dict[str, Any]]:
        """Get checkpoints for a burst run."""
        rows = self._conn.execute(
            "SELECT * FROM burst_checkpoints WHERE burst_id=? "
            "ORDER BY timestamp LIMIT 500",
            (burst_id,),
        ).fetchall()
        return [
            {
                "checkpoint_id": r[0], "burst_id": r[1], "timestamp": r[2],
                "elapsed_hours": r[3], "evolutions_completed": r[4],
                "accepted": r[5], "current_best_score": r[6],
                "estimated_cost_usd": r[7], "instances_active": r[8],
            }
            for r in rows
        ]

    def stats(self) -> Dict[str, Any]:
        """Aggregate burst statistics."""
        conn = self._conn
        total = conn.execute("SELECT COUNT(*) FROM burst_runs").fetchone()[0]
        completed = conn.execute(
            "SELECT COUNT(*) FROM burst_runs WHERE status='completed'"
        ).fetchone()[0]
        total_cost = conn.execute(
            "SELECT COALESCE(SUM(total_cost_usd), 0) FROM burst_runs"
        ).fetchone()[0]
        total_evolutions = conn.execute(
            "SELECT COALESCE(SUM(total_evolutions), 0) FROM burst_runs"
        ).fetchone()[0]
        total_accepted = conn.execute(
            "SELECT COALESCE(SUM(accepted), 0) FROM burst_runs"
        ).fetchone()[0]

        return {
            "total_bursts": total,
            "completed": completed,
            "total_cost_usd": round(total_cost, 2),
            "total_evolutions": total_evolutions,
            "total_accepted": total_accepted,
            "acceptance_rate": round(
                total_accepted / max(total_evolutions, 1), 3
            ),
        }

    def close(self) -> None:
        self._owner.close()


# ---------------------------------------------------------------------------
# Burst Orchestrator
# ---------------------------------------------------------------------------

class BurstOrchestrator:
    """Orchestrates large-scale evolution bursts.

    Manages the full lifecycle:
    provision -> deploy -> run tournaments -> checkpoint -> collect -> teardown
    """

    def __init__(
        self,
        provider: Optional[CloudProvider] = None,
        ledger: Optional[BurstLedger] = None,
        ledger_path: str | Path = "_evolution/burst.db",
        tournament_fn: Optional[Callable[[int, Dict, float], Dict]] = None,
    ):
        self._provider = provider or CloudProvider("local")
        self._ledger = ledger or BurstLedger(ledger_path)
        self._tournament_fn = tournament_fn

    def run_burst(
        self,
        config: BurstConfig,
        baseline_config: Dict[str, Any],
        baseline_score: float,
    ) -> BurstResult:
        """Execute a burst evolution run."""
        result = BurstResult(
            burst_id=f"burst_{uuid.uuid4().hex[:12]}",
            started_at=time.time(),
            config=config,
        )

        try:
            # Phase 1: Provision instances
            result.status = BurstStatus.PROVISIONING
            instances = self._provision_instances(config)
            result.instances_used = len(instances)
            result.instance_specs = instances

            if not instances:
                result.status = BurstStatus.FAILED
                result.completed_at = time.time()
                self._ledger.record_burst(result)
                return result

            # Phase 2: Deploy
            result.status = BurstStatus.DEPLOYING
            deployed = self._deploy_all(instances, baseline_config)
            if not deployed:
                result.status = BurstStatus.FAILED
                result.completed_at = time.time()
                self._ledger.record_burst(result)
                return result

            # Phase 3: Run tournaments
            result.status = BurstStatus.RUNNING
            tournament_results = self._run_tournaments(
                config, baseline_config, baseline_score,
            )
            result.total_clones = config.num_clones
            result.total_evolutions = tournament_results.get("total", 0)
            result.accepted = tournament_results.get("accepted", 0)
            result.rejected = tournament_results.get("rejected", 0)
            result.errors = tournament_results.get("errors", 0)
            result.best_improvement = tournament_results.get("best_improvement", 0.0)
            result.champion_config = tournament_results.get("champion_config", {})

            # Phase 4: Collect & checkpoint
            result.status = BurstStatus.COLLECTING
            self._record_checkpoint(result)

            # Phase 5: Teardown
            if config.auto_teardown:
                result.status = BurstStatus.TEARING_DOWN
                self._teardown_all(instances)

            result.status = BurstStatus.COMPLETED
            result.total_cost_usd = self._estimate_total_cost(instances, result)

        except Exception as exc:
            result.status = BurstStatus.FAILED
            log.warning("Burst failed: %s", exc)

        result.completed_at = time.time()
        self._ledger.record_burst(result)
        return result

    def _provision_instances(self, config: BurstConfig) -> List[InstanceSpec]:
        """Provision compute instances."""
        instances = []
        for i in range(config.max_instances):
            spec = InstanceSpec(
                gpu_type=config.gpu_type,
                ram_gb=config.min_ram_gb,
                vram_gb=config.min_vram_gb,
            )
            provisioned = self._provider.provision(spec)
            instances.append(provisioned)
        return instances

    def _deploy_all(
        self,
        instances: List[InstanceSpec],
        config: Dict[str, Any],
    ) -> bool:
        """Deploy payload to all instances."""
        payload = {"config": config, "timestamp": time.time()}
        success = True
        for inst in instances:
            if not self._provider.deploy(inst, payload):
                log.warning("Deploy failed for %s", inst.instance_id)
                success = False
        return success

    def _run_tournaments(
        self,
        config: BurstConfig,
        baseline_config: Dict[str, Any],
        baseline_score: float,
    ) -> Dict[str, Any]:
        """Run tournament evolution across instances."""
        if self._tournament_fn:
            return self._tournament_fn(
                config.num_clones, baseline_config, baseline_score,
            )

        # Default: simulate tournament results
        return {
            "total": config.num_clones,
            "accepted": 0,
            "rejected": config.num_clones,
            "errors": 0,
            "best_improvement": 0.0,
            "champion_config": baseline_config,
        }

    def _record_checkpoint(self, result: BurstResult) -> None:
        """Record a progress checkpoint."""
        cp = ProgressCheckpoint(
            checkpoint_id=f"cp_{uuid.uuid4().hex[:8]}",
            burst_id=result.burst_id,
            timestamp=time.time(),
            elapsed_hours=(time.time() - result.started_at) / 3600.0,
            evolutions_completed=result.total_evolutions,
            accepted=result.accepted,
            current_best_score=result.best_improvement,
            estimated_cost_usd=result.total_cost_usd,
            instances_active=result.instances_used,
        )
        result.checkpoints.append({
            "checkpoint_id": cp.checkpoint_id,
            "elapsed_hours": cp.elapsed_hours,
            "evolutions": cp.evolutions_completed,
        })
        self._ledger.record_checkpoint(cp)

    def _teardown_all(self, instances: List[InstanceSpec]) -> None:
        """Tear down all instances."""
        for inst in instances:
            self._provider.teardown(inst)

    def _estimate_total_cost(
        self,
        instances: List[InstanceSpec],
        result: BurstResult,
    ) -> float:
        """Estimate total cost for the burst run."""
        hours = (result.completed_at - result.started_at) / 3600.0
        return sum(
            self._provider.estimate_cost(inst, hours)
            for inst in instances
        )

    def cancel(self, burst_id: str) -> None:
        """Cancel a running burst (placeholder for future use)."""
        log.info("Burst %s cancellation requested", burst_id)

    def get_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get burst history."""
        return self._ledger.get_history(limit=limit)

    def get_checkpoints(self, burst_id: str) -> List[Dict[str, Any]]:
        """Get checkpoints for a burst."""
        return self._ledger.get_checkpoints(burst_id)

    def stats(self) -> Dict[str, Any]:
        """Get burst statistics."""
        return self._ledger.stats()

    def close(self) -> None:
        self._ledger.close()
