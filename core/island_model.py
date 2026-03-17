"""
Island Model Evolution (Sprint 29)
------------------------------------
Multiple independent tournament populations with periodic migration
of top performers between islands.

Based on:
- AlphaEvolve (DeepMind, May 2025): island model + MAP-Elites + LLM ensemble
- GigaEvo (2025): modular MAP-Elites with async evaluation
- Classical island GA: independent populations prevent premature convergence

Architecture:
  1. INIT     -> create N islands, each with its own population config
  2. EVOLVE   -> run tournament in each island independently (parallel)
  3. MIGRATE  -> top K performers from each island migrate to neighbors (ring topology)
  4. REPEAT   -> evolve again with enriched populations
  5. COMPETE  -> cross-island championship selects the global best

Key insight: independent populations explore different regions of the search
space. Migration prevents islands from getting stuck, while isolation prevents
a single dominant strategy from taking over too early.
"""

from __future__ import annotations

import copy
import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from core.sqlite_owner import SQLiteConnectionOwner

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class IslandConfig:
    """Configuration for a single island."""
    island_id: str
    population_size: int = 6
    mutation_strength: float = 1.0  # multiplier on mutation intensity


@dataclass
class Migrant:
    """A solution migrating between islands."""
    config: Dict[str, Any]
    score: float
    source_island: str


@dataclass
class IslandState:
    """Runtime state for one island."""
    island_id: str
    generation: int = 0
    best_score: float = 0.0
    best_config: Dict[str, Any] = field(default_factory=dict)
    population: List[Dict[str, Any]] = field(default_factory=list)
    scores: List[float] = field(default_factory=list)
    immigrants: List[Migrant] = field(default_factory=list)


@dataclass
class MigrationEvent:
    """Record of a single migration."""
    generation: int
    source_island: str
    target_island: str
    migrant_score: float


@dataclass
class IslandModelResult:
    """Complete result from an island model run."""
    run_id: str
    num_islands: int
    generations: int
    champion_island: str = ""
    champion_score: float = 0.0
    champion_config: Dict[str, Any] = field(default_factory=dict)
    baseline_score: float = 0.0
    island_best_scores: Dict[str, float] = field(default_factory=dict)
    migrations: List[MigrationEvent] = field(default_factory=list)
    decision: str = ""
    reason: str = ""
    elapsed_ms: float = 0.0


# ---------------------------------------------------------------------------
# Island Model Ledger
# ---------------------------------------------------------------------------

_ISLAND_SCHEMA = """
CREATE TABLE IF NOT EXISTS island_runs (
    run_id TEXT PRIMARY KEY,
    started_at REAL NOT NULL,
    completed_at REAL DEFAULT 0,
    num_islands INTEGER DEFAULT 0,
    generations INTEGER DEFAULT 0,
    champion_island TEXT DEFAULT '',
    champion_score REAL DEFAULT 0,
    champion_config_json TEXT DEFAULT '{}',
    baseline_score REAL DEFAULT 0,
    island_best_json TEXT DEFAULT '{}',
    migrations_json TEXT DEFAULT '[]',
    decision TEXT DEFAULT '',
    reason TEXT DEFAULT ''
);
"""


class IslandLedger:
    """Audit trail for island model runs."""

    def __init__(self, db_path: str | Path):
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._owner = SQLiteConnectionOwner(self._db_path)
        conn = self._owner.connect()
        conn.executescript(_ISLAND_SCHEMA)
        conn.commit()

    @property
    def _conn(self):
        return self._owner.connect()

    def record(self, result: IslandModelResult) -> None:
        """Record a completed island model run."""
        mig_data = [
            {
                "gen": m.generation,
                "src": m.source_island,
                "dst": m.target_island,
                "score": m.migrant_score,
            }
            for m in result.migrations
        ]
        self._conn.execute(
            "INSERT OR REPLACE INTO island_runs "
            "(run_id, started_at, completed_at, num_islands, generations, "
            "champion_island, champion_score, champion_config_json, "
            "baseline_score, island_best_json, migrations_json, "
            "decision, reason) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                result.run_id, time.time() - result.elapsed_ms / 1000,
                time.time(), result.num_islands, result.generations,
                result.champion_island, result.champion_score,
                json.dumps(result.champion_config, default=str),
                result.baseline_score,
                json.dumps(result.island_best_scores, default=str),
                json.dumps(mig_data, default=str),
                result.decision, result.reason,
            ),
        )
        self._conn.commit()

    def history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent island model runs."""
        rows = self._conn.execute(
            "SELECT run_id, num_islands, generations, champion_island, "
            "champion_score, baseline_score, decision, reason "
            "FROM island_runs ORDER BY completed_at DESC LIMIT ?",
            (min(limit, 500),),
        ).fetchall()
        return [
            {
                "run_id": r[0], "num_islands": r[1], "generations": r[2],
                "champion_island": r[3], "champion_score": r[4],
                "baseline_score": r[5], "decision": r[6], "reason": r[7],
            }
            for r in rows
        ]

    def close(self) -> None:
        self._owner.close()


# ---------------------------------------------------------------------------
# Island Model Runner
# ---------------------------------------------------------------------------

class IslandModelRunner:
    """Runs multiple island populations with periodic migration.

    Each island maintains its own population of configs. Every generation:
    1. Each island evaluates and selects its top configs.
    2. Top migrants from each island move to the next island (ring topology).
    3. After all generations, a cross-island championship picks the global best.
    """

    def __init__(
        self,
        num_islands: int = 4,
        population_per_island: int = 6,
        migration_count: int = 1,
        migration_interval: int = 1,
        ledger: Optional[IslandLedger] = None,
        ledger_dir: str | Path = "_evolution",
    ):
        """
        Args:
            num_islands: Number of independent islands (3-8 recommended)
            population_per_island: Configs per island per generation
            migration_count: How many top configs migrate per interval
            migration_interval: Migrate every N generations
            ledger: Optional pre-built ledger
            ledger_dir: Directory for ledger DB
        """
        self.num_islands = max(2, min(num_islands, 16))
        self.pop_size = max(2, population_per_island)
        self.migration_count = max(1, min(migration_count, self.pop_size // 2))
        self.migration_interval = max(1, migration_interval)

        ledger_path = Path(ledger_dir)
        ledger_path.mkdir(parents=True, exist_ok=True)
        self._ledger = ledger or IslandLedger(ledger_path / "island_model.db")

    def run(
        self,
        generations: int,
        baseline_config: Dict[str, Any],
        baseline_score: float,
        eval_fn: Callable[[Dict[str, Any]], float],
        mutate_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
        min_improvement: float = 0.5,
    ) -> IslandModelResult:
        """Run the island model for N generations.

        Parameters
        ----------
        generations : number of evolution generations
        baseline_config : starting configuration
        baseline_score : current eval score
        eval_fn : evaluates a config, returns score
        mutate_fn : mutates a config, returns candidate config
        min_improvement : minimum improvement over baseline to accept
        """
        t0 = time.monotonic()
        generations = max(1, min(generations, 100))

        result = IslandModelResult(
            run_id=f"island_{uuid.uuid4().hex[:12]}",
            num_islands=self.num_islands,
            generations=generations,
            baseline_score=baseline_score,
        )

        # Initialize islands
        islands = self._init_islands(baseline_config, eval_fn, mutate_fn)

        # Run generations
        for gen in range(generations):
            # Evolve each island
            for island in islands:
                self._evolve_island(
                    island, baseline_config, eval_fn, mutate_fn)
                island.generation = gen + 1

            # Migrate (ring topology)
            if (gen + 1) % self.migration_interval == 0 and gen < generations - 1:
                migrations = self._migrate_ring(islands, gen + 1)
                result.migrations.extend(migrations)

        # Cross-island championship
        champion_island = max(islands, key=lambda i: i.best_score)
        result.champion_island = champion_island.island_id
        result.champion_score = champion_island.best_score
        result.champion_config = champion_island.best_config
        result.island_best_scores = {
            i.island_id: i.best_score for i in islands
        }

        # Decision
        improvement = result.champion_score - baseline_score
        if improvement >= min_improvement:
            result.decision = "accepted"
            result.reason = (
                f"Island {champion_island.island_id} produced champion "
                f"with +{improvement:.2f} improvement"
            )
        else:
            result.decision = "rejected"
            result.reason = (
                f"Best improvement +{improvement:.2f} "
                f"< {min_improvement} required"
            )

        result.elapsed_ms = (time.monotonic() - t0) * 1000
        self._ledger.record(result)
        return result

    def _init_islands(
        self,
        baseline_config: Dict[str, Any],
        eval_fn: Callable[[Dict[str, Any]], float],
        mutate_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
    ) -> List[IslandState]:
        """Create initial island populations via mutation from baseline."""
        islands = []
        for idx in range(self.num_islands):
            island = IslandState(
                island_id=f"island_{idx:02d}",
                generation=0,
            )

            # Seed population with mutations of baseline
            for _ in range(self.pop_size):
                try:
                    candidate = mutate_fn(copy.deepcopy(baseline_config))
                except Exception:
                    candidate = copy.deepcopy(baseline_config)
                island.population.append(candidate)

            # Evaluate initial population
            for cfg in island.population:
                try:
                    score = eval_fn(cfg)
                except Exception:
                    score = 0.0
                island.scores.append(score)

            # Track best
            if island.scores:
                best_idx = max(range(len(island.scores)),
                               key=lambda i: island.scores[i])
                island.best_score = island.scores[best_idx]
                island.best_config = copy.deepcopy(
                    island.population[best_idx])

            islands.append(island)

        return islands

    def _evolve_island(
        self,
        island: IslandState,
        baseline_config: Dict[str, Any],
        eval_fn: Callable[[Dict[str, Any]], float],
        mutate_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
    ) -> None:
        """Run one generation of evolution within an island.

        Selection: keep top half + any immigrants.
        Reproduction: mutate survivors to fill remaining slots.
        """
        # Incorporate immigrants into population
        for migrant in island.immigrants:
            island.population.append(migrant.config)
            island.scores.append(migrant.score)
        island.immigrants.clear()

        # Rank and select top half
        if not island.population:
            return

        ranked = sorted(
            zip(island.scores, island.population),
            key=lambda x: x[0],
            reverse=True,
        )
        survivors = ranked[:max(1, len(ranked) // 2)]

        # Build new population from survivors + mutations
        new_pop = []
        new_scores = []
        for score, cfg in survivors:
            new_pop.append(cfg)
            new_scores.append(score)

        while len(new_pop) < self.pop_size:
            # Pick a random parent from survivors
            parent_idx = len(new_pop) % len(survivors)
            parent_cfg = survivors[parent_idx][1]
            try:
                child = mutate_fn(copy.deepcopy(parent_cfg))
            except Exception:
                child = copy.deepcopy(parent_cfg)
            try:
                child_score = eval_fn(child)
            except Exception:
                child_score = 0.0
            new_pop.append(child)
            new_scores.append(child_score)

        island.population = new_pop
        island.scores = new_scores

        # Update best
        best_idx = max(range(len(new_scores)),
                       key=lambda i: new_scores[i])
        if new_scores[best_idx] > island.best_score:
            island.best_score = new_scores[best_idx]
            island.best_config = copy.deepcopy(new_pop[best_idx])

    def _migrate_ring(
        self,
        islands: List[IslandState],
        generation: int,
    ) -> List[MigrationEvent]:
        """Migrate top configs between islands in ring topology.

        Island 0 -> Island 1 -> ... -> Island N-1 -> Island 0
        """
        migrations = []
        n = len(islands)

        # Collect migrants from each island before modifying any
        island_migrants: List[List[Migrant]] = []
        for island in islands:
            if not island.scores:
                island_migrants.append([])
                continue

            ranked = sorted(
                zip(island.scores, island.population),
                key=lambda x: x[0],
                reverse=True,
            )
            migrants = []
            for score, cfg in ranked[:self.migration_count]:
                migrants.append(Migrant(
                    config=copy.deepcopy(cfg),
                    score=score,
                    source_island=island.island_id,
                ))
            island_migrants.append(migrants)

        # Send migrants to next island in ring
        for src_idx in range(n):
            dst_idx = (src_idx + 1) % n
            for migrant in island_migrants[src_idx]:
                islands[dst_idx].immigrants.append(migrant)
                migrations.append(MigrationEvent(
                    generation=generation,
                    source_island=islands[src_idx].island_id,
                    target_island=islands[dst_idx].island_id,
                    migrant_score=migrant.score,
                ))

        return migrations

    def history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent island model run history."""
        return self._ledger.history(limit)

    def close(self) -> None:
        self._ledger.close()
