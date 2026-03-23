"""
VM Tournament Mode (Sprint 22)
-------------------------------
Scales the Evolution Runner to N parallel isolated clones with
tournament selection. Each clone runs an independent evolution cycle,
then a tournament selects the champion.

Architecture:
  1. SPAWN   -> create N isolated clones (worktrees or in-memory configs)
  2. EVOLVE  -> run evolution cycle in each clone (parallel ThreadPool)
  3. RANK    -> score all clones, rank by candidate score
  4. SELECT  -> tournament bracket: top half advances, repeat until champion
  5. ARCHIVE -> record full tournament in ledger

Gate: 10 clones run in parallel, champion selected, decision documented.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from core.evolution_runner import (
    EvolutionCycle,
    EvolutionDecision,
    EvolutionLedger,
    EvolutionRunner,
)
from core.sqlite_owner import SQLiteConnectionOwner

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class CloneResult:
    """Result from a single clone's evolution cycle."""
    clone_id: str
    cycle: EvolutionCycle
    rank: int = 0


@dataclass
class TournamentRound:
    """Record of one tournament elimination round."""
    round_num: int
    matchups: List[Dict[str, Any]] = field(default_factory=list)
    survivors: List[str] = field(default_factory=list)
    eliminated: List[str] = field(default_factory=list)


@dataclass
class TournamentResult:
    """Complete tournament outcome."""
    tournament_id: str
    started_at: float
    completed_at: float = 0.0
    num_clones: int = 0
    champion_id: str = ""
    champion_score: float = 0.0
    champion_config: Dict[str, Any] = field(default_factory=dict)
    baseline_score: float = 0.0
    rounds: List[TournamentRound] = field(default_factory=list)
    clone_results: List[CloneResult] = field(default_factory=list)
    decision: str = ""
    reason: str = ""


# ---------------------------------------------------------------------------
# Tournament Ledger (extends evolution ledger)
# ---------------------------------------------------------------------------

_TOURNAMENT_SCHEMA = """
CREATE TABLE IF NOT EXISTS tournaments (
    tournament_id TEXT PRIMARY KEY,
    started_at REAL NOT NULL,
    completed_at REAL DEFAULT 0,
    num_clones INTEGER DEFAULT 0,
    champion_id TEXT DEFAULT '',
    champion_score REAL DEFAULT 0,
    champion_config_json TEXT DEFAULT '{}',
    baseline_score REAL DEFAULT 0,
    rounds_json TEXT DEFAULT '[]',
    decision TEXT DEFAULT '',
    reason TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS clone_results (
    clone_id TEXT PRIMARY KEY,
    tournament_id TEXT NOT NULL,
    cycle_id TEXT NOT NULL,
    candidate_score REAL DEFAULT 0,
    decision TEXT DEFAULT '',
    rank INTEGER DEFAULT 0,
    config_json TEXT DEFAULT '{}'
);
"""


class TournamentLedger:
    """Audit trail for tournament runs."""

    def __init__(self, db_path: str | Path):
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._owner = SQLiteConnectionOwner(self._db_path)
        conn = self._owner.connect()
        conn.executescript(_TOURNAMENT_SCHEMA)
        conn.commit()

    @property
    def _conn(self):
        return self._owner.connect()

    def record_tournament(self, result: TournamentResult) -> None:
        """Record a complete tournament."""
        conn = self._conn
        rounds_data = []
        for r in result.rounds:
            rounds_data.append({
                "round_num": r.round_num,
                "matchups": r.matchups,
                "survivors": r.survivors,
                "eliminated": r.eliminated,
            })

        conn.execute(
            "INSERT OR REPLACE INTO tournaments "
            "(tournament_id, started_at, completed_at, num_clones, "
            "champion_id, champion_score, champion_config_json, "
            "baseline_score, rounds_json, decision, reason) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                result.tournament_id, result.started_at, result.completed_at,
                result.num_clones, result.champion_id, result.champion_score,
                json.dumps(result.champion_config, default=str),
                result.baseline_score,
                json.dumps(rounds_data, default=str),
                result.decision, result.reason,
            ),
        )

        for cr in result.clone_results:
            conn.execute(
                "INSERT OR REPLACE INTO clone_results "
                "(clone_id, tournament_id, cycle_id, candidate_score, "
                "decision, rank, config_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    cr.clone_id, result.tournament_id, cr.cycle.cycle_id,
                    cr.cycle.candidate_score, cr.cycle.decision,
                    cr.rank,
                    json.dumps(cr.cycle.candidate_config, default=str),
                ),
            )

        conn.commit()

    def get_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent tournament results."""
        rows = self._conn.execute(
            "SELECT tournament_id, started_at, completed_at, num_clones, "
            "champion_id, champion_score, baseline_score, decision, reason "
            "FROM tournaments ORDER BY started_at DESC LIMIT ?",
            (min(limit, 500),),
        ).fetchall()
        return [
            {
                "tournament_id": r[0], "started_at": r[1],
                "completed_at": r[2], "num_clones": r[3],
                "champion_id": r[4], "champion_score": r[5],
                "baseline_score": r[6], "decision": r[7], "reason": r[8],
            }
            for r in rows
        ]

    def get_clone_results(self, tournament_id: str) -> List[Dict[str, Any]]:
        """Get all clone results for a tournament."""
        rows = self._conn.execute(
            "SELECT * FROM clone_results WHERE tournament_id=? ORDER BY rank",
            (tournament_id,),
        ).fetchall()
        return [
            {
                "clone_id": r[0], "tournament_id": r[1], "cycle_id": r[2],
                "candidate_score": r[3], "decision": r[4], "rank": r[5],
                "config": json.loads(r[6] or "{}"),
            }
            for r in rows
        ]

    def stats(self) -> Dict[str, Any]:
        """Aggregate tournament statistics."""
        conn = self._conn
        total = (conn.execute("SELECT COUNT(*) FROM tournaments").fetchone() or (0,))[0]
        accepted = (conn.execute(
            "SELECT COUNT(*) FROM tournaments WHERE decision='accepted'"
        ).fetchone() or (0,))[0]

        avg_clones = (conn.execute(
            "SELECT AVG(num_clones) FROM tournaments"
        ).fetchone() or (0,))[0]

        return {
            "total_tournaments": total,
            "accepted": accepted,
            "avg_clones": round(avg_clones or 0.0, 1),
        }

    def close(self) -> None:
        self._owner.close()


# ---------------------------------------------------------------------------
# Tournament Runner
# ---------------------------------------------------------------------------

class TournamentRunner:
    """Runs N parallel evolution clones with tournament selection.

    Each clone gets a distinct mutation of the baseline config.
    All clones are evaluated in parallel via ThreadPoolExecutor.
    Tournament bracket eliminates until a single champion remains.
    Champion must beat the original baseline to be accepted.
    """

    def __init__(
        self,
        evolution_ledger: Optional[EvolutionLedger] = None,
        tournament_ledger: Optional[TournamentLedger] = None,
        ledger_dir: str | Path = "_evolution",
        repo_root: Optional[Path] = None,
        max_workers: int = 4,
    ):
        self._ledger_dir = Path(ledger_dir)
        self._ledger_dir.mkdir(parents=True, exist_ok=True)

        self._evo_ledger = evolution_ledger or EvolutionLedger(
            self._ledger_dir / "evolution.db"
        )
        self._tourn_ledger = tournament_ledger or TournamentLedger(
            self._ledger_dir / "tournament.db"
        )
        self._repo_root = repo_root or Path.cwd()
        self._max_workers = max_workers

    def run_tournament(
        self,
        num_clones: int,
        baseline_config: Dict[str, Any],
        baseline_score: float,
        eval_fn: Callable[[Dict[str, Any]], float],
        mutate_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
        regression_fn: Optional[Callable[[float, float], bool]] = None,
        min_improvement: float = 0.5,
    ) -> TournamentResult:
        """Run a full tournament with N clones.

        Parameters
        ----------
        num_clones : number of parallel evolution clones (e.g. 10)
        baseline_config : starting configuration
        baseline_score : current eval score
        eval_fn : evaluates a config, returns score
        mutate_fn : mutates a config, returns candidate config
        regression_fn : optional regression gate
        min_improvement : minimum improvement over baseline to accept champion
        """
        num_clones = max(2, min(num_clones, 100))

        result = TournamentResult(
            tournament_id=f"tourn_{uuid.uuid4().hex[:12]}",
            started_at=time.time(),
            num_clones=num_clones,
            baseline_score=baseline_score,
        )

        # Step 1: Run all clones in parallel
        clone_results = self._run_clones(
            num_clones=num_clones,
            baseline_config=baseline_config,
            baseline_score=baseline_score,
            eval_fn=eval_fn,
            mutate_fn=mutate_fn,
            regression_fn=regression_fn,
        )
        result.clone_results = clone_results

        # Filter to successful clones only
        viable = [
            cr for cr in clone_results
            if cr.cycle.decision != EvolutionDecision.ERROR
        ]

        if not viable:
            result.decision = EvolutionDecision.ERROR
            result.reason = "All clones failed"
            result.completed_at = time.time()
            self._tourn_ledger.record_tournament(result)
            return result

        # Step 2: Tournament bracket selection
        champion, rounds = self._tournament_bracket(viable)
        result.rounds = rounds

        # Step 3: Rank all clones
        sorted_clones = sorted(
            clone_results,
            key=lambda cr: cr.cycle.candidate_score,
            reverse=True,
        )
        for i, cr in enumerate(sorted_clones):
            cr.rank = i + 1

        # Step 4: Decision -- champion must beat baseline
        result.champion_id = champion.clone_id
        result.champion_score = champion.cycle.candidate_score
        result.champion_config = champion.cycle.candidate_config

        improvement = result.champion_score - baseline_score
        if improvement >= min_improvement:
            result.decision = EvolutionDecision.ACCEPTED
            result.reason = (
                f"Champion {champion.clone_id} improved by "
                f"+{improvement:.2f} (>= {min_improvement})"
            )
            log.info(
                "TOURNAMENT ACCEPTED: %.2f -> %.2f (+%.2f) from %d clones",
                baseline_score, result.champion_score, improvement, num_clones,
            )
        else:
            result.decision = EvolutionDecision.REJECTED
            result.reason = (
                f"Best clone improved by +{improvement:.2f} "
                f"(< {min_improvement} required)"
            )
            log.info(
                "TOURNAMENT REJECTED: best was %.2f (+%.2f, need +%.2f)",
                result.champion_score, improvement, min_improvement,
            )

        result.completed_at = time.time()
        self._tourn_ledger.record_tournament(result)
        return result

    def _run_clones(
        self,
        num_clones: int,
        baseline_config: Dict[str, Any],
        baseline_score: float,
        eval_fn: Callable[[Dict[str, Any]], float],
        mutate_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
        regression_fn: Optional[Callable[[float, float], bool]] = None,
    ) -> List[CloneResult]:
        """Run N clones in parallel using ThreadPoolExecutor."""
        clone_results: List[CloneResult] = []

        def _run_single_clone(clone_idx: int) -> CloneResult:
            clone_id = f"clone_{clone_idx:03d}_{uuid.uuid4().hex[:8]}"
            runner = EvolutionRunner(
                ledger=self._evo_ledger,
                repo_root=self._repo_root,
                use_worktree=False,
            )
            cycle = runner.run_cycle(
                baseline_config=baseline_config,
                baseline_score=baseline_score,
                eval_fn=eval_fn,
                mutate_fn=mutate_fn,
                regression_fn=regression_fn,
            )
            return CloneResult(clone_id=clone_id, cycle=cycle)

        workers = min(self._max_workers, num_clones)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_run_single_clone, i): i
                for i in range(num_clones)
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    cr = future.result(timeout=300)
                    clone_results.append(cr)
                except Exception as exc:
                    log.warning("Clone %d failed: %s", idx, exc)
                    # Record error clone
                    error_cycle = EvolutionCycle(
                        cycle_id=f"evo_err_{uuid.uuid4().hex[:8]}",
                        started_at=time.time(),
                        completed_at=time.time(),
                        decision=EvolutionDecision.ERROR,
                        reason=f"Clone execution failed: {exc}",
                    )
                    clone_results.append(CloneResult(
                        clone_id=f"clone_{idx:03d}_error",
                        cycle=error_cycle,
                    ))

        return clone_results

    def _tournament_bracket(
        self, clones: List[CloneResult],
    ) -> tuple[CloneResult, List[TournamentRound]]:
        """Single-elimination tournament bracket.

        Each round, pair clones and advance the higher scorer.
        If odd number, last clone gets a bye (auto-advances).
        """
        rounds: List[TournamentRound] = []
        pool = sorted(
            clones,
            key=lambda cr: cr.cycle.candidate_score,
            reverse=True,
        )
        round_num = 0

        while len(pool) > 1:
            round_num += 1
            rnd = TournamentRound(round_num=round_num)
            next_pool: List[CloneResult] = []

            for i in range(0, len(pool), 2):
                if i + 1 < len(pool):
                    a, b = pool[i], pool[i + 1]
                    if a.cycle.candidate_score >= b.cycle.candidate_score:
                        winner, loser = a, b
                    else:
                        winner, loser = b, a

                    rnd.matchups.append({
                        "a": a.clone_id,
                        "a_score": a.cycle.candidate_score,
                        "b": b.clone_id,
                        "b_score": b.cycle.candidate_score,
                        "winner": winner.clone_id,
                    })
                    rnd.survivors.append(winner.clone_id)
                    rnd.eliminated.append(loser.clone_id)
                    next_pool.append(winner)
                else:
                    # Bye -- odd clone auto-advances
                    rnd.survivors.append(pool[i].clone_id)
                    next_pool.append(pool[i])

            rounds.append(rnd)
            pool = next_pool

        return pool[0], rounds

    def get_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent tournament history."""
        return self._tourn_ledger.get_history(limit=limit)

    def get_clone_results(self, tournament_id: str) -> List[Dict[str, Any]]:
        """Get clone results for a specific tournament."""
        return self._tourn_ledger.get_clone_results(tournament_id)

    def stats(self) -> Dict[str, Any]:
        """Get tournament statistics."""
        return self._tourn_ledger.stats()

    def close(self) -> None:
        self._evo_ledger.close()
        self._tourn_ledger.close()
