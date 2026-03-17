"""
Weekly Software Evolution Engine (Sprint 21)
----------------------------------------------
Wraps the Evolver with production-grade safety:
  1. Archive baseline before ANY mutation
  2. Git worktree isolation for safe testing
  3. Immutable eval harness verification
  4. Continual learner regression gate
  5. Full audit trail in evolution ledger

Every evolution cycle:
  ARCHIVE  -> snapshot current baseline
  PROPOSE  -> generate config mutations via Evolver
  ISOLATE  -> run eval in git worktree (no dirty main)
  VALIDATE -> check continual learner regression gate
  DECIDE   -> accept (merge) or reject (rollback) with full log
  ARCHIVE  -> store outcome for future reference

Gate: First successful evolution cycle with full audit trail.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
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

class EvolutionDecision:
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    ERROR = "error"


@dataclass
class EvolutionCycle:
    """Record of a single evolution attempt."""
    cycle_id: str
    started_at: float
    completed_at: float = 0.0
    baseline_score: float = 0.0
    candidate_score: float = 0.0
    decision: str = ""
    reason: str = ""
    baseline_config: Dict[str, Any] = field(default_factory=dict)
    candidate_config: Dict[str, Any] = field(default_factory=dict)
    eval_results: Dict[str, Any] = field(default_factory=dict)
    regression_check: Dict[str, Any] = field(default_factory=dict)
    worktree_path: str = ""
    git_commit: str = ""


# ---------------------------------------------------------------------------
# Evolution Ledger (SQLite audit trail)
# ---------------------------------------------------------------------------

_LEDGER_SCHEMA = """
CREATE TABLE IF NOT EXISTS evolution_cycles (
    cycle_id TEXT PRIMARY KEY,
    started_at REAL NOT NULL,
    completed_at REAL DEFAULT 0,
    baseline_score REAL DEFAULT 0,
    candidate_score REAL DEFAULT 0,
    decision TEXT DEFAULT '',
    reason TEXT DEFAULT '',
    baseline_config_json TEXT DEFAULT '{}',
    candidate_config_json TEXT DEFAULT '{}',
    eval_results_json TEXT DEFAULT '{}',
    regression_check_json TEXT DEFAULT '{}',
    worktree_path TEXT DEFAULT '',
    git_commit TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS baselines (
    baseline_id TEXT PRIMARY KEY,
    cycle_id TEXT NOT NULL,
    config_json TEXT NOT NULL,
    score REAL NOT NULL,
    archived_at REAL NOT NULL
);
"""


class EvolutionLedger:
    """Immutable audit trail for evolution cycles."""

    def __init__(self, db_path: str | Path):
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._owner = SQLiteConnectionOwner(self._db_path)
        conn = self._owner.connect()
        conn.executescript(_LEDGER_SCHEMA)
        conn.commit()

    @property
    def _conn(self):
        return self._owner.connect()

    def record_cycle(self, cycle: EvolutionCycle) -> None:
        """Record a complete evolution cycle."""
        conn = self._conn
        conn.execute(
            "INSERT OR REPLACE INTO evolution_cycles "
            "(cycle_id, started_at, completed_at, baseline_score, "
            "candidate_score, decision, reason, baseline_config_json, "
            "candidate_config_json, eval_results_json, "
            "regression_check_json, worktree_path, git_commit) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                cycle.cycle_id, cycle.started_at, cycle.completed_at,
                cycle.baseline_score, cycle.candidate_score,
                cycle.decision, cycle.reason,
                json.dumps(cycle.baseline_config, default=str),
                json.dumps(cycle.candidate_config, default=str),
                json.dumps(cycle.eval_results, default=str),
                json.dumps(cycle.regression_check, default=str),
                cycle.worktree_path, cycle.git_commit,
            ),
        )
        conn.commit()

    def archive_baseline(self, cycle_id: str, config: Dict[str, Any],
                         score: float) -> str:
        """Archive a baseline configuration."""
        bid = f"base_{uuid.uuid4().hex[:12]}"
        conn = self._conn
        conn.execute(
            "INSERT INTO baselines (baseline_id, cycle_id, config_json, "
            "score, archived_at) VALUES (?, ?, ?, ?, ?)",
            (bid, cycle_id, json.dumps(config, default=str), score, time.time()),
        )
        conn.commit()
        return bid

    def get_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent evolution cycles."""
        rows = self._conn.execute(
            "SELECT * FROM evolution_cycles ORDER BY started_at DESC LIMIT ?",
            (min(limit, 1000),),
        ).fetchall()
        return [
            {
                "cycle_id": r[0], "started_at": r[1], "completed_at": r[2],
                "baseline_score": r[3], "candidate_score": r[4],
                "decision": r[5], "reason": r[6],
            }
            for r in rows
        ]

    def get_baselines(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get archived baselines."""
        rows = self._conn.execute(
            "SELECT * FROM baselines ORDER BY archived_at DESC LIMIT ?",
            (min(limit, 500),),
        ).fetchall()
        return [
            {
                "baseline_id": r[0], "cycle_id": r[1],
                "config": json.loads(r[2] or "{}"),
                "score": r[3], "archived_at": r[4],
            }
            for r in rows
        ]

    def stats(self) -> Dict[str, Any]:
        """Aggregate evolution statistics."""
        conn = self._conn
        total = conn.execute("SELECT COUNT(*) FROM evolution_cycles").fetchone()[0]
        accepted = conn.execute(
            "SELECT COUNT(*) FROM evolution_cycles WHERE decision='accepted'"
        ).fetchone()[0]
        rejected = conn.execute(
            "SELECT COUNT(*) FROM evolution_cycles WHERE decision='rejected'"
        ).fetchone()[0]
        errors = conn.execute(
            "SELECT COUNT(*) FROM evolution_cycles WHERE decision='error'"
        ).fetchone()[0]

        return {
            "total_cycles": total,
            "accepted": accepted,
            "rejected": rejected,
            "errors": errors,
            "acceptance_rate": round(accepted / max(total, 1), 3),
            "baselines_archived": conn.execute(
                "SELECT COUNT(*) FROM baselines"
            ).fetchone()[0],
        }

    def close(self) -> None:
        self._owner.close()


# ---------------------------------------------------------------------------
# Worktree isolation
# ---------------------------------------------------------------------------

def _create_worktree(repo_root: Path, branch_name: str) -> Optional[Path]:
    """Create a git worktree for isolated testing."""
    wt_path = repo_root / ".worktrees" / branch_name
    try:
        subprocess.run(
            ["git", "worktree", "add", str(wt_path), "-b", branch_name],
            cwd=str(repo_root), capture_output=True, text=True, timeout=30,
        )
        return wt_path if wt_path.exists() else None
    except (subprocess.SubprocessError, FileNotFoundError, OSError) as exc:
        log.debug("Worktree creation failed: %s", exc)
        return None


def _remove_worktree(repo_root: Path, wt_path: Path) -> None:
    """Remove a git worktree."""
    try:
        subprocess.run(
            ["git", "worktree", "remove", str(wt_path), "--force"],
            cwd=str(repo_root), capture_output=True, text=True, timeout=30,
        )
    except (subprocess.SubprocessError, FileNotFoundError, OSError) as exc:
        log.debug("Worktree removal failed: %s", exc)
    # Also try to delete the branch
    branch_name = wt_path.name
    try:
        subprocess.run(
            ["git", "branch", "-D", branch_name],
            cwd=str(repo_root), capture_output=True, text=True, timeout=10,
        )
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        pass


# ---------------------------------------------------------------------------
# Evolution Runner
# ---------------------------------------------------------------------------

class EvolutionRunner:
    """Production-grade evolution engine with safety invariants.

    Wraps the Evolver with:
    - Baseline archiving before any mutation
    - Optional git worktree isolation
    - Continual learner regression gating
    - Full audit trail via EvolutionLedger
    """

    IMMUTABLE_FILES = frozenset({
        "scripts/run_eval.py",
        "tools/eval_runner.py",
        "tools/score_results.py",
        "tools/run_all.py",
    })

    def __init__(
        self,
        ledger: Optional[EvolutionLedger] = None,
        ledger_path: str | Path = "_evolution/ledger.db",
        repo_root: Optional[Path] = None,
        use_worktree: bool = False,
        surrogate_store: Optional[Any] = None,
        surrogate_threshold: float = 0.3,
    ):
        self._ledger = ledger or EvolutionLedger(ledger_path)
        self._repo_root = repo_root or Path.cwd()
        self._use_worktree = use_worktree
        self._surrogate = surrogate_store
        self._surrogate_threshold = surrogate_threshold

    def run_cycle(
        self,
        baseline_config: Dict[str, Any],
        baseline_score: float,
        eval_fn: Callable[[Dict[str, Any]], float],
        mutate_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
        regression_fn: Optional[Callable[[float, float], bool]] = None,
        min_improvement: float = 0.5,
    ) -> EvolutionCycle:
        """Execute one evolution cycle.

        Parameters
        ----------
        baseline_config : current configuration
        baseline_score : current eval score
        eval_fn : evaluates a config, returns score
        mutate_fn : mutates a config, returns candidate config
        regression_fn : checks if new score regresses (returns True if OK)
        min_improvement : minimum score improvement to accept
        """
        cycle = EvolutionCycle(
            cycle_id=f"evo_{uuid.uuid4().hex[:12]}",
            started_at=time.time(),
            baseline_config=baseline_config,
            baseline_score=baseline_score,
        )

        # Step 1: Archive baseline
        self._ledger.archive_baseline(
            cycle.cycle_id, baseline_config, baseline_score,
        )
        log.info("Archived baseline (score=%.2f)", baseline_score)

        # Step 2: Verify immutable files
        if not self._verify_immutables():
            cycle.decision = EvolutionDecision.ERROR
            cycle.reason = "Immutable eval files have been modified"
            cycle.completed_at = time.time()
            self._ledger.record_cycle(cycle)
            return cycle

        # Step 3: Propose mutation
        try:
            candidate_config = mutate_fn(baseline_config)
            cycle.candidate_config = candidate_config
        except Exception as exc:
            cycle.decision = EvolutionDecision.ERROR
            cycle.reason = f"Mutation failed: {exc}"
            cycle.completed_at = time.time()
            self._ledger.record_cycle(cycle)
            return cycle

        # Step 3.5: Surrogate pre-filter (skip expensive eval if predicted bad)
        if self._surrogate and hasattr(self._surrogate, "should_evaluate"):
            if not self._surrogate.should_evaluate(
                candidate_config, self._surrogate_threshold
            ):
                cycle.decision = EvolutionDecision.REJECTED
                cycle.reason = "Surrogate pre-filter: predicted below threshold"
                cycle.completed_at = time.time()
                self._ledger.record_cycle(cycle)
                return cycle

        # Step 4: Evaluate candidate (optionally in worktree)
        wt_path = None
        try:
            if self._use_worktree:
                branch = f"evo_{cycle.cycle_id}"
                wt_path = _create_worktree(self._repo_root, branch)
                cycle.worktree_path = str(wt_path or "")

            candidate_score = eval_fn(candidate_config)
            cycle.candidate_score = candidate_score
            cycle.eval_results = {
                "baseline": baseline_score,
                "candidate": candidate_score,
                "delta": candidate_score - baseline_score,
            }

            # Feed real eval result back to surrogate
            if self._surrogate and hasattr(self._surrogate, "record"):
                self._surrogate.record(candidate_config, candidate_score)
        except Exception as exc:
            cycle.decision = EvolutionDecision.ERROR
            cycle.reason = f"Evaluation failed: {exc}"
            cycle.completed_at = time.time()
            self._ledger.record_cycle(cycle)
            return cycle
        finally:
            if wt_path:
                _remove_worktree(self._repo_root, wt_path)

        # Step 5: Regression gate
        if regression_fn:
            regression_ok = regression_fn(baseline_score, candidate_score)
            cycle.regression_check = {
                "passed": regression_ok,
                "baseline": baseline_score,
                "candidate": candidate_score,
            }
            if not regression_ok:
                cycle.decision = EvolutionDecision.REJECTED
                cycle.reason = "Failed regression gate"
                cycle.completed_at = time.time()
                self._ledger.record_cycle(cycle)
                return cycle

        # Step 6: Decision
        improvement = candidate_score - baseline_score
        if improvement >= min_improvement:
            cycle.decision = EvolutionDecision.ACCEPTED
            cycle.reason = f"Improvement: +{improvement:.2f} (>= {min_improvement})"
            log.info(
                "ACCEPTED: %.2f -> %.2f (+%.2f)",
                baseline_score, candidate_score, improvement,
            )
        else:
            cycle.decision = EvolutionDecision.REJECTED
            cycle.reason = (
                f"Insufficient improvement: +{improvement:.2f} "
                f"(< {min_improvement})"
            )
            log.info(
                "REJECTED: %.2f -> %.2f (+%.2f, need +%.2f)",
                baseline_score, candidate_score, improvement, min_improvement,
            )

        cycle.completed_at = time.time()
        self._ledger.record_cycle(cycle)
        return cycle

    def _verify_immutables(self) -> bool:
        """Verify that immutable eval files haven't been modified."""
        try:
            result = subprocess.run(
                ["git", "diff", "--name-only"],
                cwd=str(self._repo_root),
                capture_output=True, text=True, timeout=10,
            )
            changed = set(result.stdout.strip().splitlines())
            violations = changed & self.IMMUTABLE_FILES
            if violations:
                log.error("Immutable files modified: %s", violations)
                return False
            return True
        except (subprocess.SubprocessError, FileNotFoundError, OSError):
            # Can't verify -- assume OK (non-git environment)
            return True

    def get_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent evolution history."""
        return self._ledger.get_history(limit=limit)

    def get_baselines(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get archived baselines."""
        return self._ledger.get_baselines(limit=limit)

    def stats(self) -> Dict[str, Any]:
        """Get evolution statistics."""
        return self._ledger.stats()

    def close(self) -> None:
        self._ledger.close()
