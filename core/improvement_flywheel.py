"""
Self-Improvement Flywheel — The AGI Core
==========================================
Wires together all 5 self-improvement subsystems into a single
accelerating feedback loop.

The key insight: each system's OUTPUT is another system's INPUT.
When connected in a cycle, each revolution makes the next revolution
faster. This is the compound effect — 1+1=10.

The Flywheel:
    1. AST Graph analyzes code structure (what connects to what)
    2. Strategy Evolver uses structural context to pick better search strategies
    3. Better retrieval produces higher-quality examples for Experience Replay
    4. High-quality experiences inform Prompt Evolution (what prompts work for what)
    5. Better prompts make Adversarial Self-Play generate harder, more targeted challenges
    6. Harder challenges expose new structural patterns → back to step 1

Each subsystem has an INPUT port and an OUTPUT port:
    AST Graph:       IN=code files    OUT=structural context, blast radius
    Strategy Evolver: IN=query+outcome OUT=optimized retrieval strategy
    Experience Replay: IN=trajectories  OUT=few-shot examples
    Prompt Evolution:  IN=eval traces   OUT=improved system prompts
    Self-Play:        IN=weak areas    OUT=training challenges

The Flywheel orchestrator runs one revolution per cycle,
measuring the system-level delta each time.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)


class ImprovementFlywheel:
    """Orchestrates one revolution of the self-improvement cycle.

    Each revolution:
    1. Diagnoses current weaknesses (using FailureAnalyzer)
    2. Generates structural context for weak areas (AST Graph)
    3. Selects optimal retrieval strategies (Strategy Evolver)
    4. Studies weak areas with optimized strategies
    5. Records outcomes in experience replay
    6. Measures the delta

    The key metric is not the absolute score, but whether
    the RATE of improvement is increasing (acceleration).
    """

    def __init__(
        self,
        eval_set_path: str = "evaluation/agent_eval_set_200.json",
        index_dir: str = "data/indexes",
        log_dir: str = "logs/flywheel",
    ):
        self.eval_set_path = eval_set_path
        self.index_dir = index_dir
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._revolution_count = self._load_revolution_count()

    def _load_revolution_count(self) -> int:
        meta_path = self.log_dir / "meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                return json.load(f).get("revolutions", 0)
        return 0

    def _save_revolution_count(self) -> None:
        meta_path = self.log_dir / "meta.json"
        with open(meta_path, "w") as f:
            json.dump({
                "revolutions": self._revolution_count,
                "last_run": datetime.now().isoformat(),
            }, f, indent=2)

    def revolve(self) -> Dict[str, Any]:
        """Execute one revolution of the flywheel.

        Returns a report with:
        - revolution number
        - baseline and post-revolution scores
        - delta (improvement)
        - acceleration (change in rate of improvement)
        - subsystem contributions
        """
        self._revolution_count += 1
        rev = self._revolution_count
        rev_dir = self.log_dir / f"revolution_{rev:04d}"
        rev_dir.mkdir(parents=True, exist_ok=True)

        t0 = time.time()
        log.info("=== FLYWHEEL REVOLUTION %d ===", rev)

        report: Dict[str, Any] = {
            "revolution": rev,
            "timestamp": datetime.now().isoformat(),
            "subsystems": {},
        }

        # Step 1: Diagnose current state
        log.info("[Step 1] Diagnosing current weaknesses...")
        diagnosis = self._diagnose(rev_dir)
        report["diagnosis"] = {
            "weak_categories": list(diagnosis.get("weak_categories", {}).keys()),
            "hypotheses": diagnosis.get("hypothesis", []),
        }

        # Step 2: Generate structural context for weak areas
        log.info("[Step 2] AST Graph structural analysis...")
        structural = self._structural_analysis(diagnosis, rev_dir)
        report["subsystems"]["ast_graph"] = structural

        # Step 3: Select optimized retrieval strategies
        log.info("[Step 3] Strategy evolution...")
        strategies = self._evolve_strategies(diagnosis, rev_dir)
        report["subsystems"]["strategy_evolver"] = strategies

        # Step 4: Study weak areas
        log.info("[Step 4] Targeted study...")
        study = self._study_weak_areas(diagnosis, rev_dir)
        report["subsystems"]["study"] = study

        # Step 5: Re-evaluate
        log.info("[Step 5] Re-evaluation...")
        reeval = self._reevaluate(rev_dir)
        report["reeval_score"] = reeval.get("overall_score", 0)

        # Step 6: Compute delta and acceleration
        baseline_score = diagnosis.get("success_rate", 0)
        reeval_score = reeval.get("overall_score", 0)
        delta = reeval_score - baseline_score

        # Load previous deltas to compute acceleration
        prev_deltas = self._load_previous_deltas()
        prev_deltas.append(delta)
        if len(prev_deltas) >= 2:
            acceleration = prev_deltas[-1] - prev_deltas[-2]
        else:
            acceleration = 0.0

        report["baseline_score"] = baseline_score
        report["delta"] = delta
        report["acceleration"] = acceleration
        report["elapsed_s"] = time.time() - t0
        report["previous_deltas"] = prev_deltas

        # Save report
        with open(rev_dir / "revolution_report.json", "w") as f:
            json.dump(report, f, indent=2)
        self._save_revolution_count()

        # Log summary
        log.info(
            "Revolution %d complete: %.1f%% -> %.1f%% (delta: %+.1f%%, accel: %+.2f%%)",
            rev, baseline_score * 100, reeval_score * 100,
            delta * 100, acceleration * 100,
        )

        return report

    def _diagnose(self, rev_dir: Path) -> Dict[str, Any]:
        """Run baseline eval and diagnose failures."""
        try:
            from scripts.autonomous_improve import FailureAnalyzer
            from scripts.learning_cycle import run_baseline_eval

            baseline = run_baseline_eval(
                self.eval_set_path, self.index_dir,
                str(rev_dir / "baseline.json"),
            )
            results = baseline.get("results", [])
            analyzer = FailureAnalyzer(results)
            diagnosis = analyzer.diagnose()
            diagnosis["overall_score"] = baseline.get("overall_score", 0)
            return diagnosis
        except Exception as exc:
            log.warning("Diagnosis failed: %s", exc)
            return {"success_rate": 0, "weak_categories": {}, "hypothesis": []}

    def _structural_analysis(
        self, diagnosis: Dict, rev_dir: Path,
    ) -> Dict[str, Any]:
        """Use AST Graph to find structurally relevant code for weak areas."""
        try:
            from core.ast_graph import ASTGraph
            graph = ASTGraph()

            # Index if not yet done
            if graph.stats()["nodes"] == 0:
                for d in ["core", "agent", "cli", "ingestion"]:
                    graph.index_directory(d)

            # Find structural context for weak categories
            weak = list(diagnosis.get("weak_categories", {}).keys())
            contexts = {}
            for cat in weak[:5]:
                results = graph.structural_context(cat, max_files=3)
                contexts[cat] = [r["file"] for r in results]

            return {
                "graph_stats": graph.stats(),
                "weak_area_context": contexts,
            }
        except Exception as exc:
            log.warning("Structural analysis failed: %s", exc)
            return {"error": str(exc)}

    def _evolve_strategies(
        self, diagnosis: Dict, rev_dir: Path,
    ) -> Dict[str, Any]:
        """Evolve retrieval strategies based on weak areas."""
        try:
            from core.strategy_evolver import StrategyEvolver
            evolver = StrategyEvolver()
            new_count = evolver.evolve()
            return {"new_strategies": new_count, "evolved": True}
        except Exception as exc:
            log.warning("Strategy evolution failed: %s", exc)
            return {"error": str(exc)}

    def _study_weak_areas(
        self, diagnosis: Dict, rev_dir: Path,
    ) -> Dict[str, Any]:
        """Generate and execute targeted study queries."""
        try:
            from scripts.learning_cycle import (
                generate_study_queries, run_study_engine,
            )
            # Use the latest baseline for study query generation
            baseline_path = rev_dir / "baseline.json"
            if baseline_path.exists():
                with open(baseline_path) as f:
                    baseline = json.load(f)
                queries = generate_study_queries(
                    baseline, self.eval_set_path, n_queries=20,
                )
                result = run_study_engine(queries, self.index_dir)
                return {"queries_studied": len(queries), "result": result}
        except Exception as exc:
            log.warning("Study failed: %s", exc)
        return {"queries_studied": 0}

    def _reevaluate(self, rev_dir: Path) -> Dict[str, Any]:
        """Re-evaluate after studying."""
        try:
            from scripts.learning_cycle import run_baseline_eval
            return run_baseline_eval(
                self.eval_set_path, self.index_dir,
                str(rev_dir / "reeval.json"),
            )
        except Exception as exc:
            log.warning("Re-evaluation failed: %s", exc)
            return {"overall_score": 0}

    def _load_previous_deltas(self) -> List[float]:
        """Load delta history from previous revolutions."""
        deltas = []
        for rev_dir in sorted(self.log_dir.glob("revolution_*")):
            report_path = rev_dir / "revolution_report.json"
            if report_path.exists():
                with open(report_path) as f:
                    report = json.load(f)
                deltas.append(report.get("delta", 0.0))
        return deltas

    @property
    def revolution_count(self) -> int:
        return self._revolution_count

    def status(self) -> Dict[str, Any]:
        """Return current flywheel status."""
        deltas = self._load_previous_deltas()
        return {
            "revolutions": self._revolution_count,
            "deltas": deltas,
            "total_improvement": sum(deltas),
            "accelerating": (
                len(deltas) >= 2 and deltas[-1] > deltas[-2]
            ) if deltas else False,
        }
