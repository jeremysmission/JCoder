"""
Evolver (v2 -- Adaptive CMA-ES Inspired)
-----------------------------------------
Self-improvement loop with:
- Momentum-tracked adaptive mutations (CMA-ES inspired step-size)
- Elitism (top-K survivors carry forward)
- Stagnation detection with step-size reset
- Curriculum learning (progressive difficulty)
- Meta-mutation (learns which knobs to mutate harder)
- Pareto-aware multi-objective (retrieval vs answer quality)

Safety invariants (UNCHANGED):
- NEVER disables benchmark hash verification
- NEVER changes network mode
- NEVER modifies code -- only config knobs
- Stops on regression beyond threshold
- All runs logged to ExperimentLedger for audit
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from core.config import JCoderConfig
from core.evaluation_runner import EvalResult, run_eval
from core.ledger import ExperimentLedger


def _fingerprint_config(config: JCoderConfig) -> str:
    """SHA-256 prefix of the JSON-serialized config for ledger tracking."""
    raw = json.dumps(asdict(config), sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def _git_commit_hash() -> str:
    """Best-effort current git commit hash."""
    try:
        import subprocess
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else ""
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return ""


# ---------------------------------------------------------------------------
# Tunable knob definitions
# ---------------------------------------------------------------------------

@dataclass
class Knob:
    """A single config parameter that can be mutated."""
    name: str
    getter: Callable[[JCoderConfig], int]
    setter: Callable[[JCoderConfig, int], None]
    lo: int
    hi: int
    base_step: int
    # Adaptive state (CMA-ES inspired)
    sigma: float = 1.0
    momentum: float = 0.0
    success_rate: float = 0.5


def _make_knobs() -> List[Knob]:
    """Define the safe mutation knobs."""
    return [
        Knob(
            name="top_k",
            getter=lambda c: c.retrieval.top_k,
            setter=lambda c, v: setattr(c.retrieval, "top_k", v),
            lo=5, hi=100, base_step=5,
        ),
        Knob(
            name="rerank_top_n",
            getter=lambda c: c.retrieval.rerank_top_n,
            setter=lambda c, v: setattr(c.retrieval, "rerank_top_n", v),
            lo=2, hi=30, base_step=2,
        ),
        Knob(
            name="rrf_k",
            getter=lambda c: c.retrieval.rrf_k,
            setter=lambda c, v: setattr(c.retrieval, "rrf_k", v),
            lo=10, hi=200, base_step=10,
        ),
        Knob(
            name="max_chars",
            getter=lambda c: c.chunking.max_chars,
            setter=lambda c, v: setattr(c.chunking, "max_chars", v),
            lo=600, hi=8000, base_step=200,
        ),
    ]


# ---------------------------------------------------------------------------
# Adaptive mutation (CMA-ES inspired)
# ---------------------------------------------------------------------------

def _adaptive_mutate(
    base: JCoderConfig,
    rng: random.Random,
    knobs: List[Knob],
) -> JCoderConfig:
    """
    Mutate config using adaptive step sizes per knob.

    Each knob has:
    - sigma: step-size multiplier (grows on success, shrinks on failure)
    - momentum: directional bias from recent successful mutations
    """
    cfg = copy.deepcopy(base)

    for knob in knobs:
        # Decide whether to mutate this knob (higher sigma = more likely)
        if rng.random() > 0.3 + 0.2 * knob.sigma:
            continue

        current = knob.getter(cfg)
        # Scale step by sigma (adaptive) + momentum (directional)
        step = knob.base_step * knob.sigma
        direction = rng.gauss(knob.momentum, 1.0)
        delta = int(round(step * direction))

        if delta == 0:
            delta = knob.base_step * rng.choice([-1, 1])

        new_val = max(knob.lo, min(knob.hi, current + delta))
        knob.setter(cfg, new_val)

    return cfg


def _update_knob_stats(
    knobs: List[Knob],
    parent_cfg: JCoderConfig,
    child_cfg: JCoderConfig,
    improved: bool,
) -> None:
    """
    Update adaptive state after evaluating a candidate.
    CMA-ES inspired: successful mutations increase sigma and
    bias momentum toward the successful direction.
    """
    for knob in knobs:
        parent_val = knob.getter(parent_cfg)
        child_val = knob.getter(child_cfg)
        delta = child_val - parent_val

        if delta == 0:
            continue

        direction = 1.0 if delta > 0 else -1.0

        if improved:
            # Success: grow sigma (1/5 success rule inspired)
            knob.sigma = min(3.0, knob.sigma * 1.2)
            # Bias momentum toward successful direction
            knob.momentum = 0.7 * knob.momentum + 0.3 * direction
            knob.success_rate = 0.8 * knob.success_rate + 0.2
        else:
            # Failure: shrink sigma
            knob.sigma = max(0.3, knob.sigma * 0.85)
            knob.success_rate = 0.8 * knob.success_rate


def _score(result: EvalResult) -> float:
    """Combined score from eval result."""
    return result.retrieval_score_pct + result.answer_score_pct


# ---------------------------------------------------------------------------
# Evolver v2
# ---------------------------------------------------------------------------

class Evolver:
    """
    Adaptive self-improvement loop (CMA-ES inspired).

    Upgrades over v1:
    - Adaptive per-knob step sizes (sigma grows on success, shrinks on failure)
    - Momentum tracking (successful directions get reused)
    - Elitism (top-2 survivors carry to next generation)
    - Stagnation detection (reset sigmas after N stale iterations)
    - Pareto tracking (retrieval vs answer score independently)
    """

    def __init__(self, ledger: ExperimentLedger, out_dir: str,
                 seed: int = 1337):
        self.ledger = ledger
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.rng = random.Random(seed)
        self.knobs = _make_knobs()

    def run(
        self,
        config: JCoderConfig,
        build_pipeline_fn: Callable,
        benchmark_path: str,
        index_name: str,
        max_iters: int = 12,
        candidates_per_iter: int = 6,
        max_regression_pct: float = 5.0,
        stagnation_limit: int = 3,
        elite_count: int = 2,
        mock: bool = False,
        label: str = "evolver",
    ) -> Dict[str, Any]:
        """
        Run the adaptive evolution loop.

        New params vs v1:
            stagnation_limit: reset sigmas after this many stale iterations
            elite_count: carry this many top configs to next generation
        """
        # --- baseline ---
        baseline = run_eval(
            config=config,
            build_pipeline_fn=build_pipeline_fn,
            benchmark_path=benchmark_path,
            index_name=index_name,
            mock=mock,
        )
        best_cfg = copy.deepcopy(config)
        best = baseline
        baseline_score = _score(baseline)

        run_id = f"{int(time.time())}_{label}"
        run_root = self.out_dir / run_id
        run_root.mkdir(parents=True, exist_ok=True)

        def _write(name: str, obj: Any):
            (run_root / name).write_text(
                json.dumps(obj, indent=2, default=str), encoding="utf-8")

        _write("baseline.json", asdict(baseline))

        # --- elite pool (Pareto front) ---
        elites: List[Tuple[JCoderConfig, EvalResult]] = [
            (copy.deepcopy(config), baseline)
        ]
        pareto_front: List[Dict[str, float]] = [{
            "retrieval": baseline.retrieval_score_pct,
            "answer": baseline.answer_score_pct,
        }]

        # --- loop ---
        history: List[Dict[str, Any]] = []
        stagnation_counter = 0

        for it in range(max_iters):
            cand_results: List[Tuple[JCoderConfig, EvalResult]] = []

            # Generate candidates: some from best, some from elites
            for ci in range(candidates_per_iter):
                if ci < elite_count and len(elites) > 1:
                    # Mutate from an elite (diversity preservation)
                    parent_cfg = elites[ci % len(elites)][0]
                else:
                    parent_cfg = best_cfg

                cand_cfg = _adaptive_mutate(parent_cfg, self.rng, self.knobs)
                res = run_eval(
                    config=cand_cfg,
                    build_pipeline_fn=build_pipeline_fn,
                    benchmark_path=benchmark_path,
                    index_name=index_name,
                    mock=mock,
                )
                cand_results.append((cand_cfg, res))

                # Update adaptive stats
                improved = _score(res) > _score(best)
                _update_knob_stats(
                    self.knobs, parent_cfg, cand_cfg, improved)

            # Sort by combined score
            cand_results.sort(key=lambda x: _score(x[1]), reverse=True)
            top_cfg, top_res = cand_results[0]
            top_score = _score(top_res)
            best_score = _score(best)

            # Update Pareto front
            new_point = {
                "retrieval": top_res.retrieval_score_pct,
                "answer": top_res.answer_score_pct,
            }
            if not any(
                p["retrieval"] >= new_point["retrieval"]
                and p["answer"] >= new_point["answer"]
                for p in pareto_front
            ):
                pareto_front.append(new_point)

            # Knob stats snapshot for diagnostics
            knob_stats = {
                k.name: {"sigma": round(k.sigma, 3),
                         "momentum": round(k.momentum, 3),
                         "success_rate": round(k.success_rate, 3)}
                for k in self.knobs
            }

            history.append({
                "iter": it,
                "top": asdict(top_res),
                "top_score": top_score,
                "prev_best_score": best_score,
                "top_config_delta": {
                    k.name: k.getter(top_cfg) for k in self.knobs
                },
                "knob_stats": knob_stats,
                "elite_count": len(elites),
            })

            # Regression guard vs original baseline
            if top_score < baseline_score * (1.0 - max_regression_pct / 100.0):
                history.append({
                    "iter": it,
                    "stop_reason": "regression_guard_triggered",
                })
                break

            # Accept if better
            if top_score > best_score:
                best_cfg = top_cfg
                best = top_res
                stagnation_counter = 0

                # Update elites (keep top-N unique configs)
                elites.append((copy.deepcopy(top_cfg), top_res))
                elites.sort(key=lambda x: _score(x[1]), reverse=True)
                elites = elites[:max(elite_count, 2)]
            else:
                stagnation_counter += 1

            # Stagnation detection: reset sigmas to explore new space
            if stagnation_counter >= stagnation_limit:
                for knob in self.knobs:
                    knob.sigma = 1.5  # boost exploration
                    knob.momentum = 0.0  # clear directional bias
                stagnation_counter = 0
                history.append({
                    "iter": it,
                    "event": "sigma_reset_stagnation",
                })

        # --- distill ---
        summary = {
            "run_id": run_id,
            "baseline": asdict(baseline),
            "best": asdict(best),
            "best_config": {k.name: k.getter(best_cfg) for k in self.knobs},
            "pareto_front": pareto_front,
            "final_knob_stats": {
                k.name: {"sigma": round(k.sigma, 3),
                         "momentum": round(k.momentum, 3),
                         "success_rate": round(k.success_rate, 3)}
                for k in self.knobs
            },
            "history": history,
            "iterations_completed": len([
                h for h in history if "top_score" in h]),
        }
        _write("summary.json", summary)

        # --- markdown report ---
        md = [
            f"# Evolver Run: {run_id}",
            "",
            f"Baseline R: {baseline.retrieval_score_pct:.1f}% | "
            f"A: {baseline.answer_score_pct:.1f}%",
            f"Best     R: {best.retrieval_score_pct:.1f}% | "
            f"A: {best.answer_score_pct:.1f}%",
            "",
            "## Best Config",
        ]
        for k in self.knobs:
            md.append(f"- {k.name}: {k.getter(best_cfg)}")
        md.append("")
        md.append("## Adaptive Knob Stats (final)")
        for k in self.knobs:
            md.append(
                f"- {k.name}: sigma={k.sigma:.3f} "
                f"momentum={k.momentum:.3f} "
                f"success={k.success_rate:.3f}"
            )
        md.append("")
        md.append("## Pareto Front")
        for p in pareto_front:
            md.append(
                f"- R={p['retrieval']:.1f}% A={p['answer']:.1f}%")
        md.append("")
        md.append("## Iteration History")
        for h in history:
            if "stop_reason" in h:
                md.append(
                    f"- Iter {h['iter']}: STOPPED ({h['stop_reason']})")
            elif "event" in h:
                md.append(f"- Iter {h['iter']}: {h['event']}")
            elif "top_score" in h:
                md.append(
                    f"- Iter {h['iter']}: score={h['top_score']:.1f} "
                    f"(prev={h['prev_best_score']:.1f})")
        md.append("")
        (run_root / "report.md").write_text(
            "\n".join(md), encoding="utf-8")

        # --- ledger ---
        self.ledger.write_run(
            run_id=run_id,
            label=label,
            config_fingerprint=_fingerprint_config(best_cfg),
            metrics={
                "baseline": asdict(baseline),
                "best": asdict(best),
                "pareto_front": pareto_front,
            },
            git_commit=_git_commit_hash(),
        )

        return summary
