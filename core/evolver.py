"""
Evolver
-------
Self-improvement loop that tunes retrieval/chunking config via controlled
experimentation. Runs baseline -> mutated candidates -> eval -> pick best.

Safety invariants:
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
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

from core.config import JCoderConfig
from core.evaluation_runner import EvalResult, run_eval
from core.ledger import ExperimentLedger


def _fingerprint_config(config: JCoderConfig) -> str:
    """SHA-256 prefix of the JSON-serialized config for ledger tracking."""
    raw = json.dumps(asdict(config), sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def _mutate_config(base: JCoderConfig, rng: random.Random) -> JCoderConfig:
    """
    Produce a mutated copy of the config. Only safe knobs are touched:
    retrieval (top_k, rerank_top_n, rrf_k) and chunking (max_chars).
    """
    cfg = copy.deepcopy(base)

    cfg.retrieval.top_k = max(5, min(100, cfg.retrieval.top_k + rng.choice([-5, -2, 0, 2, 5])))
    cfg.retrieval.rerank_top_n = max(2, min(30, cfg.retrieval.rerank_top_n + rng.choice([-2, -1, 0, 1, 2])))
    cfg.retrieval.rrf_k = max(10, min(200, cfg.retrieval.rrf_k + rng.choice([-10, -5, 0, 5, 10])))
    cfg.chunking.max_chars = max(600, min(8000, cfg.chunking.max_chars + rng.choice([-200, -100, 0, 100, 200])))

    return cfg


def _git_commit_hash() -> str:
    """Best-effort current git commit hash."""
    try:
        import subprocess
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else ""
    except Exception:
        return ""


class Evolver:
    """
    Runs an auditable self-improvement loop:
    baseline -> N candidate mutations -> eval -> pick best -> distill.
    """

    def __init__(self, ledger: ExperimentLedger, out_dir: str, seed: int = 1337):
        self.ledger = ledger
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.rng = random.Random(seed)

    def run(
        self,
        config: JCoderConfig,
        build_pipeline_fn: Callable,
        benchmark_path: str,
        index_name: str,
        max_iters: int = 12,
        candidates_per_iter: int = 6,
        max_regression_pct: float = 5.0,
        mock: bool = False,
        label: str = "evolver",
    ) -> Dict[str, Any]:
        """
        Run the evolution loop. Returns a summary dict and writes artifacts.
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
        baseline_score = baseline.retrieval_score_pct + baseline.answer_score_pct

        run_id = f"{int(time.time())}_{label}"
        run_root = self.out_dir / run_id
        run_root.mkdir(parents=True, exist_ok=True)

        def _write(name: str, obj: Any):
            (run_root / name).write_text(
                json.dumps(obj, indent=2, default=str), encoding="utf-8")

        _write("baseline.json", asdict(baseline))

        # --- loop ---
        history: List[Dict[str, Any]] = []
        for it in range(max_iters):
            cand_results: List[Tuple[JCoderConfig, EvalResult]] = []

            for _c in range(candidates_per_iter):
                cand_cfg = _mutate_config(best_cfg, self.rng)
                res = run_eval(
                    config=cand_cfg,
                    build_pipeline_fn=build_pipeline_fn,
                    benchmark_path=benchmark_path,
                    index_name=index_name,
                    mock=mock,
                )
                cand_results.append((cand_cfg, res))

            # pick best candidate by combined score
            cand_results.sort(
                key=lambda x: x[1].retrieval_score_pct + x[1].answer_score_pct,
                reverse=True,
            )
            top_cfg, top_res = cand_results[0]
            top_score = top_res.retrieval_score_pct + top_res.answer_score_pct
            best_score = best.retrieval_score_pct + best.answer_score_pct

            history.append({
                "iter": it,
                "top": asdict(top_res),
                "top_score": top_score,
                "prev_best_score": best_score,
                "top_config_delta": {
                    "top_k": top_cfg.retrieval.top_k,
                    "rerank_top_n": top_cfg.retrieval.rerank_top_n,
                    "rrf_k": top_cfg.retrieval.rrf_k,
                    "max_chars": top_cfg.chunking.max_chars,
                },
            })

            # regression guard vs original baseline
            if top_score < baseline_score * (1.0 - max_regression_pct / 100.0):
                history.append({"iter": it, "stop_reason": "regression_guard_triggered"})
                break

            # accept if better
            if top_score > best_score:
                best_cfg = top_cfg
                best = top_res

        # --- distill ---
        summary = {
            "run_id": run_id,
            "baseline": asdict(baseline),
            "best": asdict(best),
            "best_config": {
                "top_k": best_cfg.retrieval.top_k,
                "rerank_top_n": best_cfg.retrieval.rerank_top_n,
                "rrf_k": best_cfg.retrieval.rrf_k,
                "max_chars": best_cfg.chunking.max_chars,
            },
            "history": history,
            "iterations_completed": len(history),
        }
        _write("summary.json", summary)

        # --- distilled markdown report ---
        md_lines = [
            f"# Evolver Run: {run_id}",
            "",
            f"Baseline R: {baseline.retrieval_score_pct:.1f}% | A: {baseline.answer_score_pct:.1f}%",
            f"Best     R: {best.retrieval_score_pct:.1f}% | A: {best.answer_score_pct:.1f}%",
            "",
            "## Best Config",
            f"- top_k: {best_cfg.retrieval.top_k}",
            f"- rerank_top_n: {best_cfg.retrieval.rerank_top_n}",
            f"- rrf_k: {best_cfg.retrieval.rrf_k}",
            f"- max_chars: {best_cfg.chunking.max_chars}",
            "",
            "## Iteration History",
        ]
        for h in history:
            if "stop_reason" in h:
                md_lines.append(f"- Iter {h['iter']}: STOPPED ({h['stop_reason']})")
            else:
                md_lines.append(
                    f"- Iter {h['iter']}: score={h['top_score']:.1f} "
                    f"(prev_best={h['prev_best_score']:.1f})"
                )
        md_lines.append("")
        (run_root / "report.md").write_text("\n".join(md_lines), encoding="utf-8")

        # --- ledger ---
        self.ledger.write_run(
            run_id=run_id,
            label=label,
            config_fingerprint=_fingerprint_config(best_cfg),
            metrics={"baseline": asdict(baseline), "best": asdict(best)},
            git_commit=_git_commit_hash(),
        )

        return summary
