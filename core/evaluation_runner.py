"""
Evaluation Runner
-----------------
Programmatic benchmark runner extracted from cli.commands.evaluate().
Used by the evolver to run evaluations without Click context.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict

from core.config import JCoderConfig
from core.eval_guard import verify_hashes


@dataclass
class EvalResult:
    retrieval_score_pct: float
    answer_score_pct: float
    n_questions: int
    elapsed_s: float
    failure_modes: Dict[str, int]


def run_eval(
    config: JCoderConfig,
    build_pipeline_fn: Callable,
    benchmark_path: str,
    index_name: str,
    mock: bool = False,
    eval_mode: bool = True,
) -> EvalResult:
    """
    Run a full benchmark evaluation programmatically.

    build_pipeline_fn: callable matching cli.commands._build_pipeline signature
    """
    benchmark = Path(benchmark_path)
    eval_dir = str(benchmark.parent)

    repo_root = Path(__file__).resolve().parent.parent
    manifest = str(repo_root / config.policies.benchmark_hash_manifest)

    if config.policies.benchmark_hash_verify:
        error = verify_hashes(eval_dir, hash_path=manifest)
        if error:
            raise RuntimeError(f"Benchmark hash verification failed: {error}")

    embedder, index, retriever, runtime, orchestrator = build_pipeline_fn(
        config, mock=mock, eval_mode=eval_mode)

    try:
        index.load(index_name)
        data = json.loads(benchmark.read_text(encoding="utf-8"))

        # Import scoring helpers (kept in cli for now to minimize diff)
        from cli.commands import _score_retrieval, _score_answer

        retrieval_passes = 0
        answer_passes = 0
        failure_modes: Dict[str, int] = {}
        start = time.time()

        for item in data:
            result = orchestrator.answer(item["question"])
            r_score = _score_retrieval(item, result)
            a_score = _score_answer(item, result)

            if r_score["pass"]:
                retrieval_passes += 1
            if a_score["pass"]:
                answer_passes += 1

            for fm in [r_score["failure_mode"], a_score["failure_mode"]]:
                if fm:
                    failure_modes[fm] = failure_modes.get(fm, 0) + 1

        elapsed = time.time() - start
        n = max(len(data), 1)

        return EvalResult(
            retrieval_score_pct=100.0 * retrieval_passes / n,
            answer_score_pct=100.0 * answer_passes / n,
            n_questions=len(data),
            elapsed_s=elapsed,
            failure_modes=failure_modes,
        )
    finally:
        index.close()
        embedder.close()
        runtime.close()
