"""
Offline Prompt Evolution Runner (Sprint 11)
-------------------------------------------
Runs PromptEvolver with Thompson-sampled mutation operators to evolve
the system prompt for better generation quality.

Flow:
  1. Load current system prompt (from agent/prompts.py or custom seed)
  2. Load eval queries (from eval set or telemetry weak queries)
  3. Run evolution loop (mutate, evaluate, keep/discard)
  4. Save winning prompt + operator report

Usage:
    python scripts/evolve_prompts.py                      # defaults
    python scripts/evolve_prompts.py --generations 10     # more experiments
    python scripts/evolve_prompts.py --dry-run             # preview config
    python scripts/evolve_prompts.py --seed-prompt path    # custom seed
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))


def _fix_stdout():
    if sys.platform == "win32" and hasattr(sys.stdout, "buffer"):
        import io
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace")


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_eval_queries(eval_set_path: str, max_queries: int = 50) -> List[str]:
    """Load evaluation questions from the eval set JSON."""
    path = Path(eval_set_path)
    if not path.exists():
        print(f"[WARN] Eval set not found: {eval_set_path}")
        return []
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    questions = [q["question"] for q in data if "question" in q]
    return questions[:max_queries]


def _load_failure_examples(telemetry_db: str, limit: int = 20) -> List[Dict[str, str]]:
    """Pull recent low-confidence queries from telemetry as failure examples."""
    import sqlite3
    db_path = Path(telemetry_db)
    if not db_path.exists():
        return []
    try:
        conn = sqlite3.connect(str(db_path))
        cur = conn.execute(
            "SELECT query, answer FROM query_events "
            "WHERE confidence < 0.4 "
            "ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        )
        rows = cur.fetchall()
        conn.close()
        return [{"query": r[0], "bad_answer": (r[1] or "")[:200]} for r in rows]
    except Exception:
        return []


def _default_seed_prompt() -> str:
    """Load the current code prompt from agent/prompts.py."""
    try:
        from agent.prompts import CODE_SYSTEM_PROMPT
        return CODE_SYSTEM_PROMPT
    except ImportError:
        return (
            "You are an expert coding assistant. Answer questions about code "
            "using ONLY the provided context. Be precise and include code "
            "examples when relevant. If you cannot answer from the context, "
            "say so clearly."
        )


def run_evolution(
    eval_set_path: str,
    generations: int = 5,
    evals_per_candidate: int = 3,
    seed_prompt: Optional[str] = None,
    output_dir: str = "logs/prompt_evolution",
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Run one prompt evolution session."""
    from core.runtime import Runtime
    from core.config import load_config
    from core.prompt_evolver import PromptEvolver

    config = load_config()
    prompt_text = seed_prompt or _default_seed_prompt()
    queries = _load_eval_queries(eval_set_path)

    if not queries:
        print("[FAIL] No evaluation queries loaded. Cannot run evolution.")
        return {"error": "no_queries"}

    print(f"[OK] Loaded {len(queries)} eval queries")
    print(f"[OK] Seed prompt: {len(prompt_text)} chars")
    print(f"[OK] Generations: {generations}, evals/candidate: {evals_per_candidate}")

    if dry_run:
        print("[OK] Dry run -- exiting before evolution.")
        return {"dry_run": True, "queries": len(queries)}

    runtime = Runtime(config.model, timeout=120)

    # Build eval function: use prompt as system_prompt, score the answer
    def eval_fn(candidate_prompt: str, query: str) -> float:
        try:
            answer = runtime.generate(
                question=query,
                context_chunks=[],
                system_prompt=candidate_prompt,
                temperature=0.1,
                max_tokens=256,
            )
            # Simple heuristic: reward code, length, confidence signals
            from core.cascade import estimate_answer_confidence
            return estimate_answer_confidence(answer)
        except Exception:
            return 0.0

    failures = _load_failure_examples("_telemetry/agent_events.db")
    print(f"[OK] Loaded {len(failures)} failure examples for extend mutations")

    evolver = PromptEvolver(
        runtime=runtime,
        eval_fn=eval_fn,
        db_path="_prompt_evo/lineage.db",
    )

    print(f"[OK] Starting evolution at {_timestamp()}")
    t0 = time.time()

    result = evolver.evolve(
        seed_prompt=prompt_text,
        eval_queries=queries,
        max_generations=generations,
        evals_per_candidate=evals_per_candidate,
        failure_examples=failures,
    )

    elapsed = time.time() - t0

    # Save results
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    report = {
        "timestamp": _timestamp(),
        "champion_prompt": result.champion.text,
        "champion_score": result.champion.avg_score,
        "generations_run": result.generations_run,
        "total_evaluations": result.total_evaluations,
        "kept": result.kept_count,
        "discarded": result.discarded_count,
        "score_history": result.score_history,
        "operator_stats": result.operator_stats,
        "elapsed_s": round(elapsed, 1),
    }
    report_path = out_dir / f"evolution_{ts}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Save champion prompt separately for easy use
    champion_path = out_dir / f"champion_{ts}.txt"
    with open(champion_path, "w", encoding="utf-8") as f:
        f.write(result.champion.text)

    print(f"[OK] Evolution complete in {elapsed:.1f}s")
    print(f"[OK] Champion score: {result.champion.avg_score:.3f}")
    print(f"[OK] Kept: {result.kept_count}, Discarded: {result.discarded_count}")
    print(f"[OK] Operator stats:")
    for op, stats in result.operator_stats.items():
        print(f"  {op}: mean={stats['mean']:.3f}, "
              f"keep_rate={stats['keep_rate']:.3f}, "
              f"uses={stats['total_uses']}")
    print(f"[OK] Report: {report_path}")
    print(f"[OK] Champion: {champion_path}")

    evolver.close()
    return report


def main():
    _fix_stdout()
    parser = argparse.ArgumentParser(description="Offline prompt evolution")
    parser.add_argument(
        "--eval-set", default="evaluation/agent_eval_set_200.json",
        help="Path to eval set JSON")
    parser.add_argument(
        "--generations", type=int, default=5,
        help="Number of evolution generations (experiments = generations * pop_size)")
    parser.add_argument(
        "--evals-per-candidate", type=int, default=3,
        help="Queries to test each candidate prompt on")
    parser.add_argument(
        "--seed-prompt", default=None,
        help="Path to custom seed prompt file (default: agent/prompts.py)")
    parser.add_argument(
        "--output-dir", default="logs/prompt_evolution",
        help="Output directory for reports")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview config without running evolution")
    args = parser.parse_args()

    seed = None
    if args.seed_prompt:
        p = Path(args.seed_prompt)
        if p.exists():
            seed = p.read_text(encoding="utf-8")
        else:
            print(f"[FAIL] Seed prompt file not found: {args.seed_prompt}")
            sys.exit(1)

    run_evolution(
        eval_set_path=args.eval_set,
        generations=args.generations,
        evals_per_candidate=args.evals_per_candidate,
        seed_prompt=seed,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
