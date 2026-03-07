"""Run eval set against a local Ollama model (no full agent loop).

Sends each question directly to the model, scores the response
using AgentEvalRunner's deterministic scorer.

Usage:
    cd D:\\JCoder
    python scripts/run_eval_local.py
    python scripts/run_eval_local.py --model phi4-mini --max 10
    python scripts/run_eval_local.py --eval-set evaluation/agent_eval_set.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

# Project root
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

# Windows stdout fix
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from evaluation.agent_eval_runner import AgentEvalRunner, EvalResult

try:
    import httpx
except ImportError:
    print("[FAIL] httpx not installed")
    sys.exit(1)


def _query_ollama(prompt: str, model: str, endpoint: str,
                  temperature: float = 0.1, timeout: float = 120.0) -> tuple[str, int]:
    """Send a prompt to Ollama and return (response_text, token_count)."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature},
    }
    resp = httpx.post(
        f"{endpoint}/api/generate",
        json=payload,
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    text = data.get("response", "")
    tokens = data.get("eval_count", 0) + data.get("prompt_eval_count", 0)
    return text, tokens


def _build_prompt(question: str) -> str:
    """Wrap the eval question in a coding assistant prompt."""
    return (
        "You are a coding assistant. Answer the following question clearly and "
        "concisely. Include code examples in markdown code blocks when appropriate. "
        "Include relevant imports. Cite sources if applicable.\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )


def main():
    parser = argparse.ArgumentParser(description="Run eval with local Ollama model")
    parser.add_argument("--model", default="phi4-mini", help="Ollama model name")
    parser.add_argument("--endpoint", default="http://localhost:11434", help="Ollama endpoint")
    parser.add_argument("--eval-set", default=str(_ROOT / "evaluation" / "agent_eval_set_200.json"))
    parser.add_argument("--output-dir", default="evaluation/results_local")
    parser.add_argument("--max", type=int, default=0, help="Max questions (0=all)")
    parser.add_argument("--category", action="append", help="Filter by category")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--timeout", type=float, default=120.0, help="Per-question timeout (s)")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    # Verify Ollama connectivity
    try:
        r = httpx.get(f"{args.endpoint}/api/tags", timeout=5)
        models = [m["name"] for m in r.json().get("models", [])]
        if not any(args.model in m for m in models):
            print(f"[WARN] Model '{args.model}' not found in Ollama. Available: {models}")
    except Exception as e:
        print(f"[FAIL] Cannot connect to Ollama at {args.endpoint}: {e}")
        sys.exit(1)

    runner = AgentEvalRunner(args.eval_set, agent=None, output_dir=args.output_dir)

    # Validate first
    issues = runner.validate_eval_set()
    if issues:
        print(f"[WARN] {len(issues)} eval set issues:")
        for i in issues[:5]:
            print(f"  - {i}")
        # Continue anyway -- warnings not fatal
    else:
        print(f"[OK] Eval set valid: {len(runner.questions)} questions")

    # Filter
    qs = runner.questions
    if args.category:
        cats = {c.lower() for c in args.category}
        qs = [q for q in qs if q["category"].lower() in cats]
    if args.max > 0:
        qs = qs[:args.max]

    print(f"\nRunning {len(qs)} questions with model={args.model}, temp={args.temperature}")
    print(f"Output: {args.output_dir}\n")

    # Load previous results for resume
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "eval_results.json"
    completed = {}
    if not args.no_resume and results_path.exists():
        with open(results_path, "r", encoding="utf-8") as f:
            for r in json.load(f):
                completed[r["question_id"]] = r
        print(f"Resuming: {len(completed)} previously completed\n")

    results: list[EvalResult] = []
    total = len(qs)
    t0_all = time.time()

    for idx, q in enumerate(qs, 1):
        qid = q["id"]

        # Resume check
        if qid in completed:
            prev = completed[qid]
            results.append(EvalResult(**prev))
            print(f"  [{idx}/{total}] {qid}: RESUMED (score={prev['score']:.2f})")
            continue

        prompt = _build_prompt(q["question"])
        print(f"  [{idx}/{total}] {qid}: running...", end="", flush=True)

        t0 = time.time()
        try:
            answer, tokens = _query_ollama(
                prompt, args.model, args.endpoint,
                temperature=args.temperature, timeout=args.timeout,
            )
        except Exception as e:
            answer = f"[ERROR: {e}]"
            tokens = 0
        elapsed = time.time() - t0

        sub = runner.score_answer(q, answer)
        score = sub.get("weighted_total", 0.0)
        er = EvalResult(
            question_id=qid,
            category=q["category"],
            score=round(score, 4),
            subscores=sub,
            answer=answer,
            elapsed_s=round(elapsed, 3),
            tokens_used=tokens,
            passed=score >= 0.5,
        )
        results.append(er)
        completed[qid] = asdict(er)
        tag = "PASS" if er.passed else "FAIL"
        print(f" {tag} (score={er.score:.2f}, {er.elapsed_s:.1f}s)")

        # Incremental save
        tmp = results_path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump([asdict(r) for r in results], f, indent=2)
        tmp.replace(results_path)

    total_time = time.time() - t0_all

    # Generate report
    runner._completed = completed
    report_text = runner.report(results, str(output_dir / "eval_report.md"))

    # Print summary
    s = runner.summary(results)
    print(f"\n{'='*60}")
    print(f"Model: {args.model}")
    print(f"Questions: {s['total']}")
    print(f"Passed: {s.get('passed', 0)} ({s.get('pass_rate', 0):.1%})")
    print(f"Avg score: {s.get('avg_score', 0):.4f}")
    print(f"Avg latency: {s.get('avg_latency_s', 0):.1f}s")
    print(f"Total time: {total_time:.0f}s ({total_time/60:.1f}min)")
    print(f"\nPer category:")
    for cat, info in s.get("per_category", {}).items():
        print(f"  {cat:15s}: {info['avg_score']:.3f} avg, {info['pass_rate']:.0%} pass ({info['count']}q)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
