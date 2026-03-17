"""Run eval set against an OpenAI API model (GPT-5, GPT-4.1, etc).

Sends each question to the API with optional RAG context from
federated search, scores using AgentEvalRunner's deterministic scorer.

Usage:
    cd D:\\JCoder
    python scripts/run_eval_api.py --model gpt-4.1-mini --max 5
    python scripts/run_eval_api.py --model gpt-5.4 --rag
    python scripts/run_eval_api.py --model gpt-5.4 --rag --category python
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from evaluation.agent_eval_runner import AgentEvalRunner, EvalResult

try:
    import httpx
except ImportError:
    print("[FAIL] httpx not installed")
    sys.exit(1)


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def _query_api(
    question: str,
    model: str,
    api_key: str,
    endpoint: str = "https://api.openai.com/v1",
    rag_context: str = "",
    temperature: float = 0.1,
    timeout: float = 120.0,
) -> tuple[str, int, float]:
    """Send question to OpenAI API. Returns (answer, tokens, elapsed_s)."""

    system = (
        "You are a coding assistant. Answer clearly and concisely. "
        "Include code examples in markdown code blocks when appropriate. "
        "Include relevant imports. Cite sources if applicable."
    )

    user_msg = question
    if rag_context:
        user_msg = (
            "Use the following retrieved context to inform your answer. "
            "If the context is not relevant, answer from your own knowledge.\n\n"
            f"--- RETRIEVED CONTEXT ---\n{rag_context}\n--- END CONTEXT ---\n\n"
            f"Question: {question}"
        )

    # gpt-5+, gpt-4.1, o-series require max_completion_tokens
    new_style = any(model.startswith(p) for p in ("gpt-5", "gpt-4.1", "o1", "o3", "o4"))
    token_key = "max_completion_tokens" if new_style else "max_tokens"

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg},
        ],
        "temperature": temperature,
        token_key: 4096,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    t0 = time.monotonic()
    with httpx.Client(timeout=httpx.Timeout(timeout)) as client:
        resp = client.post(f"{endpoint}/chat/completions", json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()

    elapsed = time.monotonic() - t0
    msg = data["choices"][0]["message"]
    usage = data.get("usage", {})
    tokens = usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)

    return msg.get("content", ""), tokens, elapsed


# ---------------------------------------------------------------------------
# RAG retrieval (optional)
# ---------------------------------------------------------------------------

def _build_rag(data_dir: str, max_indexes: int = 0):
    """Build federated search from FTS5 indexes for RAG context."""
    from agent.config_loader import load_agent_config, _build_federated, _discover_fts5_indexes

    config = load_agent_config()
    if data_dir:
        config.federated_data_dir = data_dir

    fed = _build_federated(config, embedding_engine=None)
    if fed:
        count = len(fed.list_indexes())
        print(f"[OK] RAG loaded: {count} FTS5 indexes")
    else:
        print("[WARN] No RAG indexes found")
    return fed


def _retrieve_context(fed, question: str, top_k: int = 5) -> str:
    """Retrieve RAG context for a question."""
    if fed is None:
        return ""
    try:
        results = fed.search(question, top_k=top_k)
        if not results:
            return ""
        lines = []
        for i, r in enumerate(results, 1):
            source = getattr(r, "source", getattr(r, "index_name", "unknown"))
            score = getattr(r, "score", 0.0)
            content = getattr(r, "content", str(r))[:600]
            lines.append(f"[{i}] (source={source}, score={score:.3f})\n{content}")
        return "\n\n".join(lines)
    except Exception as e:
        return f"[RAG error: {e}]"


# ---------------------------------------------------------------------------
# Cost tracking
# ---------------------------------------------------------------------------

# Approximate costs per 1M tokens (USD) -- update as pricing changes
_COST_TABLE = {
    "gpt-5.4":       {"input": 10.0,  "output": 30.0},
    "gpt-5":         {"input": 10.0,  "output": 30.0},
    "gpt-4.1":       {"input": 2.0,   "output": 8.0},
    "gpt-4.1-mini":  {"input": 0.4,   "output": 1.6},
    "gpt-4.1-nano":  {"input": 0.1,   "output": 0.4},
    "gpt-4o":        {"input": 2.5,   "output": 10.0},
    "gpt-4o-mini":   {"input": 0.15,  "output": 0.6},
}


def _estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate cost in USD."""
    rates = _COST_TABLE.get(model, {"input": 5.0, "output": 15.0})
    return (input_tokens * rates["input"] + output_tokens * rates["output"]) / 1_000_000


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run eval with OpenAI API model")
    parser.add_argument("--model", default="gpt-4.1-mini", help="Model name")
    parser.add_argument("--endpoint", default="https://api.openai.com/v1")
    parser.add_argument("--eval-set", default=str(_ROOT / "evaluation" / "agent_eval_set_200.json"))
    parser.add_argument("--output-dir", default="", help="Output dir (auto-named if empty)")
    parser.add_argument("--max", type=int, default=0, help="Max questions (0=all)")
    parser.add_argument("--category", action="append", help="Filter by category")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--rag", action="store_true", help="Include RAG context from FTS5 indexes")
    parser.add_argument("--rag-dir", default="", help="FTS5 index directory for RAG")
    parser.add_argument("--rag-top-k", type=int, default=5, help="RAG retrieval top-k")
    parser.add_argument("--budget", type=float, default=5.0, help="Max spend in USD (0=unlimited)")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("[FAIL] OPENAI_API_KEY not set")
        sys.exit(1)

    # Output directory
    if not args.output_dir:
        model_slug = args.model.replace(".", "_").replace(":", "_")
        rag_tag = "_rag" if args.rag else ""
        args.output_dir = f"evaluation/results_api_{model_slug}{rag_tag}"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Verify API connectivity
    print(f"[OK] Model: {args.model}")
    print(f"[OK] Endpoint: {args.endpoint}")
    try:
        answer, tokens, elapsed = _query_api(
            "Reply with exactly: HELLO", args.model, api_key,
            args.endpoint, timeout=15.0,
        )
        print(f"[OK] API verified: {answer.strip()!r} ({elapsed:.1f}s, {tokens} tokens)")
    except Exception as e:
        print(f"[FAIL] API check failed: {e}")
        sys.exit(1)

    # RAG setup
    fed = None
    if args.rag:
        rag_dir = args.rag_dir or str(_ROOT / "data" / "indexes")
        fed = _build_rag(rag_dir)

    # Eval runner
    runner = AgentEvalRunner(args.eval_set, agent=None, output_dir=args.output_dir)
    issues = runner.validate_eval_set()
    if issues:
        print(f"[WARN] {len(issues)} eval set issues (continuing)")
    else:
        print(f"[OK] Eval set valid: {len(runner.questions)} questions")

    # Filter
    qs = runner.questions
    if args.category:
        cats = {c.lower() for c in args.category}
        qs = [q for q in qs if q["category"].lower() in cats]
    if args.max > 0:
        qs = qs[:args.max]

    print(f"\nRunning {len(qs)} questions | model={args.model} | "
          f"rag={'ON' if args.rag else 'OFF'} | temp={args.temperature}")
    if args.budget > 0:
        print(f"Budget cap: ${args.budget:.2f}")
    print(f"Output: {args.output_dir}\n")

    # Resume
    results_path = output_dir / "eval_results.json"
    completed: dict[str, dict] = {}
    if not args.no_resume and results_path.exists():
        with open(results_path, "r", encoding="utf-8") as f:
            for r in json.load(f):
                completed[r["question_id"]] = r
        print(f"Resuming: {len(completed)} previously completed\n")

    results: list[EvalResult] = []
    total = len(qs)
    t0_all = time.time()
    total_input = 0
    total_output = 0
    total_cost = 0.0

    for idx, q in enumerate(qs, 1):
        qid = q["id"]

        if qid in completed:
            prev = completed[qid]
            results.append(EvalResult(**prev))
            print(f"  [{idx}/{total}] {qid}: RESUMED (score={prev['score']:.2f})")
            continue

        # Budget guard
        if args.budget > 0 and total_cost >= args.budget:
            print(f"\n[WARN] Budget cap reached (${total_cost:.2f} >= ${args.budget:.2f}). Stopping.")
            break

        # RAG retrieval
        rag_ctx = _retrieve_context(fed, q["question"], top_k=args.rag_top_k) if fed else ""

        print(f"  [{idx}/{total}] {qid}: running...", end="", flush=True)

        t0 = time.time()
        try:
            answer, tokens, api_elapsed = _query_api(
                q["question"], args.model, api_key, args.endpoint,
                rag_context=rag_ctx, temperature=args.temperature,
                timeout=args.timeout,
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

        # Cost tracking (rough estimate from token count)
        est_input = int(tokens * 0.6)
        est_output = tokens - est_input
        total_input += est_input
        total_output += est_output
        total_cost = _estimate_cost(args.model, total_input, total_output)

        tag = "PASS" if er.passed else "FAIL"
        print(f" {tag} (score={er.score:.2f}, {er.elapsed_s:.1f}s, ${total_cost:.3f})")

        # Incremental save
        tmp = results_path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump([asdict(r) for r in results], f, indent=2)
        tmp.replace(results_path)

    wall_time = time.time() - t0_all

    # Report
    runner._completed = completed
    report_path = str(output_dir / "eval_report.md")
    runner.report(results, report_path)

    # Cost summary
    cost_summary = {
        "model": args.model,
        "rag": args.rag,
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "estimated_cost_usd": round(total_cost, 4),
        "wall_time_s": round(wall_time, 1),
        "questions_run": len([r for r in results if r.elapsed_s > 0]),
    }
    cost_path = output_dir / "cost_summary.json"
    with open(cost_path, "w", encoding="utf-8") as f:
        json.dump(cost_summary, f, indent=2)

    # Print summary
    s = runner.summary(results)
    print(f"\n{'='*60}")
    print(f"Model: {args.model} | RAG: {'ON' if args.rag else 'OFF'}")
    print(f"Questions: {s['total']}")
    print(f"Passed: {s.get('passed', 0)} ({s.get('pass_rate', 0):.1%})")
    print(f"Avg score: {s.get('avg_score', 0):.4f}")
    print(f"Avg latency: {s.get('avg_latency_s', 0):.1f}s")
    print(f"Total time: {wall_time:.0f}s ({wall_time/60:.1f}min)")
    print(f"Estimated cost: ${total_cost:.4f}")
    print(f"\nPer category:")
    for cat, info in s.get("per_category", {}).items():
        print(f"  {cat:15s}: {info['avg_score']:.3f} avg, {info['pass_rate']:.0%} pass ({info['count']}q)")
    print(f"\nWorst 5:")
    for w in s.get("worst_5", []):
        print(f"  {w['id']}: {w['score']:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
