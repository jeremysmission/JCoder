"""Run eval through the FULL JCoder pipeline (retrieval + generation).

Unlike run_eval_api.py (direct API call) and run_eval_local.py (direct
Ollama call), this script instantiates the real JCoder stack:

  query -> Runtime.generate(question, retrieved_context) -> score

Supports three modes:
  --backend openai   : GPT-4o or any OpenAI-compatible API
  --backend ollama   : Local Ollama (phi4:14b, etc.)
  --backend agent    : Full Agent loop with tool use

Usage:
    # GPT-4o through JCoder Runtime (generation pipeline only)
    python scripts/run_eval_pipeline.py --backend openai --model gpt-4o --max 10

    # Phi-4 14B through JCoder Runtime
    python scripts/run_eval_pipeline.py --backend ollama --model phi4:14b --max 10

    # Full Agent loop with GPT-4o
    python scripts/run_eval_pipeline.py --backend agent --model gpt-4o --max 5

    # With RAG retrieval from FTS5 indexes
    python scripts/run_eval_pipeline.py --backend ollama --model phi4:14b --rag --max 10

    # Test cascade routing with multiple models
    python scripts/run_eval_pipeline.py --backend openai --model gpt-4o --cascade --max 10
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
# Pipeline builders
# ---------------------------------------------------------------------------

def _build_runtime(endpoint: str, model: str, api_key: str, timeout: float):
    """Build a JCoder Runtime pointing at the given endpoint."""
    from core.config import ModelConfig
    from core.network_gate import NetworkGate
    from core.runtime import Runtime

    config = ModelConfig(name=model, endpoint=endpoint)
    # Allow API endpoints; local stays localhost
    if "api.openai.com" in endpoint or "openrouter" in endpoint:
        gate = NetworkGate(mode="allowlist", allowlist=[endpoint.split("//")[1].split("/")[0]])
    else:
        gate = NetworkGate(mode="localhost")

    return Runtime(config, timeout=timeout, gate=gate, api_key=api_key)


def _build_rag_retriever(data_dir: str, top_k: int = 5):
    """Build a federated FTS5 retriever for RAG context."""
    try:
        from agent.config_loader import load_agent_config, _build_federated, _discover_fts5_indexes
        config = load_agent_config()
        if data_dir:
            config.federated_data_dir = data_dir
        fed = _build_federated(config, embedding_engine=None)
        if fed:
            print(f"[OK] RAG loaded: {len(fed.list_indexes())} FTS5 indexes")
        return fed
    except Exception as e:
        print(f"[WARN] RAG setup failed: {e}")
        return None


def _retrieve_context(fed, question: str, top_k: int = 5) -> list[dict]:
    """Retrieve RAG chunks for a question. Returns list of chunk dicts."""
    if fed is None:
        return []
    try:
        results = fed.search(question, top_k=top_k)
        chunks = []
        for r in results:
            content = getattr(r, "content", str(r))[:800]
            source = getattr(r, "source", getattr(r, "index_name", "unknown"))
            chunks.append({
                "content": content,
                "source_path": source,
                "id": f"rag_{hash(content) & 0xFFFFFF:06x}",
            })
        return chunks
    except Exception as e:
        print(f"[WARN] RAG retrieval failed: {e}")
        return []


def _build_agent(endpoint: str, model: str, api_key: str, timeout: float):
    """Build a full JCoder Agent with tools."""
    from agent.llm_backend import create_backend
    from agent.core import Agent
    from agent.tools import ToolRegistry

    backend_type = "openai"
    if "anthropic" in endpoint:
        backend_type = "anthropic"

    backend = create_backend(
        backend_type=backend_type,
        endpoint=endpoint,
        model=model,
        api_key=api_key,
        timeout_s=timeout,
    )
    tools = ToolRegistry(working_dir=str(_ROOT))
    return Agent(
        backend=backend,
        tools=tools,
        max_iterations=10,
        max_tokens_budget=100_000,
    )


def _build_cascade(endpoint: str, model: str, api_key: str, timeout: float):
    """Build a ModelCascade with the given model as the primary level."""
    from core.config import ModelConfig
    from core.network_gate import NetworkGate
    from core.cascade import ModelCascade, CascadeLevel, estimate_answer_confidence

    config = ModelConfig(name=model, endpoint=endpoint)
    if "api.openai.com" in endpoint or "openrouter" in endpoint:
        gate = NetworkGate(mode="allowlist", allowlist=[endpoint.split("//")[1].split("/")[0]])
    else:
        gate = NetworkGate(mode="localhost")

    levels = [
        CascadeLevel(name="primary", model_config=config,
                      max_complexity=1.0, timeout_s=int(timeout)),
    ]
    return ModelCascade(levels, gate=gate, confidence_fn=estimate_answer_confidence)


# ---------------------------------------------------------------------------
# Cost tracking (same as run_eval_api.py)
# ---------------------------------------------------------------------------

_COST_TABLE = {
    "gpt-5.4":       {"input": 10.0,  "output": 30.0},
    "gpt-5":         {"input": 10.0,  "output": 30.0},
    "gpt-4.1":       {"input": 2.0,   "output": 8.0},
    "gpt-4.1-mini":  {"input": 0.4,   "output": 1.6},
    "gpt-4.1-nano":  {"input": 0.1,   "output": 0.4},
    "gpt-4o":        {"input": 2.5,   "output": 10.0},
    "gpt-4o-mini":   {"input": 0.15,  "output": 0.6},
}


def _estimate_cost(model: str, tokens: int) -> float:
    rates = _COST_TABLE.get(model, {"input": 0.0, "output": 0.0})
    # Rough 60/40 input/output split
    return (tokens * 0.6 * rates["input"] + tokens * 0.4 * rates["output"]) / 1_000_000


def _is_local(model: str) -> bool:
    """Detect local models that don't have API cost."""
    return any(m in model.lower() for m in ("phi4", "phi-4", "qwen", "llama", "mistral", "devstral"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run eval through the full JCoder pipeline")
    parser.add_argument("--backend", default="openai",
                        choices=["openai", "ollama", "agent"],
                        help="Backend type")
    parser.add_argument("--model", default="gpt-4o", help="Model name")
    parser.add_argument("--endpoint", default="",
                        help="API endpoint (auto-detected from backend)")
    parser.add_argument("--eval-set",
                        default=str(_ROOT / "evaluation" / "agent_eval_set_200.json"))
    parser.add_argument("--output-dir", default="", help="Output dir")
    parser.add_argument("--max", type=int, default=0, help="Max questions")
    parser.add_argument("--category", action="append", help="Filter by category")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--rag", action="store_true",
                        help="Include RAG context from FTS5 indexes")
    parser.add_argument("--rag-dir", default="", help="FTS5 index directory")
    parser.add_argument("--rag-top-k", type=int, default=5)
    parser.add_argument("--cascade", action="store_true",
                        help="Use ModelCascade routing")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--budget", type=float, default=5.0,
                        help="Max spend in USD (0=unlimited, ignored for local)")
    args = parser.parse_args()

    # Endpoint defaults
    if not args.endpoint:
        if args.backend == "ollama":
            args.endpoint = "http://localhost:11434/v1"
        elif args.backend == "openai":
            args.endpoint = "https://api.openai.com/v1"
        elif args.backend == "agent":
            args.endpoint = "https://api.openai.com/v1"

    # API key
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not _is_local(args.model) and not api_key:
        print("[FAIL] OPENAI_API_KEY not set (required for cloud models)")
        sys.exit(1)

    # Output dir
    if not args.output_dir:
        model_slug = args.model.replace(".", "_").replace(":", "_").replace("/", "_")
        mode = f"pipeline_{args.backend}"
        if args.rag:
            mode += "_rag"
        if args.cascade:
            mode += "_cascade"
        args.output_dir = f"evaluation/results_{mode}_{model_slug}"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    is_local = _is_local(args.model)

    # --- Build pipeline components ---
    print(f"[OK] Backend: {args.backend}")
    print(f"[OK] Model: {args.model}")
    print(f"[OK] Endpoint: {args.endpoint}")

    runtime = None
    agent = None
    cascade = None
    fed = None

    if args.backend == "agent":
        print("[OK] Building full Agent pipeline...")
        agent = _build_agent(args.endpoint, args.model, api_key, args.timeout)
        print(f"[OK] Agent ready (max_iterations=10)")
    elif args.cascade:
        print("[OK] Building ModelCascade...")
        cascade = _build_cascade(args.endpoint, args.model, api_key, args.timeout)
        print(f"[OK] Cascade ready ({len(cascade.levels)} levels)")
    else:
        print("[OK] Building Runtime...")
        runtime = _build_runtime(args.endpoint, args.model, api_key, args.timeout)
        print("[OK] Runtime ready")

    # Connectivity check
    if runtime:
        try:
            answer = runtime.generate("Reply with exactly: HELLO", [])
            print(f"[OK] Runtime verified: {answer.strip()!r}")
        except Exception as e:
            print(f"[FAIL] Runtime check failed: {e}")
            sys.exit(1)
    elif cascade:
        try:
            result = cascade.route("Reply with exactly: HELLO", [])
            print(f"[OK] Cascade verified: {result.answer.strip()!r} (model={result.model_used})")
        except Exception as e:
            print(f"[FAIL] Cascade check failed: {e}")
            sys.exit(1)

    # RAG setup
    if args.rag:
        rag_dir = args.rag_dir or str(_ROOT / "data" / "indexes")
        fed = _build_rag_retriever(rag_dir, args.rag_top_k)

    # Eval runner
    runner = AgentEvalRunner(args.eval_set, agent=agent, output_dir=args.output_dir)
    issues = runner.validate_eval_set()
    if issues:
        print(f"[WARN] {len(issues)} eval set issues (continuing)")
    else:
        print(f"[OK] Eval set valid: {len(runner.questions)} questions")

    qs = runner.questions
    if args.category:
        cats = {c.lower() for c in args.category}
        qs = [q for q in qs if q["category"].lower() in cats]
    if args.max > 0:
        qs = qs[:args.max]

    print(f"\nRunning {len(qs)} questions | backend={args.backend} | "
          f"model={args.model} | rag={'ON' if args.rag else 'OFF'} | "
          f"cascade={'ON' if args.cascade else 'OFF'}")
    if not is_local and args.budget > 0:
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
    total_tokens = 0
    total_cost = 0.0

    for idx, q in enumerate(qs, 1):
        qid = q["id"]

        if qid in completed:
            prev = completed[qid]
            results.append(EvalResult(**prev))
            total_tokens += prev.get("tokens_used", 0)
            print(f"  [{idx}/{total}] {qid}: RESUMED (score={prev['score']:.2f})")
            continue

        # Budget guard (API models only)
        if not is_local and args.budget > 0 and total_cost >= args.budget:
            print(f"\n[WARN] Budget cap reached (${total_cost:.2f}). Stopping.")
            break

        print(f"  [{idx}/{total}] {qid}: running...", end="", flush=True)

        # RAG retrieval
        rag_chunks = _retrieve_context(fed, q["question"], args.rag_top_k) if fed else []
        chunk_texts = [c["content"] for c in rag_chunks]

        t0 = time.time()
        answer = ""
        tokens = 0

        try:
            if args.backend == "agent":
                # Full agent loop
                result = agent.run(q["question"])
                answer = result.summary
                tokens = result.tokens
            elif args.cascade:
                # Cascade routing
                cr = cascade.route(q["question"], chunk_texts)
                answer = cr.answer
            else:
                # Direct Runtime generation
                answer = runtime.generate(q["question"], chunk_texts)
        except Exception as e:
            answer = f"[ERROR: {e}]"

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
        total_tokens += tokens

        if not is_local:
            total_cost = _estimate_cost(args.model, total_tokens)

        tag = "PASS" if er.passed else "FAIL"
        cost_str = f", ${total_cost:.3f}" if not is_local else ""
        print(f" {tag} (score={er.score:.2f}, {er.elapsed_s:.1f}s{cost_str})")

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

    # Pipeline metadata
    meta = {
        "backend": args.backend,
        "model": args.model,
        "endpoint": args.endpoint,
        "rag": args.rag,
        "cascade": args.cascade,
        "temperature": args.temperature,
        "total_tokens": total_tokens,
        "estimated_cost_usd": round(total_cost, 4) if not is_local else 0.0,
        "wall_time_s": round(wall_time, 1),
        "questions_run": len([r for r in results if r.elapsed_s > 0]),
    }
    if args.cascade and cascade:
        meta["cascade_stats"] = cascade.stats()
    with open(output_dir / "pipeline_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # Print summary
    s = runner.summary(results)
    print(f"\n{'='*60}")
    print(f"Backend: {args.backend} | Model: {args.model}")
    print(f"RAG: {'ON' if args.rag else 'OFF'} | Cascade: {'ON' if args.cascade else 'OFF'}")
    print(f"Questions: {s['total']}")
    print(f"Passed: {s.get('passed', 0)} ({s.get('pass_rate', 0):.1%})")
    print(f"Avg score: {s.get('avg_score', 0):.4f}")
    print(f"Avg latency: {s.get('avg_latency_s', 0):.1f}s")
    print(f"Total time: {wall_time:.0f}s ({wall_time/60:.1f}min)")
    if not is_local:
        print(f"Estimated cost: ${total_cost:.4f}")
    if args.cascade and cascade:
        cs = cascade.stats()
        print(f"Cascade escalation rate: {cs['escalation_rate']:.1%}")
    print(f"\nPer category:")
    for cat, info in s.get("per_category", {}).items():
        print(f"  {cat:15s}: {info['avg_score']:.3f} avg, {info['pass_rate']:.0%} pass ({info['count']}q)")
    print(f"\nWorst 5:")
    for w in s.get("worst_5", []):
        print(f"  {w['id']}: {w['score']:.4f}")
    print(f"{'='*60}")

    # Cleanup
    if runtime:
        runtime.close()
    if cascade:
        cascade.close()


if __name__ == "__main__":
    main()
