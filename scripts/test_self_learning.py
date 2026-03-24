"""Test JCoder's self-learning pipeline end-to-end.

Exercises: telemetry -> active learner -> study -> distillation -> re-test.
Uses an outside API (GPT) as teacher and verifier.

Usage:
    cd D:\\JCoder
    python scripts/test_self_learning.py
    python scripts/test_self_learning.py --model gpt-4.1-mini --max-questions 10
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import httpx


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class LearningTestResult:
    question_id: str
    category: str
    # Before learning
    baseline_score: float
    baseline_answer_snippet: str
    # After distillation
    post_score: float
    post_answer_snippet: str
    # Attribution check
    distilled_marker: str  # the [DISTILLED: ...] fact injected
    marker_found_in_answer: bool  # did the model use the distilled fact?
    # Meta
    improvement: float
    learned_from_rag: bool  # True = improvement + marker found


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def call_api(system: str, user: str, model: str, api_key: str,
             temperature: float = 0.1, max_tokens: int = 4096) -> dict:
    """Call OpenAI API."""
    new_style = any(model.startswith(p) for p in ("gpt-5", "gpt-4.1", "o1", "o3", "o4"))
    token_key = "max_completion_tokens" if new_style else "max_tokens"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
        token_key: max_tokens,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    t0 = time.monotonic()
    with httpx.Client(timeout=httpx.Timeout(180.0)) as client:
        resp = client.post("https://api.openai.com/v1/chat/completions",
                          json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
    msg = data["choices"][0]["message"]
    usage = data.get("usage", {})
    return {
        "content": msg.get("content", ""),
        "input_tokens": usage.get("prompt_tokens", 0),
        "output_tokens": usage.get("completion_tokens", 0),
        "elapsed_s": time.monotonic() - t0,
    }


# ---------------------------------------------------------------------------
# Learning pipeline test
# ---------------------------------------------------------------------------

def load_baseline_failures(results_path: str, eval_set_path: str,
                           max_questions: int = 10) -> list[dict]:
    """Load weakest questions from baseline eval."""
    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)
    with open(eval_set_path, "r", encoding="utf-8") as f:
        questions = {q["id"]: q for q in json.load(f)}

    # Sort by score ascending (weakest first)
    results.sort(key=lambda r: r["score"])
    weak = []
    for r in results:
        if r["question_id"] in questions:
            q = questions[r["question_id"]]
            weak.append({**r, **q})
        if len(weak) >= max_questions:
            break
    return weak


def feed_to_telemetry(weak_questions: list[dict]) -> None:
    """Simulate feeding failures into telemetry store."""
    try:
        from core.telemetry import TelemetryStore, QueryEvent
        telemetry = TelemetryStore(str(_ROOT / "_telemetry" / "agent_events.db"))

        for q in weak_questions:
            event = QueryEvent(
                query_id=q["question_id"],
                query_text=q["question"],
                timestamp=time.time(),
                retrieval_latency_ms=0.0,
                generation_latency_ms=q.get("elapsed_s", 0) * 1000,
                chunk_ids=[],
                chunk_scores=[],
                source_files=[],
                answer_snippet=q.get("answer", "")[:200],
                confidence=q["score"],
                reflection_relevant=0.0,
                reflection_supported=0.0,
                reflection_useful=0.0,
                feedback="bad" if q["score"] < 0.5 else None,
            )
            telemetry.log(event)

        stats = telemetry.stats()
        print(f"[OK] Telemetry: {stats.get('total', 0)} events logged")
        low = telemetry.low_confidence_queries(threshold=0.5)
        print(f"[OK] Low-confidence queries: {len(low)}")
    except Exception as e:
        print(f"[WARN] Telemetry feed failed (non-fatal): {e}")


def distill_with_marker(question: dict, model: str, api_key: str) -> tuple[str, str]:
    """Distill knowledge with a [DISTILLED: ...] attribution marker.

    Returns (distilled_content, marker_text).
    """
    system = (
        "You are creating a reference document for a small AI model. "
        "Be self-contained, include complete code examples with imports. "
        "IMPORTANT: Include exactly ONE specific, non-obvious fact wrapped in "
        "[DISTILLED: <fact>] format. This fact should be something that cannot "
        "be trivially derived from general knowledge -- a specific version "
        "behavior, a real benchmark number, an edge case, or precise API detail. "
        "Example: [DISTILLED: Python 3.12 added the -P flag to disable "
        "sys.path[0] auto-insertion]"
    )
    user = (
        f"Create a reference document to help answer this coding question:\n\n"
        f"Question: {question['question']}\n\n"
        f"Expected keywords: {question.get('expected_keywords', [])}\n"
        f"Expected imports: {question.get('expected_imports', [])}\n"
        f"Category: {question['category']}\n\n"
        f"Write a clear, complete reference with code examples."
    )

    result = call_api(system, user, model, api_key, temperature=0.3)
    content = result["content"]

    # Extract the marker
    import re
    marker_match = re.search(r"\[DISTILLED:\s*(.+?)\]", content)
    marker = marker_match.group(1).strip() if marker_match else ""

    return content, marker


def inject_into_rag(content: str, question_id: str, category: str,
                    index_dir: str) -> None:
    """Inject distilled content into FTS5 for retrieval."""
    db_path = os.path.join(index_dir, "distilled_learning_test.fts5.db")
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks USING fts5(
            content, source, category
        )
    """)
    conn.execute(
        "INSERT INTO chunks (content, source, category) VALUES (?, ?, ?)",
        (content, f"distilled_test:{question_id}", category),
    )
    conn.commit()
    conn.close()


def retrieve_rag_context(question: str, index_dir: str, top_k: int = 5) -> str:
    """Retrieve context from the distilled test index.

    Uses OR-based FTS5 query with key terms (skip stop words).
    """
    db_path = os.path.join(index_dir, "distilled_learning_test.fts5.db")
    if not os.path.exists(db_path):
        return ""
    _STOP = {"how", "do", "you", "the", "a", "an", "is", "in", "to", "of",
             "and", "or", "for", "with", "what", "when", "where", "why",
             "this", "that", "are", "can", "using", "use", "it", "i", "be"}
    try:
        conn = sqlite3.connect(db_path)
        words = [w for w in question.lower().split()
                 if w.isalnum() and w not in _STOP and len(w) > 2]
        if not words:
            conn.close()
            return ""
        # OR-based query so any keyword can match
        fts_query = " OR ".join(words[:10])
        rows = conn.execute(
            "SELECT search_content FROM chunks WHERE chunks MATCH ? LIMIT ?",
            (fts_query, top_k),
        ).fetchall()
        conn.close()
        return "\n\n".join(row[0] for row in rows)
    except Exception:
        return ""


def simulate_small_model_with_rag(question: str, rag_context: str,
                                  model: str, api_key: str) -> str:
    """Simulate the small offline model by using a constrained prompt.

    We use gpt-4.1-nano (cheapest) with a constrained system prompt
    that mimics a small model's behavior: shorter answers, simpler reasoning.
    """
    system = (
        "You are a small coding assistant (3B parameters). "
        "Answer concisely. Use the provided context when available. "
        "Include code in markdown blocks with imports. "
        "If the context contains specific facts, USE them in your answer."
    )
    user = question
    if rag_context:
        user = (
            f"Context from knowledge base:\n{rag_context[:2000]}\n\n"
            f"Question: {question}"
        )
    result = call_api(system, user, model, api_key, temperature=0.1, max_tokens=1024)
    return result["content"]


# ---------------------------------------------------------------------------
# Main test loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Test JCoder self-learning pipeline")
    parser.add_argument("--baseline", default=str(_ROOT / "evaluation" / "results_local" / "eval_results.json"))
    parser.add_argument("--eval-set", default=str(_ROOT / "evaluation" / "agent_eval_set_200.json"))
    parser.add_argument("--model", default="gpt-4.1-mini", help="Model for distillation")
    parser.add_argument("--small-model", default="gpt-4.1-nano", help="Simulate small model")
    parser.add_argument("--max-questions", type=int, default=10)
    parser.add_argument("--index-dir", default=str(_ROOT / "data" / "indexes"))
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("[FAIL] OPENAI_API_KEY not set")
        sys.exit(1)

    print("=" * 60)
    print("JCoder Self-Learning Pipeline Test")
    print("=" * 60)

    # Step 1: Load baseline failures
    print("\n--- Step 1: Load baseline failures ---")
    weak = load_baseline_failures(args.baseline, args.eval_set, args.max_questions)
    print(f"Loaded {len(weak)} weakest questions from baseline")
    for q in weak[:5]:
        print(f"  {q['question_id']} ({q['category']}): score={q['score']:.2f}")

    # Step 2: Feed failures into telemetry
    print("\n--- Step 2: Feed failures into telemetry ---")
    feed_to_telemetry(weak)

    # Step 3: Distill knowledge for each weak question
    print("\n--- Step 3: Distill with attribution markers ---")
    from evaluation.agent_eval_runner import AgentEvalRunner
    runner = AgentEvalRunner(args.eval_set, agent=None, output_dir="evaluation/results_learning_test")

    # Clean previous test index
    test_db = os.path.join(args.index_dir, "distilled_learning_test.fts5.db")
    if os.path.exists(test_db):
        os.remove(test_db)

    results: list[LearningTestResult] = []
    total_cost = 0.0

    for idx, q in enumerate(weak, 1):
        qid = q["question_id"]
        print(f"\n  [{idx}/{len(weak)}] {qid} ({q['category']}, baseline={q['score']:.2f})")

        # 3a: Get baseline answer (simulate small model WITHOUT RAG)
        print(f"    Baseline answer (no RAG)...", end="", flush=True)
        baseline_answer = simulate_small_model_with_rag(
            q["question"], "", args.small_model, api_key)
        baseline_sub = runner.score_answer(q, baseline_answer)
        baseline_score = baseline_sub.get("weighted_total", 0.0)
        print(f" score={baseline_score:.2f}")

        # 3b: Distill knowledge with marker
        print(f"    Distilling...", end="", flush=True)
        content, marker = distill_with_marker(q, args.model, api_key)
        print(f" marker={marker[:60]!r}...")

        # 3c: Inject into RAG
        inject_into_rag(content, qid, q["category"], args.index_dir)

        # 3d: Get post-learning answer (simulate small model WITH RAG)
        rag_ctx = retrieve_rag_context(q["question"], args.index_dir)
        print(f"    Post-learning answer (with RAG, {len(rag_ctx)} chars)...", end="", flush=True)
        post_answer = simulate_small_model_with_rag(
            q["question"], rag_ctx, args.small_model, api_key)
        post_sub = runner.score_answer(q, post_answer)
        post_score = post_sub.get("weighted_total", 0.0)
        print(f" score={post_score:.2f}")

        # 3e: Check attribution
        marker_found = marker.lower() in post_answer.lower() if marker else False
        improvement = post_score - baseline_score
        learned = improvement > 0.01 and marker_found

        tag = "LEARNED" if learned else ("IMPROVED" if improvement > 0.01 else "NO CHANGE")
        print(f"    Result: {tag} | delta={improvement:+.2f} | marker={'FOUND' if marker_found else 'MISSING'}")

        results.append(LearningTestResult(
            question_id=qid,
            category=q["category"],
            baseline_score=round(baseline_score, 4),
            baseline_answer_snippet=baseline_answer[:200],
            post_score=round(post_score, 4),
            post_answer_snippet=post_answer[:200],
            distilled_marker=marker,
            marker_found_in_answer=marker_found,
            improvement=round(improvement, 4),
            learned_from_rag=learned,
        ))

    # Summary
    print(f"\n{'='*60}")
    print("SELF-LEARNING TEST RESULTS")
    print(f"{'='*60}")

    improved = sum(1 for r in results if r.improvement > 0.01)
    markers_found = sum(1 for r in results if r.marker_found_in_answer)
    confirmed_learned = sum(1 for r in results if r.learned_from_rag)
    avg_improvement = sum(r.improvement for r in results) / len(results) if results else 0

    print(f"Questions tested: {len(results)}")
    print(f"Improved: {improved}/{len(results)} ({improved/len(results)*100:.0f}%)")
    print(f"Attribution markers found: {markers_found}/{len(results)} ({markers_found/len(results)*100:.0f}%)")
    print(f"Confirmed learning (improved + marker): {confirmed_learned}/{len(results)} ({confirmed_learned/len(results)*100:.0f}%)")
    print(f"Average score improvement: {avg_improvement:+.4f}")

    # Save results
    output_path = _ROOT / "evaluation" / "learning_test_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"\nResults saved: {output_path}")

    # Per-question detail
    print(f"\nPer-question:")
    for r in results:
        tag = "LEARNED" if r.learned_from_rag else ("IMPROVED" if r.improvement > 0.01 else "same")
        print(f"  {r.question_id:15s}: {r.baseline_score:.2f} -> {r.post_score:.2f} ({r.improvement:+.2f}) "
              f"marker={'Y' if r.marker_found_in_answer else 'N'} [{tag}]")

    print(f"\n{'='*60}")


if __name__ == "__main__":
    main()
