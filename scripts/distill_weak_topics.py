"""Distill knowledge for weak eval topics via GPT-5.

Reads eval results to find the weakest questions, sends them with
RAG context to GPT-5 for expert explanation, then indexes the
enriched content back into the RAG system.

The distilled content is written for a SMALL offline model to consume:
- Self-contained (no assumed background knowledge)
- Explicit imports and complete code examples
- Step-by-step explanations
- Concrete, not abstract

Usage:
    cd D:\\JCoder
    python scripts/distill_weak_topics.py --results evaluation/results_local/eval_results.json
    python scripts/distill_weak_topics.py --results evaluation/results_local/eval_results.json --top 20
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

if sys.platform == "win32" and __name__ == "__main__":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

try:
    import httpx
except ImportError:
    print("[FAIL] httpx not installed")
    sys.exit(1)


# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------

def call_api(system: str, user: str, model: str, api_key: str,
             timeout: float = 180.0) -> dict:
    """Call OpenAI API. Returns {content, input_tokens, output_tokens, elapsed_s}."""
    new_style = any(model.startswith(p) for p in ("gpt-5", "gpt-4.1", "o1", "o3", "o4"))
    token_key = "max_completion_tokens" if new_style else "max_tokens"

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.3,
        token_key: 4096,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    t0 = time.monotonic()
    with httpx.Client(timeout=httpx.Timeout(timeout)) as client:
        resp = client.post(
            "https://api.openai.com/v1/chat/completions",
            json=payload, headers=headers,
        )
        resp.raise_for_status()
        data = resp.json()

    elapsed = time.monotonic() - t0
    msg = data["choices"][0]["message"]
    usage = data.get("usage", {})

    return {
        "content": msg.get("content", ""),
        "input_tokens": usage.get("prompt_tokens", 0),
        "output_tokens": usage.get("completion_tokens", 0),
        "elapsed_s": elapsed,
        "model": data.get("model", model),
    }


# ---------------------------------------------------------------------------
# RAG retrieval
# ---------------------------------------------------------------------------

def retrieve_context(question: str, index_dir: str, top_k: int = 5) -> str:
    """Quick FTS5 search across available indexes for context."""
    if not os.path.isdir(index_dir):
        return ""

    # Simple FTS5 search -- no need for full federated stack
    results = []
    for entry in os.scandir(index_dir):
        if not entry.name.endswith(".fts5.db"):
            continue
        try:
            conn = sqlite3.connect(entry.path)
            # Clean query for FTS5 (remove special chars)
            clean_q = " ".join(
                w for w in question.split()
                if w.isalnum() or w.replace("_", "").isalnum()
            )
            if not clean_q:
                conn.close()
                continue
            rows = conn.execute(
                "SELECT content FROM chunks WHERE chunks MATCH ? LIMIT ?",
                (clean_q, 3),
            ).fetchall()
            conn.close()
            for row in rows:
                results.append((entry.name, row[0][:500]))
        except Exception:
            continue

    if not results:
        return ""
    # Take top_k results
    lines = []
    for i, (source, content) in enumerate(results[:top_k], 1):
        lines.append(f"[{i}] ({source})\n{content}")
    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# Distillation
# ---------------------------------------------------------------------------

DISTILL_SYSTEM = """You are creating reference material for a SMALL offline language model \
(3-14B parameters) that will use this text as retrieved context when answering coding questions.

Rules for your output:
1. Be COMPLETELY self-contained. Never assume the reader knows anything.
2. Start with a one-line summary of the concept.
3. Include ALL necessary imports in code examples.
4. Show COMPLETE, runnable code -- not fragments.
5. Explain each step in plain English between code blocks.
6. Include common mistakes and how to avoid them.
7. Include the expected output or behavior.
8. Keep it under 800 words -- the small model has limited context.
9. Use markdown formatting with code blocks.
10. Do NOT use jargon without defining it first.
11. ATTRIBUTION REQUIREMENT: Include at least one specific, concrete detail that \
cannot be trivially derived from general knowledge -- a specific version number, \
a real benchmark result, a non-obvious edge case, or a precise API behavior. \
Wrap this detail in a marker: [DISTILLED: <the specific fact>]. This lets us \
verify the small model is actually using this reference material vs reasoning \
from its own training data."""


def distill_question(question: dict, rag_context: str, model: str,
                     api_key: str) -> dict:
    """Send a weak question to the API for expert explanation."""
    user_msg = f"""A coding assistant scored poorly on this question. Create a reference document \
that would help a small AI model answer it correctly.

Question: {question['question']}

Expected keywords that should appear in a good answer: {question.get('expected_keywords', [])}
Expected imports: {question.get('expected_imports', [])}
Category: {question['category']}

"""
    if rag_context:
        user_msg += f"""Here is what the RAG system currently retrieves for this question \
(this may be incomplete or low-quality -- improve on it):

{rag_context}

"""
    user_msg += "Write the reference document now."

    return call_api(DISTILL_SYSTEM, user_msg, model, api_key)


def save_to_fts5(content: str, question_id: str, category: str,
                 db_path: str) -> None:
    """Save distilled content to an FTS5 index for retrieval."""
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks USING fts5(
            content, source, category
        )
    """)
    conn.execute(
        "INSERT INTO chunks (content, source, category) VALUES (?, ?, ?)",
        (content, f"distilled:{question_id}", category),
    )
    conn.commit()
    conn.close()


def save_to_knowledge(content: str, question_id: str, category: str,
                      metadata: dict) -> Path:
    """Save to agent_knowledge as markdown."""
    knowledge_dir = _ROOT / "data" / "agent_knowledge"
    knowledge_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    filename = f"{ts}_distill_{category}_{question_id}.md"
    path = knowledge_dir / filename

    header = (
        f"# Distilled: {question_id} ({category})\n\n"
        f"Generated: {datetime.now(timezone.utc).isoformat()}\n"
        f"Model: {metadata.get('model', 'unknown')}\n"
        f"Tokens: {metadata.get('input_tokens', 0)} in, "
        f"{metadata.get('output_tokens', 0)} out\n\n---\n\n"
    )
    path.write_text(header + content, encoding="utf-8")
    return path


def run_distillation(
    eval_results: list,
    eval_set_path: str = "",
    index_dir: str = "",
    model: str = "gpt-5",
    top: int = 20,
    budget_usd: float = 2.0,
    resume: bool = True,
) -> dict:
    """Programmatic entry point for learning cycle Phase 4."""
    if not eval_set_path:
        eval_set_path = str(_ROOT / "evaluation" / "agent_eval_set_200.json")
    if not index_dir:
        index_dir = str(_ROOT / "data" / "indexes")

    # Sort by score ascending, pick weakest
    scored = [r for r in eval_results if isinstance(r.get("score"), (int, float))]
    scored.sort(key=lambda r: r["score"])
    weak = scored[:top]

    if not weak:
        return {"distilled": 0, "errors": 0, "total_cost": 0.0, "model": model}

    distill_db = Path(index_dir) / "distilled.fts5.db"
    distilled = 0
    errors = 0
    total_cost = 0.0

    api_key = os.environ.get("OPENAI_API_KEY", "")
    for item in weak:
        if total_cost >= budget_usd:
            break
        q_text = item.get("question", "")
        q_id = item.get("id", q_text[:40])
        category = item.get("category", "general")
        if not q_text:
            continue
        try:
            context = retrieve_context(q_text, index_dir)
            result = distill_question(
                {"question": q_text, "id": q_id, "category": category},
                context, model, api_key,
            )
            if result and result.get("content"):
                save_to_fts5(result["content"], q_id, category, str(distill_db))
                distilled += 1
                total_cost += result.get("cost", 0.0)
        except Exception:
            errors += 1

    return {
        "distilled": distilled,
        "skipped": len(weak) - distilled - errors,
        "errors": errors,
        "total_cost": total_cost,
        "budget_reached": total_cost >= budget_usd,
        "model": model,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Distill weak eval topics via API")
    parser.add_argument("--results", required=True, help="eval_results.json from baseline")
    parser.add_argument("--eval-set", default=str(_ROOT / "evaluation" / "agent_eval_set_200.json"))
    parser.add_argument("--top", type=int, default=20, help="Number of weakest questions to distill")
    parser.add_argument("--model", default="gpt-4.1-mini", help="Model for distillation")
    parser.add_argument("--index-dir", default=str(_ROOT / "data" / "indexes"),
                        help="FTS5 index directory for RAG context")
    parser.add_argument("--budget", type=float, default=2.0, help="Max spend in USD")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("[FAIL] OPENAI_API_KEY not set")
        sys.exit(1)

    # Load eval results and question set
    with open(args.results, "r", encoding="utf-8") as f:
        results = json.load(f)
    with open(args.eval_set, "r", encoding="utf-8") as f:
        questions = {q["id"]: q for q in json.load(f)}

    # Sort by score ascending (weakest first)
    results.sort(key=lambda r: r["score"])
    weak = results[:args.top]

    print(f"[OK] {len(weak)} weakest questions selected for distillation")
    print(f"[OK] Model: {args.model} | Budget: ${args.budget:.2f}")
    print(f"[OK] Score range: {weak[0]['score']:.3f} - {weak[-1]['score']:.3f}\n")

    # Output paths
    distill_db = str(_ROOT / "data" / "indexes" / "distilled.fts5.db")
    progress_path = _ROOT / "evaluation" / "distill_progress.json"

    # Resume
    completed = set()
    if not args.no_resume and progress_path.exists():
        with open(progress_path, "r", encoding="utf-8") as f:
            completed = set(json.load(f).get("completed", []))
        print(f"Resuming: {len(completed)} already distilled\n")

    total_cost = 0.0
    distilled = 0

    for idx, r in enumerate(weak, 1):
        qid = r["question_id"]
        if qid in completed:
            print(f"  [{idx}/{len(weak)}] {qid}: SKIPPED (already done)")
            continue

        if total_cost >= args.budget:
            print(f"\n[WARN] Budget cap reached (${total_cost:.3f})")
            break

        q = questions.get(qid)
        if not q:
            print(f"  [{idx}/{len(weak)}] {qid}: SKIPPED (not in eval set)")
            continue

        print(f"  [{idx}/{len(weak)}] {qid} ({r['category']}, score={r['score']:.2f}): "
              f"distilling...", end="", flush=True)

        # Retrieve existing context
        rag_ctx = retrieve_context(q["question"], args.index_dir, top_k=3)

        try:
            result = distill_question(q, rag_ctx, args.model, api_key)
        except Exception as e:
            print(f" ERROR: {e}")
            continue

        # Save to FTS5 index
        save_to_fts5(result["content"], qid, r["category"], distill_db)

        # Save to agent_knowledge
        save_to_knowledge(result["content"], qid, r["category"], result)

        # Cost estimate
        cost = (result["input_tokens"] * 0.4 + result["output_tokens"] * 1.6) / 1_000_000
        total_cost += cost
        distilled += 1

        completed.add(qid)
        # Save progress
        with open(progress_path, "w", encoding="utf-8") as f:
            json.dump({"completed": list(completed), "total_cost": total_cost}, f)

        print(f" OK ({result['elapsed_s']:.1f}s, ${cost:.4f})")

    print(f"\n{'='*60}")
    print(f"Distilled: {distilled} questions")
    print(f"Total cost: ${total_cost:.4f}")
    print(f"FTS5 index: {distill_db}")
    print(f"Knowledge dir: {_ROOT / 'data' / 'agent_knowledge'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
