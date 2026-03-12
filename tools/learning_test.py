"""
Learning Measurement Test
--------------------------
Measures JCoder's ability to learn by comparing answers before and after
knowledge ingestion. Runs the same questions with and without a specific
FTS5 index, then scores the delta.

Usage:
    cd D:\JCoder
    python tools/learning_test.py [--model phi4-mini] [--topic self_learning]
"""

import io
import json
import os
import sqlite3
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace"
    )

INDEX_DIR = Path(os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "indexes",
))


# ---------------------------------------------------------------------------
# Test questions: topic -> list of (question, expected_keywords)
# ---------------------------------------------------------------------------

TEST_SETS = {
    "self_learning": {
        "index": "self_learning",
        "questions": [
            {
                "q": "What is the Absolute Zero Reasoner and how does it learn?",
                "keywords": ["self-play", "no data", "no labels", "code executor",
                             "reward", "zero"],
                "weight": 1.0,
            },
            {
                "q": "Explain the PRAXIS procedural memory system for AI agents.",
                "keywords": ["state hash", "action", "outcome", "experience",
                             "web automation", "procedural"],
                "weight": 1.0,
            },
            {
                "q": "What is RISE recursive introspection and how does it improve LLMs?",
                "keywords": ["multi-turn", "MDP", "self-improve", "introspection",
                             "recursive"],
                "weight": 1.0,
            },
            {
                "q": "How does evolutionary prompt optimization (GEPA) work?",
                "keywords": ["evolutionary", "mutation", "population", "fitness",
                             "prompt", "generation"],
                "weight": 1.0,
            },
            {
                "q": "What is the Generator-Verifier-Updater (GVU) cycle in self-play?",
                "keywords": ["generator", "verifier", "updater", "self-play",
                             "cycle"],
                "weight": 1.0,
            },
            {
                "q": "Describe the Reflexion framework for language agent self-improvement.",
                "keywords": ["verbal", "reinforcement", "reflection", "memory",
                             "feedback", "trial"],
                "weight": 1.0,
            },
            {
                "q": "What is Voyager and how does it achieve open-ended learning?",
                "keywords": ["minecraft", "curriculum", "skill library", "code",
                             "exploration", "open-ended"],
                "weight": 1.0,
            },
            {
                "q": "Explain self-rewarding language models and how they train without human feedback.",
                "keywords": ["self-reward", "LLM-as-judge", "iterative",
                             "DPO", "preference"],
                "weight": 1.0,
            },
            {
                "q": "What is ExpeL and how does experience help LLM agents learn?",
                "keywords": ["experience", "insights", "accumulate", "task",
                             "trial", "natural language"],
                "weight": 1.0,
            },
            {
                "q": "How does SPIN self-play fine-tuning work without new human data?",
                "keywords": ["self-play", "discriminator", "generator", "SFT",
                             "weak", "strong"],
                "weight": 1.0,
            },
        ],
    },
}


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

@dataclass
class QuestionResult:
    question: str
    answer: str
    keywords_found: List[str]
    keywords_missed: List[str]
    score: float
    retrieval_hits: int
    latency_s: float


@dataclass
class TestReport:
    topic: str
    condition: str  # "baseline" or "learned"
    results: List[QuestionResult] = field(default_factory=list)
    avg_score: float = 0.0
    total_keywords_found: int = 0
    total_keywords_possible: int = 0
    total_latency_s: float = 0.0


def score_answer(answer: str, keywords: List[str]) -> tuple:
    """Score an answer by keyword presence. Returns (found, missed, score)."""
    answer_lower = answer.lower()
    found = [k for k in keywords if k.lower() in answer_lower]
    missed = [k for k in keywords if k.lower() not in answer_lower]
    score = len(found) / max(1, len(keywords))
    return found, missed, score


# ---------------------------------------------------------------------------
# RAG pipeline (lightweight, no full agent stack)
# ---------------------------------------------------------------------------

def search_fts5(query: str, index_names: List[str], top_k: int = 8) -> List[str]:
    """Search specific FTS5 indexes and return content chunks."""
    chunks = []
    # Strip common words and punctuation for cleaner FTS5 matching
    import re
    stop = {"what", "is", "the", "a", "an", "and", "or", "how", "does",
            "do", "it", "in", "of", "to", "for", "by", "that", "this",
            "with", "from", "its", "be", "are", "was", "were", "has",
            "have", "can", "about"}
    raw_terms = query.split()
    clean = [re.sub(r"[^a-zA-Z0-9_-]", "", t) for t in raw_terms]
    meaningful = [t for t in clean if t and t.lower() not in stop]
    # Mix: meaningful terms first, then fall back to all cleaned terms
    terms = meaningful[:8] if len(meaningful) >= 3 else clean[:8]
    terms = [t for t in terms if t]
    fts_query = " OR ".join(terms[:8])

    for idx_name in index_names:
        db_path = INDEX_DIR / f"{idx_name}.fts5.db"
        if not db_path.exists():
            continue
        try:
            conn = sqlite3.connect(str(db_path))
            rows = conn.execute(
                "SELECT search_content FROM chunks WHERE chunks MATCH ? LIMIT ?",
                (fts_query, top_k),
            ).fetchall()
            conn.close()
            for row in rows:
                if row[0]:
                    chunks.append(row[0][:1600])
        except Exception:
            pass
    return chunks


def _build_prompt(question: str, context: str) -> str:
    """Build the prompt text for any backend."""
    if context:
        return (
            f"Answer the following question using the context provided and your "
            f"own knowledge. Prioritize information from the context but supplement "
            f"with your knowledge where the context is incomplete. "
            f"Be specific and name techniques, methods, or systems mentioned.\n\n"
            f"CONTEXT:\n{context}\n\n"
            f"QUESTION: {question}\n\n"
            f"ANSWER:"
        )
    return (
        f"Answer the following question about AI and self-learning techniques. "
        f"Be specific. If you don't know, say 'I don't know'.\n\n"
        f"QUESTION: {question}\n\n"
        f"ANSWER:"
    )


def ask_llm(question: str, context: str, model: str = "phi4-mini",
            endpoint: str = "http://localhost:11434/api/generate",
            backend: str = "ollama") -> str:
    """Ask the LLM a question with optional context.

    backend: "ollama" (local), "anthropic" (direct API), or "openai" (OpenAI-compat).
    """
    import httpx

    prompt = _build_prompt(question, context)

    if backend == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            return "ERROR: ANTHROPIC_API_KEY not set"
        resp = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": model,
                "max_tokens": 512,
                "temperature": 0.1,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=120.0,
        )
        data = resp.json()
        if "content" in data and data["content"]:
            return data["content"][0].get("text", "")
        return data.get("error", {}).get("message", "ERROR: empty response")

    if backend == "openai":
        api_key = os.environ.get("OPENAI_API_KEY",
                                 os.environ.get("OPENROUTER_API_KEY", ""))
        if not api_key:
            return "ERROR: OPENAI_API_KEY not set"
        # Use OpenAI default if endpoint is still the Ollama default
        openai_endpoint = endpoint
        if "localhost" in endpoint or "11434" in endpoint:
            openai_endpoint = "https://api.openai.com/v1"
        # Newer models (gpt-5+, o3, o4) use max_completion_tokens
        token_param = "max_completion_tokens" if any(
            model.startswith(p) for p in ("gpt-5", "gpt-4.1", "o1", "o3", "o4")
        ) else "max_tokens"
        resp = httpx.post(
            f"{openai_endpoint}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "content-type": "application/json",
            },
            json={
                "model": model,
                token_param: 512,
                "temperature": 0.1,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=120.0,
        )
        data = resp.json()
        choices = data.get("choices", [])
        if choices:
            return choices[0].get("message", {}).get("content", "")
        return data.get("error", {}).get("message", "ERROR: empty response")

    # Default: Ollama raw /api/generate
    resp = httpx.post(
        endpoint,
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 512},
        },
        timeout=180.0,
    )
    return resp.json().get("response", "")


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

def run_test(
    topic: str,
    condition: str,
    index_names: List[str],
    model: str = "phi4-mini",
    backend: str = "ollama",
) -> TestReport:
    """Run all questions for a topic under a given condition."""
    test_set = TEST_SETS[topic]
    report = TestReport(topic=topic, condition=condition)

    for i, qdata in enumerate(test_set["questions"], 1):
        q = qdata["q"]
        keywords = qdata["keywords"]

        print(f"  Q{i}: {q[:70]}...", end=" ", flush=True)
        t0 = time.monotonic()

        # Retrieve context
        if index_names:
            chunks = search_fts5(q, index_names, top_k=8)
            context = "\n---\n".join(chunks[:8])
            retrieval_hits = len(chunks)
        else:
            context = ""
            retrieval_hits = 0

        # Generate answer
        try:
            answer = ask_llm(q, context, model=model, backend=backend)
        except Exception as exc:
            answer = f"ERROR: {exc}"

        elapsed = time.monotonic() - t0

        # Score
        found, missed, score = score_answer(answer, keywords)

        result = QuestionResult(
            question=q,
            answer=answer,
            keywords_found=found,
            keywords_missed=missed,
            score=score,
            retrieval_hits=retrieval_hits,
            latency_s=round(elapsed, 1),
        )
        report.results.append(result)
        report.total_keywords_found += len(found)
        report.total_keywords_possible += len(keywords)

        pct = int(score * 100)
        print(f"[{pct}%] {len(found)}/{len(keywords)} keywords, {elapsed:.0f}s")

    # Compute averages
    scores = [r.score for r in report.results]
    report.avg_score = sum(scores) / max(1, len(scores))
    report.total_latency_s = sum(r.latency_s for r in report.results)

    return report


def print_report(report: TestReport):
    """Pretty-print a test report."""
    print(f"\n{'=' * 60}")
    print(f"  {report.condition.upper()} -- Topic: {report.topic}")
    print(f"{'=' * 60}")
    print(f"  Average Score: {report.avg_score:.1%}")
    print(f"  Keywords Found: {report.total_keywords_found}/{report.total_keywords_possible}")
    print(f"  Total Latency: {report.total_latency_s:.0f}s")
    print()
    for i, r in enumerate(report.results, 1):
        pct = int(r.score * 100)
        print(f"  Q{i} [{pct:3d}%] hits={r.retrieval_hits} "
              f"found={r.keywords_found} missed={r.keywords_missed}")
    print(f"{'=' * 60}")


def print_comparison(baseline: TestReport, learned: TestReport):
    """Print side-by-side comparison."""
    print(f"\n{'=' * 60}")
    print(f"  LEARNING DELTA -- {learned.topic}")
    print(f"{'=' * 60}")
    print(f"  {'Metric':<30s} {'Baseline':>10s} {'Learned':>10s} {'Delta':>10s}")
    print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*10}")

    b_pct = f"{baseline.avg_score:.1%}"
    l_pct = f"{learned.avg_score:.1%}"
    d_pct = f"+{learned.avg_score - baseline.avg_score:.1%}"
    print(f"  {'Avg Score':<30s} {b_pct:>10s} {l_pct:>10s} {d_pct:>10s}")

    b_kw = f"{baseline.total_keywords_found}/{baseline.total_keywords_possible}"
    l_kw = f"{learned.total_keywords_found}/{learned.total_keywords_possible}"
    d_kw = f"+{learned.total_keywords_found - baseline.total_keywords_found}"
    print(f"  {'Keywords Found':<30s} {b_kw:>10s} {l_kw:>10s} {d_kw:>10s}")

    print()
    print(f"  {'Q#':<4s} {'Baseline':>10s} {'Learned':>10s} {'Delta':>10s}  Question")
    print(f"  {'-'*4} {'-'*10} {'-'*10} {'-'*10}  {'-'*40}")
    for i, (b, l) in enumerate(zip(baseline.results, learned.results), 1):
        bp = f"{b.score:.0%}"
        lp = f"{l.score:.0%}"
        dp = f"+{l.score - b.score:.0%}" if l.score >= b.score else f"{l.score - b.score:.0%}"
        print(f"  Q{i:<3d} {bp:>10s} {lp:>10s} {dp:>10s}  {b.question[:40]}")

    print(f"\n{'=' * 60}")
    improvement = learned.avg_score - baseline.avg_score
    if improvement > 0.1:
        print(f"  RESULT: JCoder LEARNED (+{improvement:.1%} improvement)")
    elif improvement > 0:
        print(f"  RESULT: Marginal improvement (+{improvement:.1%})")
    else:
        print(f"  RESULT: No measurable learning ({improvement:.1%})")
    print(f"{'=' * 60}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Measure JCoder's learning ability")
    parser.add_argument("--topic", default="self_learning",
                        choices=list(TEST_SETS.keys()))
    parser.add_argument("--model", default="phi4-mini")
    parser.add_argument("--backend", default="ollama",
                        choices=["ollama", "anthropic", "openai"],
                        help="LLM backend: ollama (local), anthropic (direct), openai (compat)")
    parser.add_argument("--baseline-only", action="store_true",
                        help="Run only the baseline (no knowledge)")
    parser.add_argument("--learned-only", action="store_true",
                        help="Run only the learned condition")
    parser.add_argument("--output", default="",
                        help="Save results JSON to this path")
    args = parser.parse_args()

    test_set = TEST_SETS[args.topic]
    target_index = test_set["index"]
    n_questions = len(test_set["questions"])

    print(f"Learning Test: {args.topic} ({n_questions} questions)")
    print(f"Model: {args.model}")
    print(f"Target index: {target_index}")
    print()

    # Phase 1: Baseline (no target index)
    baseline = None
    if not args.learned_only:
        print("[PHASE 1] BASELINE -- answering WITHOUT knowledge index")
        baseline = run_test(args.topic, "baseline", index_names=[], model=args.model, backend=args.backend)
        print_report(baseline)

    # Phase 2: Learned (with target index)
    learned = None
    if not args.baseline_only:
        print(f"\n[PHASE 2] LEARNED -- answering WITH {target_index} index")
        learned = run_test(
            args.topic, "learned",
            index_names=[target_index],
            model=args.model,
            backend=args.backend,
        )
        print_report(learned)

    # Phase 3: Comparison
    if baseline and learned:
        print_comparison(baseline, learned)

    # Save results
    if args.output:
        data = {}
        if baseline:
            data["baseline"] = {
                "avg_score": baseline.avg_score,
                "keywords_found": baseline.total_keywords_found,
                "keywords_possible": baseline.total_keywords_possible,
                "per_question": [
                    {"score": r.score, "found": r.keywords_found, "missed": r.keywords_missed}
                    for r in baseline.results
                ],
            }
        if learned:
            data["learned"] = {
                "avg_score": learned.avg_score,
                "keywords_found": learned.total_keywords_found,
                "keywords_possible": learned.total_keywords_possible,
                "per_question": [
                    {"score": r.score, "found": r.keywords_found, "missed": r.keywords_missed}
                    for r in learned.results
                ],
            }
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
