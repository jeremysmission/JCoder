"""
JCoder Self-Quiz: Periodic knowledge retention testing.

Asks JCoder questions from its own indexes and logs confidence + accuracy
for tracking retrieval quality over time.

Usage:
    python scripts/self_quiz.py                    # Quick 5-question quiz
    python scripts/self_quiz.py --count 20         # 20 questions
    python scripts/self_quiz.py --index iqt_codebase  # Quiz specific index
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = PROJECT_ROOT / "logs" / "self_quiz"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Quiz questions organized by index and difficulty
QUIZ_BANK = {
    "default": [
        "What is the EmbeddingEngine class and how does it work?",
        "How does the meta-cognitive controller select learning strategies?",
        "What are the three games in adversarial self-play?",
        "How does experience replay store and retrieve trajectories?",
        "What does the corrective retrieval module do when initial results are weak?",
        "How does the federated search combine results across multiple indexes?",
        "What is the cascade router and how does it decide model routing?",
    ],
    "iqt_codebase": [
        "What visual heuristics does IQT apply to ionogram images?",
        "How does the attention report prioritize items for human review?",
        "What is the URSI code and how is it used in site configuration?",
        "How does the timing rules engine detect stale ionograms?",
        "What dataclasses are defined in the IQT models module?",
    ],
    "career_moves": [
        "How does FSRS scheduling work in Career Moves?",
        "What are the study phases and how many tasks are in each?",
        "How does the USAJOBS connector paginate search results?",
        "What CLI commands are available for study and quiz management?",
    ],
    "canary_test": [
        "Who are the main protagonists in the Sergeant Biscuit story?",
        "What programming language was involved in the quantum dishwasher fix?",
        "What was the root cause of the sentient cereal crisis?",
    ],
}

# Trick questions that should be REFUSED (no hallucination)
TRICK_QUESTIONS = [
    ("any", "What year in the 1970s did we discover anti-gravity?"),
    ("any", "What is the capital of the country of Africa?"),
    ("any", "How many times has Python been acquired by Google?"),
]


def run_ask(question: str, index_name: str) -> dict:
    """Run a JCoder ask command and capture the result."""
    t0 = time.time()
    result = subprocess.run(
        [sys.executable, "main.py", "ask", "--index-name", index_name, question],
        capture_output=True, text=True, timeout=120, cwd=str(PROJECT_ROOT),
    )
    elapsed = time.time() - t0

    # Extract answer (stdout minus the sources table)
    lines = result.stdout.strip().split("\n")
    answer_lines = []
    for line in lines:
        if line.startswith("+--") or line.startswith("| File"):
            break
        if "Loaded index" not in line and "FAISS" not in line:
            answer_lines.append(line)

    return {
        "question": question,
        "index": index_name,
        "answer": "\n".join(answer_lines).strip(),
        "elapsed_s": round(elapsed, 1),
        "exit_code": result.returncode,
        "stderr_snippet": result.stderr[:200] if result.stderr else "",
    }


def main():
    parser = argparse.ArgumentParser(description="JCoder Self-Quiz")
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--index", default=None, help="Quiz specific index only")
    parser.add_argument("--include-tricks", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print(f"  JCoder Self-Quiz — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 60)

    # Build question pool
    pool = []
    for idx, questions in QUIZ_BANK.items():
        if args.index and idx != args.index:
            continue
        for q in questions:
            pool.append((idx, q))

    if args.include_tricks:
        for idx, q in TRICK_QUESTIONS:
            pool.append((idx if idx != "any" else "default", q))

    random.shuffle(pool)
    selected = pool[:args.count]

    results = []
    for i, (idx, question) in enumerate(selected, 1):
        print(f"\n--- Question {i}/{len(selected)} [{idx}] ---")
        print(f"Q: {question}")
        result = run_ask(question, idx)
        preview = result["answer"][:150].replace("\n", " ")
        print(f"A: {preview}...")
        print(f"   ({result['elapsed_s']}s)")
        results.append(result)

    # Save log
    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "count": len(results),
        "avg_latency_s": round(
            sum(r["elapsed_s"] for r in results) / len(results), 1
        ) if results else 0,
        "results": results,
    }
    log_path = LOG_DIR / f"quiz_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    log_path.write_text(json.dumps(log_entry, indent=2), encoding="utf-8")

    print(f"\n{'=' * 60}")
    print(f"  {len(results)} questions answered")
    print(f"  Avg latency: {log_entry['avg_latency_s']}s")
    print(f"  Log: {log_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
