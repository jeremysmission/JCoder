"""
Monkey Brain Knowledge Retention Test
---------------------------------------
Tests whether JCoder's distilled knowledge ACTUALLY helps a small model.

Design:
  - 15 pure recall questions (no reasoning, just "did you retrieve this fact?")
  - All answers come ONLY from our Phase 1-10 curriculum files
  - phi4-mini (3.8B) is the "monkey brain" -- too small to know this stuff natively

Test A: Ask phi4-mini COLD (no context). Baseline.
Test B: Ask phi4-mini WITH retrieved knowledge context. Treatment.

If B > A, the distillation taught it something real.
"""

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

sys.stdout = open(sys.stdout.fileno(), mode="w", encoding="utf-8", buffering=1)

KNOWLEDGE_DIR = Path(__file__).resolve().parent.parent / "data" / "agent_knowledge"
OLLAMA_URL = "http://localhost:11434/api/generate"

# Questions + ground truth answers + which knowledge file has the answer
QUESTIONS = [
    {
        "id": 1,
        "q": "In the Jasmine test framework bugfix pattern, what specific function call preserves the spec's `this` context?",
        "answer": "promiseBuilder.call(spec)",
        "source": "phase4b_git_&_commit_patterns",
        "keywords": "Jasmine bugfix context promiseBuilder",
    },
    {
        "id": 2,
        "q": "In Rust, what format specifier produces a quoted, escaped string literal suitable for safe embedding in JavaScript?",
        "answer": "{:?}",
        "source": "phase8a_rust_patterns",
        "keywords": "Rust format escaped string JavaScript embedding",
    },
    {
        "id": 3,
        "q": "When a Rust function expects an options/config struct, what idiomatic expression passes default settings instead of None?",
        "answer": "Default::default()",
        "source": "phase8a_rust_patterns",
        "keywords": "Rust Default options config struct sentinel",
    },
    {
        "id": 4,
        "q": "In Django's OverwritingStorage pattern, what function atomically moves a temp file to the final path with overwrite enabled?",
        "answer": "file_move_safe",
        "source": "phase4a_production_python_code",
        "keywords": "Django OverwritingStorage file_move_safe overwrite",
    },
    {
        "id": 5,
        "q": "According to security patterns, what is the example 'bad' password recovery message given as an anti-pattern?",
        "answer": "Your current password is hunter2",
        "source": "phase5c_security_patterns",
        "keywords": "password recovery bad example hunter2",
    },
    {
        "id": 6,
        "q": "What is the strongest indicator that a system stores passwords in plaintext -- redisplay after submission, or a forgot-password email containing the original password?",
        "answer": "forgot-password email containing the original password",
        "source": "phase5c_security_patterns",
        "keywords": "password plaintext indicator forgot email redisplay",
    },
    {
        "id": 7,
        "q": "In the compound interest pattern, what is the future value of depositing $250 per year at 5% interest for 3 years (end-of-year deposits)?",
        "answer": "827.8125",
        "source": "phase3b_mathematical_programming",
        "keywords": "compound interest 250 deposit 5% future value annuity",
    },
    {
        "id": 8,
        "q": "In the catch-up motion problem, if Teena is 7.5 miles behind and wants to be 15 miles ahead, what is the total distance she needs to gain?",
        "answer": "22.5",
        "source": "phase3b_mathematical_programming",
        "keywords": "catch-up motion Teena miles behind ahead relative speed",
    },
    {
        "id": 9,
        "q": "According to chain-of-thought coding patterns, what is the practical check for whether a response properly answers a code generation request?",
        "answer": "If prompt says write/create/generate code, final answer should contain code block or exact query",
        "source": "phase6b_chain-of-thought_coding",
        "keywords": "chain-of-thought code generation check deliverable artifact",
    },
    {
        "id": 10,
        "q": "In the chain-of-thought patterns, what example input list produces the output '1,4,9,8,25' when each even number is doubled and each odd number is squared?",
        "answer": "[1, 2, 3, 4, 5]",
        "source": "phase6b_chain-of-thought_coding",
        "keywords": "comma separated values list integers doubled squared",
    },
    {
        "id": 11,
        "q": "In the code review patterns, what specific constraint did the C++ Fibonacci example violate that reviewers caught?",
        "answer": "single loop statement",
        "source": "phase2b_code_review_&_quality",
        "keywords": "C++ Fibonacci constraint single loop statement reviewers",
    },
    {
        "id": 12,
        "q": "According to the master synthesis, what is pattern #1 -- the first of the top 25 cross-cutting coding patterns?",
        "answer": "Validate the problem statement before solving",
        "source": "phase10_master_synthesis",
        "keywords": "master synthesis pattern validate problem statement",
    },
    {
        "id": 13,
        "q": "In the git commit patterns, what does the example 'add a simple position component' commit add -- what specific fields?",
        "answer": "x/y fields with getters/setters",
        "source": "phase4b_git_&_commit_patterns",
        "keywords": "position component feature commit x y getters setters",
    },
    {
        "id": 14,
        "q": "In security patterns, why are unsalted password hashes dangerous even if the hashing algorithm is strong?",
        "answer": "one cracking effort can be reused across many users with the same password",
        "source": "phase5c_security_patterns",
        "keywords": "unsalted hash amortizable cracking effort reused users",
    },
    {
        "id": 15,
        "q": "According to the master synthesis, what distinguishes expert developers from juniors regarding API behavior?",
        "answer": "Experts read contracts and semantics carefully; juniors assume API behavior from names",
        "source": "phase10_master_synthesis",
        "keywords": "expert junior API behavior names contracts semantics",
    },
]


def ask_ollama(prompt: str, model: str = "phi4-mini", context: str = "") -> str:
    """Ask Ollama's local model."""
    if context:
        full_prompt = (
            f"Use ONLY the following reference material to answer the question. "
            f"Give a short, specific answer.\n\n"
            f"--- REFERENCE MATERIAL ---\n{context}\n--- END ---\n\n"
            f"Question: {prompt}\nAnswer:"
        )
    else:
        full_prompt = (
            f"Answer this question concisely. If you don't know, say 'I don't know'.\n\n"
            f"Question: {prompt}\nAnswer:"
        )

    try:
        resp = httpx.post(
            OLLAMA_URL,
            json={"model": model, "prompt": full_prompt, "stream": False,
                  "options": {"temperature": 0.1, "num_predict": 256}},
            timeout=httpx.Timeout(120.0),
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except Exception as e:
        return f"[ERROR] {e}"


def load_knowledge(source_pattern: str) -> str:
    """Load the matching knowledge file content."""
    matches = sorted(KNOWLEDGE_DIR.glob(f"*{source_pattern}*"))
    # Pick the largest file (skip empty headers)
    best = max(matches, key=lambda f: f.stat().st_size) if matches else None
    if best:
        return best.read_text(encoding="utf-8")
    return ""


def grade_answer(response: str, expected: str) -> bool:
    """Simple substring grading -- does the response contain the key answer?"""
    resp_lower = response.lower().strip()
    exp_lower = expected.lower().strip()

    # Direct substring match
    if exp_lower in resp_lower:
        return True

    # Check individual key terms (for multi-word answers)
    key_terms = [t for t in exp_lower.split() if len(t) > 3]
    if key_terms:
        matches = sum(1 for t in key_terms if t in resp_lower)
        # At least 60% of key terms present
        if matches / len(key_terms) >= 0.6:
            return True

    return False


def main():
    print("=" * 70)
    print("  MONKEY BRAIN KNOWLEDGE RETENTION TEST")
    print("  Model: phi4-mini (3.8B) -- the 'monkey brain'")
    print("  15 questions from JCoder's distilled curriculum")
    print("  Test A: COLD (no context)  |  Test B: WITH retrieved knowledge")
    print("=" * 70)

    cold_correct = 0
    warm_correct = 0
    results = []

    for q in QUESTIONS:
        print(f"\n{'─'*70}")
        print(f"  Q{q['id']}: {q['q']}")
        print(f"  Expected: {q['answer']}")
        print(f"{'─'*70}")

        # Test A: Cold (no context)
        print(f"\n  [COLD] Asking phi4-mini without context...")
        t0 = time.monotonic()
        cold_answer = ask_ollama(q["q"])
        cold_time = time.monotonic() - t0
        cold_pass = grade_answer(cold_answer, q["answer"])
        cold_correct += int(cold_pass)

        # Truncate for display
        display = cold_answer[:200].replace("\n", " ")
        print(f"  [COLD] ({cold_time:.1f}s) {'PASS' if cold_pass else 'FAIL'}")
        print(f"  >>> {display}")

        # Load knowledge context
        context = load_knowledge(q["source"])

        # Test B: With retrieved knowledge
        print(f"\n  [WARM] Asking phi4-mini WITH knowledge context...")
        t0 = time.monotonic()
        warm_answer = ask_ollama(q["q"], context=context[:4000])
        warm_time = time.monotonic() - t0
        warm_pass = grade_answer(warm_answer, q["answer"])
        warm_correct += int(warm_pass)

        display = warm_answer[:200].replace("\n", " ")
        print(f"  [WARM] ({warm_time:.1f}s) {'PASS' if warm_pass else 'FAIL'}")
        print(f"  >>> {display}")

        results.append({
            "id": q["id"],
            "question": q["q"],
            "expected": q["answer"],
            "cold_answer": cold_answer[:300],
            "cold_pass": cold_pass,
            "warm_answer": warm_answer[:300],
            "warm_pass": warm_pass,
        })

    # Final scores
    total = len(QUESTIONS)
    cold_pct = cold_correct / total * 100
    warm_pct = warm_correct / total * 100
    delta = warm_correct - cold_correct

    print(f"\n{'='*70}")
    print(f"  RESULTS")
    print(f"{'='*70}")
    print(f"  COLD (no context):   {cold_correct}/{total} ({cold_pct:.0f}%)")
    print(f"  WARM (with knowledge): {warm_correct}/{total} ({warm_pct:.0f}%)")
    print(f"  DELTA (what we taught): +{delta} questions ({warm_pct - cold_pct:.0f}%)")
    print(f"{'='*70}")

    if delta > 0:
        print(f"\n  CONCLUSION: Knowledge distillation added {delta} correct answers.")
        print(f"  The monkey brain learned something it didn't already know.")
    elif delta == 0:
        print(f"\n  CONCLUSION: No measurable improvement. Either the monkey already")
        print(f"  knew everything, or the knowledge files didn't help retrieval.")
    else:
        print(f"\n  CONCLUSION: Knowledge context actually HURT performance.")
        print(f"  The retrieved text may have confused the model.")

    # Save results
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    out_path = KNOWLEDGE_DIR / f"{ts}_monkey_brain_results.md"
    out_path.write_text(
        f"# Monkey Brain Knowledge Retention Test\n\n"
        f"Date: {datetime.now(timezone.utc).isoformat()}\n"
        f"Model: phi4-mini (3.8B)\n"
        f"Cold: {cold_correct}/{total} ({cold_pct:.0f}%)\n"
        f"Warm: {warm_correct}/{total} ({warm_pct:.0f}%)\n"
        f"Delta: +{delta}\n\n---\n\n"
        + "\n\n".join(
            f"## Q{r['id']}: {r['question']}\n"
            f"Expected: `{r['expected']}`\n"
            f"Cold ({'PASS' if r['cold_pass'] else 'FAIL'}): {r['cold_answer']}\n"
            f"Warm ({'PASS' if r['warm_pass'] else 'FAIL'}): {r['warm_answer']}\n"
            for r in results
        ),
        encoding="utf-8",
    )
    print(f"\n  Results saved: {out_path.name}")


if __name__ == "__main__":
    main()
