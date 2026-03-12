"""
JCoder Coding Challenge Runner
-------------------------------
1. Pulls high-quality code examples from FTS5 indexes
2. Has GPT-5.4 digest patterns
3. Runs 4 progressively harder coding challenges
4. Saves results to agent_knowledge
"""

import json
import os
import random
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

# Fix Windows encoding
sys.stdout = open(sys.stdout.fileno(), mode="w", encoding="utf-8", buffering=1)

JCODER_ROOT = Path(__file__).resolve().parent.parent
KNOWLEDGE_DIR = JCODER_ROOT / "data" / "agent_knowledge"
KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)

random.seed(42)


def call_gpt5(system_p: str, user_p: str, max_tok: int = 4096) -> str:
    """Call GPT-5.4 via OpenAI API."""
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("[FAIL] OPENAI_API_KEY not set")
        sys.exit(1)

    payload = {
        "model": "gpt-5.4",
        "messages": [
            {"role": "system", "content": system_p},
            {"role": "user", "content": user_p},
        ],
        "temperature": 0.2,
        "max_completion_tokens": max_tok,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-type": "application/json",
    }
    t0 = time.monotonic()
    with httpx.Client(timeout=httpx.Timeout(300.0)) as client:
        resp = client.post(
            "https://api.openai.com/v1/chat/completions",
            json=payload,
            headers=headers,
        )
        resp.raise_for_status()
        data = resp.json()

    elapsed = time.monotonic() - t0
    msg = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    print(
        f"  [{elapsed:.1f}s, {usage.get('prompt_tokens', 0)} in, "
        f"{usage.get('completion_tokens', 0)} out]"
    )
    return msg


def pull_samples() -> list:
    """Pull high-quality coding examples from FTS5 indexes."""
    print("=" * 60)
    print("STEP 1: Loading starter code from JCoder indexes")
    print("=" * 60)

    samples = []
    index_sources = [
        (JCODER_ROOT / "data/indexes/magicoder_oss_instruct.fts5.db", "magicoder", 15),
        (JCODER_ROOT / "data/indexes/code_feedback.fts5.db", "code_feedback", 10),
        (JCODER_ROOT / "data/indexes/code_exercises.fts5.db", "code_exercises", 10),
        (JCODER_ROOT / "data/indexes/codeparrot_clean.fts5.db", "codeparrot", 10),
        (JCODER_ROOT / "data/indexes/cot_code_instruct.fts5.db", "cot_instruct", 5),
    ]

    for path, name, n in index_sources:
        try:
            conn = sqlite3.connect(str(path))
            total = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            offsets = random.sample(range(total), min(n * 3, total))
            rows = []
            for off in offsets:
                row = conn.execute(
                    "SELECT search_content FROM chunks LIMIT 1 OFFSET ?", (off,)
                ).fetchone()
                if row and 200 < len(row[0]) < 3000:
                    text = row[0]
                    code_keywords = ["def ", "class ", "return ", "import ", "function"]
                    if any(kw in text.lower() for kw in code_keywords):
                        rows.append(text)
                        if len(rows) >= n:
                            break
            samples.extend(rows)
            print(f"[OK] {name}: pulled {len(rows)} quality samples (from {total:,} total)")
            conn.close()
        except Exception as e:
            print(f"[WARN] {name}: {e}")

    print(f"\n[OK] Total starter code samples: {len(samples)}")
    print(f"[OK] Total chars: {sum(len(s) for s in samples):,}")
    return samples


def digest_patterns(samples: list) -> str:
    """Have GPT-5.4 extract coding patterns from samples."""
    print()
    print("=" * 60)
    print("STEP 2: GPT-5.4 digesting coding patterns")
    print("=" * 60)

    code_context = "\n\n---NEXT EXAMPLE---\n\n".join(samples[:30])

    system = (
        "You are JCoder's coding brain. You are about to receive high-quality "
        "Python code examples from your indexed archives. Study them carefully "
        "and extract common patterns, best practices, and idioms."
    )

    digest_prompt = (
        f"Study these {min(len(samples), 30)} code examples from your archive:\n\n"
        f"{code_context[:20000]}\n\n"
        "Summarize the top 10 coding patterns and best practices you extracted. "
        "Be specific -- cite actual patterns from the examples, not generic advice."
    )

    print("[OK] Digesting patterns...")
    patterns = call_gpt5(system, digest_prompt, 2048)
    print()
    print("EXTRACTED PATTERNS:")
    print(patterns[:2000])
    if len(patterns) > 2000:
        print(f"  ... ({len(patterns) - 2000} more chars)")
    return patterns


def run_challenges(patterns: str) -> list:
    """Run 4 progressively harder coding challenges."""
    print()
    print("=" * 60)
    print("STEP 3: Coding Challenges (with learned context)")
    print("=" * 60)

    challenges = [
        {
            "name": "Easy: FizzBuzz Variant",
            "prompt": (
                "Write a Python function fizzbuzz_custom(n, rules) that takes an integer n and "
                "a list of (divisor, word) tuples. For numbers 1 to n, print the concatenated "
                "words for all matching divisors, or the number itself if none match.\n"
                'Example: fizzbuzz_custom(15, [(3,"Fizz"),(5,"Buzz")]) should print classic FizzBuzz.\n'
                "Include type hints and a docstring."
            ),
        },
        {
            "name": "Medium: LRU Cache from Scratch",
            "prompt": (
                "Implement an LRU Cache in Python from scratch (no functools.lru_cache).\n"
                "Requirements:\n"
                "- LRUCache(capacity: int) constructor\n"
                "- get(key) -> value or -1\n"
                "- put(key, value) -> None\n"
                "- O(1) time for both operations\n"
                "- Use OrderedDict or implement with a doubly-linked list + dict\n"
                "Include type hints, docstring, and at least 3 unit tests using pytest."
            ),
        },
        {
            "name": "Hard: Async Rate Limiter",
            "prompt": (
                "Write an async rate limiter class in Python using asyncio.\n"
                "Requirements:\n"
                "- AsyncRateLimiter(max_requests: int, window_seconds: float)\n"
                "- async def acquire(self) -> bool -- returns True if allowed, False if exceeded\n"
                "- async def wait_and_acquire(self) -> None -- waits until a slot is available\n"
                "- Thread-safe (use asyncio.Lock)\n"
                "- Sliding window algorithm (not fixed window)\n"
                "- Include type hints, docstring, and 3 async pytest tests\n"
                "The tests should verify: basic rate limiting works, window expiry allows new "
                "requests, and concurrent callers are properly serialized."
            ),
        },
        {
            "name": "Expert: Persistent AVL Tree",
            "prompt": (
                "Implement a persistent (immutable) AVL tree in Python.\n"
                "Requirements:\n"
                "- PersistentAVL class with insert(key, value) and search(key) methods\n"
                "- insert() returns a NEW tree (old tree remains unchanged)\n"
                "- search() returns Optional[value]\n"
                "- Proper AVL rotations (left, right, left-right, right-left)\n"
                "- O(log n) insert and search\n"
                "- Include a version() method that returns the tree version number\n"
                "- Include type hints, docstrings, and 5 unit tests proving:\n"
                "  1. Basic insert/search works\n"
                "  2. Old versions are preserved after insert\n"
                "  3. Balance is maintained (height difference <= 1)\n"
                "  4. Multiple versions can coexist\n"
                "  5. Search on empty tree returns None"
            ),
        },
    ]

    coding_system = (
        "You are JCoder, an expert Python coding assistant. You have studied thousands "
        "of code examples and extracted these patterns:\n\n"
        f"{patterns}\n\n"
        "Write clean, correct, well-tested Python code. Follow these rules:\n"
        "- Type hints on all function signatures\n"
        "- Docstrings on all public methods\n"
        "- pytest-style tests (not unittest)\n"
        "- No unnecessary complexity\n"
        "- Handle edge cases\n"
        "- Use Pythonic idioms\n\n"
        "Return ONLY the code. No explanations before or after."
    )

    results = []
    for i, challenge in enumerate(challenges):
        print(f"\n--- Challenge {i+1}/4: {challenge['name']} ---")
        print("Asking GPT-5.4...")
        answer = call_gpt5(coding_system, challenge["prompt"], 3072)
        results.append({"challenge": challenge["name"], "answer": answer})
        # Show first 60 lines
        lines = answer.split("\n")
        preview = "\n".join(lines[:60])
        print(preview)
        if len(lines) > 60:
            print(f"  ... ({len(lines) - 60} more lines)")
        print()

    return results


def save_results(samples: list, patterns: str, results: list):
    """Save everything to agent_knowledge."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    outpath = KNOWLEDGE_DIR / f"{ts}_coding_challenge_results.md"

    with open(outpath, "w", encoding="utf-8") as f:
        f.write("# JCoder Coding Challenge Results\n\n")
        f.write(f"Generated: {datetime.now(timezone.utc).isoformat()}\n")
        f.write(f"Model: gpt-5.4\nStarter examples: {len(samples)}\n\n---\n\n")
        f.write(f"## Extracted Patterns\n\n{patterns}\n\n---\n\n")
        for r in results:
            f.write(f"## {r['challenge']}\n\n```python\n{r['answer']}\n```\n\n---\n\n")

    print(f"[OK] All results saved: {outpath}")
    return outpath


def main():
    samples = pull_samples()
    if not samples:
        print("[FAIL] No samples found")
        sys.exit(1)

    patterns = digest_patterns(samples)
    results = run_challenges(patterns)
    save_results(samples, patterns, results)

    print("\n" + "=" * 60)
    print("DONE -- 4/4 challenges completed")
    print("=" * 60)


if __name__ == "__main__":
    main()
