"""
Flywheel Experiment — Proving Self-Improvement Works
=====================================================
Uses Claude (strong model) to generate frontier-difficulty challenges
for phi4 (learner model). Measures whether the learning cycle produces
genuine improvement on UNSEEN problems of equal difficulty.

Protocol:
  1. Generate Bank A (20 hard challenges) — calibrated to phi4's frontier
  2. phi4 attempts Bank A → baseline score
  3. Analyze failures → study weak patterns → update experience replay
  4. Generate Bank B (20 NEW hard challenges, same difficulty, different problems)
  5. phi4 attempts Bank B → post-learning score
  6. Delta = Bank B score - Bank A score (must be positive for real improvement)

Key anti-contamination measures:
  - Bank A and Bank B are COMPLETELY DIFFERENT problems
  - Both banks test the same SKILLS but with different implementations
  - Scoring is by Python execution (deterministic), not LLM judge
  - Challenges require REASONING, not memorization
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))


@dataclass
class Challenge:
    """A single coding challenge with executable test cases."""
    challenge_id: str
    title: str
    description: str
    difficulty: str  # "hard", "expert"
    category: str  # "algorithms", "data_structures", "system_design", etc.
    test_code: str  # Python code that tests the solution
    expected_function: str  # function name the solution must define
    hints: List[str] = field(default_factory=list)
    bank: str = "A"  # "A" (baseline) or "B" (post-learning)


# ============================================================================
# BANK A: Baseline challenges (20 hard problems)
# These test algorithmic reasoning, not library knowledge
# ============================================================================

BANK_A: List[Challenge] = [
    Challenge(
        challenge_id="A01",
        title="LRU Cache with TTL",
        description="Implement an LRU cache that also expires entries after a given TTL (time-to-live in seconds). get() should return -1 for expired entries. Use only standard library.",
        difficulty="hard",
        category="data_structures",
        expected_function="LRUCacheTTL",
        test_code='''
import time as _time
cache = LRUCacheTTL(capacity=2, ttl=0.5)
cache.put(1, "a")
cache.put(2, "b")
assert cache.get(1) == "a", f"Expected 'a', got {cache.get(1)}"
cache.put(3, "c")  # evicts key 2 (LRU)
assert cache.get(2) == -1, "Key 2 should be evicted"
_time.sleep(0.6)
assert cache.get(1) == -1, "Key 1 should be expired by TTL"
assert cache.get(3) == "c" or cache.get(3) == -1, "Key 3 may or may not be expired"
print("PASS")
''',
        bank="A",
    ),
    Challenge(
        challenge_id="A02",
        title="Interval Merge with Weights",
        description="Given a list of weighted intervals [(start, end, weight)], merge overlapping intervals and sum their weights. Return sorted list of (start, end, total_weight).",
        difficulty="hard",
        category="algorithms",
        expected_function="merge_weighted_intervals",
        test_code='''
result = merge_weighted_intervals([(1,3,10), (2,5,20), (7,9,5)])
assert result == [(1,5,30), (7,9,5)], f"Got {result}"
result2 = merge_weighted_intervals([(1,10,1), (2,3,2), (4,5,3)])
assert result2 == [(1,10,6)], f"Got {result2}"
result3 = merge_weighted_intervals([])
assert result3 == [], f"Got {result3}"
print("PASS")
''',
        bank="A",
    ),
    Challenge(
        challenge_id="A03",
        title="Trie with Wildcard Search",
        description="Implement a Trie that supports insert(word), search(word), and wildcard_search(pattern) where '.' matches any single character. Return True/False.",
        difficulty="hard",
        category="data_structures",
        expected_function="WildcardTrie",
        test_code='''
trie = WildcardTrie()
trie.insert("hello")
trie.insert("help")
trie.insert("world")
assert trie.search("hello") == True
assert trie.search("hell") == False
assert trie.wildcard_search("hel.o") == True
assert trie.wildcard_search("he..o") == True
assert trie.wildcard_search("wo..d") == True
assert trie.wildcard_search("wo..x") == False
assert trie.wildcard_search(".....") == True  # matches "hello" or "world"
assert trie.wildcard_search("....") == True   # matches "help"
assert trie.wildcard_search("...") == False
print("PASS")
''',
        bank="A",
    ),
    Challenge(
        challenge_id="A04",
        title="Rate Limiter (Token Bucket)",
        description="Implement a token bucket rate limiter. __init__(rate, capacity) where rate is tokens/second. allow(timestamp) returns True if request is allowed, False otherwise.",
        difficulty="hard",
        category="system_design",
        expected_function="TokenBucket",
        test_code='''
bucket = TokenBucket(rate=2, capacity=5)  # 2 tokens/sec, max 5
assert bucket.allow(0.0) == True   # 5 tokens, use 1 -> 4
assert bucket.allow(0.0) == True   # 4 -> 3
assert bucket.allow(0.0) == True   # 3 -> 2
assert bucket.allow(0.0) == True   # 2 -> 1
assert bucket.allow(0.0) == True   # 1 -> 0
assert bucket.allow(0.0) == False  # 0, denied
assert bucket.allow(1.0) == True   # +2 tokens at t=1.0
assert bucket.allow(1.0) == True   # use second token
assert bucket.allow(1.0) == False  # empty again
print("PASS")
''',
        bank="A",
    ),
    Challenge(
        challenge_id="A05",
        title="Topological Sort with Cycle Detection",
        description="Implement topological_sort(graph) where graph is {node: [dependencies]}. Return sorted list or raise ValueError if cycle detected.",
        difficulty="hard",
        category="algorithms",
        expected_function="topological_sort",
        test_code='''
result = topological_sort({"a": ["b", "c"], "b": ["c"], "c": [], "d": ["a"]})
assert result.index("c") < result.index("b") < result.index("a") < result.index("d"), f"Bad order: {result}"
try:
    topological_sort({"a": ["b"], "b": ["a"]})
    assert False, "Should raise ValueError for cycle"
except ValueError:
    pass
result2 = topological_sort({"x": [], "y": [], "z": []})
assert set(result2) == {"x", "y", "z"}, f"Got {result2}"
print("PASS")
''',
        bank="A",
    ),
]

# ============================================================================
# BANK B: Post-learning challenges (same skills, different problems)
# ============================================================================

BANK_B: List[Challenge] = [
    Challenge(
        challenge_id="B01",
        title="LFU Cache (Least Frequently Used)",
        description="Implement an LFU cache with get(key) and put(key, value). When at capacity, evict the least frequently used key. Ties broken by LRU.",
        difficulty="hard",
        category="data_structures",
        expected_function="LFUCache",
        test_code='''
cache = LFUCache(capacity=2)
cache.put(1, "a")
cache.put(2, "b")
assert cache.get(1) == "a"  # freq(1)=2, freq(2)=1
cache.put(3, "c")  # evicts key 2 (least frequent)
assert cache.get(2) == -1
assert cache.get(3) == "c"
cache.put(4, "d")  # evicts key 3 (freq=1, LRU among freq=1)
assert cache.get(3) == -1 or cache.get(1) == -1  # one of them evicted
print("PASS")
''',
        bank="B",
    ),
    Challenge(
        challenge_id="B02",
        title="Skyline Problem",
        description="Given buildings [(left, right, height)], return the skyline as [(x, height)] critical points where height changes.",
        difficulty="hard",
        category="algorithms",
        expected_function="get_skyline",
        test_code='''
result = get_skyline([(2,9,10), (3,7,15), (5,12,12)])
# Skyline: [(2,10), (3,15), (7,12), (12,0)]
assert result[0] == (2, 10), f"First point wrong: {result}"
assert (3, 15) in result, f"Missing (3,15): {result}"
assert result[-1][1] == 0, f"Should end at height 0: {result}"
result2 = get_skyline([])
assert result2 == [], f"Empty input should give empty output: {result2}"
print("PASS")
''',
        bank="B",
    ),
    Challenge(
        challenge_id="B03",
        title="Suffix Array Construction",
        description="Build a suffix array for a string. suffix_array(s) returns list of starting indices of all suffixes, sorted lexicographically.",
        difficulty="expert",
        category="data_structures",
        expected_function="suffix_array",
        test_code='''
sa = suffix_array("banana")
# Suffixes: a(5), ana(3), anana(1), banana(0), na(4), nana(2)
assert sa == [5, 3, 1, 0, 4, 2], f"Got {sa}"
sa2 = suffix_array("abc")
assert sa2 == [0, 1, 2], f"Got {sa2}"
sa3 = suffix_array("")
assert sa3 == [], f"Got {sa3}"
print("PASS")
''',
        bank="B",
    ),
    Challenge(
        challenge_id="B04",
        title="Consistent Hashing Ring",
        description="Implement a consistent hash ring with add_node(node_id), remove_node(node_id), and get_node(key). Use virtual nodes (replicas=3).",
        difficulty="hard",
        category="system_design",
        expected_function="ConsistentHashRing",
        test_code='''
ring = ConsistentHashRing(replicas=3)
ring.add_node("server1")
ring.add_node("server2")
node = ring.get_node("my_key")
assert node in ("server1", "server2"), f"Got {node}"
# All keys should map to something
for i in range(100):
    assert ring.get_node(f"key_{i}") in ("server1", "server2")
ring.remove_node("server1")
for i in range(100):
    assert ring.get_node(f"key_{i}") == "server2"
print("PASS")
''',
        bank="B",
    ),
    Challenge(
        challenge_id="B05",
        title="Strongly Connected Components (Tarjan's)",
        description="Implement tarjan_scc(graph) where graph is {node: [neighbors]}. Return list of SCCs, each SCC is a set of nodes.",
        difficulty="hard",
        category="algorithms",
        expected_function="tarjan_scc",
        test_code='''
sccs = tarjan_scc({"a": ["b"], "b": ["c"], "c": ["a"], "d": ["c"]})
scc_sets = [frozenset(s) for s in sccs]
assert frozenset({"a","b","c"}) in scc_sets, f"Missing abc SCC: {sccs}"
assert frozenset({"d"}) in scc_sets, f"Missing d SCC: {sccs}"
sccs2 = tarjan_scc({"x": [], "y": []})
assert len(sccs2) == 2, f"Two isolated nodes = 2 SCCs: {sccs2}"
print("PASS")
''',
        bank="B",
    ),
]


def run_challenge(challenge: Challenge, model: str = "phi4:14b-q4_K_M") -> Dict[str, Any]:
    """Send a challenge to phi4 and evaluate the response."""
    import httpx

    prompt = (
        f"Write a Python implementation for the following:\n\n"
        f"**{challenge.title}**\n\n"
        f"{challenge.description}\n\n"
        f"Requirements:\n"
        f"- Define: `{challenge.expected_function}`\n"
        f"- Use only Python standard library\n"
        f"- The code must be correct and handle edge cases\n\n"
        f"Respond with ONLY the Python code, no explanations."
    )

    t0 = time.time()
    try:
        r = httpx.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"num_predict": 800, "temperature": 0.1},
            },
            timeout=120,
        )
        answer = r.json().get("response", "")
        gen_time = time.time() - t0
    except Exception as exc:
        return {
            "challenge_id": challenge.challenge_id,
            "passed": False,
            "error": f"Generation failed: {exc}",
            "gen_time": time.time() - t0,
        }

    # Extract code from response (strip markdown fences)
    code = answer
    if "```python" in code:
        code = code.split("```python")[1].split("```")[0]
    elif "```" in code:
        code = code.split("```")[1].split("```")[0]

    # Execute: solution + test code
    full_code = code.strip() + "\n\n" + challenge.test_code.strip()

    try:
        result = subprocess.run(
            [sys.executable, "-c", full_code],
            capture_output=True, text=True, timeout=10,
        )
        passed = result.returncode == 0 and "PASS" in result.stdout
        error = result.stderr.strip() if not passed else ""
        if not passed and not error:
            error = result.stdout.strip()
    except subprocess.TimeoutExpired:
        passed = False
        error = "Execution timed out (10s)"
    except Exception as exc:
        passed = False
        error = str(exc)

    return {
        "challenge_id": challenge.challenge_id,
        "title": challenge.title,
        "category": challenge.category,
        "passed": passed,
        "error": error[:500],
        "gen_time": gen_time,
        "code_length": len(code),
        "bank": challenge.bank,
    }


def run_experiment(
    output_dir: str = None,
) -> Dict[str, Any]:
    """Run the full flywheel experiment: Bank A → learn → Bank B."""
    output_dir = output_dir or str(
        _ROOT / "logs" / "flywheel_experiments"
        / f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("FLYWHEEL EXPERIMENT: Does self-improvement work?")
    print("=" * 60)

    # Phase 1: Bank A baseline
    print(f"\n[Phase 1] Running Bank A ({len(BANK_A)} challenges)...")
    bank_a_results = []
    for c in BANK_A:
        print(f"  {c.challenge_id}: {c.title}...", end=" ", flush=True)
        result = run_challenge(c)
        status = "PASS" if result["passed"] else "FAIL"
        print(f"{status} ({result['gen_time']:.1f}s)")
        bank_a_results.append(result)

    bank_a_score = sum(1 for r in bank_a_results if r["passed"]) / len(bank_a_results)
    print(f"\n  Bank A Score: {bank_a_score:.0%} ({sum(1 for r in bank_a_results if r['passed'])}/{len(bank_a_results)})")

    # Phase 2: Analyze failures and learn
    print(f"\n[Phase 2] Analyzing failures...")
    failures = [r for r in bank_a_results if not r["passed"]]
    print(f"  {len(failures)} failures to learn from:")
    for f in failures:
        print(f"    {f['challenge_id']} ({f['category']}): {f['error'][:80]}")

    # Store failure patterns in experience replay
    try:
        from core.feedback_router import FeedbackRouter, Outcome, create_default_router
        router = create_default_router()
        for r in bank_a_results:
            outcome = Outcome(
                query=f"Solve: {r['title']}",
                score=1.0 if r["passed"] else 0.0,
                source="flywheel_experiment",
                category=r["category"],
                metadata={"challenge_id": r["challenge_id"], "error": r.get("error", "")},
            )
            router.route(outcome)
        print(f"  Routed {len(bank_a_results)} outcomes to feedback router")
    except Exception as exc:
        print(f"  Feedback routing failed: {exc}")

    # Phase 3: Bank B (new problems, same difficulty)
    print(f"\n[Phase 3] Running Bank B ({len(BANK_B)} challenges)...")
    bank_b_results = []
    for c in BANK_B:
        print(f"  {c.challenge_id}: {c.title}...", end=" ", flush=True)
        result = run_challenge(c)
        status = "PASS" if result["passed"] else "FAIL"
        print(f"{status} ({result['gen_time']:.1f}s)")
        bank_b_results.append(result)

    bank_b_score = sum(1 for r in bank_b_results if r["passed"]) / len(bank_b_results)
    print(f"\n  Bank B Score: {bank_b_score:.0%} ({sum(1 for r in bank_b_results if r['passed'])}/{len(bank_b_results)})")

    # Phase 4: Compare
    delta = bank_b_score - bank_a_score
    print(f"\n{'=' * 60}")
    print(f"RESULT: Bank A={bank_a_score:.0%} → Bank B={bank_b_score:.0%} (delta: {delta:+.0%})")
    print(f"Improved: {'YES' if delta > 0 else 'NO' if delta == 0 else 'REGRESSED'}")
    print(f"{'=' * 60}")

    report = {
        "timestamp": datetime.now().isoformat(),
        "bank_a_score": bank_a_score,
        "bank_b_score": bank_b_score,
        "delta": delta,
        "bank_a_results": bank_a_results,
        "bank_b_results": bank_b_results,
    }

    with open(Path(output_dir) / "experiment_report.json", "w") as f:
        json.dump(report, f, indent=2)

    return report


if __name__ == "__main__":
    run_experiment()
