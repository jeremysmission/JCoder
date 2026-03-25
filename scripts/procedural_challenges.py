"""
Procedural Challenge Generator
================================
Generates coding challenges algorithmically with guaranteed-correct
test cases. Impossible to memorize because problems are randomly
generated at test time.

Based on Absolute Zero methodology: external verification (Python
execution) + difficulty calibration (frontier problems) + zero
contamination (procedurally generated).

Challenge types:
  1. Array/List algorithms (sort variants, search variants)
  2. Graph algorithms (shortest path, connectivity)
  3. String algorithms (pattern matching, transformation)
  4. Math/Number theory (prime factorization, combinatorics)
  5. Data structure implementation (custom containers)

Each challenge includes:
  - A description generated from a template
  - A brute-force reference solution for generating test cases
  - Parameterized difficulty (input size, edge case complexity)
"""

from __future__ import annotations

import hashlib
import random
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple


@dataclass
class ProceduralChallenge:
    """A procedurally generated challenge with verifiable test cases."""
    challenge_id: str
    title: str
    description: str
    category: str
    difficulty: int  # 1-10
    expected_function: str
    test_cases: str  # Python code that tests the solution
    reference_answer: str = ""  # brute-force solution for verification
    seed: int = 0


def _make_id(seed: int, category: str) -> str:
    return hashlib.sha256(f"{seed}:{category}".encode()).hexdigest()[:8]


def generate_array_challenge(seed: int, difficulty: int = 5) -> ProceduralChallenge:
    """Generate an array manipulation challenge."""
    rng = random.Random(seed)

    templates = [
        {
            "title": "Custom Sort by Criterion",
            "desc": "Sort a list of tuples {tuples} by the {criterion} element, breaking ties by the {tiebreak} element.",
            "func": "custom_sort",
        },
        {
            "title": "Windowed Maximum",
            "desc": "Given array {arr} and window size {k}, return the maximum value in each sliding window.",
            "func": "windowed_max",
        },
        {
            "title": "Subarray Sum Target",
            "desc": "Find the number of contiguous subarrays in {arr} that sum to exactly {target}.",
            "func": "count_subarrays_with_sum",
        },
    ]

    template = rng.choice(templates)
    n = min(5 + difficulty * 2, 20)

    if template["func"] == "custom_sort":
        tuples = [(rng.randint(1, 100), rng.randint(1, 100)) for _ in range(n)]
        criterion = rng.choice([0, 1])
        tiebreak = 1 - criterion
        desc = f"Sort the list {tuples} by element at index {criterion}, breaking ties by element at index {tiebreak}. Return the sorted list."
        expected = sorted(tuples, key=lambda x: (x[criterion], x[tiebreak]))
        test_code = f"""
result = custom_sort({tuples}, {criterion}, {tiebreak})
assert result == {expected}, f"Expected {expected}, got {{result}}"
print("PASS")
"""

    elif template["func"] == "windowed_max":
        arr = [rng.randint(-50, 50) for _ in range(n)]
        k = rng.randint(2, min(5, n))
        expected = [max(arr[i:i + k]) for i in range(len(arr) - k + 1)]
        desc = f"Given array {arr} and window size {k}, return the maximum value in each sliding window of size {k}."
        test_code = f"""
result = windowed_max({arr}, {k})
assert result == {expected}, f"Expected {expected}, got {{result}}"
print("PASS")
"""

    else:  # count_subarrays_with_sum
        arr = [rng.randint(-5, 10) for _ in range(n)]
        target = rng.randint(1, 15)
        # Brute force count
        count = 0
        for i in range(len(arr)):
            s = 0
            for j in range(i, len(arr)):
                s += arr[j]
                if s == target:
                    count += 1
        desc = f"Count contiguous subarrays in {arr} that sum to {target}."
        test_code = f"""
result = count_subarrays_with_sum({arr}, {target})
assert result == {count}, f"Expected {count}, got {{result}}"
print("PASS")
"""

    return ProceduralChallenge(
        challenge_id=_make_id(seed, "array"),
        title=template["title"],
        description=desc,
        category="array",
        difficulty=difficulty,
        expected_function=template["func"],
        test_cases=test_code,
        seed=seed,
    )


def generate_graph_challenge(seed: int, difficulty: int = 5) -> ProceduralChallenge:
    """Generate a graph algorithm challenge."""
    rng = random.Random(seed)
    n = min(4 + difficulty, 10)

    # Generate random directed graph
    edges = []
    for i in range(n):
        for j in range(n):
            if i != j and rng.random() < 0.3:
                edges.append((i, j, rng.randint(1, 20)))

    graph = {}
    for u, v, w in edges:
        graph.setdefault(u, []).append((v, w))
    for i in range(n):
        graph.setdefault(i, [])

    # Shortest path from 0 to n-1 using Dijkstra (brute force)
    import heapq
    dist = {i: float("inf") for i in range(n)}
    dist[0] = 0
    pq = [(0, 0)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v, w in graph.get(u, []):
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                heapq.heappush(pq, (dist[v], v))
    target = n - 1
    expected = dist[target] if dist[target] != float("inf") else -1

    adj_list = {k: [(v, w) for v, w in vs] for k, vs in graph.items()}
    desc = (
        f"Find the shortest path distance from node 0 to node {target} "
        f"in the weighted directed graph: {adj_list}. "
        f"Return -1 if no path exists."
    )

    test_code = f"""
result = shortest_path({adj_list}, 0, {target})
assert result == {expected}, f"Expected {expected}, got {{result}}"
print("PASS")
"""

    return ProceduralChallenge(
        challenge_id=_make_id(seed, "graph"),
        title="Weighted Shortest Path",
        description=desc,
        category="graph",
        difficulty=difficulty,
        expected_function="shortest_path",
        test_cases=test_code,
        seed=seed,
    )


def generate_string_challenge(seed: int, difficulty: int = 5) -> ProceduralChallenge:
    """Generate a string algorithm challenge."""
    rng = random.Random(seed)

    # Longest common subsequence
    alphabet = "abcdefghij"
    n = min(5 + difficulty, 15)
    s1 = "".join(rng.choices(alphabet[:5], k=n))
    s2 = "".join(rng.choices(alphabet[:5], k=n))

    # DP brute force
    m, n2 = len(s1), len(s2)
    dp = [[0] * (n2 + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n2 + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    expected = dp[m][n2]

    desc = f'Find the length of the longest common subsequence of "{s1}" and "{s2}".'

    test_code = f"""
result = lcs_length("{s1}", "{s2}")
assert result == {expected}, f"Expected {expected}, got {{result}}"
print("PASS")
"""

    return ProceduralChallenge(
        challenge_id=_make_id(seed, "string"),
        title="Longest Common Subsequence Length",
        description=desc,
        category="string",
        difficulty=difficulty,
        expected_function="lcs_length",
        test_cases=test_code,
        seed=seed,
    )


def generate_challenge_bank(
    n: int = 10,
    seed: int = None,
    difficulty: int = 5,
) -> List[ProceduralChallenge]:
    """Generate a bank of n procedural challenges."""
    if seed is None:
        seed = int(time.time() * 1000) % (2**31)

    generators = [
        generate_array_challenge,
        generate_graph_challenge,
        generate_string_challenge,
    ]

    challenges = []
    for i in range(n):
        gen = generators[i % len(generators)]
        challenges.append(gen(seed + i, difficulty))

    return challenges


def evaluate_challenge(
    challenge: ProceduralChallenge,
    model: str = "phi4:14b-q4_K_M",
    use_lessons: bool = True,
) -> Dict[str, Any]:
    """Send challenge to model and evaluate by execution.

    When use_lessons=True, retrieves past experience from the lessons
    index before generating. This is how a 14B model competes with 100B+:
    it retrieves EXACTLY the right patterns and avoids known mistakes.
    """
    import httpx

    # Retrieve lessons from past attempts (RAG-powered self-learning)
    lesson_context = ""
    if use_lessons:
        try:
            from core.lessons_index import LessonsIndex
            lessons = LessonsIndex()
            lesson_context = lessons.build_context_prompt(
                challenge.description, category=challenge.category,
            )
        except Exception:
            pass

    prompt = (
        f"Write a Python function `{challenge.expected_function}` that solves:\n\n"
        f"{challenge.description}\n\n"
    )
    if lesson_context:
        prompt += f"{lesson_context}\n\n"
    prompt += "Respond with ONLY the Python function code, nothing else."

    t0 = time.time()
    try:
        r = httpx.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"num_predict": 500, "temperature": 0.0},
            },
            timeout=60,
        )
        code = r.json().get("response", "")
        gen_time = time.time() - t0
    except Exception as exc:
        return {"passed": False, "error": str(exc), "gen_time": time.time() - t0}

    # Clean code
    if "```python" in code:
        code = code.split("```python")[1].split("```")[0]
    elif "```" in code:
        code = code.split("```")[1].split("```")[0]

    full_code = code.strip() + "\n\n" + challenge.test_cases.strip()

    try:
        result = subprocess.run(
            [sys.executable, "-c", full_code],
            capture_output=True, text=True, timeout=10,
        )
        passed = result.returncode == 0 and "PASS" in result.stdout
        error = result.stderr[:300] if not passed else ""
    except subprocess.TimeoutExpired:
        passed = False
        error = "Timeout"
    except Exception as exc:
        passed = False
        error = str(exc)

    # Store attempt in lessons index (RAG feedback loop)
    if use_lessons:
        try:
            from core.lessons_index import LessonsIndex
            lessons = LessonsIndex()
            lessons.store_attempt(
                challenge_description=challenge.description,
                generated_code=code.strip(),
                passed=passed,
                error_message=error,
                category=challenge.category,
                difficulty=challenge.difficulty,
            )
        except Exception:
            pass

    return {
        "challenge_id": challenge.challenge_id,
        "category": challenge.category,
        "difficulty": challenge.difficulty,
        "passed": passed,
        "error": error,
        "gen_time": gen_time,
        "had_lessons": bool(lesson_context),
    }


if __name__ == "__main__":
    # Quick demo: generate and test 6 challenges
    bank = generate_challenge_bank(n=6, seed=42, difficulty=5)
    print(f"Generated {len(bank)} procedural challenges\n")

    for c in bank:
        print(f"  [{c.category}] {c.title}: {c.description[:80]}...")
        result = evaluate_challenge(c)
        status = "PASS" if result["passed"] else "FAIL"
        print(f"    -> {status} ({result['gen_time']:.1f}s)")
        if result.get("error"):
            print(f"    Error: {result['error'][:100]}")
        print()

    passed = sum(1 for c in bank for r in [evaluate_challenge(c)] if r["passed"])
    # Note: this double-evaluates, just for demo
    print(f"Score: {sum(1 for c in bank if evaluate_challenge(c)['passed'])}/{len(bank)}")
