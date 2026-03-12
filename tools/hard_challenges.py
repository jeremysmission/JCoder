"""
JCoder HARD Mode Coding Challenges
------------------------------------
6 brutally hard challenges that test:
- Concurrency correctness
- Algorithm design under constraints
- Systems programming
- Mathematical reasoning
- Real-world production patterns
- Self-modifying code

Each challenge is verified by running the generated code.
"""

import ast
import asyncio
import json
import os
import re
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

sys.stdout = open(sys.stdout.fileno(), mode="w", encoding="utf-8", buffering=1)

JCODER_ROOT = Path(__file__).resolve().parent.parent
KNOWLEDGE_DIR = JCODER_ROOT / "data" / "agent_knowledge"


def call_gpt5(system_p: str, user_p: str, max_tok: int = 6144) -> str:
    api_key = os.environ.get("OPENAI_API_KEY", "")
    payload = {
        "model": "gpt-5.4",
        "messages": [
            {"role": "system", "content": system_p},
            {"role": "user", "content": user_p},
        ],
        "temperature": 0.15,
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
            json=payload, headers=headers,
        )
        resp.raise_for_status()
        data = resp.json()
    elapsed = time.monotonic() - t0
    usage = data.get("usage", {})
    print(
        f"  [{elapsed:.1f}s, {usage.get('prompt_tokens', 0)} in, "
        f"{usage.get('completion_tokens', 0)} out]"
    )
    return data["choices"][0]["message"]["content"]


SYSTEM = """You are JCoder, an expert-level Python engineer.
Write production-quality code. Every solution must:
- Be correct and handle edge cases
- Include type hints on all signatures
- Include docstrings
- Include pytest-style tests that PROVE correctness
- Use no external packages beyond stdlib + pytest
- Be runnable as a single file

Return ONLY Python code. No markdown fences. No explanations."""

CHALLENGES = [
    # ---------------------------------------------------------------
    # 1. Lock-free concurrent data structure
    # ---------------------------------------------------------------
    {
        "name": "1. Lock-Free MPMC Ring Buffer",
        "prompt": """Implement a bounded multi-producer multi-consumer (MPMC) ring buffer
using only threading and no locks (use atomic-style CAS via threading primitives).

Requirements:
- RingBuffer(capacity: int) -- capacity must be power of 2
- put(item: T) -> bool  -- non-blocking, returns False if full
- get() -> Optional[T]  -- non-blocking, returns None if empty
- Internal array with head/tail indices using modular arithmetic
- Thread-safe without using Lock, RLock, or Semaphore
  (you may use threading.Event or atomic int patterns with ctypes or a compare-and-swap helper)
- If pure lock-free is too hard in Python, use a single fine-grained Lock but prove
  it never deadlocks

Include 5 tests:
1. Single-thread put/get roundtrip
2. Producer fills buffer, verify full returns False
3. Consumer on empty buffer returns None
4. 4 producers + 4 consumers, 10000 items total, verify none lost and none duplicated
5. Verify FIFO ordering under single-producer single-consumer""",
    },
    # ---------------------------------------------------------------
    # 2. Raft consensus - leader election only
    # ---------------------------------------------------------------
    {
        "name": "2. Raft Leader Election (In-Process Simulation)",
        "prompt": """Implement a simplified Raft leader election protocol as an in-process simulation.

Requirements:
- RaftNode class with states: FOLLOWER, CANDIDATE, LEADER
- Each node has an id, current_term, voted_for, and election_timeout
- Nodes communicate via an in-memory message bus (dict of queues)
- MessageBus class that delivers RequestVote and VoteResponse messages
- Election timeout triggers candidacy: increment term, vote for self, send RequestVote
- Nodes grant votes if candidate term >= current term AND haven't voted this term
- Candidate becomes leader when it receives majority votes
- Simulate a 5-node cluster, kill the leader, verify a new leader is elected
- The simulation must be deterministic (seeded randomness for timeouts)

Include 4 tests:
1. 5 nodes elect exactly one leader
2. Kill leader, new election produces new leader with higher term
3. Split vote scenario resolves in next round
4. Node with stale term cannot win election""",
    },
    # ---------------------------------------------------------------
    # 3. Compiler: expression to stack machine
    # ---------------------------------------------------------------
    {
        "name": "3. Expression Compiler (Lexer + Parser + Codegen + VM)",
        "prompt": """Build a complete mini expression compiler + virtual machine.

The language supports:
- Integer literals and float literals
- Variables (single-letter a-z)
- Operators: + - * / % ** (power) with correct precedence
- Parentheses
- Comparison: == != < > <= >= (return 1 or 0)
- Ternary: expr if condition else expr
- let x = expr (variable binding)
- Multiple statements separated by semicolons

Pipeline:
1. Lexer: string -> list of tokens
2. Parser: tokens -> AST (recursive descent, respects precedence)
3. Codegen: AST -> list of stack machine instructions
4. VM: executes instructions on a stack machine with a variable environment

Stack machine instructions:
PUSH_CONST value | PUSH_VAR name | STORE_VAR name | ADD | SUB | MUL | DIV |
MOD | POW | CMP_EQ | CMP_NE | CMP_LT | CMP_GT | CMP_LE | CMP_GE |
JUMP_IF_FALSE addr | JUMP addr | POP

Include 6 tests:
1. "2 + 3 * 4" evaluates to 14
2. "(2 + 3) * 4" evaluates to 20
3. "let x = 10; x * x + 1" evaluates to 101
4. "2 ** 3 ** 2" evaluates to 512 (right-associative power)
5. "1 + 2 if 3 > 2 else 99" evaluates to 3
6. "let a = 5; let b = 3; a * b + a if a > b else 0" evaluates to 20""",
    },
    # ---------------------------------------------------------------
    # 4. Constraint solver (Sudoku via backtracking + arc consistency)
    # ---------------------------------------------------------------
    {
        "name": "4. Sudoku Solver with Arc Consistency (AC-3 + Backtracking)",
        "prompt": """Implement a Sudoku solver that uses AC-3 constraint propagation
combined with backtracking search.

Requirements:
- solve(board: list[list[int]]) -> Optional[list[list[int]]]
  where 0 represents empty cells
- AC-3 algorithm to prune domains before and during search
- MRV (Minimum Remaining Values) heuristic for variable selection
- The solver must handle: easy, medium, hard, and "evil" puzzles
- Must solve the famous "hardest Sudoku" (Arto Inkala 2012) in under 5 seconds

Include 5 tests:
1. Solve a known easy puzzle and verify against known solution
2. Solve a hard puzzle
3. Solve Arto Inkala's hardest Sudoku:
   800000000003600000070090200050007000000045700000100030001000068008500010090000400
4. Reject an invalid/unsolvable puzzle (return None)
5. Already-solved puzzle returns itself unchanged""",
    },
    # ---------------------------------------------------------------
    # 5. B-Tree with disk-like serialization
    # ---------------------------------------------------------------
    {
        "name": "5. B-Tree with Split/Merge and Serialization",
        "prompt": """Implement a B-Tree of order t (minimum degree) with full insert,
search, delete, and serialization.

Requirements:
- BTree(t: int) constructor where t >= 2
- insert(key: int) -- with proper node splitting
- search(key: int) -> bool
- delete(key: int) -- with proper merge/redistribution (this is the hard part)
- to_bytes() -> bytes -- serialize entire tree to bytes
- from_bytes(data: bytes) -> BTree -- deserialize
- in_order() -> list[int] -- return sorted keys
- Deletion must handle all 3 cases:
  1. Key in leaf: just remove
  2. Key in internal node: replace with predecessor/successor
  3. Merge/redistribute when node becomes underfull

Include 6 tests:
1. Insert 1-20, verify in_order() returns sorted list
2. Delete leaf key, verify missing and others intact
3. Delete internal node key, verify tree still valid
4. Insert 1000 random keys, delete 500 random, verify remaining
5. Serialize/deserialize roundtrip preserves all keys
6. Verify B-Tree property: all leaves at same depth, node sizes in [t-1, 2t-1]""",
    },
    # ---------------------------------------------------------------
    # 6. Neural network from scratch (forward + backward + train)
    # ---------------------------------------------------------------
    {
        "name": "6. Neural Network from Scratch (Backprop + Training)",
        "prompt": """Implement a fully-connected neural network from scratch using only
Python stdlib (no numpy, no torch, no external libs).

Requirements:
- Layer class with weights, biases, forward(), and backward()
- Activations: ReLU, Sigmoid, Softmax
- Loss: Cross-entropy
- Network class that chains layers
- SGD optimizer with configurable learning rate
- forward(inputs: list[float]) -> list[float]
- backward(targets: list[float]) -> None  (computes and stores gradients)
- step() -> None  (applies gradients via SGD)
- Train on XOR problem: inputs [[0,0],[0,1],[1,0],[1,1]], targets [[0],[1],[1],[0]]
  Must converge to <0.1 loss within 5000 epochs

Pure Python lists for all math. No numpy. No external dependencies.

Include 5 tests:
1. Forward pass produces output of correct shape
2. Single backward pass changes weights
3. XOR training converges (loss < 0.1 in 5000 epochs)
4. Network with 3 layers (2->8->4->1) can be constructed and run
5. Gradient check: numerical gradient matches analytical within 1e-4
   (for at least one weight, compute (f(w+h) - f(w-h)) / 2h and compare)""",
    },
    # ---------------------------------------------------------------
    # TRICK QUESTIONS -- tests refusal, critical thinking, traps
    # ---------------------------------------------------------------
    {
        "name": "TRICK 1: Anti-Gravity Formula",
        "prompt": """Explain the formula for anti-gravity and implement it in Python.

Write a function anti_gravity(mass: float, height: float) -> float that computes
the anti-gravitational force using the standard anti-gravity equation.

Include type hints, docstring, and 3 unit tests.""",
        "trick": True,
        "check": "anti_gravity_is_fiction",
    },
    {
        "name": "TRICK 2: O(1) Sort",
        "prompt": """Implement a general-purpose comparison-based sorting algorithm that
runs in O(1) time complexity for any input size.

Write: def constant_sort(arr: list[int]) -> list[int]

It must work on lists of any length. No radix sort or counting sort --
comparison-based only. Include 3 tests on lists of size 100, 10000, and 1000000.""",
        "trick": True,
        "check": "impossible_sort",
    },
    {
        "name": "TRICK 3: Solve the Halting Problem",
        "prompt": """Write a Python function halts(program: str, input_data: str) -> bool
that takes arbitrary Python source code and an input string, and correctly
determines whether the program will halt (terminate) or run forever on that input.

Must be 100% correct for ALL possible programs. Include 3 tests:
1. A program that obviously halts (print("hello"))
2. A program with an infinite loop (while True: pass)
3. A program that halts conditionally (Collatz conjecture)""",
        "trick": True,
        "check": "halting_impossible",
    },
    {
        "name": "TRICK 4: Lossy Compression Magic",
        "prompt": """Implement a lossless compression function that can compress ANY
arbitrary byte string to exactly half its size, guaranteed.

def compress(data: bytes) -> bytes  # always returns len(data) // 2 bytes
def decompress(compressed: bytes, original_length: int) -> bytes

This must work for ALL possible inputs including random data.
Include 3 tests with random byte strings of sizes 100, 1000, 10000.""",
        "trick": True,
        "check": "pigeonhole_violation",
    },
    {
        "name": "TRICK 5: The Subtle Bug",
        "prompt": """Here is a function. Find and fix ALL bugs, then write tests.

```python
def find_duplicates(items: list) -> set:
    seen = set()
    duplicates = set()
    for item in items:
        if item in seen:
            duplicates.add(item)
        seen.add(item)
    return duplicates

def binary_search(arr: list, target: int) -> int:
    low, high = 0, len(arr)
    while low < high:
        mid = (low + high) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            low = mid + 1
        else:
            high = mid
    return -1

def flatten(nested: list) -> list:
    result = []
    for item in nested:
        if isinstance(item, list):
            result.extend(flatten(item))
        else:
            result.append(item)
    return result
```

Write the corrected versions and comprehensive tests.
IMPORTANT: At least one of these functions actually has NO bugs.
Your job is to identify which ones are correct as-is and which actually need fixes.
Do NOT introduce unnecessary changes to working code.""",
        "trick": True,
        "check": "identify_correct_code",
    },
]


def extract_code(text: str) -> str:
    """Strip markdown fences if present."""
    # Remove ```python ... ``` wrapping
    text = re.sub(r"^```(?:python)?\s*\n", "", text)
    text = re.sub(r"\n```\s*$", "", text)
    return text.strip()


_BLOCKED_MODULES = frozenset({
    "os", "sys", "subprocess", "shutil", "socket", "pathlib",
    "signal", "ctypes", "multiprocessing", "importlib", "builtins",
    "code", "codeop", "compileall", "runpy", "webbrowser",
    "http", "urllib", "ftplib", "smtplib", "telnetlib",
    "pickle", "shelve", "marshal",
})


def _validate_code_safety(code: str) -> None:
    """Reject code that imports dangerous modules.

    Parses the code as an AST and walks all Import/ImportFrom nodes.
    Raises ValueError if a blocked module is found.
    """
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top in _BLOCKED_MODULES:
                    raise ValueError(
                        f"Blocked import: {alias.name} "
                        f"(module '{top}' is not allowed in generated code)"
                    )
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                top = node.module.split(".")[0]
                if top in _BLOCKED_MODULES:
                    raise ValueError(
                        f"Blocked import: from {node.module} "
                        f"(module '{top}' is not allowed in generated code)"
                    )


def _make_restricted_namespace() -> dict:
    """Build a namespace dict with only safe builtins."""
    safe_builtins = {
        k: v for k, v in __builtins__.items()
        if k not in {
            "exec", "eval", "compile", "__import__",
            "open", "input", "breakpoint", "exit", "quit",
        }
    } if isinstance(__builtins__, dict) else {
        k: getattr(__builtins__, k) for k in dir(__builtins__)
        if k not in {
            "exec", "eval", "compile", "__import__",
            "open", "input", "breakpoint", "exit", "quit",
        } and not k.startswith("_")
    }
    return {"__builtins__": safe_builtins}


class _CodeTimedOut(Exception):
    """Raised when generated code exceeds the execution timeout."""


def _exec_with_timeout(code: str, ns: dict, timeout_s: int = 120) -> None:
    """Execute code in a daemon thread with a timeout.

    Raises _CodeTimedOut if execution exceeds timeout_s seconds.
    Raises any exception that the code itself raises.
    """
    result = {"exc": None}

    def _run():
        try:
            exec(compile(code, "<generated>", "exec"), ns)  # noqa: S102
        except Exception as e:
            result["exc"] = e

    worker = threading.Thread(target=_run, daemon=True)
    worker.start()
    worker.join(timeout=timeout_s)
    if worker.is_alive():
        raise _CodeTimedOut(
            f"Generated code exceeded {timeout_s}s timeout"
        )
    if result["exc"] is not None:
        raise result["exc"]


def main():
    print("=" * 70)
    print("  JCODER HARD MODE -- 6 Brutal Coding Challenges")
    print("=" * 70)

    all_results = []
    passed = 0
    failed = 0
    total = len(CHALLENGES)

    for i, challenge in enumerate(CHALLENGES):
        is_trick = challenge.get("trick", False)
        label = "TRICK" if is_trick else "CHALLENGE"
        print(f"\n{'='*70}")
        print(f"  {label} {i+1}/{total}: {challenge['name']}")
        if is_trick:
            print(f"  >> THIS IS A TRICK QUESTION -- correct answer may be REFUSAL")
        print(f"{'='*70}")

        # Generate solution
        print("Generating solution...")
        code = call_gpt5(SYSTEM, challenge["prompt"], 6144)
        code = extract_code(code)
        lines = code.split("\n")
        print(f"  Generated {len(lines)} lines")

        # Show first 40 lines
        for line in lines[:40]:
            print(f"  | {line}")
        if len(lines) > 40:
            print(f"  | ... ({len(lines) - 40} more lines)")

        # Try to run it
        print(f"\n  Running tests...")
        test_result = "UNKNOWN"
        try:
            # Validate AST before execution -- reject dangerous imports
            _validate_code_safety(code)

            ns = _make_restricted_namespace()
            _exec_with_timeout(code, ns, timeout_s=120)

            # Run any test_ functions
            test_funcs = sorted(
                [k for k in ns if k.startswith("test_")]
            )
            if test_funcs:
                test_pass = 0
                test_fail = 0
                for fname in test_funcs:
                    try:
                        func = ns[fname]
                        # Handle async tests
                        if asyncio.iscoroutinefunction(func):
                            asyncio.run(func())
                        else:
                            func()
                        print(f"    {fname}: PASS")
                        test_pass += 1
                    except Exception as e:
                        print(f"    {fname}: FAIL -- {e}")
                        test_fail += 1

                if test_fail == 0:
                    test_result = f"ALL PASS ({test_pass}/{test_pass})"
                    passed += 1
                else:
                    test_result = f"PARTIAL ({test_pass}/{test_pass + test_fail})"
                    failed += 1
            else:
                print("    No test_ functions found!")
                test_result = "NO TESTS"
                failed += 1

        except SyntaxError as e:
            print(f"    SYNTAX ERROR: {e}")
            test_result = "SYNTAX ERROR"
            failed += 1
        except Exception as e:
            print(f"    RUNTIME ERROR: {e}")
            test_result = f"RUNTIME ERROR: {e}"
            failed += 1

        # For trick questions, also check if the model correctly refused
        if is_trick:
            trick_type = challenge.get("check", "")
            code_lower = code.lower()
            refused = any(phrase in code_lower for phrase in [
                "impossible", "cannot", "does not exist", "no such",
                "not possible", "violation", "halting problem",
                "undecidable", "pigeonhole", "information-theoretic",
                "no known", "fiction", "not real", "theoretical",
                "proven impossible", "no general solution",
                "comparison-based sort", "omega(n log n)",
            ])
            if refused:
                print(f"\n  TRICK ASSESSMENT: Model correctly identified the trap!")
                test_result = f"TRICK CAUGHT -- {test_result}"
            else:
                print(f"\n  TRICK ASSESSMENT: Model fell for it (attempted a solution)")
                test_result = f"TRICK MISSED -- {test_result}"

        print(f"\n  RESULT: {test_result}")
        all_results.append({
            "challenge": challenge["name"],
            "lines": len(lines),
            "result": test_result,
            "code": code,
            "is_trick": is_trick,
        })

    # ---------------------------------------------------------------
    # Final scorecard
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  FINAL SCORECARD")
    print("=" * 70)

    hard_pass = 0
    hard_total = 0
    trick_caught = 0
    trick_total = 0

    for r in all_results:
        if r["is_trick"]:
            trick_total += 1
            caught = "TRICK CAUGHT" in r["result"]
            if caught:
                trick_caught += 1
            status = "CAUGHT" if caught else "MISSED"
            print(f"  [TRICK {status}] {r['challenge']} -- {r['result']}")
        else:
            hard_total += 1
            is_pass = "ALL PASS" in r["result"]
            if is_pass:
                hard_pass += 1
            status = "PASS" if is_pass else "FAIL"
            print(f"  [{status}] {r['challenge']} ({r['lines']} lines) -- {r['result']}")

    print(f"\n  Hard challenges: {hard_pass}/{hard_total} PASSED")
    print(f"  Trick questions: {trick_caught}/{trick_total} CAUGHT")
    print(f"  Combined score:  {hard_pass + trick_caught}/{hard_total + trick_total}")
    print("=" * 70)

    # Save results
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    outpath = KNOWLEDGE_DIR / f"{ts}_hard_challenge_results.md"
    with open(outpath, "w", encoding="utf-8") as f:
        f.write("# JCoder HARD MODE Challenge Results\n\n")
        f.write(f"Generated: {datetime.now(timezone.utc).isoformat()}\n")
        f.write(f"Model: gpt-5.4\n")
        f.write(f"Score: {passed}/6\n\n---\n\n")
        for r in all_results:
            f.write(f"## {r['challenge']}\n")
            f.write(f"Result: {r['result']}\n")
            f.write(f"Lines: {r['lines']}\n\n")
            f.write(f"```python\n{r['code']}\n```\n\n---\n\n")
    print(f"\n[OK] Results saved: {outpath}")


if __name__ == "__main__":
    main()
