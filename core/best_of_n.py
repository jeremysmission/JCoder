"""
Best-of-N Verified Generation (PSV Pattern)
---------------------------------------------
Generates N candidate code responses, runs local verification
on each, and selects the best one. Zero external dependencies.

Based on: PSV Framework (arxiv 2512.18160), ELHSR (2025),
Verifier-Guided Generation (ICLR 2025)

Verification signals (all local, no cloud):
1. Syntax validity (ast.parse)
2. Import resolution (can we resolve all imports?)
3. Structural quality (function count, avg length, nesting depth)
4. Pattern consistency (does it match existing codebase style?)
5. Reflection score (if reflection engine available)

The key insight from the research: generating 3-5 candidates and
picking the best with cheap verification outperforms generating
1 candidate with an expensive model.
"""

from __future__ import annotations

import ast
import re
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from core.runtime import Runtime


@dataclass
class Candidate:
    """A single generated candidate with verification scores."""
    text: str
    index: int
    syntax_valid: bool = False
    import_score: float = 0.0
    structure_score: float = 0.0
    pattern_score: float = 0.0
    reflection_score: float = 0.0
    combined_score: float = 0.0
    generation_ms: float = 0.0


@dataclass
class BestOfNResult:
    """Result from best-of-N selection."""
    answer: str
    candidates_generated: int
    selected_index: int
    selected_score: float
    all_scores: List[float] = field(default_factory=list)
    total_ms: float = 0.0


# ---------------------------------------------------------------------------
# Local verification functions (zero external dependencies)
# ---------------------------------------------------------------------------

def _check_syntax(code: str) -> bool:
    """Check if Python code has valid syntax."""
    # Extract code blocks from markdown-style output
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)\n```", code, re.DOTALL)
    if not blocks:
        # Try the whole text as code
        blocks = [code]

    for block in blocks:
        try:
            ast.parse(block)
        except SyntaxError:
            return False
    return True


def _score_imports(code: str) -> float:
    """Score based on import cleanliness (no obviously broken imports)."""
    imports = re.findall(r"(?:from|import)\s+([a-zA-Z_][\w.]*)", code)
    if not imports:
        return 1.0  # no imports = no import problems

    # Known stdlib modules (subset)
    stdlib = {
        "os", "sys", "re", "json", "math", "time", "datetime",
        "pathlib", "hashlib", "typing", "collections", "functools",
        "itertools", "copy", "abc", "dataclasses", "enum",
        "sqlite3", "subprocess", "tempfile", "unittest",
        "concurrent", "threading", "multiprocessing",
        "io", "logging", "argparse", "random", "string",
    }

    resolved = 0
    for imp in imports:
        root = imp.split(".")[0]
        if root in stdlib or root in ("core", "cli", "ingestion", "research"):
            resolved += 1
    return resolved / len(imports) if imports else 1.0


def _score_structure(code: str) -> float:
    """Score structural quality of generated code."""
    lines = code.split("\n")
    non_empty = [l for l in lines if l.strip()]
    if not non_empty:
        return 0.0

    score = 1.0

    # Penalize very long functions (>50 lines without a break)
    consecutive = 0
    max_consecutive = 0
    for line in lines:
        if line.strip():
            consecutive += 1
            max_consecutive = max(max_consecutive, consecutive)
        else:
            consecutive = 0
    if max_consecutive > 80:
        score -= 0.3

    # Penalize deep nesting (>5 levels)
    max_indent = 0
    for line in non_empty:
        indent = len(line) - len(line.lstrip())
        max_indent = max(max_indent, indent)
    if max_indent > 20:  # 5 levels * 4 spaces
        score -= 0.2

    # Reward presence of docstrings
    if '"""' in code or "'''" in code:
        score += 0.1

    # Reward function/class definitions
    defs = len(re.findall(r"\ndef\s+\w+", code))
    classes = len(re.findall(r"\nclass\s+\w+", code))
    if defs + classes > 0:
        score += 0.1

    return max(0.0, min(1.0, score))


def _score_pattern_match(code: str, context_chunks: List[str]) -> float:
    """Score how well the code matches the style of context chunks."""
    if not context_chunks:
        return 0.5

    # Check naming convention consistency
    context_text = " ".join(context_chunks)

    # Snake case vs camel case in context
    snake_count = len(re.findall(r"[a-z]+_[a-z]+", context_text))
    camel_count = len(re.findall(r"[a-z]+[A-Z][a-z]+", context_text))

    code_snake = len(re.findall(r"[a-z]+_[a-z]+", code))
    code_camel = len(re.findall(r"[a-z]+[A-Z][a-z]+", code))

    # Reward matching the dominant style
    context_style = "snake" if snake_count > camel_count else "camel"
    code_style = "snake" if code_snake > code_camel else "camel"

    style_match = 1.0 if context_style == code_style else 0.5

    return style_match


# ---------------------------------------------------------------------------
# Best-of-N generator
# ---------------------------------------------------------------------------

class BestOfNGenerator:
    """
    Generate N candidates, verify each, select the best.

    This consistently outperforms single-shot generation for code
    tasks where verification is cheap but generation is noisy.
    """

    def __init__(
        self,
        runtime: Runtime,
        n: int = 3,
        temperature_spread: Tuple[float, ...] = (0.1, 0.3, 0.5),
        reflection_fn: Optional[Callable[[str, List[Dict], str], float]] = None,
    ):
        self.runtime = runtime
        self.n = n
        self.temperatures = temperature_spread[:n]
        # Pad with last temperature if needed
        while len(self.temperatures) < n:
            self.temperatures = (*self.temperatures, self.temperatures[-1])
        self.reflection_fn = reflection_fn

    def generate(
        self,
        question: str,
        context_chunks: List[str],
        system_prompt: Optional[str] = None,
    ) -> BestOfNResult:
        """
        Generate N candidates at different temperatures, verify, select best.
        """
        t0 = time.time()
        candidates: List[Candidate] = []

        for i in range(self.n):
            t_gen = time.time()
            text = self.runtime.generate(
                question, context_chunks,
                system_prompt=system_prompt,
                temperature=self.temperatures[i],
            )
            gen_ms = (time.time() - t_gen) * 1000

            cand = Candidate(text=text, index=i, generation_ms=gen_ms)

            # Run verifications
            cand.syntax_valid = _check_syntax(text)
            cand.import_score = _score_imports(text)
            cand.structure_score = _score_structure(text)
            cand.pattern_score = _score_pattern_match(text, context_chunks)

            # Reflection if available
            if self.reflection_fn:
                try:
                    cand.reflection_score = self.reflection_fn(
                        question, context_chunks, text)
                except Exception:
                    cand.reflection_score = 0.5

            # Combined score (weighted)
            cand.combined_score = (
                (0.3 if cand.syntax_valid else 0.0) +
                0.15 * cand.import_score +
                0.15 * cand.structure_score +
                0.1 * cand.pattern_score +
                0.3 * cand.reflection_score
            )

            candidates.append(cand)

        # Select best
        candidates.sort(key=lambda c: c.combined_score, reverse=True)
        best = candidates[0]

        return BestOfNResult(
            answer=best.text,
            candidates_generated=self.n,
            selected_index=best.index,
            selected_score=best.combined_score,
            all_scores=[c.combined_score for c in candidates],
            total_ms=(time.time() - t0) * 1000,
        )
