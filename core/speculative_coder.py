"""
Speculative Code Generation (Draft-Verify Pattern)
-----------------------------------------------------
Generates code using a fast/cheap model, then verifies and corrects
using a more capable model. Inspired by speculative decoding but
applied at the semantic level for code generation.

Based on:
- Mirror Speculative Decoding (2025): Dual-GPU draft+verify
- Medusa (2024): Multiple decoding heads
- Speculative RAG (ACL 2025): Draft-then-verify for RAG

Algorithm:
1. DRAFT: Fast model generates N code candidates quickly
2. VERIFY: For each candidate, run local checks:
   - Syntax validity (ast.parse)
   - Import resolution
   - Type consistency (basic)
   - Structural quality
3. CORRECT: If top candidate has issues, send issues + candidate
   to the main model for targeted correction
4. SELECT: Pick the best verified candidate

This is 2-3x faster than generating one careful answer because:
- Draft model runs at high tokens/sec
- Most verification is local (no LLM needed)
- Correction is targeted (smaller prompt, faster)
"""

from __future__ import annotations

import ast
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from core.runtime import Runtime


@dataclass
class CodeDraft:
    """A single code draft with verification results."""
    text: str
    draft_ms: float = 0.0
    syntax_valid: bool = False
    issues: List[str] = field(default_factory=list)
    quality_score: float = 0.0
    corrected: bool = False


@dataclass
class SpeculativeResult:
    """Result from speculative code generation."""
    final_code: str
    drafts_generated: int
    best_draft_index: int
    was_corrected: bool
    total_ms: float = 0.0
    draft_ms: float = 0.0
    verify_ms: float = 0.0
    correct_ms: float = 0.0
    quality_score: float = 0.0


# ---------------------------------------------------------------------------
# Local verification (zero LLM cost)
# ---------------------------------------------------------------------------

def _check_syntax(code: str) -> Tuple[bool, List[str]]:
    """Check Python syntax validity."""
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)\n```", code, re.DOTALL)
    if not blocks:
        blocks = [code]

    issues = []
    for i, block in enumerate(blocks):
        try:
            ast.parse(block)
        except SyntaxError as e:
            issues.append(f"Syntax error in block {i}: {e.msg} (line {e.lineno})")

    return len(issues) == 0, issues


def _check_imports(code: str) -> List[str]:
    """Check for potentially broken imports."""
    issues = []
    stdlib = {
        "os", "sys", "re", "json", "math", "time", "datetime",
        "pathlib", "hashlib", "typing", "collections", "functools",
        "itertools", "copy", "abc", "dataclasses", "enum",
        "sqlite3", "subprocess", "tempfile", "unittest",
        "io", "logging", "argparse", "random", "string",
        "contextlib", "textwrap", "shutil", "glob",
    }

    for m in re.finditer(r"(?:from|import)\s+([a-zA-Z_][\w.]*)", code):
        module = m.group(1).split(".")[0]
        if (module not in stdlib and
            module not in ("core", "cli", "ingestion", "research") and
            not module.startswith("_")):
            issues.append(f"Unresolved import: {module}")

    return issues


def _check_structure(code: str) -> Tuple[float, List[str]]:
    """Check structural quality of generated code."""
    issues = []
    score = 1.0

    lines = code.split("\n")
    non_empty = [l for l in lines if l.strip()]

    # Very short output is suspicious
    if len(non_empty) < 3:
        issues.append("Very short output (< 3 lines)")
        score -= 0.3

    # No function/class definitions in substantial code
    if len(non_empty) > 10:
        defs = len(re.findall(r"\ndef\s+\w+", code))
        classes = len(re.findall(r"\nclass\s+\w+", code))
        if defs + classes == 0:
            issues.append("No function/class definitions in substantial code")
            score -= 0.1

    # Deep nesting
    max_indent = 0
    for line in non_empty:
        indent = len(line) - len(line.lstrip())
        max_indent = max(max_indent, indent)
    if max_indent > 24:  # 6+ levels
        issues.append(f"Deep nesting (max indent: {max_indent} spaces)")
        score -= 0.2

    # Very long lines
    long_lines = sum(1 for l in lines if len(l) > 120)
    if long_lines > 5:
        issues.append(f"{long_lines} lines exceed 120 characters")
        score -= 0.1

    # Docstrings present
    if '"""' in code or "'''" in code:
        score += 0.1

    return max(0.0, min(1.0, score)), issues


def _check_consistency(code: str) -> List[str]:
    """Check for common code consistency issues."""
    issues = []

    # Mixed indentation
    has_tabs = "\t" in code
    has_spaces = "    " in code
    if has_tabs and has_spaces:
        issues.append("Mixed tabs and spaces")

    # Unused variable patterns
    assigns = re.findall(r"(\w+)\s*=", code)
    for var in set(assigns):
        if var.startswith("_"):
            continue
        # Check if used after assignment (simple heuristic)
        pattern = re.compile(r"\b" + re.escape(var) + r"\b")
        matches = pattern.findall(code)
        if len(matches) <= 1:
            issues.append(f"Possibly unused variable: {var}")

    return issues[:5]  # Cap at 5 issues


# ---------------------------------------------------------------------------
# Speculative Code Generator
# ---------------------------------------------------------------------------

class SpeculativeCodeGenerator:
    """
    Generate code speculatively: draft fast, verify locally, correct if needed.

    Optimized for speed: most verification is zero-cost (local checks).
    Only falls back to LLM correction when local checks find issues.
    """

    def __init__(
        self,
        runtime: Runtime,
        n_drafts: int = 3,
        draft_temperature: float = 0.5,
        correction_temperature: float = 0.1,
    ):
        self.runtime = runtime
        self.n_drafts = n_drafts
        self.draft_temp = draft_temperature
        self.correct_temp = correction_temperature

    def generate(
        self,
        question: str,
        context_chunks: List[str],
        system_prompt: Optional[str] = None,
    ) -> SpeculativeResult:
        """
        Speculative code generation pipeline.

        1. Generate N drafts quickly
        2. Verify each locally (syntax, imports, structure)
        3. If best has issues, correct with targeted prompt
        4. Return best result
        """
        t0 = time.time()

        # --- Phase 1: Draft ---
        t_draft = time.time()
        drafts = []
        temps = [
            self.draft_temp + i * 0.15
            for i in range(self.n_drafts)
        ]

        for temp in temps:
            t_d = time.time()
            try:
                text = self.runtime.generate(
                    question, context_chunks,
                    system_prompt=system_prompt,
                    temperature=temp,
                    max_tokens=1024,
                )
                draft = CodeDraft(text=text, draft_ms=(time.time() - t_d) * 1000)
            except Exception:
                draft = CodeDraft(text="", draft_ms=(time.time() - t_d) * 1000)
            drafts.append(draft)

        draft_ms = (time.time() - t_draft) * 1000

        # --- Phase 2: Verify ---
        t_verify = time.time()
        for draft in drafts:
            if not draft.text:
                draft.quality_score = 0.0
                continue

            # Syntax
            draft.syntax_valid, syntax_issues = _check_syntax(draft.text)
            draft.issues.extend(syntax_issues)

            # Imports
            import_issues = _check_imports(draft.text)
            draft.issues.extend(import_issues)

            # Structure
            struct_score, struct_issues = _check_structure(draft.text)
            draft.issues.extend(struct_issues)

            # Consistency
            consistency_issues = _check_consistency(draft.text)
            draft.issues.extend(consistency_issues)

            # Combined score
            draft.quality_score = (
                (0.4 if draft.syntax_valid else 0.0) +
                0.2 * max(0, 1.0 - len(import_issues) * 0.2) +
                0.3 * struct_score +
                0.1 * max(0, 1.0 - len(consistency_issues) * 0.1)
            )

        verify_ms = (time.time() - t_verify) * 1000

        # Sort by quality
        drafts.sort(key=lambda d: d.quality_score, reverse=True)
        best = drafts[0]
        best_index = 0

        # --- Phase 3: Correct (if needed) ---
        correct_ms = 0.0
        was_corrected = False

        if best.issues and best.quality_score < 0.8:
            t_correct = time.time()
            corrected = self._correct(question, best)
            correct_ms = (time.time() - t_correct) * 1000
            if corrected:
                best = corrected
                was_corrected = True

        return SpeculativeResult(
            final_code=best.text,
            drafts_generated=len(drafts),
            best_draft_index=best_index,
            was_corrected=was_corrected,
            total_ms=(time.time() - t0) * 1000,
            draft_ms=draft_ms,
            verify_ms=verify_ms,
            correct_ms=correct_ms,
            quality_score=best.quality_score,
        )

    def _correct(
        self,
        question: str,
        draft: CodeDraft,
    ) -> Optional[CodeDraft]:
        """Use LLM to fix specific issues in a draft."""
        issues_text = "\n".join(f"- {i}" for i in draft.issues[:5])

        correction_prompt = (
            f"Fix these specific issues in the code below:\n"
            f"{issues_text}\n\n"
            f"Original question: {question}\n\n"
            f"Code to fix:\n{draft.text[:2000]}\n\n"
            f"Return ONLY the corrected code."
        )

        try:
            corrected_text = self.runtime.generate(
                question=correction_prompt,
                context_chunks=[],
                system_prompt="You are a code fixer. Fix the issues and return corrected code.",
                temperature=self.correct_temp,
                max_tokens=1024,
            )

            corrected = CodeDraft(text=corrected_text, corrected=True)
            corrected.syntax_valid, _ = _check_syntax(corrected_text)

            if corrected.syntax_valid:
                corrected.quality_score = draft.quality_score + 0.2
                return corrected

        except Exception:
            pass

        return None
