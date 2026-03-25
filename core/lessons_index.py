"""
Lessons Learned Index — RAG-Powered Self-Learning
===================================================
Stores every coding challenge attempt (success AND failure) in a
searchable FTS5 index. Before phi4 attempts a new challenge, it
retrieves similar past attempts — including what went wrong and
how it was fixed.

This is how a 14B model competes with 100B+ models:
    Large model: relies on parametric knowledge (training data)
    Small model + RAG: retrieves EXACTLY the right pattern at query time

The index stores:
    - Challenge description
    - Generated code (what phi4 wrote)
    - Test result (PASS/FAIL + error message)
    - Lesson learned (what the mistake was, how to fix it)
    - Category tags for retrieval matching

When phi4 faces a new challenge, we retrieve:
    1. Similar SUCCESSFUL past solutions (few-shot examples)
    2. Similar FAILED attempts with error patterns (what to avoid)
    3. Expert corrections for common mistakes

This gives phi4 "experience" it never had in training.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS lessons
USING fts5(
    challenge_description,
    generated_code,
    test_result,
    error_message,
    lesson_learned,
    category,
    difficulty,
    timestamp_str
);
"""


class LessonsIndex:
    """Searchable index of coding challenge attempts and lessons learned.

    Every attempt — success or failure — is stored. Before the next
    challenge, retrieve() finds relevant past experience to inject
    as context for the LLM.
    """

    def __init__(self, db_path: str = "data/indexes/lessons_learned.fts5.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path))

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def store_attempt(
        self,
        challenge_description: str,
        generated_code: str,
        passed: bool,
        error_message: str = "",
        lesson_learned: str = "",
        category: str = "",
        difficulty: int = 5,
    ) -> None:
        """Store a challenge attempt (success or failure) for future retrieval."""
        test_result = "PASS" if passed else "FAIL"

        # Auto-generate lesson from error if not provided
        if not lesson_learned and not passed and error_message:
            lesson_learned = self._extract_lesson(error_message, generated_code)

        with self._connect() as conn:
            conn.execute(
                "INSERT INTO lessons "
                "(challenge_description, generated_code, test_result, "
                "error_message, lesson_learned, category, difficulty, timestamp_str) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    challenge_description[:2000],
                    generated_code[:5000],
                    test_result,
                    error_message[:1000],
                    lesson_learned[:1000],
                    category,
                    str(difficulty),
                    str(int(time.time())),
                ),
            )

    def retrieve(
        self,
        query: str,
        category: str = "",
        top_k: int = 3,
        include_failures: bool = True,
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant past attempts for a new challenge.

        Returns both successes (few-shot examples) and failures
        (error patterns to avoid).
        """
        # Build FTS5 query
        words = [w for w in query.split() if len(w) > 2 and w.isalnum()]
        if category:
            words.append(category)
        if not words:
            return []

        fts_query = " OR ".join(words[:15])

        with self._connect() as conn:
            try:
                if include_failures:
                    rows = conn.execute(
                        "SELECT challenge_description, generated_code, "
                        "test_result, error_message, lesson_learned, "
                        "category, rank "
                        "FROM lessons WHERE lessons MATCH ? "
                        "ORDER BY rank LIMIT ?",
                        (fts_query, top_k * 2),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        "SELECT challenge_description, generated_code, "
                        "test_result, error_message, lesson_learned, "
                        "category, rank "
                        "FROM lessons WHERE lessons MATCH ? "
                        "AND test_result = 'PASS' "
                        "ORDER BY rank LIMIT ?",
                        (fts_query, top_k),
                    ).fetchall()
            except Exception:
                return []

        results = []
        for row in rows:
            results.append({
                "description": row[0][:500],
                "code": row[1][:2000],
                "result": row[2],
                "error": row[3][:500],
                "lesson": row[4][:500],
                "category": row[5],
            })

        # Prioritize: successes first, then failures with lessons
        successes = [r for r in results if r["result"] == "PASS"]
        failures = [r for r in results if r["result"] == "FAIL" and r["lesson"]]

        # Return mix: up to top_k, preferring successes
        combined = successes[:top_k] + failures[:max(0, top_k - len(successes))]
        return combined[:top_k]

    def build_context_prompt(
        self,
        challenge_description: str,
        category: str = "",
    ) -> str:
        """Build a context prompt from past lessons for the LLM.

        This is the key function: it turns past experience into
        few-shot examples and warnings that help phi4 avoid
        repeating mistakes.
        """
        # RELEVANCE GATE: only inject top-1 most relevant lesson.
        # Context pollution finding (2026-03-25): too much mixed context
        # CONFUSES phi4. Quality > quantity. If no highly relevant
        # result exists, return empty (no context > bad context).
        lessons = self.retrieve(
            challenge_description, category=category, top_k=1,
        )

        if not lessons:
            return ""

        # Additional relevance check: does the lesson description
        # share significant keyword overlap with the challenge?
        challenge_words = set(
            w.lower() for w in challenge_description.split()
            if len(w) > 3 and w.isalnum()
        )
        lesson_words = set(
            w.lower() for w in lessons[0]["description"].split()
            if len(w) > 3 and w.isalnum()
        )
        overlap = len(challenge_words & lesson_words)
        # Require at least 1 keyword overlap (was 2 — too strict,
        # gated out valid lessons for topo sort and LRU cache)
        if overlap < 1:
            return ""  # Not relevant enough — no context > bad context

        parts = ["## Relevant Past Experience\n"]

        for i, lesson in enumerate(lessons, 1):
            if lesson["result"] == "PASS":
                parts.append(
                    f"### Example {i} (SUCCESSFUL)\n"
                    f"Similar problem: {lesson['description'][:200]}\n"
                    f"Working solution:\n```python\n{lesson['code'][:1000]}\n```\n"
                )
            else:
                parts.append(
                    f"### Warning {i} (PAST FAILURE)\n"
                    f"Similar problem: {lesson['description'][:200]}\n"
                    f"What went wrong: {lesson['error'][:200]}\n"
                    f"Lesson: {lesson['lesson'][:200]}\n"
                )

        parts.append(
            "\nUse these examples to guide your solution. "
            "Avoid the mistakes shown in past failures.\n"
        )

        return "\n".join(parts)

    @staticmethod
    def _extract_lesson(error_message: str, code: str) -> str:
        """Auto-extract a lesson from an error message."""
        error_lower = error_message.lower()

        if "assertionerror" in error_lower or "assert" in error_lower:
            return "Output did not match expected result. Check algorithm logic and edge cases."
        elif "typeerror" in error_lower:
            return "Type mismatch. Check function signatures, return types, and argument types."
        elif "indexerror" in error_lower:
            return "Array/list index out of bounds. Check loop boundaries and empty input handling."
        elif "keyerror" in error_lower:
            return "Dictionary key not found. Initialize default values or check key existence."
        elif "timeout" in error_lower:
            return "Solution too slow. Optimize algorithm complexity (avoid O(n^2) or worse)."
        elif "nameerror" in error_lower:
            return "Undefined variable or function. Check spelling and scope."
        elif "recursionerror" in error_lower:
            return "Infinite recursion. Add base case or convert to iterative approach."
        else:
            return f"Runtime error: {error_message[:200]}"

    def stats(self) -> Dict[str, Any]:
        """Return index statistics."""
        with self._connect() as conn:
            total = conn.execute("SELECT COUNT(*) FROM lessons").fetchone()[0]
            passes = conn.execute(
                "SELECT COUNT(*) FROM lessons WHERE test_result = 'PASS'"
            ).fetchone()[0]
            fails = total - passes
        return {"total": total, "passes": passes, "fails": fails}
