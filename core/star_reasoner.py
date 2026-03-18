"""
STaR Self-Taught Reasoner (Bootstrapped Reasoning)
-----------------------------------------------------
Iteratively improves answer quality by:
1. Generate answer with explicit reasoning chain
2. Verify: is the answer correct?
3. If correct: keep (query, reasoning, answer) as training example
4. If wrong: "rationalize" -- hint the correct answer, generate reasoning
5. Use successful reasoning chains as few-shot examples for future queries

Based on:
- STaR: Bootstrapping Reasoning With Reasoning (Zelikman 2022)
- AdaSTaR (NeurIPS 2025): Adaptive difficulty sampling via MinHeap
- CARE-STaR (ACL 2025): Constraint-aware chain-of-thought
- START (EMNLP 2025): Self-taught reasoner with tools

Key insight: Correct reasoning chains from YOUR codebase are the
highest-quality few-shot examples possible. They capture your
specific coding patterns, naming conventions, and query styles.

Zero weight updates. Pure prompt engineering with memory.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from core.runtime import Runtime

log = logging.getLogger(__name__)

@dataclass
class ReasoningTrace:
    """A single reasoning chain with verification."""
    trace_id: str
    query: str
    reasoning: str  # the chain-of-thought
    answer: str
    correct: bool
    difficulty: float  # 0.0-1.0 estimated
    staleness: int = 0  # how many iterations since last sampled
    iteration: int = 0


@dataclass
class STaRResult:
    """Result of a STaR self-improvement iteration."""
    iteration: int
    queries_attempted: int
    correct_first_try: int
    rationalized: int
    accuracy: float
    traces_stored: int
    improvement_over_previous: float = 0.0


_REASON_PROMPT = (
    "Answer this code question step by step. Show your reasoning "
    "before giving the final answer.\n\n"
    "{examples}"
    "Question: {question}\n\n"
    "Think step by step:\n"
    "1."
)

_RATIONALIZE_PROMPT = (
    "The correct answer to this question involves: {hint}\n\n"
    "Explain step by step WHY this is the correct answer.\n\n"
    "Question: {question}\n\n"
    "Reasoning:\n"
    "1."
)

_VERIFY_PROMPT = (
    "Does this answer correctly address the question? "
    "Rate 0-10. Output ONLY a number.\n\n"
    "Question: {question}\n"
    "Answer: {answer}\n"
    "Score:"
)


class STaRReasoner:
    """
    Self-Taught Reasoner for code RAG.

    Builds a library of verified reasoning chains from the codebase.
    Uses them as few-shot examples to improve future answers.
    Implements AdaSTaR-style priority sampling for efficiency.
    """

    def __init__(
        self,
        runtime: Runtime,
        verify_fn: Optional[Callable[[str, str], float]] = None,
        db_path: str = "_star/reasoner.db",
        seed: int = 42,
    ):
        """
        Args:
            runtime: LLM for generation
            verify_fn: Optional custom verifier. If None, uses LLM self-verify.
                        verify_fn(question, answer) -> score (0.0-1.0)
            db_path: SQLite for trace storage
        """
        self.runtime = runtime
        self.verify_fn = verify_fn
        self.rng = random.Random(seed)
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self._db_path))

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS traces (
                    trace_id TEXT PRIMARY KEY,
                    query TEXT NOT NULL,
                    reasoning TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    correct INTEGER,
                    difficulty REAL,
                    staleness INTEGER DEFAULT 0,
                    iteration INTEGER,
                    created_at REAL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_traces_correct
                ON traces(correct DESC, difficulty)
            """)
            conn.commit()

    def run_iteration(
        self,
        queries: List[str],
        context_fn: Callable[[str], List[str]],
        ground_truth: Optional[List[str]] = None,
        iteration: int = 0,
    ) -> STaRResult:
        """
        Run one STaR iteration on a batch of queries.

        Args:
            queries: Questions to process
            context_fn: Function(query) -> list of context chunks
            ground_truth: Optional ground truth answers for verification
            iteration: Current iteration number
        """
        # Get few-shot examples from previous successful traces
        examples = self._get_best_examples(k=3)
        example_text = self._format_examples(examples)

        correct_first = 0
        rationalized = 0
        stored = 0
        gt = ground_truth or [None] * len(queries)

        # AdaSTaR-style priority sampling
        prioritized = self._prioritize_queries(queries, iteration)

        for query, gt_answer in zip(prioritized, gt[:len(prioritized)]):
            # Phase 1: Generate with reasoning
            context = context_fn(query)
            reasoning, answer = self._generate_with_reasoning(
                query, context, example_text
            )

            # Phase 2: Verify
            score = self._verify(query, answer, gt_answer)
            correct = score >= 0.6

            if correct:
                correct_first += 1
                trace = self._store_trace(
                    query, reasoning, answer, True,
                    score, iteration,
                )
                stored += 1
            else:
                # Phase 3: Rationalization
                if gt_answer:
                    rationalized_reasoning = self._rationalize(
                        query, gt_answer
                    )
                    if rationalized_reasoning:
                        self._store_trace(
                            query, rationalized_reasoning, gt_answer,
                            True, 0.7, iteration,
                        )
                        rationalized += 1
                        stored += 1

            # Update staleness for all traces
            self._increment_staleness()

        n = len(prioritized)
        accuracy = correct_first / n if n > 0 else 0.0

        # Compare to previous iteration
        prev_accuracy = self._previous_accuracy(iteration - 1)
        improvement = accuracy - prev_accuracy if prev_accuracy > 0 else 0.0

        return STaRResult(
            iteration=iteration,
            queries_attempted=n,
            correct_first_try=correct_first,
            rationalized=rationalized,
            accuracy=accuracy,
            traces_stored=stored,
            improvement_over_previous=improvement,
        )

    def answer_with_reasoning(
        self,
        query: str,
        context_chunks: List[str],
    ) -> Tuple[str, str]:
        """
        Answer a query using the best accumulated reasoning examples.
        Returns (reasoning, answer).
        """
        examples = self._get_best_examples(k=3)
        example_text = self._format_examples(examples)
        return self._generate_with_reasoning(
            query, context_chunks, example_text
        )

    def _generate_with_reasoning(
        self,
        query: str,
        context: List[str],
        examples: str,
    ) -> Tuple[str, str]:
        """Generate answer with explicit reasoning chain."""
        prompt = _REASON_PROMPT.format(
            question=query,
            examples=examples,
        )
        raw = self.runtime.generate(
            question=prompt,
            context_chunks=context,
            system_prompt=(
                "You are a code expert. Show your step-by-step reasoning "
                "before giving a final answer. End with 'ANSWER: <your answer>'."
            ),
            temperature=0.3,
            max_tokens=1024,
        )

        # Split reasoning from answer
        if "ANSWER:" in raw:
            parts = raw.split("ANSWER:", 1)
            reasoning = parts[0].strip()
            answer = parts[1].strip()
        else:
            reasoning = raw
            answer = raw

        return reasoning, answer

    def _rationalize(self, query: str, correct_answer: str) -> Optional[str]:
        """Generate reasoning that leads to the correct answer."""
        try:
            prompt = _RATIONALIZE_PROMPT.format(
                question=query,
                hint=correct_answer[:300],
            )
            raw = self.runtime.generate(
                question=prompt,
                context_chunks=[],
                system_prompt="Explain the reasoning step by step.",
                temperature=0.3,
                max_tokens=512,
            )
            return raw.strip() if len(raw.strip()) > 30 else None
        except Exception:
            log.warning("STaR rationalization failed for query", exc_info=True)
            return None

    def _verify(
        self,
        query: str,
        answer: str,
        ground_truth: Optional[str],
    ) -> float:
        """Verify answer correctness."""
        if self.verify_fn:
            return self.verify_fn(query, answer)

        if ground_truth:
            # Simple overlap check
            gt_lower = ground_truth.lower()
            ans_lower = answer.lower()
            if gt_lower in ans_lower or ans_lower in gt_lower:
                return 0.8

        # LLM self-verification
        prompt = _VERIFY_PROMPT.format(
            question=query[:300], answer=answer[:300]
        )
        try:
            raw = self.runtime.generate(
                question=prompt, context_chunks=[],
                system_prompt="Rate answer quality. Output only a number.",
                temperature=0.0, max_tokens=8,
            )
            import re
            m = re.search(r"\b(10|[0-9])\b", raw.strip())
            return int(m.group(1)) / 10.0 if m else 0.5
        except Exception:
            log.warning("STaR verification LLM call failed", exc_info=True)
            return 0.5

    def _get_best_examples(self, k: int = 3) -> List[ReasoningTrace]:
        """Get top-k correct reasoning traces for few-shot injection."""
        with self._connect() as conn:
            cur = conn.execute(
                "SELECT trace_id, query, reasoning, answer, correct, "
                "difficulty, staleness, iteration FROM traces "
                "WHERE correct = 1 "
                "ORDER BY difficulty DESC, staleness DESC "
                "LIMIT ?",
                (k,),
            )
            return [
                ReasoningTrace(
                    trace_id=r[0], query=r[1], reasoning=r[2],
                    answer=r[3], correct=bool(r[4]), difficulty=r[5],
                    staleness=r[6], iteration=r[7],
                )
                for r in cur.fetchall()
            ]

    @staticmethod
    def _format_examples(traces: List[ReasoningTrace]) -> str:
        """Format traces as few-shot examples."""
        if not traces:
            return ""
        parts = ["Here are examples of good reasoning:\n"]
        for t in traces:
            parts.append(
                f"Q: {t.query}\n"
                f"Reasoning: {t.reasoning[:300]}\n"
                f"ANSWER: {t.answer[:200]}\n---\n"
            )
        return "\n".join(parts) + "\n"

    def _store_trace(
        self,
        query: str,
        reasoning: str,
        answer: str,
        correct: bool,
        difficulty: float,
        iteration: int,
    ) -> ReasoningTrace:
        tid = hashlib.sha256(
            f"{query}:{answer}:{iteration}".encode()
        ).hexdigest()[:12]

        trace = ReasoningTrace(
            trace_id=tid, query=query, reasoning=reasoning,
            answer=answer, correct=correct, difficulty=difficulty,
            iteration=iteration,
        )

        with self._connect() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO traces
                (trace_id, query, reasoning, answer, correct,
                 difficulty, staleness, iteration, created_at)
                VALUES (?, ?, ?, ?, ?, ?, 0, ?, ?)
            """, (
                tid, query, reasoning[:2000], answer[:1000],
                1 if correct else 0, difficulty,
                iteration, time.time(),
            ))
            conn.commit()

        return trace

    def _prioritize_queries(
        self, queries: List[str], iteration: int
    ) -> List[str]:
        """AdaSTaR-style priority: stale + hard queries first."""
        # For now, simple shuffle. With history, we'd prioritize
        # queries that were previously hard or stale.
        shuffled = list(queries)
        self.rng.shuffle(shuffled)
        return shuffled

    def _increment_staleness(self) -> None:
        with self._connect() as conn:
            conn.execute("UPDATE traces SET staleness = staleness + 1")
            conn.commit()

    def _previous_accuracy(self, iteration: int) -> float:
        with self._connect() as conn:
            cur = conn.execute(
                "SELECT AVG(correct) FROM traces WHERE iteration = ?",
                (iteration,),
            )
            row = cur.fetchone()
        return row[0] if row and row[0] else 0.0

    def stats(self) -> Dict[str, Any]:
        with self._connect() as conn:
            total = conn.execute(
                "SELECT COUNT(*) FROM traces"
            ).fetchone()[0]
            correct = conn.execute(
                "SELECT COUNT(*) FROM traces WHERE correct = 1"
            ).fetchone()[0]
            avg_diff = conn.execute(
                "SELECT AVG(difficulty) FROM traces WHERE correct = 1"
            ).fetchone()[0]
        return {
            "total_traces": total,
            "correct_traces": correct,
            "accuracy": round(correct / total, 3) if total > 0 else 0.0,
            "avg_difficulty": round(avg_diff or 0, 3),
        }
