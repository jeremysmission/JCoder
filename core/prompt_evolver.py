"""
Prompt Evolver (EvoPrompt + OPRO Hybrid)
-----------------------------------------
Automatically evolves system prompts using the LLM itself as both
mutator and evaluator. The highest-leverage self-improvement mechanism
because the system prompt controls ALL generation quality.

Based on:
- EvoPrompt (ACL 2024): Evolutionary prompt optimization with LLM-driven crossover
- OPRO (NeurIPS 2024): Optimization by PROmpting, using past trajectories
- PromptBreeder (ICLR 2024): Self-referential prompt evolution

Key insight: Instead of hand-tuning prompts, let the LLM generate
prompt variants, score them on real queries, and evolve the best ones.

Algorithm:
1. Start with current system prompt + variants
2. Use LLM to generate mutations (rephrase, extend, compress, invert)
3. Evaluate each variant on held-out queries from telemetry
4. Tournament selection: best 50% survive
5. Crossover: combine elements of top prompts
6. Repeat for N generations
7. Champion prompt replaces the system prompt

Storage: SQLite for prompt lineage tracking.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from core.runtime import Runtime


@dataclass
class PromptCandidate:
    """A single prompt variant in the evolutionary pool."""
    prompt_id: str
    text: str
    generation: int  # which evolution generation created this
    parent_ids: List[str] = field(default_factory=list)
    mutation_type: str = ""  # "seed" | "rephrase" | "extend" | "compress" | "crossover"
    avg_score: float = 0.0
    eval_count: int = 0
    scores: List[float] = field(default_factory=list)


@dataclass
class EvolutionResult:
    """Result of a prompt evolution run."""
    champion: PromptCandidate
    generations_run: int
    total_candidates: int
    total_evaluations: int
    score_history: List[float]  # best score per generation
    total_ms: float = 0.0


# ---------------------------------------------------------------------------
# Mutation operators (LLM-powered)
# ---------------------------------------------------------------------------

_REPHRASE_PROMPT = (
    "Rephrase the following system prompt to be clearer and more effective "
    "at guiding a code assistant. Keep the same meaning but improve "
    "the wording. Output ONLY the new prompt, nothing else.\n\n"
    "Original prompt:\n{prompt}"
)

_EXTEND_PROMPT = (
    "Improve this system prompt by adding ONE specific, actionable rule "
    "that would make a code assistant give better answers. Add it naturally. "
    "Output ONLY the complete new prompt.\n\n"
    "Current prompt:\n{prompt}\n\n"
    "Recent failure example (the assistant gave a bad answer to this):\n"
    "Query: {query}\nBad answer snippet: {bad_answer}"
)

_COMPRESS_PROMPT = (
    "Make this system prompt shorter and more focused. Remove any "
    "redundant or low-value instructions. The prompt should be at "
    "most 70% of the original length. Output ONLY the new prompt.\n\n"
    "Original prompt:\n{prompt}"
)

_CROSSOVER_PROMPT = (
    "Combine the best elements of these two system prompts into one "
    "superior prompt. Take the strongest rules from each. "
    "Output ONLY the new combined prompt.\n\n"
    "Prompt A (score: {score_a:.2f}):\n{prompt_a}\n\n"
    "Prompt B (score: {score_b:.2f}):\n{prompt_b}"
)


def _prompt_id(text: str) -> str:
    """Stable hash for a prompt variant."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


class PromptEvolver:
    """
    Evolutionary prompt optimization using the LLM as both mutator and evaluator.

    The core loop:
    1. Maintain a population of prompt variants
    2. Evaluate each on held-out queries (using reflection scoring)
    3. Tournament selection + crossover + mutation
    4. Champion replaces system prompt
    """

    def __init__(
        self,
        runtime: Runtime,
        eval_fn: Callable[[str, str], float],
        db_path: str = "_prompt_evo/lineage.db",
        population_size: int = 8,
        seed: int = 42,
    ):
        """
        Args:
            runtime: LLM runtime for generating mutations
            eval_fn: Function(prompt, query) -> score (0.0-1.0)
                     Evaluates how well a prompt performs on a query.
            db_path: SQLite path for lineage tracking
            population_size: Number of prompt variants per generation
        """
        self.runtime = runtime
        self.eval_fn = eval_fn
        self.pop_size = population_size
        self.rng = random.Random(seed)
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self._db_path))

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS prompt_lineage (
                    prompt_id TEXT PRIMARY KEY,
                    text TEXT NOT NULL,
                    generation INTEGER,
                    parent_ids_json TEXT,
                    mutation_type TEXT,
                    avg_score REAL,
                    eval_count INTEGER,
                    created_at REAL
                )
            """)
            conn.commit()

    def _log_candidate(self, cand: PromptCandidate) -> None:
        with self._connect() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO prompt_lineage
                (prompt_id, text, generation, parent_ids_json,
                 mutation_type, avg_score, eval_count, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                cand.prompt_id, cand.text, cand.generation,
                json.dumps(cand.parent_ids), cand.mutation_type,
                cand.avg_score, cand.eval_count, time.time(),
            ))
            conn.commit()

    def evolve(
        self,
        seed_prompt: str,
        eval_queries: List[str],
        max_generations: int = 5,
        evals_per_candidate: int = 3,
        failure_examples: Optional[List[Dict[str, str]]] = None,
    ) -> EvolutionResult:
        """
        Run prompt evolution.

        Args:
            seed_prompt: Starting system prompt
            eval_queries: Queries to evaluate prompts against
            max_generations: Number of evolution generations
            evals_per_candidate: How many queries to test each candidate
            failure_examples: Optional list of {"query": ..., "bad_answer": ...}
                              used to guide the _extend mutation

        Returns:
            EvolutionResult with the champion prompt
        """
        t0 = time.time()
        failures = failure_examples or []
        score_history = []
        total_evals = 0

        # --- Initialize population with seed + mutations ---
        population = self._init_population(seed_prompt, failures)

        for gen in range(max_generations):
            # --- Evaluate ---
            test_queries = self._sample_queries(eval_queries, evals_per_candidate)
            for cand in population:
                for q in test_queries:
                    try:
                        score = self.eval_fn(cand.text, q)
                        cand.scores.append(score)
                        total_evals += 1
                    except Exception:
                        cand.scores.append(0.0)
                cand.avg_score = (
                    sum(cand.scores) / len(cand.scores) if cand.scores else 0.0
                )

            # Sort by score
            population.sort(key=lambda c: c.avg_score, reverse=True)
            best_score = population[0].avg_score
            score_history.append(best_score)

            # Log all candidates
            for cand in population:
                cand.eval_count = len(cand.scores)
                self._log_candidate(cand)

            # --- Selection + Reproduction ---
            if gen < max_generations - 1:
                survivors = population[:self.pop_size // 2]
                children = self._reproduce(
                    survivors, gen + 1, failures)
                population = survivors + children
                # Reset scores for new generation
                for cand in population:
                    cand.scores = []

        # Final sort
        population.sort(key=lambda c: c.avg_score, reverse=True)
        champion = population[0]

        return EvolutionResult(
            champion=champion,
            generations_run=max_generations,
            total_candidates=sum(
                self.pop_size for _ in range(max_generations)),
            total_evaluations=total_evals,
            score_history=score_history,
            total_ms=(time.time() - t0) * 1000,
        )

    def _init_population(
        self,
        seed: str,
        failures: List[Dict[str, str]],
    ) -> List[PromptCandidate]:
        """Create initial population from seed prompt."""
        seed_cand = PromptCandidate(
            prompt_id=_prompt_id(seed),
            text=seed,
            generation=0,
            mutation_type="seed",
        )
        population = [seed_cand]

        # Generate mutations to fill population
        mutations = ["rephrase", "compress", "extend"]
        while len(population) < self.pop_size:
            mut_type = mutations[len(population) % len(mutations)]
            mutated = self._mutate(seed_cand, mut_type, 0, failures)
            if mutated and mutated.prompt_id != seed_cand.prompt_id:
                population.append(mutated)
            else:
                # Fallback: add seed with slight variation
                break

        return population

    def _reproduce(
        self,
        survivors: List[PromptCandidate],
        generation: int,
        failures: List[Dict[str, str]],
    ) -> List[PromptCandidate]:
        """Create children via mutation and crossover."""
        children = []
        target = self.pop_size - len(survivors)

        for i in range(target):
            if i % 3 == 0 and len(survivors) >= 2:
                # Crossover
                parents = self.rng.sample(survivors, 2)
                child = self._crossover(parents[0], parents[1], generation)
            else:
                # Mutation
                parent = survivors[i % len(survivors)]
                mut_type = self.rng.choice(["rephrase", "extend", "compress"])
                child = self._mutate(parent, mut_type, generation, failures)

            if child:
                children.append(child)

        return children

    def _mutate(
        self,
        parent: PromptCandidate,
        mutation_type: str,
        generation: int,
        failures: List[Dict[str, str]],
    ) -> Optional[PromptCandidate]:
        """Apply a single mutation operator."""
        try:
            if mutation_type == "rephrase":
                prompt = _REPHRASE_PROMPT.format(prompt=parent.text)
            elif mutation_type == "compress":
                prompt = _COMPRESS_PROMPT.format(prompt=parent.text)
            elif mutation_type == "extend":
                if failures:
                    fail = self.rng.choice(failures)
                    prompt = _EXTEND_PROMPT.format(
                        prompt=parent.text,
                        query=fail.get("query", ""),
                        bad_answer=fail.get("bad_answer", "")[:200],
                    )
                else:
                    prompt = _REPHRASE_PROMPT.format(prompt=parent.text)
            else:
                return None

            raw = self.runtime.generate(
                question=prompt,
                context_chunks=[],
                system_prompt="You are a prompt engineer. Follow instructions exactly.",
                temperature=0.7,
                max_tokens=512,
            )

            text = raw.strip()
            if len(text) < 20:
                return None

            return PromptCandidate(
                prompt_id=_prompt_id(text),
                text=text,
                generation=generation,
                parent_ids=[parent.prompt_id],
                mutation_type=mutation_type,
            )
        except Exception:
            return None

    def _crossover(
        self,
        parent_a: PromptCandidate,
        parent_b: PromptCandidate,
        generation: int,
    ) -> Optional[PromptCandidate]:
        """Combine two prompts using LLM-driven crossover."""
        try:
            prompt = _CROSSOVER_PROMPT.format(
                prompt_a=parent_a.text,
                prompt_b=parent_b.text,
                score_a=parent_a.avg_score,
                score_b=parent_b.avg_score,
            )
            raw = self.runtime.generate(
                question=prompt,
                context_chunks=[],
                system_prompt="You are a prompt engineer. Follow instructions exactly.",
                temperature=0.5,
                max_tokens=512,
            )
            text = raw.strip()
            if len(text) < 20:
                return None

            return PromptCandidate(
                prompt_id=_prompt_id(text),
                text=text,
                generation=generation,
                parent_ids=[parent_a.prompt_id, parent_b.prompt_id],
                mutation_type="crossover",
            )
        except Exception:
            return None

    def _sample_queries(self, queries: List[str], n: int) -> List[str]:
        """Sample evaluation queries."""
        if len(queries) <= n:
            return queries
        return self.rng.sample(queries, n)

    def champion_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Return top prompts from the lineage database."""
        with self._connect() as conn:
            cur = conn.execute(
                "SELECT prompt_id, text, generation, mutation_type, "
                "avg_score, eval_count FROM prompt_lineage "
                "ORDER BY avg_score DESC LIMIT ?",
                (limit,),
            )
            return [
                {
                    "prompt_id": r[0], "text": r[1],
                    "generation": r[2], "mutation_type": r[3],
                    "avg_score": r[4], "eval_count": r[5],
                }
                for r in cur.fetchall()
            ]
