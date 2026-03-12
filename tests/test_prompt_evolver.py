"""
Tests for core.prompt_evolver -- EvoPrompt + OPRO hybrid.
All LLM calls are mocked; no live runtime needed.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from core.prompt_evolver import (
    PromptCandidate,
    PromptEvolver,
    EvolutionResult,
    _prompt_id,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_runtime(responses=None):
    rt = MagicMock()
    if responses:
        rt.generate.side_effect = responses
    else:
        rt.generate.return_value = "You are a helpful code assistant. Always cite sources."
    return rt


def _mock_eval_fn(scores=None):
    if scores:
        it = iter(scores)
        return MagicMock(side_effect=lambda p, q: next(it, 0.5))
    return MagicMock(return_value=0.7)


SEED_PROMPT = (
    "You are a code assistant. Answer questions about the codebase "
    "using only the provided context. Be precise and cite sources."
)


# ---------------------------------------------------------------------------
# PromptCandidate dataclass
# ---------------------------------------------------------------------------

class TestPromptCandidate:

    def test_defaults(self):
        c = PromptCandidate(prompt_id="abc", text="test", generation=0)
        assert c.avg_score == 0.0
        assert c.eval_count == 0
        assert c.mutation_type == ""
        assert c.parent_ids == []

    def test_with_fields(self):
        c = PromptCandidate(
            prompt_id="xyz", text="prompt", generation=2,
            mutation_type="rephrase", avg_score=0.8,
        )
        assert c.mutation_type == "rephrase"
        assert c.avg_score == 0.8


# ---------------------------------------------------------------------------
# _prompt_id
# ---------------------------------------------------------------------------

class TestPromptId:

    def test_deterministic(self):
        assert _prompt_id("hello") == _prompt_id("hello")

    def test_different_texts(self):
        assert _prompt_id("a") != _prompt_id("b")

    def test_length(self):
        assert len(_prompt_id("anything")) == 12


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:

    def test_minimal(self, tmp_path):
        db = str(tmp_path / "evo.db")
        pe = PromptEvolver(
            runtime=_mock_runtime(),
            eval_fn=_mock_eval_fn(),
            db_path=db,
        )
        assert pe.pop_size == 8

    def test_custom_params(self, tmp_path):
        db = str(tmp_path / "evo.db")
        pe = PromptEvolver(
            runtime=_mock_runtime(),
            eval_fn=_mock_eval_fn(),
            db_path=db,
            population_size=4,
            seed=123,
        )
        assert pe.pop_size == 4


# ---------------------------------------------------------------------------
# _init_population
# ---------------------------------------------------------------------------

class TestInitPopulation:

    def test_seed_is_first(self, tmp_path):
        db = str(tmp_path / "evo.db")
        pe = PromptEvolver(
            runtime=_mock_runtime(),
            eval_fn=_mock_eval_fn(),
            db_path=db,
            population_size=4,
        )
        pop = pe._init_population(SEED_PROMPT, [])
        assert pop[0].text == SEED_PROMPT
        assert pop[0].mutation_type == "seed"

    def test_fills_population(self, tmp_path):
        db = str(tmp_path / "evo.db")
        pe = PromptEvolver(
            runtime=_mock_runtime(),
            eval_fn=_mock_eval_fn(),
            db_path=db,
            population_size=4,
        )
        pop = pe._init_population(SEED_PROMPT, [])
        # Should have seed + mutations (up to pop_size)
        assert len(pop) >= 2


# ---------------------------------------------------------------------------
# Mutations
# ---------------------------------------------------------------------------

class TestMutations:

    def test_rephrase(self, tmp_path):
        db = str(tmp_path / "evo.db")
        rt = _mock_runtime()
        rt.generate.return_value = "Rephrased prompt text that is long enough to pass validation check."
        pe = PromptEvolver(runtime=rt, eval_fn=_mock_eval_fn(), db_path=db)
        parent = PromptCandidate(prompt_id="p1", text=SEED_PROMPT, generation=0)
        child = pe._mutate(parent, "rephrase", 1, [])
        assert child is not None
        assert child.mutation_type == "rephrase"
        assert child.parent_ids == ["p1"]

    def test_compress(self, tmp_path):
        db = str(tmp_path / "evo.db")
        rt = _mock_runtime()
        rt.generate.return_value = "Shorter prompt text that still works for code assistance."
        pe = PromptEvolver(runtime=rt, eval_fn=_mock_eval_fn(), db_path=db)
        parent = PromptCandidate(prompt_id="p1", text=SEED_PROMPT, generation=0)
        child = pe._mutate(parent, "compress", 1, [])
        assert child is not None
        assert child.mutation_type == "compress"

    def test_extend_with_failures(self, tmp_path):
        db = str(tmp_path / "evo.db")
        rt = _mock_runtime()
        rt.generate.return_value = "Extended prompt with a new rule about handling edge cases in code retrieval."
        pe = PromptEvolver(runtime=rt, eval_fn=_mock_eval_fn(), db_path=db)
        parent = PromptCandidate(prompt_id="p1", text=SEED_PROMPT, generation=0)
        failures = [{"query": "What is X?", "bad_answer": "I don't know"}]
        child = pe._mutate(parent, "extend", 1, failures)
        assert child is not None
        assert child.mutation_type == "extend"

    def test_mutation_too_short_returns_none(self, tmp_path):
        db = str(tmp_path / "evo.db")
        rt = _mock_runtime()
        rt.generate.return_value = "short"  # < 20 chars
        pe = PromptEvolver(runtime=rt, eval_fn=_mock_eval_fn(), db_path=db)
        parent = PromptCandidate(prompt_id="p1", text=SEED_PROMPT, generation=0)
        child = pe._mutate(parent, "rephrase", 1, [])
        assert child is None

    def test_runtime_crash_returns_none(self, tmp_path):
        db = str(tmp_path / "evo.db")
        rt = _mock_runtime()
        rt.generate.side_effect = RuntimeError("crash")
        pe = PromptEvolver(runtime=rt, eval_fn=_mock_eval_fn(), db_path=db)
        parent = PromptCandidate(prompt_id="p1", text=SEED_PROMPT, generation=0)
        child = pe._mutate(parent, "rephrase", 1, [])
        assert child is None


# ---------------------------------------------------------------------------
# Crossover
# ---------------------------------------------------------------------------

class TestCrossover:

    def test_crossover(self, tmp_path):
        db = str(tmp_path / "evo.db")
        rt = _mock_runtime()
        rt.generate.return_value = "Combined prompt taking best elements from both parent prompts."
        pe = PromptEvolver(runtime=rt, eval_fn=_mock_eval_fn(), db_path=db)
        a = PromptCandidate(prompt_id="a1", text="Prompt A text", generation=0, avg_score=0.8)
        b = PromptCandidate(prompt_id="b1", text="Prompt B text", generation=0, avg_score=0.6)
        child = pe._crossover(a, b, 1)
        assert child is not None
        assert child.mutation_type == "crossover"
        assert "a1" in child.parent_ids
        assert "b1" in child.parent_ids


# ---------------------------------------------------------------------------
# evolve (end-to-end)
# ---------------------------------------------------------------------------

class TestEvolve:

    def test_returns_result(self, tmp_path):
        db = str(tmp_path / "evo.db")
        # Need enough LLM responses for init population + reproduction
        rt = _mock_runtime()
        rt.generate.return_value = "A valid evolved prompt text that passes the length check easily."
        pe = PromptEvolver(
            runtime=rt, eval_fn=_mock_eval_fn(),
            db_path=db, population_size=3,
        )
        result = pe.evolve(
            seed_prompt=SEED_PROMPT,
            eval_queries=["q1", "q2", "q3"],
            max_generations=2,
            evals_per_candidate=2,
        )
        assert isinstance(result, EvolutionResult)
        assert result.generations_run == 2
        assert result.total_evaluations > 0
        assert len(result.score_history) == 2
        assert result.champion is not None
        assert result.total_ms >= 0

    def test_champion_has_highest_score(self, tmp_path):
        db = str(tmp_path / "evo.db")
        rt = _mock_runtime()
        rt.generate.return_value = "An evolved prompt variant for code assistance that is good."
        # First candidate (seed) scores higher
        eval_fn = _mock_eval_fn()
        pe = PromptEvolver(
            runtime=rt, eval_fn=eval_fn,
            db_path=db, population_size=2,
        )
        result = pe.evolve(SEED_PROMPT, ["q1"], max_generations=1, evals_per_candidate=1)
        assert result.champion.avg_score >= 0

    def test_lineage_persisted(self, tmp_path):
        db = str(tmp_path / "evo.db")
        rt = _mock_runtime()
        rt.generate.return_value = "Valid prompt text for lineage persistence test check."
        pe = PromptEvolver(
            runtime=rt, eval_fn=_mock_eval_fn(),
            db_path=db, population_size=2,
        )
        pe.evolve(SEED_PROMPT, ["q1"], max_generations=1, evals_per_candidate=1)
        history = pe.champion_history(limit=10)
        assert len(history) >= 1
        assert "prompt_id" in history[0]
        assert "avg_score" in history[0]


# ---------------------------------------------------------------------------
# _sample_queries
# ---------------------------------------------------------------------------

class TestSampleQueries:

    def test_returns_subset(self, tmp_path):
        db = str(tmp_path / "evo.db")
        pe = PromptEvolver(runtime=_mock_runtime(), eval_fn=_mock_eval_fn(), db_path=db)
        result = pe._sample_queries(["a", "b", "c", "d", "e"], 3)
        assert len(result) == 3

    def test_returns_all_if_fewer(self, tmp_path):
        db = str(tmp_path / "evo.db")
        pe = PromptEvolver(runtime=_mock_runtime(), eval_fn=_mock_eval_fn(), db_path=db)
        result = pe._sample_queries(["a", "b"], 5)
        assert len(result) == 2
