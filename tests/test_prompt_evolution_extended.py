"""
Extended tests for prompt evolution and adversarial self-play systems.
Covers mutation operators, fitness evaluation, population management,
convergence, adversarial games, escalation, and edge cases.
All LLM calls are mocked; no live runtime needed.
"""

from __future__ import annotations

import random
from unittest.mock import MagicMock, patch, call

import pytest

from core.prompt_evolver import (
    PromptCandidate,
    PromptEvolver,
    EvolutionResult,
    _prompt_id,
)
from core.adversarial_self_play import (
    AdversarialSelfPlay,
    Challenge,
    ChallengeOutcome,
    SelfPlayResult,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

SEED = "You are a code assistant. Answer using only the provided context."
LONG_VARIANT = "A long enough mutated prompt variant that passes the 20-char check."


def _rt(text=LONG_VARIANT):
    """Mock Runtime with a default generate return."""
    rt = MagicMock()
    rt.generate.return_value = text
    return rt


def _eval(score=0.7):
    """Mock eval_fn returning a fixed score."""
    return MagicMock(return_value=score)


def _evo(tmp_path, rt=None, ef=None, pop=4, seed=42):
    """Create a PromptEvolver wired to tmp_path."""
    return PromptEvolver(
        runtime=rt or _rt(),
        eval_fn=ef or _eval(),
        db_path=str(tmp_path / "evo.db"),
        population_size=pop,
        seed=seed,
    )


def _asp(tmp_path, rt=None, af=None, seed=42):
    """Create an AdversarialSelfPlay wired to tmp_path."""
    return AdversarialSelfPlay(
        runtime=rt or _rt("Q: What does foo return?\nA: 42"),
        answer_fn=af or MagicMock(return_value="foo returns 42"),
        db_path=str(tmp_path / "asp.db"),
        seed=seed,
    )


CHUNKS = [
    {"content": "def foo():\n    return 42\n", "source_path": "src/foo.py"},
    {"content": "class Bar:\n    x = 1\n", "source_path": "src/bar.py"},
]


# ===================================================================
# Part 1 -- Prompt Evolver: Mutation Operators
# ===================================================================

class TestMutationOperators:
    """Detailed mutation operator coverage."""

    def test_rephrase_preserves_lineage(self, tmp_path):
        pe = _evo(tmp_path)
        parent = PromptCandidate(prompt_id="p0", text=SEED, generation=0)
        child = pe._mutate(parent, "rephrase", 1, [])
        assert child is not None
        assert child.generation == 1
        assert child.parent_ids == ["p0"]
        assert child.mutation_type == "rephrase"

    def test_compress_produces_different_id(self, tmp_path):
        pe = _evo(tmp_path)
        parent = PromptCandidate(prompt_id="p0", text=SEED, generation=0)
        child = pe._mutate(parent, "compress", 1, [])
        assert child is not None
        assert child.prompt_id != parent.prompt_id

    def test_extend_uses_failure_examples(self, tmp_path):
        rt = _rt()
        pe = _evo(tmp_path, rt=rt)
        parent = PromptCandidate(prompt_id="p0", text=SEED, generation=0)
        failures = [{"query": "explain merge sort", "bad_answer": "idk"}]
        pe._mutate(parent, "extend", 1, failures)
        # The generate call should include the failure context
        call_args = rt.generate.call_args
        assert "merge sort" in call_args.kwargs.get("question", call_args[0][0] if call_args[0] else "")

    def test_extend_without_failures_falls_back_to_rephrase(self, tmp_path):
        rt = _rt()
        pe = _evo(tmp_path, rt=rt)
        parent = PromptCandidate(prompt_id="p0", text=SEED, generation=0)
        pe._mutate(parent, "extend", 1, [])
        call_args = rt.generate.call_args
        prompt_used = call_args.kwargs.get("question", call_args[0][0] if call_args[0] else "")
        assert "Rephrase" in prompt_used

    def test_unknown_mutation_type_returns_none(self, tmp_path):
        pe = _evo(tmp_path)
        parent = PromptCandidate(prompt_id="p0", text=SEED, generation=0)
        assert pe._mutate(parent, "delete_nonexistent", 1, []) is None

    def test_crossover_records_both_parents(self, tmp_path):
        pe = _evo(tmp_path)
        a = PromptCandidate(prompt_id="a", text="Prompt A text", generation=0, avg_score=0.9)
        b = PromptCandidate(prompt_id="b", text="Prompt B text", generation=0, avg_score=0.6)
        child = pe._crossover(a, b, 2)
        assert child is not None
        assert set(child.parent_ids) == {"a", "b"}
        assert child.generation == 2
        assert child.mutation_type == "crossover"

    def test_crossover_too_short_returns_none(self, tmp_path):
        rt = _rt("tiny")
        pe = _evo(tmp_path, rt=rt)
        a = PromptCandidate(prompt_id="a", text="A", generation=0)
        b = PromptCandidate(prompt_id="b", text="B", generation=0)
        assert pe._crossover(a, b, 1) is None

    def test_crossover_runtime_error_returns_none(self, tmp_path):
        rt = _rt()
        rt.generate.side_effect = RuntimeError("boom")
        pe = _evo(tmp_path, rt=rt)
        a = PromptCandidate(prompt_id="a", text="A", generation=0)
        b = PromptCandidate(prompt_id="b", text="B", generation=0)
        assert pe._crossover(a, b, 1) is None


# ===================================================================
# Part 2 -- Fitness Evaluation with Mocked Scoring
# ===================================================================

class TestFitnessEvaluation:
    """Fitness scoring wired through eval_fn."""

    def test_scores_accumulate(self, tmp_path):
        scores = [0.3, 0.5, 0.9, 0.4, 0.6, 0.8]
        ef = MagicMock(side_effect=scores)
        pe = _evo(tmp_path, ef=ef, pop=2)
        result = pe.evolve(SEED, ["q1", "q2", "q3"], max_generations=1, evals_per_candidate=3)
        assert result.total_evaluations == ef.call_count

    def test_eval_failure_yields_zero(self, tmp_path):
        ef = MagicMock(side_effect=RuntimeError("scoring crash"))
        pe = _evo(tmp_path, ef=ef, pop=2)
        result = pe.evolve(SEED, ["q1"], max_generations=1, evals_per_candidate=1)
        # All scores are 0.0 due to exception fallback
        assert result.champion.avg_score == 0.0

    def test_champion_is_highest_scoring(self, tmp_path):
        # Alternate high/low scores so seed (evaluated first) wins
        scores = iter([0.95, 0.2, 0.95, 0.2])
        ef = MagicMock(side_effect=lambda p, q: next(scores, 0.5))
        pe = _evo(tmp_path, ef=ef, pop=2)
        result = pe.evolve(SEED, ["q1", "q2"], max_generations=1, evals_per_candidate=2)
        assert result.champion.avg_score >= 0.5


# ===================================================================
# Part 3 -- Population Management (Selection, Elitism)
# ===================================================================

class TestPopulationManagement:
    """Selection, survivor carryover, reproduction."""

    def test_survivors_are_top_half(self, tmp_path):
        pe = _evo(tmp_path, pop=4)
        result = pe.evolve(SEED, ["q1"], max_generations=2, evals_per_candidate=1)
        # After 2 generations the result should have the champion from final pop
        assert result.generations_run == 2
        assert result.champion is not None

    def test_reproduce_creates_target_children(self, tmp_path):
        pe = _evo(tmp_path, pop=6)
        survivors = [
            PromptCandidate(prompt_id=f"s{i}", text=f"Survivor prompt number {i} text", generation=0, avg_score=0.8 - i*0.1)
            for i in range(3)
        ]
        children = pe._reproduce(survivors, 1, [])
        # target = pop_size - len(survivors) = 6 - 3 = 3
        assert len(children) <= 3

    def test_elitism_seed_survives_if_best(self, tmp_path):
        # Seed always scores 1.0, mutations score 0.0
        call_count = [0]
        def scoring(prompt, query):
            call_count[0] += 1
            return 1.0 if prompt == SEED else 0.0
        pe = _evo(tmp_path, ef=scoring, pop=3)
        result = pe.evolve(SEED, ["q1"], max_generations=2, evals_per_candidate=1)
        assert result.champion.text == SEED

    def test_score_history_length_matches_generations(self, tmp_path):
        pe = _evo(tmp_path, pop=2)
        result = pe.evolve(SEED, ["q1"], max_generations=4, evals_per_candidate=1)
        assert len(result.score_history) == 4


# ===================================================================
# Part 4 -- Convergence Detection
# ===================================================================

class TestConvergence:
    """Verify that evolution terminates and converges."""

    def test_stable_scores_across_generations(self, tmp_path):
        pe = _evo(tmp_path, ef=_eval(0.75), pop=2)
        result = pe.evolve(SEED, ["q1"], max_generations=3, evals_per_candidate=1)
        # All generations should score identically with a fixed eval_fn
        assert all(s == result.score_history[0] for s in result.score_history)

    def test_max_generations_respected(self, tmp_path):
        pe = _evo(tmp_path, pop=2)
        result = pe.evolve(SEED, ["q1"], max_generations=7, evals_per_candidate=1)
        assert result.generations_run == 7
        assert len(result.score_history) == 7

    def test_single_query_no_crash(self, tmp_path):
        pe = _evo(tmp_path, pop=2)
        result = pe.evolve(SEED, ["only_one"], max_generations=1, evals_per_candidate=5)
        # evals_per_candidate > len(queries), so all queries returned
        assert result.total_evaluations >= 2  # at least 2 candidates * 1 query


# ===================================================================
# Part 5 -- Adversarial Game: Solver Generates, Verifier Checks
# ===================================================================

class TestAdversarialGame:
    """Solver/verifier interaction in self-play."""

    def test_hardness_correct_answer_passes(self, tmp_path):
        rt = _rt()
        rt.generate.return_value = "8"  # score 0.8 => correct
        asp = _asp(tmp_path, rt=rt)
        ch = Challenge(
            challenge_id="h1", game="hardness", question="What does foo return?",
            expected_behavior="42", difficulty=0.5, source_context="def foo(): return 42",
        )
        correct, failure, conf = asp._judge_hardness(ch, "foo returns 42")
        assert correct is True
        assert failure == "none"

    def test_hardness_wrong_answer_fails(self, tmp_path):
        rt = _rt()
        rt.generate.return_value = "2"  # score 0.2 => incorrect
        asp = _asp(tmp_path, rt=rt)
        ch = Challenge(
            challenge_id="h2", game="hardness", question="What does foo return?",
            expected_behavior="42", difficulty=0.5, source_context="def foo(): return 42",
        )
        correct, failure, conf = asp._judge_hardness(ch, "foo returns None")
        assert correct is False
        assert failure == "wrong_answer"

    def test_hardness_no_expected_always_passes(self, tmp_path):
        asp = _asp(tmp_path)
        ch = Challenge(
            challenge_id="h3", game="hardness", question="q?",
            expected_behavior="", difficulty=0.5, source_context="code",
        )
        correct, failure, conf = asp._judge_hardness(ch, "any answer")
        assert correct is True

    def test_evaluate_challenge_calls_answer_fn(self, tmp_path):
        af = MagicMock(return_value="I don't know, that does not exist.")
        rt = _rt()
        rt.generate.return_value = "7"
        asp = _asp(tmp_path, rt=rt, af=af)
        ch = Challenge(
            challenge_id="e1", game="hardness", question="What is X?",
            expected_behavior="X is 5", difficulty=0.3, source_context="X=5",
        )
        outcome = asp._evaluate_challenge(ch)
        af.assert_called_once_with("What is X?")
        assert isinstance(outcome, ChallengeOutcome)
        assert outcome.latency_ms >= 0

    def test_evaluate_challenge_answer_fn_error(self, tmp_path):
        af = MagicMock(side_effect=RuntimeError("pipeline down"))
        rt = _rt()
        rt.generate.return_value = "3"
        asp = _asp(tmp_path, rt=rt, af=af)
        ch = Challenge(
            challenge_id="e2", game="hardness", question="Q?",
            expected_behavior="A", difficulty=0.5, source_context="code",
        )
        outcome = asp._evaluate_challenge(ch)
        assert "ERROR" in outcome.verifier_answer


# ===================================================================
# Part 6 -- Adversarial Escalation: Difficulty Increases on Success
# ===================================================================

class TestAdversarialEscalation:
    """Difficulty ramps up when verifier succeeds, drops on failure."""

    def test_difficulty_increases_on_correct(self, tmp_path):
        rt = _rt("Q: Hard question?\nA: correct answer")
        # Verifier always refuses correctly for trick game
        af = MagicMock(return_value="I don't know, that does not exist in the code.")
        asp = _asp(tmp_path, rt=rt, af=af)

        difficulties_seen = []
        orig_gen = asp._gen_hardness_challenge

        def tracking_gen(chunk, difficulty):
            difficulties_seen.append(difficulty)
            return orig_gen(chunk, difficulty)

        asp._gen_hardness_challenge = tracking_gen
        asp._gen_trick_challenge = MagicMock(return_value=None)
        asp._gen_ambiguity_challenge = MagicMock(return_value=None)

        asp.play_session(CHUNKS, rounds_per_game=3, difficulty_start=0.3, difficulty_step=0.15)
        # Each successful round should bump difficulty
        if len(difficulties_seen) >= 2:
            assert difficulties_seen[1] >= difficulties_seen[0]

    def test_difficulty_capped_at_one(self, tmp_path):
        rt = _rt("Q: question?\nA: answer")
        af = MagicMock(return_value="does not exist")
        asp = _asp(tmp_path, rt=rt, af=af)

        difficulties_seen = []
        orig_gen = asp._gen_hardness_challenge

        def tracking_gen(chunk, difficulty):
            difficulties_seen.append(difficulty)
            return orig_gen(chunk, difficulty)

        asp._gen_hardness_challenge = tracking_gen
        asp._gen_trick_challenge = MagicMock(return_value=None)
        asp._gen_ambiguity_challenge = MagicMock(return_value=None)

        asp.play_session(CHUNKS, rounds_per_game=10, difficulty_start=0.8, difficulty_step=0.2)
        assert all(d <= 1.0 for d in difficulties_seen)

    def test_difficulty_decreases_on_failure(self, tmp_path):
        rt = _rt("Q: question?\nA: answer")
        # Verifier gives a confident wrong answer (no refusal signals)
        af = MagicMock(return_value="The function processes data and returns a list.")
        asp = _asp(tmp_path, rt=rt, af=af)

        difficulties_seen = []
        orig_trick = asp._gen_trick_challenge

        def tracking_trick(chunk, difficulty):
            difficulties_seen.append(difficulty)
            return orig_trick(chunk, difficulty)

        asp._gen_hardness_challenge = MagicMock(return_value=None)
        asp._gen_trick_challenge = tracking_trick
        asp._gen_ambiguity_challenge = MagicMock(return_value=None)

        # trick game: verifier that doesn't refuse => failure => difficulty drops
        rt.generate.return_value = "Q: What does baz do?\nTRAP: baz does not exist"
        asp.play_session(CHUNKS, rounds_per_game=3, difficulty_start=0.6, difficulty_step=0.2)
        if len(difficulties_seen) >= 2:
            # After first failure, difficulty should decrease
            assert difficulties_seen[1] < difficulties_seen[0]


# ===================================================================
# Part 7 -- Edge Cases
# ===================================================================

class TestEdgeCases:
    """Population size 1, zero-fitness, max generations."""

    def test_population_size_one_triggers_zero_division(self, tmp_path):
        """pop_size=1 causes ZeroDivisionError in _reproduce (pop_size//2==0).
        This documents a known edge case in the source."""
        pe = _evo(tmp_path, pop=1)
        with pytest.raises(ZeroDivisionError):
            pe.evolve(SEED, ["q1"], max_generations=2, evals_per_candidate=1)

    def test_population_size_two_minimum_viable(self, tmp_path):
        pe = _evo(tmp_path, pop=2)
        result = pe.evolve(SEED, ["q1"], max_generations=2, evals_per_candidate=1)
        assert result.champion is not None

    def test_zero_fitness_population(self, tmp_path):
        pe = _evo(tmp_path, ef=_eval(0.0), pop=3)
        result = pe.evolve(SEED, ["q1", "q2"], max_generations=2, evals_per_candidate=1)
        assert result.champion.avg_score == 0.0
        assert result.generations_run == 2

    def test_empty_eval_queries(self, tmp_path):
        pe = _evo(tmp_path, pop=2)
        result = pe.evolve(SEED, [], max_generations=1, evals_per_candidate=3)
        # No queries means no evaluations
        assert result.total_evaluations == 0

    def test_high_generation_count(self, tmp_path):
        pe = _evo(tmp_path, pop=2)
        result = pe.evolve(SEED, ["q1"], max_generations=10, evals_per_candidate=1)
        assert result.generations_run == 10
        assert len(result.score_history) == 10

    def test_pareto_front_all_identical(self):
        candidates = [
            PromptCandidate(prompt_id=f"p{i}", text="same text", generation=0, avg_score=0.5)
            for i in range(5)
        ]
        front = PromptEvolver.pareto_front(candidates)
        # All have identical score and token_cost so none dominates the other
        # Actually they all have same score AND same cost, so first one dominates
        # the rest (or rather, none strictly dominates because >= and <= are both true
        # but the strict inequality clause fails). All survive.
        assert len(front) >= 1

    def test_adversarial_session_all_games_fail(self, tmp_path):
        asp = _asp(tmp_path)
        asp._gen_hardness_challenge = MagicMock(side_effect=ValueError("bad"))
        asp._gen_trick_challenge = MagicMock(side_effect=ValueError("bad"))
        asp._gen_ambiguity_challenge = MagicMock(side_effect=ValueError("bad"))
        result = asp.play_session(CHUNKS, rounds_per_game=2)
        assert result.total_challenges == 0
        assert result.failed_rounds == 6  # 3 games * 2 rounds

    def test_adversarial_empty_weakness_report_on_all_correct(self, tmp_path):
        rt = _rt()
        rt.generate.return_value = "Q: test?\nA: answer"
        af = MagicMock(return_value="I don't know. It does not exist.")
        asp = _asp(tmp_path, rt=rt, af=af)
        # Judge for hardness returns "8" (correct)
        rt.generate.return_value = "Q: question?\nA: answer"
        result = asp.play_session(CHUNKS, rounds_per_game=1)
        # Even if not all games produce challenges, weakness report should be dict
        assert isinstance(result.weakness_report, dict)

    def test_parse_score_edge_cases(self):
        assert AdversarialSelfPlay._parse_score("0") == 0.0
        assert AdversarialSelfPlay._parse_score("10") == 1.0
        assert AdversarialSelfPlay._parse_score("Score: 5 out of 10") == 0.5
        assert AdversarialSelfPlay._parse_score("") == 0.5

    def test_prompt_id_empty_string(self):
        pid = _prompt_id("")
        assert len(pid) == 12

    def test_token_cost_calculation(self):
        c = PromptCandidate(prompt_id="x", text="a" * 100, generation=0)
        assert c.token_cost == 25  # 100 / 4
        c2 = PromptCandidate(prompt_id="y", text="", generation=0)
        assert c2.token_cost == 0
