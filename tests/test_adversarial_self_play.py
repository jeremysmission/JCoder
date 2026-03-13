"""
Tests for core.adversarial_self_play -- Sol-Ver adversarial games.
All LLM calls are mocked; no live runtime needed.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest

from core.adversarial_self_play import (
    AdversarialSelfPlay,
    Challenge,
    ChallengeOutcome,
    SelfPlayResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_runtime(response="Q: What does foo return?\nA: It returns 42"):
    rt = MagicMock()
    rt.generate.return_value = response
    return rt


def _mock_answer_fn(answer="foo returns 42"):
    return MagicMock(return_value=answer)


def _sample_chunks():
    return [
        {"content": "def foo():\n    return 42\n", "source_path": "src/foo.py"},
        {"content": "class Bar:\n    pass\n", "source_path": "src/bar.py"},
    ]


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

class TestDataclasses:

    def test_challenge_fields(self):
        c = Challenge(
            challenge_id="c1", game="hardness", question="test?",
            expected_behavior="answer", difficulty=0.5, source_context="code",
        )
        assert c.generated_at == 0.0

    def test_outcome_fields(self):
        o = ChallengeOutcome(
            challenge_id="c1", verifier_answer="42",
            correct_behavior=True, confidence=0.9,
            failure_mode="none",
        )
        assert o.latency_ms == 0.0

    def test_result_accuracy(self):
        r = SelfPlayResult(
            total_challenges=10, correct=7, accuracy=0.7,
            weakness_report={"wrong_answer": 3},
            hardest_failures=[],
        )
        assert r.accuracy == 0.7
        assert r.failed_rounds == 0


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:

    def test_creates_db(self, tmp_path):
        db = str(tmp_path / "play.db")
        asp = AdversarialSelfPlay(
            runtime=_mock_runtime(),
            answer_fn=_mock_answer_fn(),
            db_path=db,
        )
        assert asp.runtime is not None

    def test_custom_seed(self, tmp_path):
        db = str(tmp_path / "play.db")
        asp = AdversarialSelfPlay(
            runtime=_mock_runtime(),
            answer_fn=_mock_answer_fn(),
            db_path=db,
            seed=999,
        )
        assert asp.rng is not None


# ---------------------------------------------------------------------------
# Challenge generators
# ---------------------------------------------------------------------------

class TestChallengeGeneration:

    def test_hardness_challenge(self, tmp_path):
        db = str(tmp_path / "play.db")
        rt = _mock_runtime("Q: What does foo return?\nA: 42")
        asp = AdversarialSelfPlay(runtime=rt, answer_fn=_mock_answer_fn(), db_path=db)
        chunk = {"content": "def foo(): return 42", "source_path": "src/foo.py"}
        challenge = asp._gen_hardness_challenge(chunk, 0.5)
        assert challenge is not None
        assert challenge.game == "hardness"
        assert "foo" in challenge.question.lower() or len(challenge.question) > 0

    def test_trick_challenge(self, tmp_path):
        db = str(tmp_path / "play.db")
        rt = _mock_runtime("Q: What does baz_method do?\nTRAP: baz_method does not exist")
        asp = AdversarialSelfPlay(runtime=rt, answer_fn=_mock_answer_fn(), db_path=db)
        chunk = {"content": "def foo(): return 42", "source_path": "src/foo.py"}
        challenge = asp._gen_trick_challenge(chunk, 0.5)
        assert challenge is not None
        assert challenge.game == "trick"
        assert "REFUSE" in challenge.expected_behavior

    def test_ambiguity_challenge(self, tmp_path):
        db = str(tmp_path / "play.db")
        rt = _mock_runtime("Q: What does get do?\nAMBIGUITY: Could mean get_data or get_config")
        asp = AdversarialSelfPlay(runtime=rt, answer_fn=_mock_answer_fn(), db_path=db)
        chunk = {"content": "def get(): pass", "source_path": "src/get.py"}
        challenge = asp._gen_ambiguity_challenge(chunk, 0.5)
        assert challenge is not None
        assert challenge.game == "ambiguity"
        assert "CLARIFY" in challenge.expected_behavior

    def test_empty_chunk_returns_none(self, tmp_path):
        db = str(tmp_path / "play.db")
        asp = AdversarialSelfPlay(
            runtime=_mock_runtime(), answer_fn=_mock_answer_fn(), db_path=db,
        )
        assert asp._gen_hardness_challenge({"content": "", "source_path": ""}, 0.5) is None

    def test_no_q_line_returns_none(self, tmp_path):
        db = str(tmp_path / "play.db")
        rt = _mock_runtime("No question format here")
        asp = AdversarialSelfPlay(runtime=rt, answer_fn=_mock_answer_fn(), db_path=db)
        chunk = {"content": "def foo(): pass", "source_path": "foo.py"}
        assert asp._gen_hardness_challenge(chunk, 0.5) is None


# ---------------------------------------------------------------------------
# Judging
# ---------------------------------------------------------------------------

class TestJudging:

    def test_trick_refusal_detected(self, tmp_path):
        db = str(tmp_path / "play.db")
        asp = AdversarialSelfPlay(
            runtime=_mock_runtime(), answer_fn=_mock_answer_fn(), db_path=db,
        )
        challenge = Challenge(
            challenge_id="t1", game="trick", question="What does baz do?",
            expected_behavior="REFUSE: baz does not exist",
            difficulty=0.5, source_context="code",
        )
        # Answer contains refusal signal
        correct, failure, conf = asp._judge_trick(
            challenge, "I don't have information about baz. It does not exist in the codebase."
        )
        assert correct is True
        assert failure == "none"

    def test_trick_not_refused(self, tmp_path):
        db = str(tmp_path / "play.db")
        asp = AdversarialSelfPlay(
            runtime=_mock_runtime(), answer_fn=_mock_answer_fn(), db_path=db,
        )
        challenge = Challenge(
            challenge_id="t2", game="trick", question="What does baz do?",
            expected_behavior="REFUSE: baz does not exist",
            difficulty=0.5, source_context="code",
        )
        correct, failure, conf = asp._judge_trick(
            challenge, "baz processes data and returns a list of results."
        )
        assert correct is False
        assert failure == "missed_refusal"

    def test_ambiguity_clarification_detected(self, tmp_path):
        db = str(tmp_path / "play.db")
        asp = AdversarialSelfPlay(
            runtime=_mock_runtime(), answer_fn=_mock_answer_fn(), db_path=db,
        )
        challenge = Challenge(
            challenge_id="a1", game="ambiguity", question="What does get do?",
            expected_behavior="CLARIFY: multiple meanings",
            difficulty=0.5, source_context="code",
        )
        correct, failure, conf = asp._judge_ambiguity(
            challenge, "Could you clarify which 'get' function you mean?"
        )
        assert correct is True

    def test_ambiguity_not_clarified(self, tmp_path):
        db = str(tmp_path / "play.db")
        asp = AdversarialSelfPlay(
            runtime=_mock_runtime(), answer_fn=_mock_answer_fn(), db_path=db,
        )
        challenge = Challenge(
            challenge_id="a2", game="ambiguity", question="What does get do?",
            expected_behavior="CLARIFY: multiple meanings",
            difficulty=0.5, source_context="code",
        )
        correct, failure, conf = asp._judge_ambiguity(
            challenge, "get retrieves data from the database and returns it."
        )
        assert correct is False
        assert failure == "hallucination"


# ---------------------------------------------------------------------------
# _parse_score
# ---------------------------------------------------------------------------

class TestParseScore:

    def test_single_digit(self):
        assert AdversarialSelfPlay._parse_score("7") == 0.7

    def test_ten(self):
        assert AdversarialSelfPlay._parse_score("10") == 1.0

    def test_no_number(self):
        assert AdversarialSelfPlay._parse_score("great") == 0.5


# ---------------------------------------------------------------------------
# play_session
# ---------------------------------------------------------------------------

class TestPlaySession:

    def test_returns_result(self, tmp_path):
        db = str(tmp_path / "play.db")
        rt = _mock_runtime("Q: What does foo return?\nA: 42")
        # Judge always returns "7"
        rt.generate.return_value = "Q: test question?\nA: test answer"
        asp = AdversarialSelfPlay(
            runtime=rt,
            answer_fn=_mock_answer_fn("I don't know, that function does not exist."),
            db_path=db,
        )
        result = asp.play_session(_sample_chunks(), rounds_per_game=1)
        assert isinstance(result, SelfPlayResult)
        assert result.total_challenges >= 0
        assert result.total_ms >= 0

    def test_empty_chunks(self, tmp_path):
        db = str(tmp_path / "play.db")
        asp = AdversarialSelfPlay(
            runtime=_mock_runtime(), answer_fn=_mock_answer_fn(), db_path=db,
        )
        result = asp.play_session([], rounds_per_game=1)
        assert result.total_challenges == 0

    def test_failed_rounds_counted_and_logged(self, tmp_path, caplog):
        db = str(tmp_path / "play.db")
        asp = AdversarialSelfPlay(
            runtime=_mock_runtime(), answer_fn=_mock_answer_fn(), db_path=db,
        )
        asp._gen_hardness_challenge = MagicMock(side_effect=RuntimeError("hard fail"))
        asp._gen_trick_challenge = MagicMock(side_effect=RuntimeError("trick fail"))
        asp._gen_ambiguity_challenge = MagicMock(side_effect=RuntimeError("ambig fail"))

        with caplog.at_level(logging.WARNING, logger="core.adversarial_self_play"):
            result = asp.play_session(_sample_chunks(), rounds_per_game=1)

        assert result.total_challenges == 0
        assert result.failed_rounds == 3
        assert any("Game round failed" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# weakness_analysis
# ---------------------------------------------------------------------------

class TestWeaknessAnalysis:

    def test_empty_db(self, tmp_path):
        db = str(tmp_path / "play.db")
        asp = AdversarialSelfPlay(
            runtime=_mock_runtime(), answer_fn=_mock_answer_fn(), db_path=db,
        )
        analysis = asp.weakness_analysis()
        assert "weaknesses" in analysis
        assert analysis["weaknesses"] == []
