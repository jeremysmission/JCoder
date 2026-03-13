"""
Adversarial Self-Play (Sol-Ver Pattern)
-----------------------------------------
The system plays BOTH sides: Generator creates challenging questions,
Verifier tries to answer them. Failures become training signal.

Based on:
- Sol-Ver (ACL 2025): Solution-Verification adversarial generation
- Self-Play Fine-Tuning (SPIN, ICML 2024): Iterative self-competition
- Meta-Rewarding (2024): LLM judges its own outputs to create reward signal

Three adversarial games:

Game 1: Question Hardness Escalation
    Generator creates progressively harder questions about the codebase.
    Verifier tries to answer using retrieval. Questions that stump the
    Verifier identify retrieval/generation weaknesses.

Game 2: Trick Question Detection
    Generator creates questions that LOOK answerable but aren't
    (referencing nonexistent functions, wrong file paths, etc.)
    Verifier must correctly identify and refuse trick questions.

Game 3: Ambiguity Challenge
    Generator creates deliberately ambiguous questions.
    Verifier must request clarification rather than guessing.

Each game produces structured training data that feeds back into
the experience replay store and prompt evolver.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import re
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from core.runtime import Runtime

logger = logging.getLogger(__name__)


@dataclass
class Challenge:
    """A single adversarial challenge."""
    challenge_id: str
    game: str  # "hardness" | "trick" | "ambiguity"
    question: str
    expected_behavior: str  # what the Verifier SHOULD do
    difficulty: float  # 0.0-1.0
    source_context: str  # code/chunk that inspired the challenge
    generated_at: float = 0.0


@dataclass
class ChallengeOutcome:
    """Result of the Verifier attempting a challenge."""
    challenge_id: str
    verifier_answer: str
    correct_behavior: bool  # did the Verifier do the right thing?
    confidence: float
    failure_mode: str  # "none" | "hallucination" | "missed_refusal" | "wrong_answer"
    latency_ms: float = 0.0


@dataclass
class SelfPlayResult:
    """Summary of a self-play session."""
    total_challenges: int
    correct: int
    accuracy: float
    weakness_report: Dict[str, int]  # failure_mode -> count
    hardest_failures: List[Dict[str, str]]  # worst failures for review
    failed_rounds: int = 0
    total_ms: float = 0.0


# ---------------------------------------------------------------------------
# Challenge generators (one per game)
# ---------------------------------------------------------------------------

_HARDNESS_PROMPT = (
    "You are generating increasingly difficult technical questions about "
    "a codebase. Given this code snippet, generate a question that requires "
    "deep understanding of the code's logic, edge cases, or interactions "
    "with other parts of the system.\n\n"
    "Difficulty target: {difficulty}/10\n\n"
    "Code from {source_path}:\n```\n{code}\n```\n\n"
    "Generate ONE question. Format: Q: <your question>\n"
    "Then on a new line: A: <the correct answer based on the code>"
)

_TRICK_PROMPT = (
    "You are generating a trick question about a codebase. The question "
    "should reference something that does NOT exist in the code but sounds "
    "plausible. A good system should recognize this and refuse to answer.\n\n"
    "Real code from {source_path}:\n```\n{code}\n```\n\n"
    "Generate a plausible-sounding question about something NOT in this code.\n"
    "Format: Q: <trick question>\n"
    "Then: TRAP: <what doesn't actually exist>"
)

_AMBIGUITY_PROMPT = (
    "You are generating a deliberately ambiguous question about code. "
    "The question should have multiple valid interpretations. A good system "
    "should ask for clarification rather than guessing.\n\n"
    "Code from {source_path}:\n```\n{code}\n```\n\n"
    "Generate an ambiguous question where at least 2 interpretations exist.\n"
    "Format: Q: <ambiguous question>\n"
    "Then: AMBIGUITY: <explain the multiple interpretations>"
)

# Verifier assessment prompts
_VERIFY_ANSWER_PROMPT = (
    "Evaluate this answer to a code question.\n\n"
    "Question: {question}\n"
    "Expected answer: {expected}\n"
    "Actual answer: {actual}\n\n"
    "Is the actual answer correct? Rate 0-10. Output ONLY a number."
)

_VERIFY_REFUSAL_PROMPT = (
    "Did this response correctly refuse to answer a trick question?\n\n"
    "Trick question: {question}\n"
    "Trap: {trap}\n"
    "Response: {actual}\n\n"
    "Rate 0-10 how well the response avoided the trap. "
    "10 = correctly refused, 0 = fell for it. Output ONLY a number."
)


class AdversarialSelfPlay:
    """
    Runs adversarial self-play games to discover system weaknesses.

    The Generator creates challenges, the Verifier (the actual RAG pipeline)
    attempts them, and outcomes feed back into training.
    """

    def __init__(
        self,
        runtime: Runtime,
        answer_fn: Callable[[str], str],
        db_path: str = "_self_play/games.db",
        seed: int = 42,
    ):
        """
        Args:
            runtime: LLM for generating challenges and judging
            answer_fn: The actual RAG pipeline answer function
                       answer_fn(question) -> answer_text
            db_path: SQLite for game history
        """
        self.runtime = runtime
        self.answer_fn = answer_fn
        self.rng = random.Random(seed)
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self._db_path))

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS challenges (
                    challenge_id TEXT PRIMARY KEY,
                    game TEXT,
                    question TEXT,
                    expected_behavior TEXT,
                    difficulty REAL,
                    source_context TEXT,
                    generated_at REAL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS outcomes (
                    challenge_id TEXT PRIMARY KEY,
                    verifier_answer TEXT,
                    correct_behavior INTEGER,
                    confidence REAL,
                    failure_mode TEXT,
                    latency_ms REAL,
                    evaluated_at REAL
                )
            """)
            conn.commit()

    def play_session(
        self,
        code_chunks: List[Dict[str, str]],
        rounds_per_game: int = 5,
        difficulty_start: float = 0.3,
        difficulty_step: float = 0.15,
    ) -> SelfPlayResult:
        """
        Run a full self-play session across all three games.

        Args:
            code_chunks: List of {"content": ..., "source_path": ...}
            rounds_per_game: Challenges per game type
            difficulty_start: Starting difficulty (0.0-1.0)
            difficulty_step: Difficulty increase per round
        """
        t0 = time.time()
        outcomes: List[ChallengeOutcome] = []
        failed_rounds = 0

        games = [
            ("hardness", self._gen_hardness_challenge),
            ("trick", self._gen_trick_challenge),
            ("ambiguity", self._gen_ambiguity_challenge),
        ]

        for game_name, gen_fn in games:
            difficulty = difficulty_start
            for _ in range(rounds_per_game):
                chunk = self.rng.choice(code_chunks) if code_chunks else {}
                try:
                    challenge = gen_fn(chunk, min(1.0, difficulty))
                    if challenge:
                        outcome = self._evaluate_challenge(challenge)
                        outcomes.append(outcome)
                        self._persist(challenge, outcome)

                        # Adaptive difficulty: increase on success, decrease on failure
                        if outcome.correct_behavior:
                            difficulty += difficulty_step
                        else:
                            difficulty = max(0.1, difficulty - difficulty_step * 0.5)
                except Exception as exc:
                    failed_rounds += 1
                    logger.warning(
                        "Game round failed (%s, difficulty=%.2f): %s",
                        game_name,
                        difficulty,
                        exc,
                    )

        # Build weakness report
        weakness = {}
        hardest_failures = []
        correct = 0

        for oc in outcomes:
            if oc.correct_behavior:
                correct += 1
            else:
                weakness[oc.failure_mode] = weakness.get(oc.failure_mode, 0) + 1
                if len(hardest_failures) < 5:
                    hardest_failures.append({
                        "challenge_id": oc.challenge_id,
                        "answer_snippet": oc.verifier_answer[:200],
                        "failure_mode": oc.failure_mode,
                    })

        total = len(outcomes)
        return SelfPlayResult(
            total_challenges=total,
            correct=correct,
            accuracy=correct / total if total > 0 else 0.0,
            weakness_report=weakness,
            hardest_failures=hardest_failures,
            failed_rounds=failed_rounds,
            total_ms=(time.time() - t0) * 1000,
        )

    def _gen_hardness_challenge(
        self, chunk: Dict[str, str], difficulty: float
    ) -> Optional[Challenge]:
        """Generate a hard factual question about code."""
        code = chunk.get("content", "")[:1200]
        source = chunk.get("source_path", "unknown")
        if not code:
            return None

        prompt = _HARDNESS_PROMPT.format(
            difficulty=int(difficulty * 10),
            source_path=source,
            code=code,
        )
        raw = self.runtime.generate(
            question=prompt, context_chunks=[],
            system_prompt="Generate a code question.",
            temperature=0.7 + difficulty * 0.2,
            max_tokens=256,
        )

        # Parse Q: and A: lines
        q_match = re.search(r"Q:\s*(.+?)(?:\n|$)", raw)
        a_match = re.search(r"A:\s*(.+?)(?:\n|$)", raw, re.DOTALL)
        if not q_match:
            return None

        question = q_match.group(1).strip()
        expected = a_match.group(1).strip() if a_match else ""
        cid = hashlib.sha256(f"{question}:{time.time()}".encode()).hexdigest()[:12]

        return Challenge(
            challenge_id=cid,
            game="hardness",
            question=question,
            expected_behavior=expected,
            difficulty=difficulty,
            source_context=code[:500],
            generated_at=time.time(),
        )

    def _gen_trick_challenge(
        self, chunk: Dict[str, str], difficulty: float
    ) -> Optional[Challenge]:
        """Generate a trick question (unanswerable)."""
        code = chunk.get("content", "")[:1200]
        source = chunk.get("source_path", "unknown")
        if not code:
            return None

        prompt = _TRICK_PROMPT.format(source_path=source, code=code)
        raw = self.runtime.generate(
            question=prompt, context_chunks=[],
            system_prompt="Generate a trick question about code.",
            temperature=0.8,
            max_tokens=256,
        )

        q_match = re.search(r"Q:\s*(.+?)(?:\n|$)", raw)
        trap_match = re.search(r"TRAP:\s*(.+?)(?:\n|$)", raw)
        if not q_match:
            return None

        question = q_match.group(1).strip()
        trap = trap_match.group(1).strip() if trap_match else "unknown trap"
        cid = hashlib.sha256(f"trick:{question}".encode()).hexdigest()[:12]

        return Challenge(
            challenge_id=cid,
            game="trick",
            question=question,
            expected_behavior=f"REFUSE: {trap}",
            difficulty=difficulty,
            source_context=code[:500],
            generated_at=time.time(),
        )

    def _gen_ambiguity_challenge(
        self, chunk: Dict[str, str], difficulty: float
    ) -> Optional[Challenge]:
        """Generate an ambiguous question."""
        code = chunk.get("content", "")[:1200]
        source = chunk.get("source_path", "unknown")
        if not code:
            return None

        prompt = _AMBIGUITY_PROMPT.format(source_path=source, code=code)
        raw = self.runtime.generate(
            question=prompt, context_chunks=[],
            system_prompt="Generate an ambiguous question about code.",
            temperature=0.8,
            max_tokens=256,
        )

        q_match = re.search(r"Q:\s*(.+?)(?:\n|$)", raw)
        amb_match = re.search(r"AMBIGUITY:\s*(.+?)(?:\n|$)", raw, re.DOTALL)
        if not q_match:
            return None

        question = q_match.group(1).strip()
        ambiguity = amb_match.group(1).strip() if amb_match else ""
        cid = hashlib.sha256(f"ambig:{question}".encode()).hexdigest()[:12]

        return Challenge(
            challenge_id=cid,
            game="ambiguity",
            question=question,
            expected_behavior=f"CLARIFY: {ambiguity}",
            difficulty=difficulty,
            source_context=code[:500],
            generated_at=time.time(),
        )

    def _evaluate_challenge(self, challenge: Challenge) -> ChallengeOutcome:
        """Run the actual RAG pipeline on a challenge and judge the result."""
        t0 = time.time()
        try:
            answer = self.answer_fn(challenge.question)
        except Exception as e:
            answer = f"ERROR: {e}"
        latency = (time.time() - t0) * 1000

        # Judge based on game type
        if challenge.game == "hardness":
            correct, failure, confidence = self._judge_hardness(challenge, answer)
        elif challenge.game == "trick":
            correct, failure, confidence = self._judge_trick(challenge, answer)
        elif challenge.game == "ambiguity":
            correct, failure, confidence = self._judge_ambiguity(challenge, answer)
        else:
            correct, failure, confidence = False, "unknown_game", 0.0

        return ChallengeOutcome(
            challenge_id=challenge.challenge_id,
            verifier_answer=answer[:1000],
            correct_behavior=correct,
            confidence=confidence,
            failure_mode=failure,
            latency_ms=latency,
        )

    def _judge_hardness(
        self, challenge: Challenge, answer: str
    ) -> Tuple[bool, str, float]:
        """Judge a hardness challenge answer."""
        if not challenge.expected_behavior:
            return True, "none", 0.5

        prompt = _VERIFY_ANSWER_PROMPT.format(
            question=challenge.question,
            expected=challenge.expected_behavior[:300],
            actual=answer[:300],
        )
        raw = self.runtime.generate(
            question=prompt, context_chunks=[],
            system_prompt="You are an answer evaluator.",
            temperature=0.0, max_tokens=8,
        )
        score = self._parse_score(raw)
        correct = score >= 0.5
        failure = "none" if correct else "wrong_answer"
        return correct, failure, score

    def _judge_trick(
        self, challenge: Challenge, answer: str
    ) -> Tuple[bool, str, float]:
        """Judge a trick question response."""
        # Check for refusal indicators
        refusal_signals = [
            "not found", "doesn't exist", "does not exist",
            "no such", "cannot find", "don't have", "not available",
            "no information", "unclear", "cannot determine",
            "i don't know", "not in the", "not present",
        ]
        answer_lower = answer.lower()
        refused = any(s in answer_lower for s in refusal_signals)

        if refused:
            return True, "none", 0.8

        # Fell for the trick
        return False, "missed_refusal", 0.2

    def _judge_ambiguity(
        self, challenge: Challenge, answer: str
    ) -> Tuple[bool, str, float]:
        """Judge an ambiguity challenge response."""
        clarification_signals = [
            "could you clarify", "do you mean", "ambiguous",
            "multiple interpretations", "please specify",
            "could refer to", "which one", "not clear",
            "depends on", "several possibilities",
        ]
        answer_lower = answer.lower()
        sought_clarification = any(s in answer_lower for s in clarification_signals)

        if sought_clarification:
            return True, "none", 0.8

        # Guessed instead of clarifying
        return False, "hallucination", 0.3

    @staticmethod
    def _parse_score(text: str) -> float:
        """Extract 0-10 score from LLM output."""
        m = re.search(r"\b(10|[0-9])\b", text.strip())
        return int(m.group(1)) / 10.0 if m else 0.5

    def _persist(self, challenge: Challenge, outcome: ChallengeOutcome) -> None:
        try:
            with self._connect() as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO challenges
                    (challenge_id, game, question, expected_behavior,
                     difficulty, source_context, generated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    challenge.challenge_id, challenge.game,
                    challenge.question, challenge.expected_behavior,
                    challenge.difficulty, challenge.source_context[:500],
                    challenge.generated_at,
                ))
                conn.execute("""
                    INSERT OR REPLACE INTO outcomes
                    (challenge_id, verifier_answer, correct_behavior,
                     confidence, failure_mode, latency_ms, evaluated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    outcome.challenge_id, outcome.verifier_answer[:500],
                    1 if outcome.correct_behavior else 0,
                    outcome.confidence, outcome.failure_mode,
                    outcome.latency_ms, time.time(),
                ))
                conn.commit()
        except Exception as exc:
            logger.warning(
                "Failed to persist challenge %s: %s",
                challenge.challenge_id,
                exc,
            )

    def weakness_analysis(self, limit: int = 50) -> Dict[str, Any]:
        """Analyze historical weaknesses from all past games."""
        with self._connect() as conn:
            cur = conn.execute("""
                SELECT c.game, o.failure_mode, COUNT(*) as cnt,
                       AVG(o.confidence) as avg_conf
                FROM outcomes o JOIN challenges c ON o.challenge_id = c.challenge_id
                WHERE o.correct_behavior = 0
                GROUP BY c.game, o.failure_mode
                ORDER BY cnt DESC LIMIT ?
            """, (limit,))
            rows = cur.fetchall()

        return {
            "weaknesses": [
                {"game": r[0], "failure_mode": r[1],
                 "count": r[2], "avg_confidence": round(r[3], 3)}
                for r in rows
            ]
        }
