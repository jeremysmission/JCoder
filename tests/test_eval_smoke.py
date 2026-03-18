"""Fast smoke tests for the eval infrastructure.

Validates scoring logic, question loading, and edge cases
without requiring Ollama, GPU, or any live LLM backend.
Target: <5 seconds wall-clock in CI.
"""
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List
from unittest.mock import MagicMock

import pytest

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.agent_eval_runner import AgentEvalRunner, EvalResult

EVAL_SET_PATH = str(PROJECT_ROOT / "evaluation" / "agent_eval_set_200.json")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def runner(tmp_path_factory):
    """Runner pointed at a temp output dir so we don't pollute real results."""
    out = tmp_path_factory.mktemp("eval_results")
    return AgentEvalRunner(EVAL_SET_PATH, agent=None, output_dir=str(out))


@pytest.fixture(scope="module")
def questions():
    with open(EVAL_SET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def sample_20(questions):
    rng = random.Random(42)
    return rng.sample(questions, min(20, len(questions)))


# ---------------------------------------------------------------------------
# Canned answers for deterministic scoring
# ---------------------------------------------------------------------------

PERFECT_ANSWER = """\
You can use `pathlib.Path.rglob` to recursively find files.

```python
from pathlib import Path

for f in Path(".").rglob("*.py"):
    print(f)
```

See https://docs.python.org/3/library/pathlib.html for details.
"""

GOOD_ANSWER_NO_CITE = """\
Here's an example:

```python
from pathlib import Path
list(Path(".").rglob("*.py"))
```
"""

EMPTY_ANSWER = ""

VERY_LONG_ANSWER = "This is filler. " * 5000  # ~80 KB of text, no code

IRRELEVANT_ANSWER = "I like turtles. The weather is nice today. 42 is the answer."


# ---------------------------------------------------------------------------
# 1. Golden question loading and validation
# ---------------------------------------------------------------------------

class TestEvalSetLoading:
    def test_loads_questions(self, runner):
        assert len(runner.questions) > 0, "Eval set should not be empty"

    def test_all_ids_unique(self, runner):
        ids = [q["id"] for q in runner.questions]
        assert len(ids) == len(set(ids)), "Duplicate question IDs found"

    def test_question_map_matches(self, runner):
        assert len(runner.question_map) == len(runner.questions)

    def test_required_fields_present(self, questions):
        required = {"id", "category", "question", "expected_keywords", "scoring"}
        for q in questions:
            missing = required - set(q.keys())
            assert not missing, f"{q.get('id', '?')}: missing {missing}"

    def test_scoring_weights_sum_to_one(self, questions):
        for q in questions:
            wt = sum(q["scoring"].values())
            assert abs(wt - 1.0) < 0.02, (
                f"{q['id']}: scoring weights sum to {wt:.4f}"
            )

    def test_validate_eval_set_clean(self, runner):
        issues = runner.validate_eval_set()
        assert issues == [], f"Validation issues: {issues}"

    def test_categories_non_empty(self, questions):
        cats = {q["category"] for q in questions}
        assert len(cats) >= 2, "Expected multiple categories"

    def test_difficulties_present(self, questions):
        diffs = {q.get("difficulty") for q in questions}
        assert diffs - {None}, "Expected at least one difficulty value"


# ---------------------------------------------------------------------------
# 2. Scoring logic -- 20-question sample with canned responses
# ---------------------------------------------------------------------------

class TestScoringWith20Questions:
    """Mock the LLM and score each of the 20 sampled questions."""

    def _make_canned_answer(self, q: Dict) -> str:
        """Build a plausible canned answer that hits the expected keywords."""
        keywords = q.get("expected_keywords", [])
        imports = q.get("expected_imports", [])
        import_lines = "\n".join(f"import {m}" for m in imports) if imports else ""
        keyword_lines = " ".join(keywords)
        return (
            f"Here is how you do it using {keyword_lines}.\n\n"
            f"```python\n{import_lines}\n"
            f"# example using {', '.join(keywords)}\n"
            f"print('done')\n```\n\n"
            f"Reference: https://docs.python.org/3/library/index.html\n"
        )

    def test_all_20_produce_valid_scores(self, runner, sample_20):
        for q in sample_20:
            answer = self._make_canned_answer(q)
            sub = runner.score_answer(q, answer)
            total = sub["weighted_total"]
            assert 0.0 <= total <= 1.0, (
                f"{q['id']}: score {total} out of range"
            )
            for key in ("has_code", "has_correct_api", "has_imports",
                        "is_runnable", "cites_source", "weighted_total"):
                assert key in sub, f"{q['id']}: missing subscore '{key}'"

    def test_perfect_answers_score_high(self, runner, sample_20):
        high_count = 0
        for q in sample_20:
            answer = self._make_canned_answer(q)
            sub = runner.score_answer(q, answer)
            if sub["weighted_total"] >= 0.5:
                high_count += 1
        assert high_count >= len(sample_20) * 0.7, (
            f"Only {high_count}/{len(sample_20)} canned answers scored >= 0.5"
        )

    def test_subscores_are_floats(self, runner, sample_20):
        for q in sample_20:
            answer = self._make_canned_answer(q)
            sub = runner.score_answer(q, answer)
            for k, v in sub.items():
                assert isinstance(v, (int, float)), (
                    f"{q['id']}.{k}: expected numeric, got {type(v)}"
                )


# ---------------------------------------------------------------------------
# 3. Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def _first_question(self, runner) -> Dict:
        return runner.questions[0]

    def test_empty_answer(self, runner):
        q = self._first_question(runner)
        sub = runner.score_answer(q, EMPTY_ANSWER)
        assert sub["weighted_total"] == 0.0, "Empty answer should score 0"
        assert sub["has_code"] == 0.0

    def test_very_long_answer(self, runner):
        q = self._first_question(runner)
        sub = runner.score_answer(q, VERY_LONG_ANSWER)
        total = sub["weighted_total"]
        assert 0.0 <= total <= 1.0, f"Long answer score {total} out of range"
        assert sub["has_code"] == 0.0, "Filler text should have no code block"

    def test_irrelevant_answer(self, runner):
        q = self._first_question(runner)
        sub = runner.score_answer(q, IRRELEVANT_ANSWER)
        assert sub["weighted_total"] < 0.5, (
            "Irrelevant answer should score below 0.5"
        )
        assert sub["has_correct_api"] < 1.0

    def test_perfect_answer_scores_high(self, runner):
        q = self._first_question(runner)
        sub = runner.score_answer(q, PERFECT_ANSWER)
        assert sub["weighted_total"] >= 0.8, (
            f"Perfect answer scored only {sub['weighted_total']}"
        )
        assert sub["has_code"] == 1.0
        assert sub["cites_source"] == 1.0

    def test_good_answer_no_citation(self, runner):
        q = self._first_question(runner)
        sub = runner.score_answer(q, GOOD_ANSWER_NO_CITE)
        assert sub["has_code"] == 1.0
        # cites_source may still be 1.0 because _cites_source matches file
        # extensions too; just check the total is reasonable
        assert sub["weighted_total"] >= 0.5

    def test_answer_with_bad_python(self, runner):
        q = self._first_question(runner)
        bad_code_answer = "```python\ndef broken(\n```"
        sub = runner.score_answer(q, bad_code_answer)
        assert sub["has_code"] == 1.0, "Code block present even if invalid"
        assert sub["is_runnable"] == 0.0, "Invalid Python should not be runnable"


# ---------------------------------------------------------------------------
# 4. EvalResult dataclass and run_single without agent
# ---------------------------------------------------------------------------

class TestRunSingleNoAgent:
    def test_no_agent_returns_zero(self, runner):
        qid = runner.questions[0]["id"]
        result = runner.run_single(qid)
        assert isinstance(result, EvalResult)
        assert result.score == 0.0
        assert result.passed is False
        assert result.category == runner.questions[0]["category"]

    def test_unknown_id_raises(self, runner):
        with pytest.raises(KeyError, match="Unknown question ID"):
            runner.run_single("NONEXISTENT_999")

    def test_eval_result_serializable(self, runner):
        qid = runner.questions[0]["id"]
        result = runner.run_single(qid)
        d = asdict(result)
        assert isinstance(d, dict)
        roundtrip = json.dumps(d)
        assert isinstance(roundtrip, str)


# ---------------------------------------------------------------------------
# 5. Mock agent integration
# ---------------------------------------------------------------------------

class TestMockAgent:
    def _mock_agent(self, answer_text: str, tokens: int = 100):
        agent = MagicMock()
        resp = MagicMock()
        resp.answer = answer_text
        resp.tokens_used = tokens
        agent.run.return_value = resp
        return agent

    def test_mock_agent_run_single(self, tmp_path):
        agent = self._mock_agent(PERFECT_ANSWER, tokens=150)
        r = AgentEvalRunner(EVAL_SET_PATH, agent=agent,
                            output_dir=str(tmp_path))
        qid = r.questions[0]["id"]
        result = r.run_single(qid)
        assert result.score > 0.0
        assert result.tokens_used == 150
        assert result.answer == PERFECT_ANSWER
        agent.run.assert_called_once()

    def test_mock_agent_run_all_max3(self, tmp_path):
        agent = self._mock_agent(PERFECT_ANSWER)
        r = AgentEvalRunner(EVAL_SET_PATH, agent=agent,
                            output_dir=str(tmp_path))
        results = r.run_all(max_questions=3, resume=False)
        assert len(results) == 3
        assert all(isinstance(x, EvalResult) for x in results)
        assert agent.run.call_count == 3


# ---------------------------------------------------------------------------
# 6. Summary / report helpers
# ---------------------------------------------------------------------------

class TestSummaryReport:
    def _make_results(self) -> List[EvalResult]:
        return [
            EvalResult("q1", "python", 0.9, {}, "ans", 1.0, 100, True),
            EvalResult("q2", "python", 0.3, {}, "ans", 0.5, 80, False),
            EvalResult("q3", "git", 0.7, {}, "ans", 0.8, 120, True),
        ]

    def test_summary_structure(self):
        s = AgentEvalRunner.summary(self._make_results())
        assert s["total"] == 3
        assert s["passed"] == 2
        assert 0.0 <= s["pass_rate"] <= 1.0
        assert "per_category" in s
        assert "python" in s["per_category"]

    def test_summary_empty_list(self):
        s = AgentEvalRunner.summary([])
        assert s == {"total": 0}

    def test_report_writes_file(self, runner, tmp_path):
        results = self._make_results()
        path = str(tmp_path / "report.md")
        text = runner.report(results, output_path=path)
        assert Path(path).exists()
        assert "Agent Evaluation Report" in text
        assert "python" in text


# ---------------------------------------------------------------------------
# 7. Static helper methods
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_has_code_block(self):
        assert AgentEvalRunner._has_code_block("```python\nprint()```")
        assert not AgentEvalRunner._has_code_block("no code here")

    def test_token_ratio(self):
        assert AgentEvalRunner._token_ratio("Path rglob *.py", ["Path", "rglob"]) == 1.0
        assert AgentEvalRunner._token_ratio("nothing", ["Path", "rglob"]) == 0.0
        assert AgentEvalRunner._token_ratio("anything", []) == 1.0

    def test_extract_python(self):
        text = "Hi\n```python\nprint('hello')\n```\nBye"
        assert "print" in AgentEvalRunner._extract_python(text)
        assert AgentEvalRunner._extract_python("no code") == ""

    def test_is_valid_python(self):
        assert AgentEvalRunner._is_valid_python("x = 1")
        assert not AgentEvalRunner._is_valid_python("def (")
        assert not AgentEvalRunner._is_valid_python("")

    def test_cites_source(self):
        assert AgentEvalRunner._cites_source("see https://example.com")
        assert AgentEvalRunner._cites_source("file utils.py has it")
        assert not AgentEvalRunner._cites_source("no links here at all")
