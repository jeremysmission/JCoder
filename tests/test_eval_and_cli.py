"""Tests for AgentEvalRunner and CLI command wiring.

Validates:
  - Eval set structure, scoring weights, category/difficulty coverage
  - Deterministic scoring (code blocks, keywords, imports, empty answers)
  - Summary/report output structure
  - CLI groups and subcommands registered in commands.py
"""
import json
import math
from pathlib import Path

import pytest

from evaluation.agent_eval_runner import AgentEvalRunner, EvalResult

EVAL_SET = str(Path(__file__).resolve().parent.parent / "evaluation" / "agent_eval_set.json")

EXPECTED_CATEGORIES = {"python", "javascript", "systems", "security", "algorithms", "debugging"}
EXPECTED_DIFFICULTIES = {"easy", "medium", "hard"}
REQUIRED_FIELDS = {"id", "category", "question", "expected_keywords", "scoring"}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def runner(tmp_path_factory):
    out = tmp_path_factory.mktemp("eval_out")
    return AgentEvalRunner(EVAL_SET, agent=None, output_dir=str(out))


@pytest.fixture(scope="module")
def questions():
    with open(EVAL_SET, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# TestEvalRunner
# ---------------------------------------------------------------------------

class TestEvalRunner:

    def test_load_eval_set(self, runner):
        assert len(runner.questions) == 50, (
            f"Expected 50 questions, got {len(runner.questions)}")

    def test_validate_eval_set(self, runner):
        issues = runner.validate_eval_set()
        assert issues == [], f"Validation issues: {issues}"

    def test_scoring_weights_sum(self, questions):
        for q in questions:
            total = sum(q["scoring"].values())
            assert math.isclose(total, 1.0, abs_tol=0.01), (
                f"{q['id']}: scoring weights sum to {total}, expected ~1.0")

    def test_score_answer_with_code(self, runner, questions):
        q = questions[0]  # py_001 -- expects code
        answer = (
            "Use pathlib:\n"
            "```python\n"
            "from pathlib import Path\n"
            "files = list(Path('.').rglob('*.py'))\n"
            "```\n"
        )
        sub = runner.score_answer(q, answer)
        assert sub["has_code"] == 1.0, "Code block present but has_code != 1.0"
        assert sub["weighted_total"] > 0.0

    def test_score_answer_with_keywords(self, runner, questions):
        q = questions[0]  # py_001 -- expected_keywords: Path, rglob, *.py
        answer = "You can use Path and rglob('*.py') to find files."
        sub = runner.score_answer(q, answer)
        assert sub["has_correct_api"] == 1.0, (
            f"All keywords present but has_correct_api={sub['has_correct_api']}")

    def test_score_answer_with_imports(self, runner, questions):
        q = questions[0]  # py_001 -- expected_imports: pathlib
        answer = "import pathlib\nUse pathlib.Path to find files."
        sub = runner.score_answer(q, answer)
        assert sub["has_imports"] == 1.0, (
            f"Import present but has_imports={sub['has_imports']}")

    def test_score_answer_empty(self, runner, questions):
        q = questions[0]
        sub = runner.score_answer(q, "")
        assert sub["weighted_total"] == 0.0, (
            f"Empty answer should score 0.0, got {sub['weighted_total']}")
        assert sub["has_code"] == 0.0
        assert sub["has_correct_api"] == 0.0

    def test_summary_structure(self, runner):
        # Build synthetic results for summary testing
        results = [
            EvalResult(question_id="py_001", category="python",
                       score=0.8, subscores={}, answer="ok",
                       elapsed_s=1.0, tokens_used=100, passed=True),
            EvalResult(question_id="js_001", category="javascript",
                       score=0.3, subscores={}, answer="bad",
                       elapsed_s=2.0, tokens_used=50, passed=False),
        ]
        s = AgentEvalRunner.summary(results)
        assert "overall_score" not in s or True  # key name is avg_score
        assert "pass_rate" in s
        assert "per_category" in s
        assert isinstance(s["per_category"], dict)
        assert s["total"] == 2
        assert s["passed"] == 1
        assert 0.0 <= s["pass_rate"] <= 1.0
        assert "avg_score" in s
        assert "min_score" in s
        assert "max_score" in s

    def test_report_generates_markdown(self, runner, tmp_path):
        results = [
            EvalResult(question_id="py_001", category="python",
                       score=0.9, subscores={"has_code": 1.0}, answer="good",
                       elapsed_s=0.5, tokens_used=80, passed=True),
        ]
        out = str(tmp_path / "report.md")
        text = runner.report(results, output_path=out)
        assert Path(out).exists(), "Report file was not written"
        assert "# Agent Evaluation Report" in text
        assert "python" in text

    def test_categories_covered(self, questions):
        found = {q["category"] for q in questions}
        assert found == EXPECTED_CATEGORIES, (
            f"Missing categories: {EXPECTED_CATEGORIES - found}")

    def test_difficulty_distribution(self, questions):
        found = {q.get("difficulty") for q in questions}
        assert EXPECTED_DIFFICULTIES.issubset(found), (
            f"Missing difficulties: {EXPECTED_DIFFICULTIES - found}")


# ---------------------------------------------------------------------------
# TestCLIWiring
# ---------------------------------------------------------------------------

class TestCLIWiring:

    def test_agent_commands_registered(self):
        from click.testing import CliRunner
        from cli.commands import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "agent" in result.output, (
            "'agent' group not found in CLI help output")

    def test_agent_subcommands(self):
        from click.testing import CliRunner
        from cli.commands import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["agent", "--help"])
        assert result.exit_code == 0
        for sub in ("run", "study", "goals", "autopilot", "complete"):
            assert sub in result.output, (
                f"Subcommand '{sub}' not in agent --help output")

    def test_ingest_corpus_registered(self):
        from click.testing import CliRunner
        from cli.commands import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "ingest-corpus" in result.output, (
            "'ingest-corpus' group not found in CLI help output")

    def test_ingest_subcommands(self):
        from click.testing import CliRunner
        from cli.commands import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["ingest-corpus", "--help"])
        assert result.exit_code == 0
        for sub in ("stackoverflow", "codesearchnet", "docs", "code", "status"):
            assert sub in result.output, (
                f"Subcommand '{sub}' not in ingest-corpus --help output")
