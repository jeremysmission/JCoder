"""
Tests for evaluation.agent_eval_runner -- scoring, validation, reporting.
No live agent required; exercises scoring logic and data integrity.
"""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import pytest

from evaluation.agent_eval_runner import AgentEvalRunner, EvalResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_question(
    qid: str = "test_001",
    category: str = "python",
    question: str = "How do I read a file?",
    keywords: List[str] = None,
    imports: List[str] = None,
    scoring: Dict[str, float] = None,
) -> Dict:
    return {
        "id": qid,
        "category": category,
        "question": question,
        "expected_keywords": keywords or ["open", "read"],
        "expected_imports": imports or ["pathlib"],
        "expected_code": True,
        "scoring": scoring or {
            "has_code": 0.3,
            "has_correct_api": 0.3,
            "has_imports": 0.1,
            "is_runnable": 0.2,
            "cites_source": 0.1,
        },
    }


def _write_eval_set(tmp_path: Path, questions: List[Dict]) -> str:
    path = str(tmp_path / "eval_set.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(questions, f)
    return path


@pytest.fixture()
def runner_one(tmp_path):
    """Runner with one simple question, no agent."""
    path = _write_eval_set(tmp_path, [_make_question()])
    return AgentEvalRunner(path, agent=None, output_dir=str(tmp_path / "results"))


@pytest.fixture()
def runner_multi(tmp_path):
    """Runner with 5 questions across 3 categories."""
    qs = [
        _make_question("py_001", "python", "How to read a file?", ["open", "read"], ["pathlib"]),
        _make_question("py_002", "python", "How to write JSON?", ["json", "dumps"], ["json"]),
        _make_question("js_001", "javascript", "Explain closures", ["closure", "scope"], []),
        _make_question("sec_001", "security", "SQL injection", ["parameterized", "bind"], []),
        _make_question("alg_001", "algorithms", "Binary search", ["mid", "low", "high"], []),
    ]
    path = _write_eval_set(tmp_path, qs)
    return AgentEvalRunner(path, agent=None, output_dir=str(tmp_path / "results"))


# ---------------------------------------------------------------------------
# Scoring logic
# ---------------------------------------------------------------------------

class TestScoring:

    def test_perfect_answer(self, runner_one):
        """Answer with code block, all keywords, imports, valid python, and URL scores 1.0."""
        answer = (
            "Use pathlib to read:\n"
            "```python\n"
            "from pathlib import Path\n"
            "text = Path('file.txt').read_text()\n"
            "# open and read\n"
            "```\n"
            "See https://docs.python.org/3/library/pathlib.html"
        )
        q = runner_one.questions[0]
        sub = runner_one.score_answer(q, answer)
        assert sub["has_code"] == 1.0
        assert sub["has_correct_api"] == 1.0
        assert sub["has_imports"] == 1.0
        assert sub["is_runnable"] == 1.0
        assert sub["cites_source"] == 1.0
        assert sub["weighted_total"] == 1.0

    def test_no_code_block(self, runner_one):
        """Answer without code block scores 0 on has_code and is_runnable."""
        answer = "Just use open() and read() the file. Import pathlib."
        q = runner_one.questions[0]
        sub = runner_one.score_answer(q, answer)
        assert sub["has_code"] == 0.0
        assert sub["has_correct_api"] == 1.0  # keywords present
        assert sub["has_imports"] == 1.0       # import mentioned

    def test_partial_keywords(self, runner_one):
        """Answer with only half the keywords gets 0.5 on has_correct_api."""
        answer = "```python\nf = open('file.txt')\n```"
        q = runner_one.questions[0]
        sub = runner_one.score_answer(q, answer)
        assert sub["has_correct_api"] == 0.5  # "open" found, "read" missing

    def test_invalid_python(self, runner_one):
        """Code block with syntax errors scores 0 on is_runnable."""
        answer = "```python\ndef read(:\n  pass\n```"
        q = runner_one.questions[0]
        sub = runner_one.score_answer(q, answer)
        assert sub["has_code"] == 1.0
        assert sub["is_runnable"] == 0.0

    def test_empty_answer(self, runner_one):
        """Empty answer scores 0 on everything."""
        q = runner_one.questions[0]
        sub = runner_one.score_answer(q, "")
        assert sub["has_code"] == 0.0
        assert sub["has_correct_api"] == 0.0
        assert sub["has_imports"] == 0.0
        assert sub["cites_source"] == 0.0
        assert sub["weighted_total"] == 0.0

    def test_citation_detects_url(self, runner_one):
        """URL in answer counts as citation."""
        answer = "See https://example.com for details"
        q = runner_one.questions[0]
        sub = runner_one.score_answer(q, answer)
        assert sub["cites_source"] == 1.0

    def test_citation_detects_filepath(self, runner_one):
        """File path in answer counts as citation."""
        answer = "Check src/utils/reader.py for the implementation."
        q = runner_one.questions[0]
        sub = runner_one.score_answer(q, answer)
        assert sub["cites_source"] == 1.0

    def test_weights_sum_correctly(self, runner_one):
        """Weighted total uses scoring weights from question."""
        answer = (
            "```python\n"
            "from pathlib import Path\n"
            "text = open('f').read()\n"
            "```\n"
            "Ref: https://docs.python.org"
        )
        q = runner_one.questions[0]
        sub = runner_one.score_answer(q, answer)
        # All subscores are 1.0 here
        assert abs(sub["weighted_total"] - 1.0) < 0.01


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestValidation:

    def test_valid_set_no_issues(self, runner_one):
        issues = runner_one.validate_eval_set()
        assert issues == []

    def test_missing_field_detected(self, tmp_path):
        bad = [{"id": "x", "category": "py", "question": "Q"}]
        path = _write_eval_set(tmp_path, bad)
        runner = AgentEvalRunner(path, output_dir=str(tmp_path / "r"))
        issues = runner.validate_eval_set()
        assert any("expected_keywords" in i for i in issues)
        assert any("scoring" in i for i in issues)

    def test_duplicate_id_detected(self, tmp_path):
        qs = [_make_question("dup"), _make_question("dup")]
        path = _write_eval_set(tmp_path, qs)
        runner = AgentEvalRunner(path, output_dir=str(tmp_path / "r"))
        issues = runner.validate_eval_set()
        assert any("duplicate" in i for i in issues)

    def test_weight_sum_mismatch_detected(self, tmp_path):
        q = _make_question()
        q["scoring"] = {"has_code": 0.5, "has_correct_api": 0.8}
        path = _write_eval_set(tmp_path, [q])
        runner = AgentEvalRunner(path, output_dir=str(tmp_path / "r"))
        issues = runner.validate_eval_set()
        assert any("weights sum" in i for i in issues)

    def test_no_keywords_detected(self, tmp_path):
        q = _make_question()
        q["expected_keywords"] = []
        path = _write_eval_set(tmp_path, [q])
        runner = AgentEvalRunner(path, output_dir=str(tmp_path / "r"))
        issues = runner.validate_eval_set()
        assert any("no expected_keywords" in i for i in issues)


# ---------------------------------------------------------------------------
# Run without agent
# ---------------------------------------------------------------------------

class TestRunNoAgent:

    def test_single_without_agent_returns_zero(self, runner_one):
        """Without an agent, run_single returns score 0."""
        result = runner_one.run_single("test_001")
        assert result.score == 0.0
        assert result.passed is False
        assert "no agent" in result.answer.lower()

    def test_unknown_id_raises(self, runner_one):
        with pytest.raises(KeyError, match="Unknown"):
            runner_one.run_single("nonexistent_999")

    def test_run_all_no_agent(self, runner_multi):
        results = runner_multi.run_all()
        assert len(results) == 5
        assert all(r.score == 0.0 for r in results)

    def test_category_filter(self, runner_multi):
        results = runner_multi.run_all(categories=["python"])
        assert len(results) == 2
        assert all(r.category == "python" for r in results)

    def test_max_questions(self, runner_multi):
        results = runner_multi.run_all(max_questions=2)
        assert len(results) == 2


# ---------------------------------------------------------------------------
# Summary and reporting
# ---------------------------------------------------------------------------

class TestSummary:

    def test_empty_results(self):
        s = AgentEvalRunner.summary([])
        assert s["total"] == 0

    def test_summary_structure(self, runner_multi):
        results = runner_multi.run_all()
        s = AgentEvalRunner.summary(results)
        assert s["total"] == 5
        assert "per_category" in s
        assert "worst_5" in s
        assert s["pass_rate"] == 0.0  # no agent = all zero

    def test_per_category_breakdown(self, runner_multi):
        results = runner_multi.run_all()
        s = AgentEvalRunner.summary(results)
        cats = s["per_category"]
        assert "python" in cats
        assert cats["python"]["count"] == 2
        assert "javascript" in cats

    def test_report_writes_file(self, runner_multi, tmp_path):
        results = runner_multi.run_all()
        out = str(tmp_path / "report.md")
        text = runner_multi.report(results, output_path=out)
        assert Path(out).exists()
        assert "# Agent Evaluation Report" in text
        assert "Per Category" in text


# ---------------------------------------------------------------------------
# Crash-safe resume
# ---------------------------------------------------------------------------

class TestResume:

    def test_save_and_reload(self, tmp_path):
        qs = [_make_question("r1"), _make_question("r2")]
        path = _write_eval_set(tmp_path, qs)
        out = str(tmp_path / "results")

        runner1 = AgentEvalRunner(path, agent=None, output_dir=out)
        results1 = runner1.run_all(max_questions=1, resume=False)
        assert len(results1) == 1

        # Second runner loads completed results
        runner2 = AgentEvalRunner(path, agent=None, output_dir=out)
        results2 = runner2.run_all(resume=True)
        assert len(results2) == 2
        # First result was resumed, second was new
        assert results2[0].question_id == "r1"
        assert results2[1].question_id == "r2"

    def test_no_resume_discards_previous(self, tmp_path):
        qs = [_make_question("d1")]
        path = _write_eval_set(tmp_path, qs)
        out = str(tmp_path / "results")

        runner1 = AgentEvalRunner(path, agent=None, output_dir=out)
        runner1.run_all(resume=False)

        runner2 = AgentEvalRunner(path, agent=None, output_dir=out)
        # With resume=False in run_all, it re-runs even if completed exists
        results = runner2.run_all(resume=False)
        assert len(results) == 1


# ---------------------------------------------------------------------------
# Golden eval set integrity
# ---------------------------------------------------------------------------

class TestGoldenSetIntegrity:
    """Validate the actual 200-question eval set ships with no issues."""

    EVAL_200 = Path(__file__).resolve().parent.parent / "evaluation" / "agent_eval_set_200.json"

    @pytest.mark.skipif(
        not (Path(__file__).resolve().parent.parent / "evaluation" / "agent_eval_set_200.json").exists(),
        reason="200-question eval set not found",
    )
    def test_200_set_validates_clean(self):
        runner = AgentEvalRunner(str(self.EVAL_200))
        issues = runner.validate_eval_set()
        assert issues == [], f"Validation issues: {issues}"

    @pytest.mark.skipif(
        not (Path(__file__).resolve().parent.parent / "evaluation" / "agent_eval_set_200.json").exists(),
        reason="200-question eval set not found",
    )
    def test_200_set_has_200_questions(self):
        runner = AgentEvalRunner(str(self.EVAL_200))
        assert len(runner.questions) == 200

    @pytest.mark.skipif(
        not (Path(__file__).resolve().parent.parent / "evaluation" / "agent_eval_set_200.json").exists(),
        reason="200-question eval set not found",
    )
    def test_200_set_covers_all_categories(self):
        runner = AgentEvalRunner(str(self.EVAL_200))
        cats = {q["category"] for q in runner.questions}
        expected = {"python", "javascript", "systems", "security",
                    "algorithms", "debugging", "go", "rust", "shell"}
        assert cats == expected, f"Missing categories: {expected - cats}"

    @pytest.mark.skipif(
        not (Path(__file__).resolve().parent.parent / "evaluation" / "agent_eval_set_200.json").exists(),
        reason="200-question eval set not found",
    )
    def test_200_set_no_duplicate_ids(self):
        runner = AgentEvalRunner(str(self.EVAL_200))
        ids = [q["id"] for q in runner.questions]
        assert len(ids) == len(set(ids)), "Duplicate question IDs found"
