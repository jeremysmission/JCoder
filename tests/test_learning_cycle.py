"""
Tests for scripts.learning_cycle -- Automated learning cycle.
Uses temp dirs and mocked FTS5 indexes.
"""

from __future__ import annotations

import json
import sqlite3

import pytest

from scripts.learning_cycle import (
    _keyword_score,
    compare_and_report,
    generate_study_queries,
    run_baseline_eval,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def eval_set(tmp_path):
    """Create a minimal eval set file."""
    questions = [
        {"id": "q1", "question": "What is a function?", "category": "basics",
         "expected_keywords": ["def", "function", "callable"]},
        {"id": "q2", "question": "How does async work?", "category": "async",
         "expected_keywords": ["async", "await", "coroutine"]},
        {"id": "q3", "question": "What is a class?", "category": "basics",
         "expected_keywords": ["class", "object", "instance"]},
        {"id": "q4", "question": "Explain decorators", "category": "advanced",
         "expected_keywords": ["decorator", "wrapper", "functools"]},
        {"id": "q5", "question": "What is list comprehension?", "category": "basics",
         "expected_keywords": ["list", "comprehension", "for"]},
    ]
    path = tmp_path / "eval_set.json"
    path.write_text(json.dumps(questions), encoding="utf-8")
    return str(path)


@pytest.fixture
def index_dir(tmp_path):
    """Create a minimal FTS5 index."""
    idx_dir = tmp_path / "indexes"
    idx_dir.mkdir()

    db_path = idx_dir / "test.fts5.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE VIRTUAL TABLE chunks USING fts5(content, source)")
    conn.execute(
        "INSERT INTO chunks (content, source) VALUES (?, ?)",
        ("A function is defined with def keyword. It is callable.", "test.py"),
    )
    conn.execute(
        "INSERT INTO chunks (content, source) VALUES (?, ?)",
        ("A class is an object template. Instance creation uses __init__.", "test.py"),
    )
    conn.execute(
        "INSERT INTO chunks (content, source) VALUES (?, ?)",
        ("List comprehension uses for loop syntax in a list expression.", "test.py"),
    )
    conn.commit()
    conn.close()
    return str(idx_dir)


# ---------------------------------------------------------------------------
# _keyword_score
# ---------------------------------------------------------------------------

class TestKeywordScore:

    def test_all_match(self):
        assert _keyword_score(["def function callable code"], ["def", "function"]) == 1.0

    def test_partial_match(self):
        score = _keyword_score(["def function"], ["def", "function", "lambda"])
        assert score == pytest.approx(2 / 3, abs=0.01)

    def test_no_match(self):
        assert _keyword_score(["some text"], ["async", "await"]) == 0.0

    def test_empty_expected(self):
        assert _keyword_score(["some text"], []) == 0.5

    def test_no_context(self):
        assert _keyword_score([], ["keyword"]) == 0.0


# ---------------------------------------------------------------------------
# run_baseline_eval
# ---------------------------------------------------------------------------

class TestBaselineEval:

    def test_produces_report(self, tmp_path, eval_set, index_dir):
        output = str(tmp_path / "baseline.json")
        report = run_baseline_eval(eval_set, index_dir, output)
        assert "overall_score" in report
        assert "category_scores" in report
        assert report["total_questions"] == 5
        assert 0.0 <= report["overall_score"] <= 1.0

    def test_saves_to_file(self, tmp_path, eval_set, index_dir):
        output = str(tmp_path / "baseline.json")
        run_baseline_eval(eval_set, index_dir, output)
        with open(output) as f:
            saved = json.load(f)
        assert saved["total_questions"] == 5

    def test_has_category_breakdown(self, tmp_path, eval_set, index_dir):
        output = str(tmp_path / "baseline.json")
        report = run_baseline_eval(eval_set, index_dir, output)
        assert "basics" in report["category_scores"]

    def test_empty_index(self, tmp_path, eval_set):
        empty_idx = str(tmp_path / "empty_indexes")
        (tmp_path / "empty_indexes").mkdir()
        output = str(tmp_path / "baseline.json")
        report = run_baseline_eval(eval_set, str(empty_idx), output)
        assert report["overall_score"] == 0.0


# ---------------------------------------------------------------------------
# generate_study_queries
# ---------------------------------------------------------------------------

class TestGenerateStudyQueries:

    def test_generates_queries(self, tmp_path, eval_set, index_dir):
        output = str(tmp_path / "baseline.json")
        baseline = run_baseline_eval(eval_set, index_dir, output)
        study = generate_study_queries(baseline, eval_set, n_queries=3)
        assert len(study) <= 3
        assert all("question" in sq for sq in study)
        assert all("category" in sq for sq in study)

    def test_targets_weak_categories(self, tmp_path, eval_set, index_dir):
        output = str(tmp_path / "baseline.json")
        baseline = run_baseline_eval(eval_set, index_dir, output)
        study = generate_study_queries(
            baseline, eval_set, n_queries=10, weakness_threshold=1.0,
        )
        # Should include all categories since threshold is 1.0
        assert len(study) > 0


# ---------------------------------------------------------------------------
# compare_and_report
# ---------------------------------------------------------------------------

class TestCompareAndReport:

    def test_generates_comparison(self, tmp_path):
        baseline = {
            "overall_score": 0.5,
            "category_scores": {"basics": 0.6, "async": 0.3, "advanced": 0.4},
        }
        reeval = {
            "overall_score": 0.6,
            "category_scores": {"basics": 0.7, "async": 0.5, "advanced": 0.4},
        }
        report = compare_and_report(baseline, reeval, str(tmp_path))
        assert report["overall_delta"] == pytest.approx(0.1, abs=0.01)
        assert report["categories_improved"] == 2
        assert report["categories_regressed"] == 0

    def test_handles_regression(self, tmp_path):
        baseline = {
            "overall_score": 0.7,
            "category_scores": {"basics": 0.8},
        }
        reeval = {
            "overall_score": 0.5,
            "category_scores": {"basics": 0.5},
        }
        report = compare_and_report(baseline, reeval, str(tmp_path))
        assert report["overall_delta"] < 0
        assert report["categories_regressed"] >= 1

    def test_saves_report_file(self, tmp_path):
        baseline = {"overall_score": 0.5, "category_scores": {"a": 0.5}}
        reeval = {"overall_score": 0.6, "category_scores": {"a": 0.6}}
        compare_and_report(baseline, reeval, str(tmp_path))
        report_path = tmp_path / "comparison_report.json"
        assert report_path.exists()
