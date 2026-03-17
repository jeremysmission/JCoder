"""Tests for scripts/evolve_prompts.py -- offline prompt evolution runner."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _patch_runtime():
    """Return a mock Runtime that generates predictable answers."""
    rt = MagicMock()
    rt.generate.return_value = (
        "Here is how to sort a list:\n"
        "```python\ndef sort_list(items):\n    return sorted(items)\n```\n"
        "This uses Python's built-in sorted() function which returns a new list."
    )
    return rt


class TestLoadEvalQueries:

    def test_loads_questions(self, tmp_path):
        from scripts.evolve_prompts import _load_eval_queries
        qs = [
            {"id": "q1", "question": "How to sort?"},
            {"id": "q2", "question": "What is a dict?"},
        ]
        path = tmp_path / "eval.json"
        path.write_text(json.dumps(qs), encoding="utf-8")
        result = _load_eval_queries(str(path))
        assert len(result) == 2
        assert result[0] == "How to sort?"

    def test_max_queries(self, tmp_path):
        from scripts.evolve_prompts import _load_eval_queries
        qs = [{"question": f"q{i}"} for i in range(100)]
        path = tmp_path / "eval.json"
        path.write_text(json.dumps(qs), encoding="utf-8")
        result = _load_eval_queries(str(path), max_queries=10)
        assert len(result) == 10

    def test_missing_file(self):
        from scripts.evolve_prompts import _load_eval_queries
        result = _load_eval_queries("/nonexistent/path.json")
        assert result == []


class TestLoadFailureExamples:

    def test_no_db(self):
        from scripts.evolve_prompts import _load_failure_examples
        result = _load_failure_examples("/nonexistent/db.sqlite")
        assert result == []

    def test_loads_from_telemetry(self, tmp_path):
        import sqlite3
        from scripts.evolve_prompts import _load_failure_examples
        db_path = tmp_path / "events.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE query_events (
                query TEXT, answer TEXT, confidence REAL, timestamp REAL
            )
        """)
        conn.execute(
            "INSERT INTO query_events VALUES (?, ?, ?, ?)",
            ("bad query", "bad answer", 0.1, 1.0),
        )
        conn.commit()
        conn.close()

        result = _load_failure_examples(str(db_path))
        assert len(result) == 1
        assert result[0]["query"] == "bad query"


class TestDefaultSeedPrompt:

    def test_returns_string(self):
        from scripts.evolve_prompts import _default_seed_prompt
        prompt = _default_seed_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 20


class TestRunEvolutionDryRun:

    def test_dry_run_returns_config(self, tmp_path):
        from scripts.evolve_prompts import run_evolution
        qs = [{"question": f"q{i}"} for i in range(5)]
        path = tmp_path / "eval.json"
        path.write_text(json.dumps(qs), encoding="utf-8")

        result = run_evolution(
            eval_set_path=str(path),
            dry_run=True,
        )
        assert result["dry_run"] is True
        assert result["queries"] == 5

    def test_no_queries_returns_error(self):
        from scripts.evolve_prompts import run_evolution
        result = run_evolution(
            eval_set_path="/nonexistent.json",
            dry_run=True,
        )
        assert result.get("error") == "no_queries"
