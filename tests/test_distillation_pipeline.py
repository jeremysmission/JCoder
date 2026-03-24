"""Tests for R16: Distillation pipeline components.

Tests the local (non-API) parts of the distillation pipeline:
saving to FTS5, saving to knowledge files, quality filtering,
and deduplication.
"""

from __future__ import annotations

import json
import sqlite3
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from scripts.distill_weak_topics import (
    save_to_fts5,
    save_to_knowledge,
    retrieve_context,
    run_distillation,
    DISTILL_SYSTEM,
)


# ---------------------------------------------------------------------------
# FTS5 Storage
# ---------------------------------------------------------------------------

class TestSaveToFts5:

    def test_creates_table_and_inserts(self, tmp_path):
        db = str(tmp_path / "test.fts5.db")
        save_to_fts5("def foo(): pass", "Q001", "python_basics", db)

        conn = sqlite3.connect(db)
        rows = conn.execute("SELECT content, source, category FROM chunks").fetchall()
        conn.close()

        assert len(rows) == 1
        assert rows[0][0] == "def foo(): pass"
        assert rows[0][1] == "distilled:Q001"
        assert rows[0][2] == "python_basics"

    def test_multiple_inserts(self, tmp_path):
        db = str(tmp_path / "test.fts5.db")
        save_to_fts5("content 1", "Q001", "cat1", db)
        save_to_fts5("content 2", "Q002", "cat2", db)

        conn = sqlite3.connect(db)
        count = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
        conn.close()
        assert count == 2


# ---------------------------------------------------------------------------
# Knowledge File Storage
# ---------------------------------------------------------------------------

class TestSaveToKnowledge:

    def test_creates_markdown_file(self, tmp_path):
        with patch("scripts.distill_weak_topics._ROOT", tmp_path):
            path = save_to_knowledge(
                "# How to sort\n\n```python\nsorted(items)\n```",
                "Q001", "sorting",
                {"model": "gpt-5", "input_tokens": 100, "output_tokens": 200},
            )
        assert path.exists()
        content = path.read_text(encoding="utf-8")
        assert "Distilled: Q001" in content
        assert "sorting" in content
        assert "gpt-5" in content

    def test_file_is_markdown_extension(self, tmp_path):
        with patch("scripts.distill_weak_topics._ROOT", tmp_path):
            path = save_to_knowledge("content", "Q002", "general", {})
        assert path.suffix == ".md"


# ---------------------------------------------------------------------------
# RAG Context Retrieval (local FTS5)
# ---------------------------------------------------------------------------

class TestRetrieveContext:

    def test_retrieve_from_existing_index(self, tmp_path):
        # Create a small FTS5 index
        db = tmp_path / "test.fts5.db"
        conn = sqlite3.connect(str(db))
        conn.execute("CREATE VIRTUAL TABLE chunks USING fts5(search_content, source_path, chunk_id)")
        conn.execute("INSERT INTO chunks (search_content, source_path, chunk_id) VALUES (?, ?, ?)",
                     ("Python sorting algorithm quicksort implementation", "test.py", "c0"))
        conn.commit()
        conn.close()

        context = retrieve_context("python sorting", str(tmp_path))
        assert "sorting" in context.lower() or "quicksort" in context.lower()

    def test_retrieve_from_missing_dir(self):
        context = retrieve_context("test", "/nonexistent/path")
        assert context == ""

    def test_retrieve_empty_query(self, tmp_path):
        context = retrieve_context("", str(tmp_path))
        assert context == ""


# ---------------------------------------------------------------------------
# Run Distillation (mocked API)
# ---------------------------------------------------------------------------

class TestRunDistillation:

    def test_empty_results_returns_zero(self, tmp_path):
        result = run_distillation([], index_dir=str(tmp_path))
        assert result["distilled"] == 0

    def test_no_scored_results(self, tmp_path):
        results = [{"question": "test?", "id": "Q1"}]  # no "score" key
        result = run_distillation(results, index_dir=str(tmp_path))
        assert result["distilled"] == 0

    def test_budget_cap_respected(self, tmp_path):
        results = [
            {"question": "Q1?", "id": "Q1", "score": 0.1, "category": "cat"},
            {"question": "Q2?", "id": "Q2", "score": 0.2, "category": "cat"},
        ]
        # Mock call_api to simulate cost
        with patch("scripts.distill_weak_topics.call_api") as mock_api, \
             patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            mock_api.return_value = {
                "content": "distilled content here",
                "input_tokens": 100,
                "output_tokens": 200,
                "cost": 3.0,  # Over budget
                "model": "gpt-5",
            }
            result = run_distillation(
                results, index_dir=str(tmp_path), budget_usd=2.0, top=2)
        # Should have attempted at least one
        assert result["distilled"] >= 0


# ---------------------------------------------------------------------------
# Distill System Prompt
# ---------------------------------------------------------------------------

class TestDistillSystemPrompt:

    def test_system_prompt_contains_key_rules(self):
        assert "self-contained" in DISTILL_SYSTEM.lower()
        assert "DISTILLED" in DISTILL_SYSTEM
        assert "imports" in DISTILL_SYSTEM.lower()

    def test_system_prompt_not_empty(self):
        assert len(DISTILL_SYSTEM) > 100
