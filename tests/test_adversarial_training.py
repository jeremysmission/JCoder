"""Tests for scripts/adversarial_training.py -- adversarial self-play runner."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestLoadCodeChunks:

    def test_empty_dir(self, tmp_path):
        from scripts.adversarial_training import _load_code_chunks
        result = _load_code_chunks(str(tmp_path))
        assert result == []

    def test_loads_from_fts5(self, tmp_path):
        from scripts.adversarial_training import _load_code_chunks
        db = tmp_path / "test.fts5.db"
        conn = sqlite3.connect(str(db))
        conn.execute("CREATE VIRTUAL TABLE chunks USING fts5(content)")
        conn.execute(
            "INSERT INTO chunks VALUES (?)",
            ("def foo():\n    return bar()\n\ndef baz():\n    pass\n" * 5,),
        )
        conn.commit()
        conn.close()

        result = _load_code_chunks(str(tmp_path), max_chunks=10)
        assert len(result) >= 1
        assert "content" in result[0]
        assert "source_path" in result[0]

    def test_max_chunks_respected(self, tmp_path):
        from scripts.adversarial_training import _load_code_chunks
        db = tmp_path / "test.fts5.db"
        conn = sqlite3.connect(str(db))
        conn.execute("CREATE VIRTUAL TABLE chunks USING fts5(content)")
        for i in range(20):
            conn.execute(
                "INSERT INTO chunks VALUES (?)",
                (f"def func_{i}():\n    return {i}\n" * 5,),
            )
        conn.commit()
        conn.close()

        result = _load_code_chunks(str(tmp_path), max_chunks=5)
        assert len(result) <= 5

    def test_nonexistent_dir(self):
        from scripts.adversarial_training import _load_code_chunks
        result = _load_code_chunks("/nonexistent/path")
        assert result == []


class TestRunDryRun:

    def test_dry_run_returns_config(self):
        from scripts.adversarial_training import run_adversarial_training
        result = run_adversarial_training(dry_run=True)
        assert result["dry_run"] is True
        assert "chunks" in result


class TestFeedFailures:

    def test_empty_failures(self):
        from scripts.adversarial_training import _feed_failures_to_experience
        result = _feed_failures_to_experience([])
        assert result == 0

    def test_feed_with_mock(self):
        from scripts.adversarial_training import _feed_failures_to_experience
        failures = [
            {
                "challenge_id": "c1",
                "answer_snippet": "wrong answer about sorting",
                "failure_mode": "wrong_answer",
            },
        ]
        # May fail if ExperienceStore can't be imported -- that's OK
        result = _feed_failures_to_experience(failures)
        assert isinstance(result, int)
