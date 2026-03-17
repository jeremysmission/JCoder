"""Tests for scripts/weekly_scraper.py -- weekly knowledge scraper."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestStripHtml:

    def test_removes_tags(self):
        from scripts.weekly_scraper import _strip_html
        assert _strip_html("<p>hello <b>world</b></p>") == "hello world"

    def test_collapses_whitespace(self):
        from scripts.weekly_scraper import _strip_html
        assert _strip_html("  hello   world  ") == "hello world"

    def test_empty(self):
        from scripts.weekly_scraper import _strip_html
        assert _strip_html("") == ""


class TestChunkId:

    def test_deterministic(self):
        from scripts.weekly_scraper import _chunk_id
        a = _chunk_id("hello world")
        b = _chunk_id("hello world")
        assert a == b

    def test_different_content(self):
        from scripts.weekly_scraper import _chunk_id
        a = _chunk_id("hello")
        b = _chunk_id("world")
        assert a != b


class TestIngestChunks:

    def test_creates_db(self, tmp_path):
        from scripts.weekly_scraper import ingest_chunks
        db_path = tmp_path / "fresh.fts5.db"
        chunks = [
            {"content": "Python 3.14 released", "source": "rss", "title": "Release"},
        ]
        count = ingest_chunks(chunks, str(db_path))
        assert count == 1
        assert db_path.exists()

    def test_deduplicates(self, tmp_path):
        from scripts.weekly_scraper import ingest_chunks
        db_path = tmp_path / "fresh.fts5.db"
        chunks = [
            {"content": "same content", "source": "a", "title": "A"},
        ]
        ingest_chunks(chunks, str(db_path))
        count = ingest_chunks(chunks, str(db_path))
        assert count == 0  # duplicate skipped

    def test_rotation(self, tmp_path):
        from scripts.weekly_scraper import ingest_chunks
        db_path = tmp_path / "fresh.fts5.db"
        # Insert 5 chunks with max 3
        for i in range(5):
            ingest_chunks(
                [{"content": f"chunk {i} " * 20, "source": "test", "title": f"C{i}"}],
                str(db_path),
                max_chunks=3,
            )
        conn = sqlite3.connect(str(db_path))
        count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        conn.close()
        assert count <= 3

    def test_empty_chunks(self, tmp_path):
        from scripts.weekly_scraper import ingest_chunks
        db_path = tmp_path / "fresh.fts5.db"
        count = ingest_chunks([], str(db_path))
        assert count == 0


class TestRunWeeklyScrapeDryRun:

    def test_dry_run(self):
        from scripts.weekly_scraper import run_weekly_scrape
        # Dry run with no network calls by using a nonexistent scraper list
        result = run_weekly_scrape(scrapers=[], dry_run=True)
        assert result["ingested"] == 0

    def test_report_structure(self, tmp_path):
        from scripts.weekly_scraper import run_weekly_scrape
        result = run_weekly_scrape(
            scrapers=[],
            dry_run=True,
            output_dir=str(tmp_path / "reports"),
        )
        assert "timestamp" in result
        assert "total_collected" in result
        assert "elapsed_s" in result
