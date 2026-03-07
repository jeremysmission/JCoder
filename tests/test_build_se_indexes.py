"""
Tests for scripts/build_se_indexes.py
--------------------------------------
Covers: completion marker logic, streaming parse from file path,
and resume/skip behaviour.
"""

from __future__ import annotations

import sqlite3
import textwrap
from pathlib import Path

import pytest

# Import the functions under test.
# build_se_indexes replaces sys.stdout at module level on win32, which
# breaks pytest capture.  Guard against that by temporarily pretending
# we're not on Windows during import.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

_real_platform = sys.platform
sys.platform = "linux"  # suppress stdout wrapper
try:
    from build_se_indexes import (
        _is_build_complete,
        _mark_build_complete,
        _parse_and_index,
    )
finally:
    sys.platform = _real_platform


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_posts_xml(tmp_path: Path, rows: str) -> Path:
    """Write a minimal Posts.xml file and return its path."""
    xml = (f'<?xml version="1.0" encoding="utf-8"?>\n'
           f'<posts>\n{rows}\n</posts>\n')
    p = tmp_path / "Posts.xml"
    p.write_text(xml, encoding="utf-8")
    return p


def _sample_rows() -> str:
    """A question (score 5) with two answers (score 4, score 10)."""
    return (
        '<row Id="1" PostTypeId="1" Score="5" '
        'Title="How to sort a list?" Tags="&lt;python&gt;" '
        'Body="&lt;p&gt;I want to sort a list in Python.&lt;/p&gt;" />\n'
        '<row Id="2" PostTypeId="2" ParentId="1" Score="4" '
        'Body="&lt;p&gt;Use sorted() built-in function.&lt;/p&gt;" />\n'
        '<row Id="3" PostTypeId="2" ParentId="1" Score="10" '
        'Body="&lt;p&gt;Use list.sort() for in-place sorting.&lt;/p&gt;" />\n'
    )


# ---------------------------------------------------------------------------
# Completion marker
# ---------------------------------------------------------------------------

class TestCompletionMarker:

    def test_incomplete_db_not_marked(self, tmp_path):
        db = tmp_path / "test.fts5.db"
        conn = sqlite3.connect(str(db))
        conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS chunks "
            "USING fts5(search_content, source_path, chunk_id)"
        )
        conn.close()
        assert not _is_build_complete(db)

    def test_complete_db_is_marked(self, tmp_path):
        db = tmp_path / "test.fts5.db"
        conn = sqlite3.connect(str(db))
        conn.execute(
            "CREATE TABLE _build_meta (key TEXT PRIMARY KEY, value TEXT)"
        )
        _mark_build_complete(conn)
        conn.close()
        assert _is_build_complete(db)

    def test_nonexistent_db(self, tmp_path):
        assert not _is_build_complete(tmp_path / "nope.db")


# ---------------------------------------------------------------------------
# Streaming parse from file path
# ---------------------------------------------------------------------------

class TestParseAndIndex:

    def test_basic_qa_pair(self, tmp_path):
        xml_path = _make_posts_xml(tmp_path, _sample_rows())
        db_path = tmp_path / "test.fts5.db"

        entries, chunks, skipped = _parse_and_index(xml_path, "test", db_path)

        assert entries == 1
        assert chunks >= 1
        assert skipped == 0

        # DB should be marked complete
        assert _is_build_complete(db_path)

        # Verify content was written
        conn = sqlite3.connect(str(db_path))
        rows = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        conn.close()
        assert rows == chunks

    def test_low_score_filtered(self, tmp_path):
        """Questions with score < MIN_SCORE should not appear."""
        rows = (
            '<row Id="1" PostTypeId="1" Score="1" '
            'Title="Low score" Tags="" '
            'Body="&lt;p&gt;Tiny question.&lt;/p&gt;" />\n'
        )
        xml_path = _make_posts_xml(tmp_path, rows)
        db_path = tmp_path / "filtered.fts5.db"

        entries, chunks, skipped = _parse_and_index(xml_path, "test", db_path)
        assert entries == 0
        assert chunks == 0

    def test_answer_without_question_skipped(self, tmp_path):
        """Answers whose parent question didn't qualify are ignored."""
        rows = (
            '<row Id="99" PostTypeId="2" ParentId="999" Score="5" '
            'Body="&lt;p&gt;Orphan answer body text here.&lt;/p&gt;" />\n'
        )
        xml_path = _make_posts_xml(tmp_path, rows)
        db_path = tmp_path / "orphan.fts5.db"

        entries, chunks, skipped = _parse_and_index(xml_path, "test", db_path)
        assert entries == 0
        assert chunks == 0

    def test_accepts_path_not_bytes(self, tmp_path):
        """_parse_and_index must accept a Path, not bytes."""
        xml_path = _make_posts_xml(tmp_path, _sample_rows())
        db_path = tmp_path / "pathtest.fts5.db"

        # Should not raise -- if it tried io.BytesIO it would fail on Path
        entries, chunks, _ = _parse_and_index(xml_path, "test", db_path)
        assert entries >= 1
