"""Tests for ingestion/ast_fts5_builder.py -- AST-aware FTS5 index building."""

import sqlite3
import sys
import os

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingestion.ast_fts5_builder import ASTIndexBuilder, _normalize_for_fts5, _detect_language


class TestNormalize:
    def test_lowercases(self):
        assert _normalize_for_fts5("Hello World") == "hello world"

    def test_replaces_separators(self):
        assert _normalize_for_fts5("foo_bar-baz.qux") == "foo bar baz qux"

    def test_preserves_content(self):
        result = _normalize_for_fts5("def hello(): pass")
        assert "def hello" in result
        assert "pass" in result


class TestDetectLanguage:
    def test_python(self):
        assert _detect_language("src/main.py") == "python"

    def test_rust(self):
        assert _detect_language("lib.rs") == "rust"

    def test_javascript(self):
        assert _detect_language("app.js") == "javascript"

    def test_unknown(self):
        assert _detect_language("data.csv") is None

    def test_markdown_returns_none(self):
        assert _detect_language("README.md") is None


class TestASTIndexBuilder:
    def test_add_code_python(self, tmp_path):
        db = tmp_path / "test.fts5.db"
        builder = ASTIndexBuilder(db, max_chunk_chars=2000)
        code = '''
def hello():
    """Say hello."""
    print("hello")

def world():
    """Say world."""
    print("world")

class Greeter:
    def greet(self):
        return "hi"
'''
        added = builder.add_code(code, "test.py", language="python")
        entries, chunks, size_mb = builder.finish()
        assert entries == 1
        assert chunks >= 2  # At least hello + world or more AST nodes
        assert size_mb > 0

    def test_add_code_no_language(self, tmp_path):
        db = tmp_path / "test.fts5.db"
        builder = ASTIndexBuilder(db)
        added = builder.add_code("some code here\nmore code", "unknown.xyz")
        entries, chunks, _ = builder.finish()
        assert entries == 1
        assert chunks >= 1  # Char fallback

    def test_add_text(self, tmp_path):
        db = tmp_path / "test.fts5.db"
        builder = ASTIndexBuilder(db)
        builder.add_text("This is a long documentation string " * 20, "doc_123")
        entries, chunks, _ = builder.finish()
        assert entries == 1
        assert chunks >= 1

    def test_empty_input_skipped(self, tmp_path):
        db = tmp_path / "test.fts5.db"
        builder = ASTIndexBuilder(db)
        assert builder.add_code("", "empty.py") == 0
        assert builder.add_code("   ", "whitespace.py") == 0
        assert builder.add_text("", "empty") == 0
        entries, chunks, _ = builder.finish()
        assert entries == 0
        assert chunks == 0

    def test_fts5_searchable(self, tmp_path):
        db = tmp_path / "test.fts5.db"
        builder = ASTIndexBuilder(db)
        builder.add_code(
            'def calculate_fibonacci(n):\n    """Calculate fibonacci."""\n    if n <= 1: return n\n    return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)\n',
            "math.py", language="python"
        )
        builder.finish()
        conn = sqlite3.connect(str(db))
        rows = conn.execute(
            "SELECT search_content FROM chunks WHERE chunks MATCH 'fibonacci'"
        ).fetchall()
        conn.close()
        assert len(rows) >= 1
        assert "fibonacci" in rows[0][0]

    def test_context_manager(self, tmp_path):
        db = tmp_path / "test.fts5.db"
        with ASTIndexBuilder(db) as builder:
            builder.add_code("def foo(): pass", "foo.py", language="python")
        conn = sqlite3.connect(str(db))
        count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        conn.close()
        assert count >= 1

    def test_large_code_splits(self, tmp_path):
        db = tmp_path / "test.fts5.db"
        builder = ASTIndexBuilder(db, max_chunk_chars=200)
        big_func = "def big():\n" + "    x = 1\n" * 100
        builder.add_code(big_func, "big.py", language="python")
        entries, chunks, _ = builder.finish()
        assert chunks >= 2  # Should split oversized function

    def test_batch_flush(self, tmp_path):
        db = tmp_path / "test.fts5.db"
        builder = ASTIndexBuilder(db, batch_size=5)
        for i in range(20):
            builder.add_text(f"Document number {i} with enough text to pass the minimum length threshold for indexing", f"doc_{i}")
        entries, chunks, _ = builder.finish()
        assert entries == 20
        assert chunks == 20

    def test_detects_language_from_source_id(self, tmp_path):
        db = tmp_path / "test.fts5.db"
        builder = ASTIndexBuilder(db)
        # Should auto-detect Python from .py extension
        builder.add_code("def auto(): pass\ndef detect(): pass", "auto.py")
        entries, chunks, _ = builder.finish()
        assert entries == 1
        assert chunks >= 1
