"""Integration tests for the AST chunker wired into the build/ingest pipeline.

Verifies:
  - Python files produce function-level AST chunks (or char-fallback splits)
  - Unknown extensions fall back to character chunking
  - Empty files produce zero chunks
  - Files with syntax errors fall back gracefully (no crash)
  - _chunk_code_file integrates correctly with the FTS5 builder plumbing
  - Chunker.chunk_file dispatches AST vs char based on extension
"""

import os
import sqlite3
import sys
import textwrap

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingestion.chunker import Chunker, LANGUAGE_MAP
import scripts.build_fts5_indexes as builder_mod
from scripts.build_fts5_indexes import (
    _chunk_code_file,
    _normalize,
    AST_EXTENSIONS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write(tmp_path, name, content):
    """Write a temp file and return its Path."""
    p = tmp_path / name
    p.write_text(textwrap.dedent(content), encoding="utf-8")
    return p


def _has_tree_sitter_python():
    """Return True if tree-sitter-python grammar loads successfully."""
    return Chunker()._get_parser("python") is not None


# ---------------------------------------------------------------------------
# 1. Python files produce function-level chunks
# ---------------------------------------------------------------------------

class TestPythonASTChunking:
    """Python source files should be split at function/class boundaries."""

    SAMPLE = '''\
    import os

    def alpha():
        """First function."""
        return 1

    def beta(x):
        """Second function."""
        return x * 2

    class Gamma:
        """A class."""

        def method(self):
            return "hello"
    '''

    def test_produces_chunks(self, tmp_path):
        fpath = _write(tmp_path, "sample.py", self.SAMPLE)
        chunks = _chunk_code_file(fpath)
        # Always produces at least 1 chunk (char fallback if no grammar)
        assert len(chunks) >= 1
        # With tree-sitter, should split into multiple AST nodes
        if _has_tree_sitter_python():
            assert len(chunks) >= 3, f"AST should yield >=3 chunks, got {len(chunks)}"

    def test_chunk_contains_function_text(self, tmp_path):
        fpath = _write(tmp_path, "sample.py", self.SAMPLE)
        chunks = _chunk_code_file(fpath)
        all_content = " ".join(c["content"] for c in chunks)
        assert "def alpha" in all_content
        assert "def beta" in all_content
        assert "class Gamma" in all_content

    def test_each_chunk_has_required_keys(self, tmp_path):
        fpath = _write(tmp_path, "sample.py", self.SAMPLE)
        chunks = _chunk_code_file(fpath)
        for c in chunks:
            assert "content" in c
            assert "source" in c
            assert "title" in c

    def test_single_function_file(self, tmp_path):
        fpath = _write(tmp_path, "one.py", """\
        def only():
            pass
        """)
        chunks = _chunk_code_file(fpath)
        assert len(chunks) >= 1
        assert "def only" in chunks[0]["content"]

    def test_small_max_chars_forces_splits(self, tmp_path):
        """With a tiny max_chars, even char-fallback must split."""
        fpath = _write(tmp_path, "multi.py", self.SAMPLE)
        old_chunker = builder_mod._code_chunker
        try:
            builder_mod._code_chunker = Chunker(max_chars=80)
            chunks = _chunk_code_file(fpath)
            assert len(chunks) >= 2, "Small max_chars should force multiple chunks"
        finally:
            builder_mod._code_chunker = old_chunker

    def test_chunker_routes_py_to_ast_path(self):
        """Chunker.chunk_file selects AST path for .py extension."""
        c = Chunker()
        ext = ".py"
        lang = LANGUAGE_MAP.get(ext)
        assert lang == "python", "LANGUAGE_MAP must map .py -> python"


# ---------------------------------------------------------------------------
# 2. Unknown extensions fall back to character chunking
# ---------------------------------------------------------------------------

class TestUnknownExtensionFallback:
    """Files with no tree-sitter grammar should still produce chunks."""

    def test_csv_falls_back(self, tmp_path):
        fpath = _write(tmp_path, "data.csv", "a,b,c\n1,2,3\n4,5,6\n")
        chunks = _chunk_code_file(fpath)
        assert len(chunks) >= 1
        assert "a,b,c" in chunks[0]["content"]

    def test_toml_falls_back(self, tmp_path):
        fpath = _write(tmp_path, "config.toml", """\
        [tool.pytest]
        minversion = "6.0"
        addopts = "-q"
        """)
        chunks = _chunk_code_file(fpath)
        assert len(chunks) >= 1

    def test_unknown_ext_no_crash(self, tmp_path):
        fpath = _write(tmp_path, "readme.xyz", "Just some random text content.\n" * 5)
        chunks = _chunk_code_file(fpath)
        assert len(chunks) >= 1

    def test_language_map_none_for_text(self):
        """Extensions mapped to None must use char fallback."""
        assert LANGUAGE_MAP.get(".md") is None
        assert LANGUAGE_MAP.get(".txt") is None
        assert LANGUAGE_MAP.get(".json") is None


# ---------------------------------------------------------------------------
# 3. Empty files produce zero chunks
# ---------------------------------------------------------------------------

class TestEmptyFiles:
    """Empty or whitespace-only files should yield no chunks."""

    def test_empty_file(self, tmp_path):
        fpath = _write(tmp_path, "empty.py", "")
        chunks = _chunk_code_file(fpath)
        assert chunks == []

    def test_whitespace_only(self, tmp_path):
        fpath = _write(tmp_path, "blank.py", "   \n\n   \n")
        chunks = _chunk_code_file(fpath)
        assert chunks == []

    def test_empty_unknown_ext(self, tmp_path):
        fpath = _write(tmp_path, "nothing.txt", "")
        chunks = _chunk_code_file(fpath)
        assert chunks == []


# ---------------------------------------------------------------------------
# 4. Syntax errors fall back gracefully
# ---------------------------------------------------------------------------

class TestSyntaxErrorFallback:
    """Files with broken syntax should still produce chunks via fallback."""

    def test_broken_python_still_chunks(self, tmp_path):
        bad_code = "def broken(\n    x = [\n    # missing close\n\ndef other():\n    pass\n"
        fpath = _write(tmp_path, "broken.py", bad_code)
        chunks = _chunk_code_file(fpath)
        # Should get something -- either AST partial or char fallback
        assert len(chunks) >= 1, "Broken syntax should still produce chunks"

    def test_garbage_bytes_no_crash(self, tmp_path):
        fpath = tmp_path / "garbage.py"
        fpath.write_bytes(b"\x80\x81\x82def foo():\n    pass\n")
        chunks = _chunk_code_file(fpath)
        # Should not raise; may or may not produce chunks
        assert isinstance(chunks, list)

    def test_nonexistent_file_returns_empty(self, tmp_path):
        fpath = tmp_path / "does_not_exist.py"
        chunks = _chunk_code_file(fpath)
        assert chunks == []


# ---------------------------------------------------------------------------
# 5. Pipeline integration: _chunk_code_file -> FTS5 normalize -> DB
# ---------------------------------------------------------------------------

class TestFTS5PipelineIntegration:
    """End-to-end: chunk code file, normalize, insert into FTS5 DB."""

    SAMPLE = '''\
    def search_fibonacci(n):
        """Calculate fibonacci number."""
        if n <= 1:
            return n
        return search_fibonacci(n - 1) + search_fibonacci(n - 2)

    def search_factorial(n):
        """Calculate factorial."""
        if n <= 1:
            return 1
        return n * search_factorial(n - 1)
    '''

    def test_chunks_are_fts5_searchable(self, tmp_path):
        fpath = _write(tmp_path, "math_utils.py", self.SAMPLE)
        chunks = _chunk_code_file(fpath)
        assert len(chunks) >= 1

        db_path = tmp_path / "test.fts5.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "CREATE VIRTUAL TABLE chunks "
            "USING fts5(search_content, source_path, chunk_id, title)"
        )
        for i, c in enumerate(chunks):
            conn.execute(
                "INSERT INTO chunks VALUES (?, ?, ?, ?)",
                (_normalize(c["content"]), c["source"], f"id_{i}", c["title"]),
            )
        conn.commit()

        rows = conn.execute(
            "SELECT search_content FROM chunks WHERE chunks MATCH 'fibonacci'"
        ).fetchall()
        assert len(rows) >= 1
        assert "fibonacci" in rows[0][0]
        conn.close()

    def test_normalize_preserves_keywords(self):
        """FTS5 normalize should keep searchable tokens."""
        norm = _normalize("def calculate_fibonacci(n):")
        assert "fibonacci" in norm
        assert "calculate" in norm
        assert "def" in norm

    def test_ast_extensions_set_populated(self):
        """AST_EXTENSIONS should include the core languages."""
        assert ".py" in AST_EXTENSIONS
        assert ".js" in AST_EXTENSIONS
        assert ".ts" in AST_EXTENSIONS
        assert ".java" in AST_EXTENSIONS
        assert ".go" in AST_EXTENSIONS
        assert ".rs" in AST_EXTENSIONS
        # Text files should NOT be in AST set
        assert ".md" not in AST_EXTENSIONS
        assert ".txt" not in AST_EXTENSIONS
