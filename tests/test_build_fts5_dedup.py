"""Tests for R12.3 (cross-index dedup) and R12.5 (quality scoring) in build_fts5_indexes."""

from __future__ import annotations

import sqlite3
import pytest
from pathlib import Path
from unittest.mock import patch

# Import the functions under test
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.build_fts5_indexes import (
    build_fts5_index,
    _estimate_quality,
    _normalize,
)
from ingestion.dedup import MinHashDedup


# ---------------------------------------------------------------------------
# Quality scoring
# ---------------------------------------------------------------------------

class TestEstimateQuality:

    def test_empty_content_low_score(self):
        score = _estimate_quality("x = 1", "unknown_source.py")
        assert score <= 2

    def test_function_with_docstring_high_score(self):
        code = '''def process(items: list) -> list:
    """Process items and return filtered results."""
    import re
    return [x for x in items if x]'''
        score = _estimate_quality(code, "python_docs/stdlib.py")
        assert score >= 4

    def test_curated_source_bonus(self):
        content = "Some documentation text that is long enough."
        score_curated = _estimate_quality(content * 3, "python_docs/io.md")
        score_generic = _estimate_quality(content * 3, "random_repo/foo.py")
        assert score_curated >= score_generic

    def test_score_capped_at_5(self):
        code = '''import os
def mega_func(items: list) -> dict:
    """Super documented."""
    return {}'''
        score = _estimate_quality(code, "python_docs/x.py")
        assert score <= 5

    def test_very_short_content_penalized(self):
        score = _estimate_quality("x", "python_docs/x.py")
        assert score < 5


# ---------------------------------------------------------------------------
# Dedup integration
# ---------------------------------------------------------------------------

class TestDedupInBuild:

    def _create_source(self, tmp_path, source_name, files):
        """Create a fake source directory with Q&A markdown files."""
        clean = tmp_path / "clean_source" / source_name
        clean.mkdir(parents=True)
        for name, content in files.items():
            (clean / name).write_text(content, encoding="utf-8")
        return clean

    def test_dedup_removes_exact_duplicates(self, tmp_path):
        """Identical chunks from different files should be deduped."""
        same_content = "# Question\n\nHow to sort a list in Python?\n\n```python\ndef sort_list(items):\n    return sorted(items)\n```"
        self._create_source(tmp_path, "test_src", {
            "a.md": same_content,
            "b.md": same_content,
            "c.md": same_content,
        })

        dedup = MinHashDedup(num_perm=128, threshold=0.8)
        config = {"index": "test_idx", "type": "qa"}

        with patch("scripts.build_fts5_indexes.CLEAN_DIR", tmp_path / "clean_source"), \
             patch("scripts.build_fts5_indexes.INDEX_DIR", tmp_path / "indexes"):
            stats = build_fts5_index("test_src", config, dedup=dedup)

        assert stats["dedup_hits"] >= 2  # At least 2 of 3 should be deduped
        assert stats["chunks"] == 1  # Only 1 unique chunk

    def test_no_dedup_keeps_all(self, tmp_path):
        """Without dedup, all chunks are kept."""
        same_content = "# Question\n\nHow to sort?\n\n```python\ndef sort_list(items):\n    return sorted(items)\n```"
        self._create_source(tmp_path, "test_src", {
            "a.md": same_content,
            "b.md": same_content,
        })

        config = {"index": "test_idx2", "type": "qa"}

        with patch("scripts.build_fts5_indexes.CLEAN_DIR", tmp_path / "clean_source"), \
             patch("scripts.build_fts5_indexes.INDEX_DIR", tmp_path / "indexes"):
            stats = build_fts5_index("test_src", config, dedup=None)

        assert stats["chunks"] == 2  # Both kept

    def test_near_duplicates_caught(self, tmp_path):
        """Content with minor variations should be caught as near-dupes."""
        base = "# Sorting\n\nSort a list in Python using the built-in sorted function.\n\n```python\ndef sort_items(data):\n    return sorted(data)\n```"
        variant = "# Sorting\n\nSort a list in Python using the builtin sorted function.\n\n```python\ndef sort_items(data):\n    return sorted(data)\n```"
        self._create_source(tmp_path, "test_src", {
            "a.md": base,
            "b.md": variant,
        })

        dedup = MinHashDedup(num_perm=128, threshold=0.8)
        config = {"index": "test_near", "type": "qa"}

        with patch("scripts.build_fts5_indexes.CLEAN_DIR", tmp_path / "clean_source"), \
             patch("scripts.build_fts5_indexes.INDEX_DIR", tmp_path / "indexes"):
            stats = build_fts5_index("test_src", config, dedup=dedup)

        # At least one should be deduped
        assert stats["dedup_hits"] >= 1

    def test_unique_content_passes_dedup(self, tmp_path):
        """Completely different content should all pass dedup."""
        self._create_source(tmp_path, "test_src", {
            "a.md": "# Alpha\n\nHow to use decorators in Python for function wrapping?",
            "b.md": "# Beta\n\nExplain garbage collection in the JVM and its impact on latency.",
            "c.md": "# Gamma\n\nWhat is the difference between TCP and UDP protocols?",
        })

        dedup = MinHashDedup(num_perm=128, threshold=0.8)
        config = {"index": "test_unique", "type": "qa"}

        with patch("scripts.build_fts5_indexes.CLEAN_DIR", tmp_path / "clean_source"), \
             patch("scripts.build_fts5_indexes.INDEX_DIR", tmp_path / "indexes"):
            stats = build_fts5_index("test_src", config, dedup=dedup)

        assert stats["dedup_hits"] == 0
        assert stats["chunks"] == 3


# ---------------------------------------------------------------------------
# Quality score in FTS5 schema
# ---------------------------------------------------------------------------

class TestQualityScoreInIndex:

    def test_quality_score_stored_in_db(self, tmp_path):
        """Quality score should be queryable from the FTS5 index."""
        content = "# Code\n\n```python\nimport os\ndef read_file(path: str) -> str:\n    \"\"\"Read file contents.\"\"\"\n    return open(path).read()\n```"
        clean = tmp_path / "clean_source" / "test_src"
        clean.mkdir(parents=True)
        (clean / "a.md").write_text(content, encoding="utf-8")

        config = {"index": "test_qs", "type": "qa"}

        with patch("scripts.build_fts5_indexes.CLEAN_DIR", tmp_path / "clean_source"), \
             patch("scripts.build_fts5_indexes.INDEX_DIR", tmp_path / "indexes"):
            build_fts5_index("test_src", config, dedup=None)

        db_path = tmp_path / "indexes" / "test_qs.fts5.db"
        assert db_path.exists()
        conn = sqlite3.connect(str(db_path))
        rows = conn.execute("SELECT quality_score FROM chunks").fetchall()
        conn.close()
        assert len(rows) >= 1
        assert all(int(r[0]) >= 0 for r in rows)


# ---------------------------------------------------------------------------
# Dedup persistence
# ---------------------------------------------------------------------------

class TestDedupPersistence:

    def test_dedup_state_persists(self, tmp_path):
        """Dedup state should save and reload for incremental builds."""
        persist = str(tmp_path / "dedup.json")
        d1 = MinHashDedup(num_perm=128, threshold=0.8, persist_path=persist)
        d1.add("hello world this is a test document with enough content")
        d1.save()

        d2 = MinHashDedup(num_perm=128, threshold=0.8, persist_path=persist)
        assert d2.is_duplicate("hello world this is a test document with enough content")


# ---------------------------------------------------------------------------
# Normalize
# ---------------------------------------------------------------------------

class TestNormalize:

    def test_camel_case_split(self):
        assert "mock reranker" in _normalize("MockReranker")

    def test_underscore_to_space(self):
        assert "foo bar" in _normalize("foo_bar")

    def test_path_separators(self):
        norm = _normalize("src/core/engine.py")
        assert "src" in norm
        assert "engine" in norm
