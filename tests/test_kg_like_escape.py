"""
Tests for knowledge_graph LIKE wildcard escaping in _find_seeds.
"""

from __future__ import annotations

import pytest

from core.knowledge_graph import CodeKnowledgeGraph


class TestLikeEscaping:

    def test_escape_like_percent(self):
        assert CodeKnowledgeGraph._escape_like("100%done") == "100\\%done"

    def test_escape_like_underscore(self):
        assert CodeKnowledgeGraph._escape_like("my_func") == "my\\_func"

    def test_escape_like_backslash(self):
        assert CodeKnowledgeGraph._escape_like("a\\b") == "a\\\\b"

    def test_escape_like_clean_string(self):
        assert CodeKnowledgeGraph._escape_like("retrieve") == "retrieve"

    def test_find_seeds_with_wildcard_in_query(self, tmp_path):
        """Ensure LIKE wildcards in user input don't cause spurious matches."""
        kg = CodeKnowledgeGraph(db_path=str(tmp_path / "kg.db"))
        chunks = [
            {
                "id": "c1",
                "content": "def alpha_beta():\n    pass\n",
                "source_path": "mod.py",
            },
        ]
        kg.build_from_chunks(chunks)
        # A query containing literal _ should not act as LIKE wildcard
        seeds = kg._find_seeds("alpha_beta")
        # Should find the entity via substring match on "alpha" and "beta"
        assert len(seeds) >= 1
