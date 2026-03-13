"""
Tests for core.knowledge_graph -- CodeKnowledgeGraph.
Uses in-memory temp databases; no persistent state needed.
"""

from __future__ import annotations

import logging
import os
import tempfile

import pytest

from core.knowledge_graph import CodeKnowledgeGraph, Entity, Relation


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_db(tmp_path):
    """Create a KG with a temp database."""
    db_path = str(tmp_path / "test_kg.db")
    return CodeKnowledgeGraph(db_path=db_path)


def _code_chunks():
    """Sample code chunks for graph building."""
    return [
        {
            "id": "chunk_1",
            "content": (
                "from pathlib import Path\n"
                "from core.utils import helper\n\n"
                "class RetrievalEngine:\n"
                '    """Main retrieval engine."""\n'
                "    def retrieve(self, query):\n"
                "        results = helper(query)\n"
                "        return results\n"
            ),
            "source_path": "core/retrieval.py",
        },
        {
            "id": "chunk_2",
            "content": (
                "def helper(text):\n"
                '    """Helper function for text processing."""\n'
                "    return text.lower()\n\n"
                "def validate(data):\n"
                "    return helper(data)\n"
            ),
            "source_path": "core/utils.py",
        },
        {
            "id": "chunk_3",
            "content": (
                "class SmartRetriever(RetrievalEngine):\n"
                "    def smart_retrieve(self, query):\n"
                "        base = self.retrieve(query)\n"
                "        return validate(base)\n"
            ),
            "source_path": "core/smart.py",
        },
    ]


# ---------------------------------------------------------------------------
# Entity / Relation dataclasses
# ---------------------------------------------------------------------------

class TestDataclasses:

    def test_entity_defaults(self):
        e = Entity(entity_id="a", name="foo", entity_type="function", source_file="x.py")
        assert e.line_number == 0
        assert e.docstring == ""
        assert e.chunk_id == ""

    def test_relation_defaults(self):
        r = Relation(source_id="a", target_id="b", relation_type="calls")
        assert r.weight == 1.0


# ---------------------------------------------------------------------------
# Construction and DB init
# ---------------------------------------------------------------------------

class TestConstruction:

    def test_creates_db(self, tmp_db):
        stats = tmp_db.stats()
        assert stats["entities"] == 0
        assert stats["relations"] == 0
        assert stats["communities"] == 0

    def test_db_path_created(self, tmp_path):
        db_path = str(tmp_path / "sub" / "dir" / "kg.db")
        kg = CodeKnowledgeGraph(db_path=db_path)
        assert os.path.exists(db_path)


# ---------------------------------------------------------------------------
# Entity extraction
# ---------------------------------------------------------------------------

class TestEntityExtraction:

    def test_extracts_functions(self, tmp_db):
        code = "def foo():\n    pass\n\ndef bar(x):\n    return x\n"
        entities = tmp_db._extract_entities(code, "test.py", "c1")
        names = [e.name for e in entities]
        assert "foo" in names
        assert "bar" in names
        assert all(e.entity_type == "function" for e in entities if e.name in ("foo", "bar"))

    def test_extracts_classes(self, tmp_db):
        code = "class MyClass:\n    pass\n\nclass Other(Base):\n    pass\n"
        entities = tmp_db._extract_entities(code, "test.py", "c1")
        names = [e.name for e in entities]
        assert "MyClass" in names
        assert "Other" in names

    def test_extracts_imports(self, tmp_db):
        code = "from os.path import join\nfrom core.utils import helper\n"
        entities = tmp_db._extract_entities(code, "test.py", "c1")
        import_entities = [e for e in entities if e.entity_type == "import"]
        assert len(import_entities) == 2
        names = [e.name for e in import_entities]
        assert "join" in names
        assert "helper" in names

    def test_extracts_docstrings(self, tmp_db):
        code = 'def documented():\n    """This is a docstring."""\n    pass\n'
        entities = tmp_db._extract_entities(code, "test.py", "c1")
        func = [e for e in entities if e.name == "documented"][0]
        assert "docstring" in func.docstring.lower()


# ---------------------------------------------------------------------------
# Relation extraction
# ---------------------------------------------------------------------------

class TestRelationExtraction:

    def test_extracts_calls(self, tmp_db):
        code = "def outer():\n    inner()\n    process(data)\n"
        rels = tmp_db._extract_relations(code, "test.py")
        call_targets = [r.target_id for r in rels if r.relation_type == "calls"]
        assert any("inner" in t for t in call_targets)
        assert any("process" in t for t in call_targets)

    def test_extracts_inheritance(self, tmp_db):
        code = "class Child(Parent):\n    pass\n"
        rels = tmp_db._extract_relations(code, "test.py")
        inherits = [r for r in rels if r.relation_type == "inherits"]
        assert len(inherits) == 1
        assert "Parent" in inherits[0].target_id

    def test_extracts_import_relations(self, tmp_db):
        code = "from core.utils import helper\n"
        rels = tmp_db._extract_relations(code, "test.py")
        imports = [r for r in rels if r.relation_type == "imports"]
        assert len(imports) == 1

    def test_filters_builtins(self, tmp_db):
        code = "def foo():\n    print(len(str(42)))\n    my_func()\n"
        rels = tmp_db._extract_relations(code, "test.py")
        call_targets = [r.target_id for r in rels if r.relation_type == "calls"]
        # print, len, str should be filtered out
        assert not any("print" in t for t in call_targets)
        assert not any("::len" in t for t in call_targets)
        assert any("my_func" in t for t in call_targets)


# ---------------------------------------------------------------------------
# build_from_chunks
# ---------------------------------------------------------------------------

class TestBuildFromChunks:

    def test_builds_graph(self, tmp_db):
        stats = tmp_db.build_from_chunks(_code_chunks())
        assert stats["entities"] > 0
        assert stats["relations"] > 0
        assert stats["chunks_processed"] == 3

    def test_stats_after_build(self, tmp_db):
        tmp_db.build_from_chunks(_code_chunks())
        stats = tmp_db.stats()
        assert stats["entities"] > 0
        assert "function" in stats["by_type"]

    def test_empty_chunks(self, tmp_db):
        stats = tmp_db.build_from_chunks([])
        assert stats["entities"] == 0
        assert stats["chunks_processed"] == 0


class _BrokenConnection:

    def __enter__(self):
        raise RuntimeError("db locked")

    def __exit__(self, exc_type, exc, tb):
        return False


class TestLogging:

    def test_write_failures_are_logged(self, tmp_db, monkeypatch, caplog):
        monkeypatch.setattr(tmp_db, "_connect", lambda: _BrokenConnection())
        entity = Entity(entity_id="e1", name="foo", entity_type="function", source_file="x.py")
        relation = Relation(source_id="a", target_id="b", relation_type="calls")

        with caplog.at_level(logging.WARNING, logger="core.knowledge_graph"):
            tmp_db._add_entity(entity)
            tmp_db._add_relation(relation)

        assert any("Failed to add entity" in rec.message for rec in caplog.records)
        assert any("Failed to add relation" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# Query (Personalized PageRank)
# ---------------------------------------------------------------------------

class TestQuery:

    def test_query_returns_results(self, tmp_db):
        tmp_db.build_from_chunks(_code_chunks())
        results = tmp_db.query("retrieve helper")
        assert len(results) > 0
        assert "score" in results[0]
        assert "name" in results[0]

    def test_query_no_match(self, tmp_db):
        tmp_db.build_from_chunks(_code_chunks())
        results = tmp_db.query("zzz_nonexistent_term")
        assert results == []

    def test_query_top_k(self, tmp_db):
        tmp_db.build_from_chunks(_code_chunks())
        results = tmp_db.query("helper retrieve validate", top_k=2)
        assert len(results) <= 2

    def test_query_empty_graph(self, tmp_db):
        results = tmp_db.query("anything")
        assert results == []


# ---------------------------------------------------------------------------
# get_neighbors
# ---------------------------------------------------------------------------

class TestGetNeighbors:

    def test_finds_neighbors(self, tmp_db):
        tmp_db.build_from_chunks(_code_chunks())
        # Find an entity that should have relations
        stats = tmp_db.stats()
        if stats["relations"] > 0:
            # Get first entity with known relations
            with tmp_db._connect() as conn:
                row = conn.execute(
                    "SELECT source_id FROM relations LIMIT 1"
                ).fetchone()
            if row:
                neighbors = tmp_db.get_neighbors(row[0])
                assert len(neighbors) > 0
                assert "relation" in neighbors[0]

    def test_no_neighbors_for_unknown(self, tmp_db):
        neighbors = tmp_db.get_neighbors("nonexistent_entity")
        assert neighbors == []


# ---------------------------------------------------------------------------
# Communities
# ---------------------------------------------------------------------------

class TestCommunities:

    def test_communities_built(self, tmp_db):
        tmp_db.build_from_chunks(_code_chunks())
        stats = tmp_db.stats()
        # Should have at least 1 community if there are relations
        if stats["relations"] > 0:
            assert stats["communities"] >= 1

    def test_empty_graph_no_communities(self, tmp_db):
        stats = tmp_db.stats()
        assert stats["communities"] == 0
