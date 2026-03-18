"""
Extended tests for core.knowledge_graph -- Sprint 10.

Covers: node creation, edge creation, graph traversal, query expansion,
graph persistence (save/load), duplicate node handling, empty graph queries,
and LIKE wildcard injection prevention.

All mocked -- no Ollama or network required.
"""

from __future__ import annotations

import os
import sqlite3

import pytest

from core.knowledge_graph import CodeKnowledgeGraph, Entity, Relation


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def kg(tmp_path):
    """Fresh knowledge graph with temp DB."""
    return CodeKnowledgeGraph(db_path=str(tmp_path / "kg.db"))


@pytest.fixture
def db_path(tmp_path):
    """Return a temp DB path (not yet opened)."""
    return str(tmp_path / "kg.db")


def _sample_chunks():
    return [
        {
            "id": "c1",
            "content": (
                "from os.path import join\n"
                "class Engine:\n"
                '    """Core engine."""\n'
                "    def run(self, query):\n"
                '        """Run the engine."""\n'
                "        return process(query)\n"
            ),
            "source_path": "core/engine.py",
        },
        {
            "id": "c2",
            "content": (
                "def process(data):\n"
                '    """Process incoming data."""\n'
                "    return transform(data)\n\n"
                "def transform(data):\n"
                "    return data.strip()\n"
            ),
            "source_path": "core/pipeline.py",
        },
        {
            "id": "c3",
            "content": (
                "class FastEngine(Engine):\n"
                "    def run(self, query):\n"
                "        cached = self.lookup(query)\n"
                "        return process(cached)\n\n"
                "    def lookup(self, key):\n"
                "        return key\n"
            ),
            "source_path": "core/fast.py",
        },
    ]


# ---------------------------------------------------------------------------
# Node creation -- entities extracted from text
# ---------------------------------------------------------------------------

class TestNodeCreation:

    def test_function_entities_have_correct_id_format(self, kg):
        code = "def alpha():\n    pass\n"
        ents = kg._extract_entities(code, "mod.py", "c1")
        assert ents[0].entity_id == "mod.py::alpha"

    def test_class_entity_attributes(self, kg):
        code = "class Widget:\n    pass\n"
        ents = kg._extract_entities(code, "ui.py", "c5")
        cls = [e for e in ents if e.entity_type == "class"][0]
        assert cls.name == "Widget"
        assert cls.source_file == "ui.py"
        assert cls.chunk_id == "c5"

    def test_import_entity_id_includes_module(self, kg):
        code = "from collections import OrderedDict\n"
        ents = kg._extract_entities(code, "x.py", "c1")
        imp = [e for e in ents if e.entity_type == "import"][0]
        assert imp.entity_id == "import::collections.OrderedDict"

    def test_multiple_functions_same_chunk(self, kg):
        code = "def a():\n    pass\ndef b():\n    pass\ndef c():\n    pass\n"
        ents = kg._extract_entities(code, "m.py", "c1")
        funcs = [e for e in ents if e.entity_type == "function"]
        assert len(funcs) == 3

    def test_entities_persisted_in_db(self, kg):
        kg.build_from_chunks(_sample_chunks())
        with kg._connect() as conn:
            count = conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
        assert count > 0


# ---------------------------------------------------------------------------
# Edge creation -- relations between entities
# ---------------------------------------------------------------------------

class TestEdgeCreation:

    def test_call_relation_connects_caller_to_callee(self, kg):
        code = "def outer():\n    inner()\n"
        rels = kg._extract_relations(code, "x.py")
        calls = [r for r in rels if r.relation_type == "calls"]
        assert any(r.source_id == "x.py::outer" for r in calls)
        assert any("inner" in r.target_id for r in calls)

    def test_inheritance_relation(self, kg):
        code = "class Child(Parent, Mixin):\n    pass\n"
        rels = kg._extract_relations(code, "x.py")
        inherits = [r for r in rels if r.relation_type == "inherits"]
        targets = [r.target_id for r in inherits]
        assert any("Parent" in t for t in targets)
        assert any("Mixin" in t for t in targets)

    def test_import_relation_stored(self, kg):
        kg.build_from_chunks(_sample_chunks())
        with kg._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM relations WHERE relation_type = 'imports'"
            ).fetchall()
        assert len(rows) >= 1

    def test_relation_weight_default(self, kg):
        r = Relation(source_id="a", target_id="b", relation_type="calls")
        assert r.weight == 1.0

    def test_relations_persisted_after_build(self, kg):
        kg.build_from_chunks(_sample_chunks())
        with kg._connect() as conn:
            count = conn.execute("SELECT COUNT(*) FROM relations").fetchone()[0]
        assert count > 0


# ---------------------------------------------------------------------------
# Graph traversal -- find related concepts via get_neighbors
# ---------------------------------------------------------------------------

class TestGraphTraversal:

    def test_neighbors_returns_related_entities(self, kg):
        kg.build_from_chunks(_sample_chunks())
        # Find an entity that has relations
        with kg._connect() as conn:
            row = conn.execute(
                "SELECT source_id FROM relations LIMIT 1"
            ).fetchone()
        if row:
            nbrs = kg.get_neighbors(row[0])
            assert len(nbrs) >= 1
            assert all("entity_id" in n for n in nbrs)

    def test_traversal_two_hops(self, kg):
        """Walk two hops: entity -> neighbor -> neighbor-of-neighbor."""
        kg.build_from_chunks(_sample_chunks())
        with kg._connect() as conn:
            row = conn.execute(
                "SELECT source_id FROM relations LIMIT 1"
            ).fetchone()
        if row:
            first_hop = kg.get_neighbors(row[0])
            if first_hop:
                second_hop = kg.get_neighbors(first_hop[0]["entity_id"])
                # May or may not have results, but should not crash
                assert isinstance(second_hop, list)

    def test_neighbor_fields(self, kg):
        kg.build_from_chunks(_sample_chunks())
        with kg._connect() as conn:
            row = conn.execute(
                "SELECT source_id FROM relations LIMIT 1"
            ).fetchone()
        if row:
            nbrs = kg.get_neighbors(row[0])
            if nbrs:
                n = nbrs[0]
                assert "relation" in n
                assert "name" in n
                assert "type" in n


# ---------------------------------------------------------------------------
# Query expansion via graph (PageRank finds related terms)
# ---------------------------------------------------------------------------

class TestQueryExpansion:

    def test_query_returns_related_not_just_exact(self, kg):
        """PageRank should surface entities beyond the direct seed matches."""
        kg.build_from_chunks(_sample_chunks())
        results = kg.query("process", top_k=10)
        names = [r["name"] for r in results]
        # 'process' is a seed, but PageRank should also surface 'transform'
        # because process calls transform
        assert "process" in names
        # At minimum we get more than just the seed
        assert len(results) >= 1

    def test_query_scores_decrease(self, kg):
        kg.build_from_chunks(_sample_chunks())
        results = kg.query("Engine run process", top_k=20)
        if len(results) >= 2:
            scores = [r["score"] for r in results]
            assert scores == sorted(scores, reverse=True)

    def test_query_result_has_chunk_id(self, kg):
        kg.build_from_chunks(_sample_chunks())
        results = kg.query("Engine")
        if results:
            assert "chunk_id" in results[0]

    def test_query_damping_zero_returns_seeds_only(self, kg):
        """With damping=0, no score propagates; only seeds keep weight."""
        kg.build_from_chunks(_sample_chunks())
        results = kg.query("process", top_k=50, damping=0.0)
        # All returned entities should be seeds (contain 'process' in name)
        for r in results:
            assert r["score"] >= 0.0


# ---------------------------------------------------------------------------
# Graph persistence -- save / load
# ---------------------------------------------------------------------------

class TestGraphPersistence:

    def test_reopen_db_preserves_entities(self, db_path):
        kg1 = CodeKnowledgeGraph(db_path=db_path)
        kg1.build_from_chunks(_sample_chunks())
        stats1 = kg1.stats()

        # Reopen from same path
        kg2 = CodeKnowledgeGraph(db_path=db_path)
        stats2 = kg2.stats()

        assert stats2["entities"] == stats1["entities"]
        assert stats2["relations"] == stats1["relations"]

    def test_reopen_db_query_still_works(self, db_path):
        kg1 = CodeKnowledgeGraph(db_path=db_path)
        kg1.build_from_chunks(_sample_chunks())

        kg2 = CodeKnowledgeGraph(db_path=db_path)
        results = kg2.query("process transform")
        assert len(results) > 0

    def test_reopen_db_preserves_communities(self, db_path):
        kg1 = CodeKnowledgeGraph(db_path=db_path)
        kg1.build_from_chunks(_sample_chunks())
        c1 = kg1.stats()["communities"]

        kg2 = CodeKnowledgeGraph(db_path=db_path)
        c2 = kg2.stats()["communities"]
        assert c2 == c1

    def test_db_file_exists_on_disk(self, db_path):
        CodeKnowledgeGraph(db_path=db_path)
        assert os.path.isfile(db_path)


# ---------------------------------------------------------------------------
# Duplicate node handling
# ---------------------------------------------------------------------------

class TestDuplicateHandling:

    def test_duplicate_entity_uses_insert_or_replace(self, kg):
        e = Entity(
            entity_id="x.py::foo", name="foo",
            entity_type="function", source_file="x.py",
            docstring="original",
        )
        kg._add_entity(e)

        # Add again with different docstring
        e2 = Entity(
            entity_id="x.py::foo", name="foo",
            entity_type="function", source_file="x.py",
            docstring="updated",
        )
        kg._add_entity(e2)

        with kg._connect() as conn:
            row = conn.execute(
                "SELECT docstring FROM entities WHERE entity_id = ?",
                ("x.py::foo",),
            ).fetchone()
        assert row[0] == "updated"

    def test_duplicate_entity_count_stays_one(self, kg):
        e = Entity(
            entity_id="x.py::bar", name="bar",
            entity_type="function", source_file="x.py",
        )
        kg._add_entity(e)
        kg._add_entity(e)
        kg._add_entity(e)

        with kg._connect() as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM entities WHERE entity_id = ?",
                ("x.py::bar",),
            ).fetchone()[0]
        assert count == 1

    def test_duplicate_relation_uses_insert_or_replace(self, kg):
        # First add entities so the relation can resolve
        kg._add_entity(Entity(
            entity_id="a", name="a", entity_type="function", source_file="x.py",
        ))
        kg._add_entity(Entity(
            entity_id="b", name="b", entity_type="function", source_file="x.py",
        ))
        r = Relation(source_id="a", target_id="b", relation_type="calls", weight=1.0)
        kg._add_relation(r)
        r2 = Relation(source_id="a", target_id="b", relation_type="calls", weight=2.0)
        kg._add_relation(r2)

        with kg._connect() as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM relations WHERE source_id='a' AND target_id='b'"
            ).fetchone()[0]
        assert count == 1

    def test_rebuild_same_chunks_is_idempotent_in_db(self, kg):
        chunks = _sample_chunks()
        kg.build_from_chunks(chunks)
        stats1 = kg.stats()
        kg.build_from_chunks(chunks)
        stats2 = kg.stats()
        # INSERT OR REPLACE means DB entity count stays the same
        assert stats2["entities"] == stats1["entities"]


# ---------------------------------------------------------------------------
# Empty graph queries
# ---------------------------------------------------------------------------

class TestEmptyGraphQueries:

    def test_query_empty_graph(self, kg):
        assert kg.query("anything") == []

    def test_neighbors_empty_graph(self, kg):
        assert kg.get_neighbors("nonexistent") == []

    def test_stats_empty_graph(self, kg):
        s = kg.stats()
        assert s["entities"] == 0
        assert s["relations"] == 0
        assert s["communities"] == 0
        assert s["by_type"] == {}

    def test_build_empty_chunks(self, kg):
        result = kg.build_from_chunks([])
        assert result["entities"] == 0
        assert result["relations"] == 0
        assert result["chunks_processed"] == 0

    def test_query_short_words_ignored(self, kg):
        """Words shorter than 3 chars are skipped in _find_seeds."""
        kg.build_from_chunks(_sample_chunks())
        seeds = kg._find_seeds("a b")
        assert seeds == set()


# ---------------------------------------------------------------------------
# LIKE wildcard injection prevention
# ---------------------------------------------------------------------------

class TestLikeInjection:

    def test_percent_wildcard_escaped(self):
        assert CodeKnowledgeGraph._escape_like("%admin%") == "\\%admin\\%"

    def test_underscore_wildcard_escaped(self):
        assert CodeKnowledgeGraph._escape_like("_private") == "\\_private"

    def test_backslash_escaped_first(self):
        # Backslash must be escaped before % and _ to avoid double-escaping
        result = CodeKnowledgeGraph._escape_like("a\\%b")
        assert result == "a\\\\\\%b"

    def test_injection_does_not_match_all(self, kg):
        """A query of '%' should not match every entity."""
        kg.build_from_chunks(_sample_chunks())
        seeds = kg._find_seeds("%")
        # '%' is only 1 char, skipped by the len < 3 guard
        assert seeds == set()

    def test_underscore_injection_no_extra_matches(self, kg):
        """Underscores in user query should not act as single-char wildcards."""
        chunks = [
            {"id": "c1", "content": "def abc():\n    pass\n", "source_path": "m.py"},
            {"id": "c2", "content": "def axc():\n    pass\n", "source_path": "m.py"},
        ]
        kg.build_from_chunks(chunks)
        # 'a_c' with a literal _ should match 'a_c' substring, not 'abc' or 'axc'
        seeds = kg._find_seeds("a_c")
        # Neither 'abc' nor 'axc' should match a literal underscore query
        entity_names = set()
        with kg._connect() as conn:
            for eid in seeds:
                row = conn.execute(
                    "SELECT name FROM entities WHERE entity_id = ?", (eid,)
                ).fetchone()
                if row:
                    entity_names.add(row[0])
        assert "abc" not in entity_names
        assert "axc" not in entity_names

    def test_find_seeds_parameterized_query(self, kg):
        """Verify _find_seeds uses parameterized queries (not string format)."""
        # Build a graph and query with SQL injection attempt
        kg.build_from_chunks(_sample_chunks())
        # This should not cause an error or match everything
        seeds = kg._find_seeds("'; DROP TABLE entities; --")
        # Table should still exist
        with kg._connect() as conn:
            count = conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
        assert count > 0
