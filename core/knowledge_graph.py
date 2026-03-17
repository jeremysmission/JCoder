"""
Knowledge Graph RAG (HippoRAG-Inspired)
------------------------------------------
Builds and traverses a knowledge graph over the codebase for
associative retrieval. Goes beyond flat vector search by
connecting entities across documents.

Based on:
- HippoRAG 2 (ICML 2025): From RAG to Memory via knowledge graphs
- GraphRAG (Microsoft, 2024): Community detection + summarization
- RAPTOR (2024): Recursive abstractive processing for tree-organized retrieval

The key insight: Standard RAG retrieves isolated chunks. Knowledge
graph RAG retrieves CONNECTED information -- if function A calls
function B, and B uses class C, the graph links them even if
they're in different files.

Graph construction (all local, SQLite):
1. Entity extraction: functions, classes, imports, variables
2. Relation extraction: calls, imports, inherits, uses
3. Personalized PageRank: walk the graph from query-relevant nodes
4. Community detection: find clusters of related code

Zero external dependencies beyond the LLM.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class Entity:
    """A code entity in the knowledge graph."""
    entity_id: str
    name: str
    entity_type: str  # "function" | "class" | "module" | "variable" | "import"
    source_file: str
    line_number: int = 0
    docstring: str = ""
    chunk_id: str = ""


@dataclass
class Relation:
    """A relationship between two entities."""
    source_id: str
    target_id: str
    relation_type: str  # "calls" | "imports" | "inherits" | "uses" | "defines"
    weight: float = 1.0


class CodeKnowledgeGraph:
    """
    Knowledge graph over a codebase with Personalized PageRank retrieval.

    Build phase:
    - Extract entities (functions, classes, imports) from code chunks
    - Extract relations (calls, imports, inherits) from code structure
    - Store in SQLite for persistence

    Query phase:
    - Find seed entities matching the query
    - Run Personalized PageRank from seed nodes
    - Return top-ranked entities and their associated chunks
    """

    def __init__(self, db_path: str = "_kg/knowledge_graph.db"):
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self._db_path))

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS entities (
                    entity_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    entity_type TEXT NOT NULL,
                    source_file TEXT,
                    line_number INTEGER DEFAULT 0,
                    docstring TEXT,
                    chunk_id TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS relations (
                    source_id TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    relation_type TEXT NOT NULL,
                    weight REAL DEFAULT 1.0,
                    PRIMARY KEY (source_id, target_id, relation_type)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS communities (
                    community_id INTEGER,
                    entity_id TEXT,
                    PRIMARY KEY (community_id, entity_id)
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_entity_name
                ON entities(name)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_entity_type
                ON entities(entity_type)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_rel_source
                ON relations(source_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_rel_target
                ON relations(target_id)
            """)
            conn.commit()

    def build_from_chunks(self, chunks: List[Dict[str, str]]) -> Dict[str, int]:
        """
        Build knowledge graph from code chunks.

        Args:
            chunks: List of {"content": ..., "source_path": ..., "id": ...}

        Returns:
            Stats dict with entity and relation counts.
        """
        entities_added = 0
        relations_added = 0

        for chunk in chunks:
            content = chunk.get("content", "")
            source = chunk.get("source_path", "")
            chunk_id = chunk.get("id", "")

            # Extract entities
            new_entities = self._extract_entities(content, source, chunk_id)
            for ent in new_entities:
                self._add_entity(ent)
                entities_added += 1

            # Extract relations
            new_relations = self._extract_relations(content, source)
            for rel in new_relations:
                self._add_relation(rel)
                relations_added += 1

        # Build communities (simple connected components)
        self._build_communities()

        return {
            "entities": entities_added,
            "relations": relations_added,
            "chunks_processed": len(chunks),
        }

    def query(
        self,
        query: str,
        top_k: int = 10,
        damping: float = 0.85,
        max_iterations: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Query the knowledge graph using Personalized PageRank.

        Args:
            query: Natural language query
            top_k: Number of results to return
            damping: PageRank damping factor (0.85 standard)
            max_iterations: Max PageRank iterations

        Returns:
            List of {"entity": ..., "score": ..., "chunk_id": ...}
        """
        # Find seed entities matching query terms
        seeds = self._find_seeds(query)
        if not seeds:
            return []

        # Run Personalized PageRank
        scores = self._personalized_pagerank(
            seeds, damping, max_iterations
        )

        # Get top-k entities with their metadata
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        results = []

        with self._connect() as conn:
            for entity_id, score in ranked[:top_k]:
                cur = conn.execute(
                    "SELECT name, entity_type, source_file, docstring, chunk_id "
                    "FROM entities WHERE entity_id = ?",
                    (entity_id,),
                )
                row = cur.fetchone()
                if row:
                    results.append({
                        "entity_id": entity_id,
                        "name": row[0],
                        "type": row[1],
                        "source_file": row[2],
                        "docstring": row[3] or "",
                        "chunk_id": row[4] or "",
                        "score": round(score, 6),
                    })

        return results

    def get_neighbors(self, entity_id: str) -> List[Dict[str, str]]:
        """Get all entities connected to a given entity."""
        with self._connect() as conn:
            cur = conn.execute("""
                SELECT r.target_id, r.relation_type, e.name, e.entity_type
                FROM relations r JOIN entities e ON r.target_id = e.entity_id
                WHERE r.source_id = ?
                UNION
                SELECT r.source_id, r.relation_type, e.name, e.entity_type
                FROM relations r JOIN entities e ON r.source_id = e.entity_id
                WHERE r.target_id = ?
            """, (entity_id, entity_id))
            return [
                {
                    "entity_id": r[0], "relation": r[1],
                    "name": r[2], "type": r[3],
                }
                for r in cur.fetchall()
            ]

    def _extract_entities(
        self, code: str, source: str, chunk_id: str
    ) -> List[Entity]:
        """Extract code entities from a chunk using regex."""
        entities = []

        # Functions
        for m in re.finditer(r"def\s+(\w+)\s*\(", code):
            eid = f"{source}::{m.group(1)}"
            docstring = ""
            # Try to find docstring
            after = code[m.end():]
            doc_m = re.search(r'"""(.*?)"""', after[:300], re.DOTALL)
            if doc_m:
                docstring = doc_m.group(1).strip()[:200]
            entities.append(Entity(
                entity_id=eid, name=m.group(1),
                entity_type="function", source_file=source,
                docstring=docstring, chunk_id=chunk_id,
            ))

        # Classes
        for m in re.finditer(r"class\s+(\w+)[\s(:]", code):
            eid = f"{source}::{m.group(1)}"
            entities.append(Entity(
                entity_id=eid, name=m.group(1),
                entity_type="class", source_file=source,
                chunk_id=chunk_id,
            ))

        # Imports
        for m in re.finditer(r"from\s+([\w.]+)\s+import\s+(\w+)", code):
            module = m.group(1)
            name = m.group(2)
            eid = f"import::{module}.{name}"
            entities.append(Entity(
                entity_id=eid, name=name,
                entity_type="import", source_file=source,
                chunk_id=chunk_id,
            ))

        return entities

    def _extract_relations(
        self, code: str, source: str
    ) -> List[Relation]:
        """Extract relationships between entities."""
        relations = []

        # Function calls: name(
        for m in re.finditer(r"(\w+)\s*\(", code):
            caller_context = code[:m.start()]
            # Find enclosing function
            func_m = list(re.finditer(r"def\s+(\w+)", caller_context))
            if func_m:
                caller = f"{source}::{func_m[-1].group(1)}"
                callee_name = m.group(1)
                if callee_name not in ("if", "for", "while", "with", "print",
                                        "range", "len", "str", "int", "float",
                                        "list", "dict", "set", "tuple", "type",
                                        "isinstance", "hasattr", "getattr"):
                    relations.append(Relation(
                        source_id=caller,
                        target_id=f"*::{callee_name}",
                        relation_type="calls",
                    ))

        # Inheritance: class X(Y):
        for m in re.finditer(r"class\s+(\w+)\s*\(([^)]+)\)", code):
            child = f"{source}::{m.group(1)}"
            parents = [p.strip() for p in m.group(2).split(",")]
            for parent in parents:
                parent_name = parent.split(".")[-1]
                if parent_name and parent_name not in ("object",):
                    relations.append(Relation(
                        source_id=child,
                        target_id=f"*::{parent_name}",
                        relation_type="inherits",
                    ))

        # Import relations
        for m in re.finditer(r"from\s+([\w.]+)\s+import\s+(\w+)", code):
            module = m.group(1)
            name = m.group(2)
            relations.append(Relation(
                source_id=f"{source}::*",
                target_id=f"import::{module}.{name}",
                relation_type="imports",
            ))

        return relations

    def _add_entity(self, entity: Entity) -> None:
        try:
            with self._connect() as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO entities
                    (entity_id, name, entity_type, source_file,
                     line_number, docstring, chunk_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    entity.entity_id, entity.name, entity.entity_type,
                    entity.source_file, entity.line_number,
                    entity.docstring, entity.chunk_id,
                ))
                conn.commit()
        except Exception as exc:
            logger.warning(
                "Failed to add entity %s (%s): %s",
                entity.entity_id,
                entity.entity_type,
                exc,
            )

    def _add_relation(self, rel: Relation) -> None:
        try:
            with self._connect() as conn:
                # Resolve wildcard targets
                if rel.target_id.startswith("*::"):
                    name = rel.target_id.split("::")[1]
                    cur = conn.execute(
                        "SELECT entity_id FROM entities WHERE name = ? LIMIT 1",
                        (name,),
                    )
                    row = cur.fetchone()
                    if row:
                        rel.target_id = row[0]
                    else:
                        return  # Target not found

                conn.execute("""
                    INSERT OR REPLACE INTO relations
                    (source_id, target_id, relation_type, weight)
                    VALUES (?, ?, ?, ?)
                """, (rel.source_id, rel.target_id, rel.relation_type, rel.weight))
                conn.commit()
        except Exception as exc:
            logger.warning(
                "Failed to add relation %s -> %s (%s): %s",
                rel.source_id,
                rel.target_id,
                rel.relation_type,
                exc,
            )

    @staticmethod
    def _escape_like(term: str) -> str:
        """Escape LIKE wildcard characters so user input is treated literally."""
        return term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")

    def _find_seeds(self, query: str) -> Set[str]:
        """Find entities matching query terms."""
        words = re.findall(r"[a-zA-Z_]\w*", query)
        seeds = set()

        with self._connect() as conn:
            for word in words:
                if len(word) < 3:
                    continue
                cur = conn.execute(
                    "SELECT entity_id FROM entities WHERE name LIKE ? ESCAPE '\\'",
                    (f"%{self._escape_like(word)}%",),
                )
                for row in cur.fetchall():
                    seeds.add(row[0])

        return seeds

    def _personalized_pagerank(
        self,
        seeds: Set[str],
        damping: float,
        max_iterations: int,
    ) -> Dict[str, float]:
        """
        Personalized PageRank from seed nodes.
        Teleport probability goes back to seed nodes, not uniformly.
        """
        # Build adjacency
        adj: Dict[str, List[str]] = defaultdict(list)
        with self._connect() as conn:
            cur = conn.execute("SELECT source_id, target_id FROM relations")
            for row in cur.fetchall():
                adj[row[0]].append(row[1])
                adj[row[1]].append(row[0])  # undirected

        all_nodes = set(adj.keys()) | seeds
        n = len(all_nodes)
        if n == 0:
            return {}

        # Initialize: uniform over seeds
        scores = {node: 0.0 for node in all_nodes}
        seed_weight = 1.0 / len(seeds) if seeds else 0.0
        for s in seeds:
            scores[s] = seed_weight

        # Iterate
        for _ in range(max_iterations):
            new_scores = {node: 0.0 for node in all_nodes}
            for node in all_nodes:
                neighbors = adj.get(node, [])
                if neighbors:
                    share = scores[node] * damping / len(neighbors)
                    for neighbor in neighbors:
                        if neighbor in new_scores:
                            new_scores[neighbor] += share

                # Teleport to seeds
                teleport = (1.0 - damping) * seed_weight
                if node in seeds:
                    new_scores[node] += teleport

            scores = new_scores

        return scores

    def _build_communities(self) -> None:
        """Simple connected components as communities."""
        adj: Dict[str, Set[str]] = defaultdict(set)
        with self._connect() as conn:
            cur = conn.execute("SELECT source_id, target_id FROM relations")
            for row in cur.fetchall():
                adj[row[0]].add(row[1])
                adj[row[1]].add(row[0])

        visited = set()
        community_id = 0

        with self._connect() as conn:
            conn.execute("DELETE FROM communities")

            for node in adj:
                if node in visited:
                    continue
                # BFS
                queue = [node]
                component = set()
                while queue:
                    current = queue.pop(0)
                    if current in visited:
                        continue
                    visited.add(current)
                    component.add(current)
                    for neighbor in adj[current]:
                        if neighbor not in visited:
                            queue.append(neighbor)

                for entity_id in component:
                    conn.execute(
                        "INSERT OR REPLACE INTO communities VALUES (?, ?)",
                        (community_id, entity_id),
                    )
                community_id += 1

            conn.commit()

    def stats(self) -> Dict[str, Any]:
        with self._connect() as conn:
            entities = conn.execute(
                "SELECT COUNT(*) FROM entities"
            ).fetchone()[0]
            relations = conn.execute(
                "SELECT COUNT(*) FROM relations"
            ).fetchone()[0]
            communities = conn.execute(
                "SELECT COUNT(DISTINCT community_id) FROM communities"
            ).fetchone()[0]
            by_type = conn.execute(
                "SELECT entity_type, COUNT(*) FROM entities "
                "GROUP BY entity_type"
            ).fetchall()

        return {
            "entities": entities,
            "relations": relations,
            "communities": communities,
            "by_type": {r[0]: r[1] for r in by_type},
        }
