"""
Build a CodeKnowledgeGraph from indexed code chunks.

Usage:
    python scripts/build_knowledge_graph.py [--db-path PATH] [--index-dir DIR] [--limit N]

Reads FTS5 indexes or chunk files and feeds them into the knowledge graph.
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.knowledge_graph import CodeKnowledgeGraph

log = logging.getLogger(__name__)


def load_chunks_from_fts(index_dir: str, limit: int = 0) -> list:
    """Load code chunks from FTS5 index databases."""
    chunks = []
    idx_path = Path(index_dir)
    if not idx_path.exists():
        log.warning("Index dir not found: %s", index_dir)
        return chunks

    db_files = sorted(idx_path.glob("*.db"))
    log.info("Found %d index databases in %s", len(db_files), index_dir)

    for db_file in db_files:
        try:
            conn = sqlite3.connect(str(db_file))
            # Try common FTS5 table names
            for table in ("chunks", "documents", "content"):
                try:
                    cur = conn.execute(
                        f"SELECT rowid, content, source_path FROM {table}"
                        + (f" LIMIT {limit}" if limit else "")
                    )
                    for row in cur.fetchall():
                        chunks.append({
                            "id": f"{db_file.stem}:{row[0]}",
                            "content": row[1] or "",
                            "source_path": row[2] or str(db_file),
                        })
                    break
                except sqlite3.OperationalError:
                    continue
            conn.close()
        except Exception as exc:
            log.warning("Failed to read %s: %s", db_file, exc)

    return chunks


def load_chunks_from_json(json_path: str) -> list:
    """Load chunks from a JSON file."""
    with open(json_path) as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    return data.get("chunks", [])


def main():
    parser = argparse.ArgumentParser(description="Build CodeKnowledgeGraph")
    parser.add_argument("--db-path", default="_kg/agent_knowledge_graph.db",
                        help="Output knowledge graph database path")
    parser.add_argument("--index-dir", default="data/indexes",
                        help="Directory containing FTS5 index databases")
    parser.add_argument("--json", default=None,
                        help="Load chunks from a JSON file instead of FTS5")
    parser.add_argument("--limit", type=int, default=0,
                        help="Max chunks per index (0 = unlimited)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    t0 = time.time()

    # Load chunks
    if args.json:
        log.info("Loading chunks from %s", args.json)
        chunks = load_chunks_from_json(args.json)
    else:
        log.info("Loading chunks from FTS5 indexes in %s", args.index_dir)
        chunks = load_chunks_from_fts(args.index_dir, args.limit)

    if not chunks:
        log.warning("No chunks loaded -- nothing to build")
        sys.exit(1)

    log.info("Loaded %d chunks in %.1fs", len(chunks), time.time() - t0)

    # Build graph
    kg = CodeKnowledgeGraph(db_path=args.db_path)
    stats = kg.build_from_chunks(chunks)

    elapsed = time.time() - t0
    log.info("[OK] Knowledge graph built in %.1fs", elapsed)
    log.info("  Entities: %d", stats["entities"])
    log.info("  Relations: %d", stats["relations"])
    log.info("  Chunks processed: %d", stats["chunks_processed"])

    # Print final stats
    full_stats = kg.stats()
    log.info("  Communities: %d", full_stats.get("communities", 0))
    if full_stats.get("by_type"):
        for etype, count in full_stats["by_type"].items():
            log.info("    %s: %d", etype, count)


if __name__ == "__main__":
    main()
