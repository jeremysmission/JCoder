"""
Agent Memory
------------
Personal RAG for the JCoder agent -- stores and retrieves learned knowledge.

Non-programmer explanation:
Every time the agent completes a task, it can store what it learned.
Later, when it faces a similar task, it searches its own memory for
relevant past knowledge. This is like an engineer's personal notebook,
except it's searchable by meaning (vectors) and by keywords (FTS5).

Works in two modes:
  - Full mode: vector + keyword hybrid search (requires embedding server)
  - Fallback mode: keyword-only search (works offline without any server)
"""

import json
import logging
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np

from agent.core import AgentResult
from core.config import StorageConfig
from core.embedding_engine import EmbeddingEngine
from core.index_engine import IndexEngine

log = logging.getLogger(__name__)

_DEDUP_THRESHOLD = 0.95  # cosine similarity above this = duplicate


@dataclass
class MemoryEntry:
    """A single piece of learned knowledge."""
    id: str
    content: str
    source_task: str
    tags: List[str]
    confidence: float
    timestamp: str
    tokens_used: int


class AgentMemory:
    """The agent's personal RAG -- stores and retrieves its own learned knowledge.

    Operates in two modes:
      - Full hybrid (embedding_engine provided): FAISS vectors + FTS5 keywords
      - FTS5-only (no embedding_engine): keyword search only, no server needed
    """

    def __init__(
        self,
        embedding_engine: EmbeddingEngine = None,
        index_dir: str = "data/indexes",
        index_name: str = "agent_memory",
        dimension: int = 768,
        knowledge_dir: str = "data/agent_knowledge",
    ):
        self._embedder = embedding_engine
        self._index_name = index_name
        self._knowledge_dir = knowledge_dir
        self._sparse_only = embedding_engine is None

        storage = StorageConfig(data_dir=os.path.dirname(index_dir), index_dir=index_dir)
        self._index = IndexEngine(
            dimension=dimension,
            storage=storage,
            sparse_only=self._sparse_only,
        )

        os.makedirs(knowledge_dir, exist_ok=True)
        os.makedirs(index_dir, exist_ok=True)

        # Load existing index if present
        faiss_path = os.path.join(index_dir, index_name + ".faiss")
        meta_path = os.path.join(index_dir, index_name + ".meta.json")
        if not self._sparse_only and os.path.exists(faiss_path) and os.path.exists(meta_path):
            try:
                self._index.load(index_name)
                log.info("[OK] Loaded agent memory: %d entries", self._index.count)
            except Exception as exc:
                log.warning("[WARN] Failed to load agent memory index: %s", exc)
        elif self._sparse_only and os.path.exists(meta_path):
            # FTS5-only: load metadata and rebuild keyword index
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    self._index.metadata = json.load(f)
                self._index._db_path = os.path.join(index_dir, index_name + ".fts5.db")
                self._index._build_fts5()
                log.info("[OK] Loaded agent memory (FTS5-only): %d entries", self._index.count)
            except Exception as exc:
                log.warning("[WARN] Failed to load FTS5 memory: %s", exc)

    def ingest(
        self,
        content: str,
        source_task: str,
        tags: List[str] = None,
        confidence: float = 1.0,
        tokens_used: int = 0,
    ) -> MemoryEntry:
        """Add a knowledge entry to the memory store."""
        if not content or not content.strip():
            raise ValueError("Cannot ingest empty content")

        # Deduplication: check cosine similarity against existing entries
        if self._embedder is not None and self._index.count > 0:
            query_vec = self._embedder.embed_single(content)
            hits = self._index.search_vectors(query_vec, k=1)
            if hits:
                _, score = hits[0]
                if score > _DEDUP_THRESHOLD:
                    idx = hits[0][0]
                    existing = self._index.metadata[idx]
                    log.info("[WARN] Skipping duplicate (score=%.3f) of entry %s",
                             score, existing.get("id", "?"))
                    return MemoryEntry(
                        id=existing["id"],
                        content=existing.get("content", ""),
                        source_task=existing.get("source_task", ""),
                        tags=existing.get("tags", []),
                        confidence=existing.get("confidence", 1.0),
                        timestamp=existing.get("timestamp", ""),
                        tokens_used=existing.get("tokens_used", 0),
                    )

        entry = MemoryEntry(
            id=str(uuid.uuid4()),
            content=content.strip(),
            source_task=source_task,
            tags=tags or [],
            confidence=confidence,
            timestamp=datetime.now(timezone.utc).isoformat(),
            tokens_used=tokens_used,
        )

        meta = asdict(entry)

        if self._embedder is not None:
            vec = self._embedder.embed_single(content)
            self._index.add_vectors(vec.reshape(1, -1), [meta])
        else:
            self._index.metadata.append(meta)

        # Save index to disk
        self._save()

        # Write human-readable .md backup
        self._write_md(entry)

        log.info("[OK] Ingested memory %s (%d chars, tags=%s)", entry.id[:8], len(content), entry.tags)
        return entry

    def ingest_task_result(self, task: str, result: AgentResult) -> Optional[MemoryEntry]:
        """Auto-ingest an agent task result into memory.

        Only ingests successful results. Builds content from the task
        description, result summary, and key tool steps.
        """
        if not result.success:
            return None
        if not result.summary or not result.summary.strip():
            return None

        # Build structured content
        lines = [f"Task: {task}", "", f"Result: {result.summary}"]
        if result.steps:
            lines.append("")
            lines.append("Key steps:")
            for step in result.steps:
                status = "OK" if step.tool_success else "FAIL"
                lines.append(f"  - [{status}] {step.tool_name} ({step.elapsed_s:.1f}s)")

        content = "\n".join(lines)

        # Extract tags from tool names and task text
        tags = _extract_tags(task, result)
        tokens_used = result.total_input_tokens + result.total_output_tokens

        return self.ingest(
            content=content,
            source_task=task,
            tags=tags,
            confidence=1.0 if not result.timed_out else 0.5,
            tokens_used=tokens_used,
        )

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search the agent's memory for relevant past knowledge.

        Returns list of dicts with keys:
          content, source_task, tags, confidence, score, timestamp
        """
        if self._index.count == 0:
            return []

        if self._embedder is not None:
            query_vec = self._embedder.embed_single(query)
            raw = self._index.hybrid_search(query_vec, query, k=top_k)
        else:
            raw = self._index.search_keywords(query, k=top_k)

        results = []
        for item in raw:
            if self._embedder is not None:
                score, meta = item  # hybrid_search returns (score, meta_dict)
            else:
                idx, score = item  # search_keywords returns (idx, score)
                meta = self._index.metadata[idx]

            results.append({
                "content": meta.get("content", ""),
                "source_task": meta.get("source_task", ""),
                "tags": meta.get("tags", []),
                "confidence": meta.get("confidence", 1.0),
                "score": score,
                "timestamp": meta.get("timestamp", ""),
            })
        return results

    def forget(self, entry_id: str) -> bool:
        """Remove a specific memory entry by ID.

        Rebuilds the index without the entry. Returns True if found and removed.
        """
        idx_to_remove = None
        for i, m in enumerate(self._index.metadata):
            if m.get("id") == entry_id:
                idx_to_remove = i
                break
        if idx_to_remove is None:
            return False

        # Remove metadata entry
        self._index.metadata.pop(idx_to_remove)

        # Rebuild FAISS index if we have embeddings
        if self._embedder is not None and self._index.count > 0:
            # Re-embed all remaining entries
            texts = [m.get("content", "") for m in self._index.metadata]
            vectors = self._embedder.embed(texts)
            # Reset FAISS index
            self._index.index.reset()
            self._index.index.add(vectors)

        self._save()

        # Delete .md backup file
        md_path = os.path.join(self._knowledge_dir, f"{entry_id}.md")
        if os.path.exists(md_path):
            os.remove(md_path)

        log.info("[OK] Forgot memory %s", entry_id[:8])
        return True

    def stats(self) -> Dict:
        """Return memory statistics."""
        entries = self._index.metadata
        total_tokens = sum(m.get("tokens_used", 0) for m in entries)
        all_tags = set()
        for m in entries:
            all_tags.update(m.get("tags", []))

        timestamps = [m.get("timestamp", "") for m in entries if m.get("timestamp")]
        oldest = min(timestamps) if timestamps else None
        newest = max(timestamps) if timestamps else None

        return {
            "total_entries": len(entries),
            "total_tokens_used": total_tokens,
            "topics": sorted(all_tags),
            "oldest": oldest,
            "newest": newest,
            "mode": "hybrid" if self._embedder else "fts5_only",
        }

    def export_knowledge(self, output_path: str):
        """Export all knowledge to a single markdown file for human review."""
        lines = ["# Agent Knowledge Export", ""]
        for m in self._index.metadata:
            lines.append(f"## {m.get('source_task', 'Unknown task')}")
            lines.append(f"**ID:** {m.get('id', '?')}")
            lines.append(f"**Tags:** {', '.join(m.get('tags', []))}")
            lines.append(f"**Confidence:** {m.get('confidence', 1.0)}")
            lines.append(f"**Timestamp:** {m.get('timestamp', '?')}")
            lines.append(f"**Tokens:** {m.get('tokens_used', 0)}")
            lines.append("")
            lines.append(m.get("content", ""))
            lines.append("")
            lines.append("---")
            lines.append("")

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        log.info("[OK] Exported %d entries to %s", len(self._index.metadata), output_path)

    def close(self):
        """Release resources."""
        self._index.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    # -- Internal helpers --------------------------------------------------

    def _save(self):
        """Persist index to disk."""
        if self._sparse_only:
            # FTS5-only: save metadata JSON and rebuild FTS5 DB
            idx_dir = self._index.storage.index_dir
            meta_path = os.path.join(idx_dir, self._index_name + ".meta.json")
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(self._index.metadata, f, ensure_ascii=False)
            self._index._db_path = os.path.join(idx_dir, self._index_name + ".fts5.db")
            self._index._build_fts5()
        else:
            self._index.save(self._index_name)

    def _write_md(self, entry: MemoryEntry):
        """Write a human-readable .md backup of an entry."""
        lines = [
            f"# {entry.source_task}",
            "",
            f"- **ID:** {entry.id}",
            f"- **Tags:** {', '.join(entry.tags)}",
            f"- **Confidence:** {entry.confidence}",
            f"- **Timestamp:** {entry.timestamp}",
            f"- **Tokens used:** {entry.tokens_used}",
            "",
            entry.content,
        ]
        path = os.path.join(self._knowledge_dir, f"{entry.id}.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))


def _extract_tags(task: str, result: AgentResult) -> List[str]:
    """Extract topic tags from a task description and its result steps."""
    tags = set()

    # Tags from tool names used
    for step in result.steps:
        name = step.tool_name.lower()
        if "read" in name or "file" in name:
            tags.add("file_io")
        elif "write" in name or "edit" in name:
            tags.add("code_edit")
        elif "exec" in name or "bash" in name or "shell" in name:
            tags.add("execution")
        elif "search" in name or "grep" in name:
            tags.add("search")
        elif "test" in name:
            tags.add("testing")

    # Tags from task text keywords
    task_lower = task.lower()
    keyword_map = {
        "bug": "bugfix", "fix": "bugfix", "error": "bugfix",
        "test": "testing", "pytest": "testing", "assert": "testing",
        "refactor": "refactoring", "rename": "refactoring",
        "config": "configuration", "yaml": "configuration",
        "api": "api", "endpoint": "api", "server": "api",
        "index": "indexing", "embed": "indexing", "vector": "indexing",
        "search": "search", "retriev": "search", "query": "search",
    }
    for keyword, tag in keyword_map.items():
        if keyword in task_lower:
            tags.add(tag)

    return sorted(tags)
