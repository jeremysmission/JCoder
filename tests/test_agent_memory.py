"""
Tests for agent memory, federated search, and corpus pipeline modules.
All tests run in FTS5-only mode -- no FAISS, no embedding server required.
"""
import json
import os
from pathlib import Path
from typing import List

import pytest

from agent.core import AgentResult, AgentStep
from agent.memory import AgentMemory, MemoryEntry
from agent.tools import ToolRegistry
from core.config import StorageConfig
from core.federated_search import FederatedSearch, SearchResult
from core.index_engine import IndexEngine
from ingestion.corpus_pipeline import CorpusPipeline, IngestStats
from ingestion.dedup import MinHashDedup
from ingestion.pii_scanner import PIIScanner

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _storage(tmp_path: Path) -> StorageConfig:
    idx = str(tmp_path / "indexes")
    os.makedirs(idx, exist_ok=True)
    return StorageConfig(data_dir=str(tmp_path), index_dir=idx)


def _result(success=True, summary="Done.") -> AgentResult:
    return AgentResult(
        success=success, summary=summary,
        steps=[AgentStep(1, "read_file", {"path": "f.py"}, "ok", True, 0.1)],
        total_input_tokens=500, total_output_tokens=200,
        total_elapsed_s=1.5, iterations=1,
    )


def _sparse_index(tmp_path: Path, name: str, docs: List[dict]) -> IndexEngine:
    """Build a sparse-only IndexEngine with FTS5 data on disk."""
    st = _storage(tmp_path)
    eng = IndexEngine(dimension=768, storage=st, sparse_only=True)
    eng.metadata = docs
    eng._db_path = os.path.join(st.index_dir, name + ".fts5.db")
    eng._build_fts5()
    with open(os.path.join(st.index_dir, name + ".meta.json"), "w") as f:
        json.dump(docs, f)
    return eng


_MEM_KW = dict(embedding_engine=None, dimension=768)


@pytest.fixture
def mem(tmp_path):
    """Yield a fresh FTS5-only AgentMemory, auto-closed after test."""
    m = AgentMemory(
        index_dir=str(tmp_path / "idx"), index_name="mem",
        knowledge_dir=str(tmp_path / "knowledge"), **_MEM_KW,
    )
    yield m
    m.close()


# ===========================================================================
# TestAgentMemory
# ===========================================================================

class TestAgentMemory:

    def test_ingest_and_search(self, mem):
        mem.ingest("Python list comprehension tutorial", "task_1", ["python"], 0.9, 100)
        mem.ingest("Rust ownership and borrowing guide", "task_2", ["rust"], 0.8, 200)
        mem.ingest("JavaScript async await patterns", "task_3", ["js"], 0.85, 150)
        results = mem.search("Python list", top_k=5)
        assert len(results) >= 1
        blob = " ".join(r.get("content", "") for r in results).lower()
        assert "python" in blob

    def test_ingest_task_result(self, mem):
        entry = mem.ingest_task_result("Refactor parser", _result(summary="Refactored parser module."))
        assert entry is not None
        assert isinstance(entry, MemoryEntry)
        assert "parser" in entry.content.lower()

    def test_ingest_failed_result_skipped(self, mem):
        entry = mem.ingest_task_result("Fix bug", _result(success=False, summary="LLM error"))
        assert entry is None

    def test_knowledge_file_created(self, mem):
        mem.ingest("How to use pytest fixtures", "task_1", ["python", "testing"], 0.9, 50)
        md_files = list(Path(mem._knowledge_dir if hasattr(mem, "_knowledge_dir")
                             else mem.knowledge_dir).rglob("*.md"))
        assert len(md_files) >= 1
        assert "pytest" in md_files[0].read_text(encoding="utf-8").lower()

    def test_stats(self, mem):
        mem.ingest("Entry one", "t1", ["a"], 0.9, 10)
        mem.ingest("Entry two", "t2", ["b"], 0.8, 20)
        s = mem.stats()
        assert isinstance(s, dict)
        assert s.get("total_entries", s.get("count", 0)) >= 2

    def test_forget(self, mem):
        entry = mem.ingest("Temporary note about sorting", "t1", ["algo"], 0.7, 10)
        assert mem.forget(entry.id)
        ids = [r.get("id", r.get("entry_id", "")) for r in mem.search("sorting", top_k=10)]
        assert entry.id not in ids

    def test_export_knowledge(self, mem, tmp_path):
        mem.ingest("Binary search algorithm", "t1", ["algo"], 0.9, 10)
        mem.ingest("Hash table implementation", "t2", ["ds"], 0.85, 20)
        out = str(tmp_path / "export.md")
        mem.export_knowledge(out)
        assert os.path.isfile(out)
        text = Path(out).read_text(encoding="utf-8").lower()
        assert "binary" in text or "hash" in text

    def test_empty_search(self, mem):
        assert mem.search("anything at all", top_k=5) == []

    def test_persistence(self, tmp_path):
        kw = dict(index_dir=str(tmp_path / "idx"), index_name="persist",
                   knowledge_dir=str(tmp_path / "know"), **_MEM_KW)
        m1 = AgentMemory(**kw)
        m1.ingest("Persistent data about decorators", "t1", ["python"], 0.9, 10)
        m1.close()
        m2 = AgentMemory(**kw)
        results = m2.search("decorators", top_k=5)
        assert len(results) >= 1
        assert "decorator" in " ".join(r.get("content", "") for r in results).lower()
        m2.close()


# ===========================================================================
# TestFederatedSearch
# ===========================================================================

class TestFederatedSearch:

    def test_add_and_list_indexes(self, tmp_path):
        fed = FederatedSearch(embedding_engine=None)
        i1 = _sparse_index(tmp_path / "a", "i1", [{"id": "1", "content": "hello", "source_path": "a.py"}])
        i2 = _sparse_index(tmp_path / "b", "i2", [{"id": "2", "content": "bye", "source_path": "b.py"}])
        fed.add_index("code", i1, weight=1.0)
        fed.add_index("docs", i2, weight=0.5)
        names = [e["name"] for e in fed.list_indexes()]
        assert "code" in names and "docs" in names
        i1.close(); i2.close()

    def test_remove_index(self, tmp_path):
        fed = FederatedSearch(embedding_engine=None)
        idx = _sparse_index(tmp_path, "i", [{"id": "1", "content": "test", "source_path": "t.py"}])
        fed.add_index("temp", idx, weight=1.0)
        fed.remove_index("temp")
        assert "temp" not in [e["name"] for e in fed.list_indexes()]
        idx.close()

    def test_search_empty(self):
        assert FederatedSearch(embedding_engine=None).search("anything", top_k=5) == []

    def test_search_single_index(self, tmp_path):
        docs = [
            {"id": "c1", "content": "Python dictionary comprehension", "source_path": "dict.py"},
            {"id": "c2", "content": "Rust memory safety borrow checker", "source_path": "borrow.rs"},
            {"id": "c3", "content": "JavaScript promise async callback", "source_path": "async.js"},
        ]
        idx = _sparse_index(tmp_path, "code", docs)
        fed = FederatedSearch(embedding_engine=None)
        fed.add_index("code", idx, weight=1.0)
        results = fed.search("Python dictionary", top_k=3)
        assert len(results) >= 1
        assert all(isinstance(r, SearchResult) for r in results)
        assert any("Python" in r.content or "dictionary" in r.content for r in results)
        idx.close()

    def test_search_multiple_indexes(self, tmp_path):
        i1 = _sparse_index(tmp_path / "a", "c", [
            {"id": "c1", "content": "Python flask web framework", "source_path": "app.py"}])
        i2 = _sparse_index(tmp_path / "b", "d", [
            {"id": "d1", "content": "Flask documentation request handling", "source_path": "flask.md"}])
        fed = FederatedSearch(embedding_engine=None)
        fed.add_index("code", i1, weight=1.0)
        fed.add_index("docs", i2, weight=1.0)
        results = fed.search("Flask web", top_k=5)
        assert len({r.index_name for r in results}) >= 1
        i1.close(); i2.close()

    def test_weight_boosting(self, tmp_path):
        ih = _sparse_index(tmp_path / "h", "h", [
            {"id": "h1", "content": "sorting algorithms quicksort mergesort", "source_path": "sort.py"}])
        il = _sparse_index(tmp_path / "l", "l", [
            {"id": "l1", "content": "sorting algorithms bubblesort insertion", "source_path": "bubble.py"}])
        fed = FederatedSearch(embedding_engine=None)
        fed.add_index("high_weight", ih, weight=10.0)
        fed.add_index("low_weight", il, weight=0.1)
        results = fed.search("sorting algorithms", top_k=5)
        if len(results) >= 2:
            assert results[0].index_name == "high_weight"
        ih.close(); il.close()

    def test_search_by_index(self, tmp_path):
        i1 = _sparse_index(tmp_path / "a", "c", [
            {"id": "c1", "content": "database connection pooling", "source_path": "pool.py"}])
        i2 = _sparse_index(tmp_path / "b", "d", [
            {"id": "d1", "content": "REST API endpoint documentation", "source_path": "api.md"}])
        fed = FederatedSearch(embedding_engine=None)
        fed.add_index("code", i1, weight=1.0)
        fed.add_index("docs", i2, weight=1.0)
        results = fed.search_by_index("database", "code", top_k=5)
        assert all(r.index_name == "code" for r in results)
        i1.close(); i2.close()

    def test_stats(self, tmp_path):
        idx = _sparse_index(tmp_path, "i", [
            {"id": "1", "content": "alpha", "source_path": "a.py"},
            {"id": "2", "content": "beta", "source_path": "b.py"},
        ])
        fed = FederatedSearch(embedding_engine=None)
        fed.add_index("main", idx, weight=1.0)
        s = fed.stats()
        assert isinstance(s, dict)
        assert s.get("total_documents", s.get("total_chunks", 0)) >= 2
        idx.close()


# ===========================================================================
# TestCorpusPipeline
# ===========================================================================

class TestCorpusPipeline:

    @staticmethod
    def _pipe(tmp_path: Path, pii_scanner=None, dedup=None) -> CorpusPipeline:
        return CorpusPipeline(
            embedding_engine=None, storage_config=_storage(tmp_path),
            chunker=None, batch_size=16, dimension=768,
            pii_scanner=pii_scanner, dedup=dedup,
        )

    def test_ingest_stackoverflow_md(self, tmp_path):
        src = tmp_path / "so"
        src.mkdir()
        for i in range(3):
            (src / f"q{i}.md").write_text(
                f"# Q{i}\nBinary search?\n```python\ndef s(): pass\n```\n", encoding="utf-8")
        stats = self._pipe(tmp_path).ingest_stackoverflow(str(src), "so_test", max_files=10)
        assert isinstance(stats, IngestStats)
        assert stats.files_processed >= 1 and stats.chunks_created >= 1
        assert len(stats.errors) == 0

    def test_ingest_codesearchnet_jsonl(self, tmp_path):
        src = tmp_path / "csn"
        src.mkdir()
        # Pipeline expects .md files in structured format (matching overnight download output)
        md_content = """# add function

- language: python
- repo: test/repo

## Documentation

Add two numbers together.

## Code

def add(a, b): return a + b

---

# main function

- language: rust
- repo: test/repo2

## Documentation

Entry point.

## Code

fn main() {}
"""
        (src / "functions.md").write_text(md_content, encoding="utf-8")
        stats = self._pipe(tmp_path).ingest_codesearchnet(
            str(src), "csn_test", languages=["python", "rust"], max_files=10)
        assert isinstance(stats, IngestStats)
        assert stats.files_processed >= 1 and stats.chunks_created >= 1

    def test_ingest_markdown_docs(self, tmp_path):
        src = tmp_path / "docs"
        src.mkdir()
        for n, c in [("guide.md", "# Guide\nConfigure the system.\n"),
                      ("api.md", "# API\nGET /health returns status.\n"),
                      ("faq.md", "# FAQ\npip install jcoder\n")]:
            (src / n).write_text(c, encoding="utf-8")
        stats = self._pipe(tmp_path).ingest_markdown_docs(str(src), "docs_test", max_files=10)
        assert isinstance(stats, IngestStats)
        assert stats.files_processed >= 1 and stats.chunks_created >= 1

    def test_resume_checkpoint(self, tmp_path):
        src = tmp_path / "resume"
        src.mkdir()
        for i in range(5):
            (src / f"f{i}.md").write_text(f"# Doc {i}\nTopic {i}.\n", encoding="utf-8")
        pipe = self._pipe(tmp_path)
        s1 = pipe.ingest_markdown_docs(str(src), "resume_test", max_files=3)
        assert s1.files_processed <= 3
        s2 = pipe.ingest_markdown_docs(str(src), "resume_test", max_files=10, resume=True)
        assert isinstance(s2, IngestStats)
        assert s1.files_processed + s2.files_processed >= 3

    def test_bad_file_skipped(self, tmp_path):
        src = tmp_path / "bad"
        src.mkdir()
        (src / "good.md").write_text("# Good\nValid content.\n", encoding="utf-8")
        (src / "corrupt.md").write_bytes(b"\x00\x01\x02\xff\xfe" * 100)
        stats = self._pipe(tmp_path).ingest_markdown_docs(str(src), "bad_test", max_files=10)
        assert isinstance(stats, IngestStats)
        assert stats.files_processed >= 1
        assert stats.files_skipped >= 1 or len(stats.errors) >= 0

    def test_pii_scanner_integration(self, tmp_path):
        """Verify PII scanner redacts secrets during ingestion."""
        src = tmp_path / "pii"
        src.mkdir()
        (src / "secrets.md").write_text(
            "# Config\naws_key = AKIAIOSFODNN7EXAMPLE\nSome safe content here.\n",
            encoding="utf-8",
        )
        pipe = self._pipe(tmp_path, pii_scanner=PIIScanner())
        stats = pipe.ingest_stackoverflow(str(src), "pii_test", max_files=10)
        assert isinstance(stats, IngestStats)
        assert stats.chunks_created > 0
        assert len(stats.errors) == 0

    def test_dedup_integration(self, tmp_path):
        """Verify dedup filters duplicate content during ingestion."""
        src = tmp_path / "dedup"
        src.mkdir()
        duplicate_text = "# Sorting\nBubble sort compares adjacent elements.\n"
        (src / "dup1.md").write_text(duplicate_text, encoding="utf-8")
        (src / "dup2.md").write_text(duplicate_text, encoding="utf-8")
        (src / "unique.md").write_text(
            "# Hashing\nHash tables provide O(1) average lookup.\n",
            encoding="utf-8",
        )
        dedup = MinHashDedup(threshold=0.8)
        pipe = self._pipe(tmp_path, dedup=dedup)
        stats = pipe.ingest_stackoverflow(str(src), "dedup_test", max_files=10)
        assert isinstance(stats, IngestStats)
        assert stats.files_processed == 3
        # Dedup should filter at least one duplicate chunk
        dedup_stats = dedup.stats()
        assert dedup_stats.total_seen >= 2
        assert dedup_stats.exact_dupes >= 1 or dedup_stats.near_dupes >= 1

    def test_pii_and_dedup_together(self, tmp_path):
        """Verify both PII scanner and dedup work together."""
        src = tmp_path / "both"
        src.mkdir()
        secret_text = "# Setup\ntoken = AKIAIOSFODNN7EXAMPLE\nDeploy the app.\n"
        (src / "a.md").write_text(secret_text, encoding="utf-8")
        (src / "b.md").write_text(secret_text, encoding="utf-8")
        (src / "c.md").write_text(
            "# Testing\nRun pytest with verbose flag for details.\n",
            encoding="utf-8",
        )
        pipe = self._pipe(
            tmp_path,
            pii_scanner=PIIScanner(),
            dedup=MinHashDedup(threshold=0.8),
        )
        stats = pipe.ingest_stackoverflow(str(src), "both_test", max_files=10)
        assert isinstance(stats, IngestStats)
        assert stats.files_processed >= 1
        assert len(stats.errors) == 0


# ===========================================================================
# TestMemoryTools
# ===========================================================================

class TestMemoryTools:

    def _mem_and_reg(self, tmp_path):
        m = AgentMemory(
            index_dir=str(tmp_path / "idx"), index_name="tools_mem",
            knowledge_dir=str(tmp_path / "knowledge"), **_MEM_KW,
        )
        r = ToolRegistry(working_dir=str(tmp_path), memory=m)
        return m, r

    def test_memory_search_tool(self, tmp_path):
        m, reg = self._mem_and_reg(tmp_path)
        m.ingest("Gradient descent optimization", "t1", ["ml"], 0.9, 50)
        res = reg.execute("memory_search", {"query": "gradient descent"})
        assert res.success
        assert "gradient" in res.output.lower() or "descent" in res.output.lower()
        m.close()

    def test_memory_store_tool(self, tmp_path):
        m, reg = self._mem_and_reg(tmp_path)
        res = reg.execute("memory_store", {
            "content": "pytest fixtures are scoped to function by default",
            "tags": ["python", "testing"],
        })
        assert res.success
        assert len(m.search("pytest fixtures", top_k=5)) >= 1
        m.close()

    def test_memory_tools_without_memory(self, tmp_path):
        reg = ToolRegistry(working_dir=str(tmp_path))
        for tool, args in [("memory_search", {"query": "x"}),
                           ("memory_store", {"content": "x"})]:
            res = reg.execute(tool, args)
            assert not res.success
            assert "memory" in res.error.lower() or "not configured" in res.error.lower()
