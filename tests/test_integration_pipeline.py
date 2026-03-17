"""Integration tests wiring real JCoder pipeline components together.

Tests the full flow: IndexEngine -> RetrievalEngine -> Orchestrator
with real SQLite FTS5 indexes (no mocks for storage layer).
"""

from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import MagicMock

from core.config import StorageConfig, ModelConfig, RetrievalConfig
from core.index_engine import IndexEngine
from core.orchestrator import Orchestrator, AnswerResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def populated_index(tmp_path):
    """IndexEngine with real FTS5 data, sparse-only mode."""
    storage = StorageConfig(
        data_dir=str(tmp_path),
        index_dir=str(tmp_path / "idx"),
    )
    engine = IndexEngine(dimension=4, storage=storage, sparse_only=True)
    engine.metadata = [
        {
            "id": "chunk_auth_1",
            "content": "def authenticate(username, password):\n"
                       "    \"\"\"Verify credentials against the database.\"\"\"\n"
                       "    user = db.find_user(username)\n"
                       "    return bcrypt.verify(password, user.hash)\n",
            "source_path": "auth/login.py",
        },
        {
            "id": "chunk_config_1",
            "content": "class DatabaseConfig:\n"
                       "    host: str = 'localhost'\n"
                       "    port: int = 5432\n"
                       "    max_connections: int = 10\n",
            "source_path": "config/database.py",
        },
        {
            "id": "chunk_test_1",
            "content": "def test_login_success():\n"
                       "    result = authenticate('admin', 'correct_password')\n"
                       "    assert result is True\n",
            "source_path": "tests/test_auth.py",
        },
        {
            "id": "chunk_utils_1",
            "content": "def parse_csv(filepath):\n"
                       "    \"\"\"Parse a CSV file into list of dicts.\"\"\"\n"
                       "    import csv\n"
                       "    with open(filepath) as f:\n"
                       "        return list(csv.DictReader(f))\n",
            "source_path": "utils/parsers.py",
        },
    ]
    engine._db_path = str(tmp_path / "idx" / "test.fts5.db")
    engine._build_fts5()
    try:
        yield engine
    finally:
        engine.close()


def _mock_runtime(response_text: str = "Here is the answer."):
    """Build a mock Runtime that returns fixed text."""
    rt = MagicMock()
    rt.generate.return_value = response_text
    return rt


def _mock_embedder():
    """Build a mock embedder that returns zero vectors."""
    emb = MagicMock()
    emb.embed.return_value = np.zeros(4, dtype=np.float32)
    return emb


# ---------------------------------------------------------------------------
# FTS5 Search Integration
# ---------------------------------------------------------------------------

class TestFTS5SearchIntegration:

    def test_keyword_search_finds_auth(self, populated_index):
        results = populated_index.search_keywords("authenticate password", k=5)
        assert len(results) > 0
        # Should find the auth chunk
        found_ids = {populated_index.metadata[idx]["id"] for idx, _ in results}
        assert "chunk_auth_1" in found_ids

    def test_keyword_search_finds_config(self, populated_index):
        results = populated_index.search_keywords("database config host port", k=5)
        assert len(results) > 0
        found_ids = {populated_index.metadata[idx]["id"] for idx, _ in results}
        assert "chunk_config_1" in found_ids

    def test_keyword_search_csv_parser(self, populated_index):
        results = populated_index.search_keywords("parse csv file", k=5)
        assert len(results) > 0
        found_ids = {populated_index.metadata[idx]["id"] for idx, _ in results}
        assert "chunk_utils_1" in found_ids

    def test_irrelevant_query_returns_empty_or_low(self, populated_index):
        results = populated_index.search_keywords("quantum entanglement photon", k=5)
        # Either empty or very low scores
        assert len(results) <= 1


class TestHybridSearchIntegration:

    def test_sparse_only_hybrid(self, populated_index):
        """In sparse-only mode, hybrid_search uses FTS5 only."""
        dummy_vec = np.zeros(4, dtype=np.float32)
        results = populated_index.hybrid_search(
            dummy_vec, "authenticate login password", k=3,
        )
        assert len(results) > 0
        top_sources = {r[1]["source_path"] for r in results}
        assert "auth/login.py" in top_sources

    def test_path_prior_boost(self, populated_index):
        """Config-intent queries boost config paths."""
        dummy_vec = np.zeros(4, dtype=np.float32)
        results = populated_index.hybrid_search(
            dummy_vec, "database config port", k=4,
        )
        # Config chunk should rank high due to path boost
        top_sources = [r[1]["source_path"] for r in results[:2]]
        assert "config/database.py" in top_sources


# ---------------------------------------------------------------------------
# Orchestrator Integration
# ---------------------------------------------------------------------------

class TestOrchestratorIntegration:

    def test_full_pipeline_returns_answer(self, populated_index):
        """Orchestrator wires retriever + runtime into a single answer."""
        # Build a minimal retriever that wraps the index
        class _SimpleRetriever:
            def __init__(self, index):
                self._index = index

            def retrieve(self, question, top_k=5):
                # Use keyword search directly to avoid hybrid edge cases
                kw_results = self._index.search_keywords(question, k=top_k)
                return [
                    self._index.metadata[idx]
                    for idx, _ in kw_results
                    if 0 <= idx < len(self._index.metadata)
                ]

        retriever = _SimpleRetriever(populated_index)
        runtime = _mock_runtime("The authenticate function checks credentials via bcrypt.")

        orch = Orchestrator(retriever, runtime, timeout=10.0)
        result = orch.answer("authenticate password database")

        assert isinstance(result, AnswerResult)
        assert "authenticate" in result.answer.lower() or "bcrypt" in result.answer.lower()
        assert result.chunk_count > 0
        assert len(result.sources) > 0

    def test_empty_retrieval_returns_no_code_found(self, populated_index):
        """When nothing matches, orchestrator returns graceful message."""
        class _EmptyRetriever:
            def retrieve(self, question, top_k=5):
                return []

        retriever = _EmptyRetriever()
        runtime = _mock_runtime()

        orch = Orchestrator(retriever, runtime, timeout=10.0)
        result = orch.answer("What is quantum computing?")

        assert "no relevant" in result.answer.lower()
        assert result.chunk_count == 0
        # Runtime should NOT have been called
        runtime.generate.assert_not_called()

    def test_runtime_receives_retrieved_context(self, populated_index):
        """Verify the runtime actually gets the retrieved chunk text."""
        class _SimpleRetriever:
            def __init__(self, index):
                self._index = index

            def retrieve(self, question, top_k=5):
                kw_results = self._index.search_keywords(question, k=top_k)
                return [
                    self._index.metadata[idx]
                    for idx, _ in kw_results
                    if 0 <= idx < len(self._index.metadata)
                ]

        retriever = _SimpleRetriever(populated_index)
        runtime = _mock_runtime("Answer about CSV parsing.")

        orch = Orchestrator(retriever, runtime, timeout=10.0)
        orch.answer("parse csv file")

        # Runtime.generate should have been called with question + context
        assert runtime.generate.call_count == 1
        call_args = runtime.generate.call_args
        question_arg = call_args[0][0]
        chunks_arg = call_args[0][1]
        assert "csv" in question_arg.lower()
        assert len(chunks_arg) > 0
        assert any("csv" in c.lower() for c in chunks_arg)
