"""Tests for core/orchestrator.py -- the main answer pipeline."""

import concurrent.futures
import time
from unittest.mock import MagicMock, patch

import pytest

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.orchestrator import Orchestrator, AnswerResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_retriever():
    r = MagicMock()
    r.retrieve.return_value = [
        {"content": "def hello(): pass", "source_path": "src/hello.py"},
        {"content": "def world(): pass", "source_path": "src/world.py"},
    ]
    return r


@pytest.fixture
def mock_runtime():
    rt = MagicMock()
    rt.generate.return_value = "The hello function prints a greeting."
    return rt


@pytest.fixture
def orchestrator(mock_retriever, mock_runtime):
    return Orchestrator(mock_retriever, mock_runtime, timeout=10.0)


# ---------------------------------------------------------------------------
# AnswerResult
# ---------------------------------------------------------------------------

class TestAnswerResult:
    def test_basic_construction(self):
        r = AnswerResult(answer="test", sources=["a.py"], chunk_count=1)
        assert r.answer == "test"
        assert r.sources == ["a.py"]
        assert r.chunk_count == 1
        assert r.chunks is None

    def test_with_chunks(self):
        chunks = [{"content": "x", "source_path": "a.py"}]
        r = AnswerResult(answer="y", sources=["a.py"], chunk_count=1, chunks=chunks)
        assert r.chunks == chunks


# ---------------------------------------------------------------------------
# Orchestrator.answer
# ---------------------------------------------------------------------------

class TestOrchestrator:
    def test_answer_returns_result(self, orchestrator, mock_retriever, mock_runtime):
        result = orchestrator.answer("What does hello do?")
        assert isinstance(result, AnswerResult)
        assert result.answer == "The hello function prints a greeting."
        assert result.chunk_count == 2
        mock_retriever.retrieve.assert_called_once_with("What does hello do?")
        mock_runtime.generate.assert_called_once()

    def test_answer_extracts_sources(self, orchestrator):
        result = orchestrator.answer("test")
        assert sorted(result.sources) == ["src/hello.py", "src/world.py"]

    def test_answer_deduplicates_sources(self, mock_runtime):
        retriever = MagicMock()
        retriever.retrieve.return_value = [
            {"content": "a", "source_path": "src/same.py"},
            {"content": "b", "source_path": "src/same.py"},
        ]
        orch = Orchestrator(retriever, mock_runtime)
        result = orch.answer("q")
        assert result.sources == ["src/same.py"]

    def test_answer_no_chunks_returns_fallback(self, mock_runtime):
        retriever = MagicMock()
        retriever.retrieve.return_value = []
        orch = Orchestrator(retriever, mock_runtime)
        result = orch.answer("unknown topic")
        assert "No relevant code" in result.answer
        assert result.chunk_count == 0
        assert result.sources == []
        mock_runtime.generate.assert_not_called()

    def test_answer_passes_chunk_texts_to_runtime(self, orchestrator, mock_runtime):
        orchestrator.answer("q")
        call_args = mock_runtime.generate.call_args
        chunk_texts = call_args[0][1]
        assert chunk_texts == ["def hello(): pass", "def world(): pass"]

    def test_answer_includes_chunks_in_result(self, orchestrator):
        result = orchestrator.answer("q")
        assert result.chunks is not None
        assert len(result.chunks) == 2

    def test_answer_handles_missing_source_path(self, mock_runtime):
        retriever = MagicMock()
        retriever.retrieve.return_value = [
            {"content": "code"},
        ]
        orch = Orchestrator(retriever, mock_runtime)
        result = orch.answer("q")
        assert result.sources == ["unknown"]

    def test_timeout_raises(self, mock_runtime):
        """Verify pipeline timeout fires when retrieval is slow."""
        slow_retriever = MagicMock()
        def slow_retrieve(q):
            time.sleep(5)
            return []
        slow_retriever.retrieve.side_effect = slow_retrieve
        orch = Orchestrator(slow_retriever, mock_runtime, timeout=0.5)
        with pytest.raises(TimeoutError, match="exceeded"):
            orch.answer("slow query")

    def test_retriever_exception_propagates(self, mock_runtime):
        retriever = MagicMock()
        retriever.retrieve.side_effect = RuntimeError("connection lost")
        orch = Orchestrator(retriever, mock_runtime)
        with pytest.raises(RuntimeError, match="connection lost"):
            orch.answer("q")

    def test_runtime_exception_propagates(self, mock_retriever):
        rt = MagicMock()
        rt.generate.side_effect = ValueError("bad prompt")
        orch = Orchestrator(mock_retriever, rt)
        with pytest.raises(ValueError, match="bad prompt"):
            orch.answer("q")

    def test_custom_timeout(self, mock_retriever, mock_runtime):
        orch = Orchestrator(mock_retriever, mock_runtime, timeout=42.0)
        assert orch._timeout == 42.0
