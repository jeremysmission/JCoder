"""Tests for R14: Retrieval Eval Harness.

Tests the eval scoring logic without needing live indexes or Ollama.
"""

from __future__ import annotations

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from scripts.run_retrieval_eval import evaluate_query, load_golden_set, _normalize


# ---------------------------------------------------------------------------
# Normalize
# ---------------------------------------------------------------------------

class TestNormalize:

    def test_underscores_become_spaces(self):
        assert "chunk file" in _normalize("chunk_file")

    def test_paths_become_spaces(self):
        assert "ingestion" in _normalize("ingestion/chunker.py")

    def test_lowercase(self):
        assert _normalize("CHUNKER") == "chunker"


# ---------------------------------------------------------------------------
# Golden set loading
# ---------------------------------------------------------------------------

class TestLoadGoldenSet:

    def test_loads_default_golden_set(self):
        golden = load_golden_set()
        assert len(golden) >= 10
        assert all("question" in q for q in golden)

    def test_loads_custom_path(self, tmp_path):
        data = [{"id": "T1", "question": "test?"}]
        p = tmp_path / "test.json"
        p.write_text(json.dumps(data))
        loaded = load_golden_set(str(p))
        assert len(loaded) == 1
        assert loaded[0]["id"] == "T1"


# ---------------------------------------------------------------------------
# Evaluate query scoring
# ---------------------------------------------------------------------------

class TestEvaluateQuery:

    def _mock_index(self, results):
        """Create a mock IndexEngine that returns given results."""
        index = MagicMock()
        index.search_fts5_direct = MagicMock(return_value=results)
        index.metadata = []
        return index

    def test_file_hit_at_rank_1(self):
        results = [
            (0.9, {"content": "class Chunker:\n    def chunk_file(self):", "source_path": "ingestion/chunker.py"}),
        ]
        index = self._mock_index(results)
        item = {"id": "G001", "question": "Where is Chunker?",
                "expected_file_contains": "ingestion/chunker.py", "expected_symbols": ["Chunker"]}
        r = evaluate_query(index, item, top_k=5)
        assert r["file_hit"] is True
        assert r["file_rank"] == 1
        assert r["mrr"] == 1.0

    def test_file_hit_at_rank_3(self):
        results = [
            (0.9, {"content": "noise", "source_path": "other.py"}),
            (0.8, {"content": "more noise", "source_path": "other2.py"}),
            (0.7, {"content": "class Chunker:", "source_path": "ingestion/chunker.py"}),
        ]
        index = self._mock_index(results)
        item = {"id": "G001", "question": "Where is Chunker?",
                "expected_file_contains": "ingestion/chunker.py", "expected_symbols": []}
        r = evaluate_query(index, item, top_k=5)
        assert r["file_hit"] is True
        assert r["file_rank"] == 3
        assert abs(r["mrr"] - 1/3) < 0.01

    def test_file_miss(self):
        results = [
            (0.9, {"content": "unrelated", "source_path": "other.py"}),
        ]
        index = self._mock_index(results)
        item = {"id": "G001", "question": "Where is Chunker?",
                "expected_file_contains": "ingestion/chunker.py", "expected_symbols": []}
        r = evaluate_query(index, item, top_k=5)
        assert r["file_hit"] is False
        assert r["mrr"] == 0.0

    def test_symbol_hit(self):
        results = [
            (0.9, {"content": "def chunk_file(self):\n    LANGUAGE_MAP = {}", "source_path": "chunker.py"}),
        ]
        index = self._mock_index(results)
        item = {"id": "G002", "question": "What does chunk_file do?",
                "expected_file_contains": "", "expected_symbols": ["chunk_file", "LANGUAGE_MAP"]}
        r = evaluate_query(index, item, top_k=5)
        assert r["symbol_rate"] == 1.0
        assert all(s["hit"] for s in r["symbol_hits"])

    def test_partial_symbol_hit(self):
        results = [
            (0.9, {"content": "def chunk_file(): pass", "source_path": "x.py"}),
        ]
        index = self._mock_index(results)
        item = {"id": "G002", "question": "test",
                "expected_file_contains": "", "expected_symbols": ["chunk_file", "LANGUAGE_MAP"]}
        r = evaluate_query(index, item, top_k=5)
        assert r["symbol_rate"] == 0.5

    def test_empty_results(self):
        index = self._mock_index([])
        item = {"id": "G001", "question": "test",
                "expected_file_contains": "foo.py", "expected_symbols": ["bar"]}
        r = evaluate_query(index, item, top_k=5)
        assert r["file_hit"] is False
        assert r["symbol_rate"] == 0.0
        assert r["results_count"] == 0

    def test_no_expected_symbols_defaults_to_1(self):
        results = [(0.9, {"content": "code", "source_path": "x.py"})]
        index = self._mock_index(results)
        item = {"id": "G010", "question": "test", "expected_file_contains": "", "expected_symbols": []}
        r = evaluate_query(index, item, top_k=5)
        assert r["symbol_rate"] == 1.0


# ---------------------------------------------------------------------------
# Run eval (mock)
# ---------------------------------------------------------------------------

class TestRunEval:

    def test_run_eval_returns_summary_structure(self, tmp_path):
        golden = [{"id": "T1", "question": "test?", "expected_file_contains": "", "expected_symbols": []}]
        golden_path = tmp_path / "golden.json"
        golden_path.write_text(json.dumps(golden))

        mock_index = MagicMock()
        mock_index.search_fts5_direct.return_value = []
        mock_index.count = 0

        with patch("scripts.run_retrieval_eval.IndexEngine") as MockIE:
            instance = MockIE.return_value
            instance.load.return_value = None
            instance.count = 0
            instance.search_fts5_direct.return_value = []
            instance.metadata = []

            from scripts.run_retrieval_eval import run_eval
            summary = run_eval(golden_path=str(golden_path), index_dir=str(tmp_path))

        # Should have error or results
        if "error" not in summary:
            assert "metrics" in summary
            assert "per_question" in summary
