"""Tests for core.reranker -- cross-encoder rescoring."""

from unittest.mock import MagicMock, patch

import pytest

from core.config import ModelConfig
from core.network_gate import NetworkGate
from core.reranker import Reranker


def _mock_score_response(scores):
    """Build a mock httpx response for /score endpoint."""
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        "data": [{"index": i, "score": s} for i, s in enumerate(scores)]
    }
    return resp


class TestDisabledReranker:

    def test_passthrough_when_disabled(self):
        config = ModelConfig(name="test", endpoint="http://localhost:8002/v1", enabled=False)
        rr = Reranker(config)
        result = rr.rerank("query", ["doc1", "doc2", "doc3"], top_n=2)
        assert result == [(0, 1.0), (1, 1.0)]

    def test_passthrough_empty_docs(self):
        config = ModelConfig(name="test", endpoint="http://localhost:8002/v1", enabled=True)
        rr = Reranker(config)
        result = rr.rerank("query", [], top_n=5)
        assert result == []


class TestEnabledReranker:

    def test_rerank_sorts_by_score(self):
        config = ModelConfig(name="test-reranker", endpoint="http://localhost:8002/v1")
        rr = Reranker(config)
        mock_resp = _mock_score_response([0.3, 0.9, 0.1, 0.7])
        rr._client = MagicMock()
        rr._client.post.return_value = mock_resp

        result = rr.rerank("how does auth work?", ["a", "b", "c", "d"], top_n=3)

        assert len(result) == 3
        # Highest score first
        assert result[0] == (1, 0.9)
        assert result[1] == (3, 0.7)
        assert result[2] == (0, 0.3)

    def test_rerank_respects_top_n(self):
        config = ModelConfig(name="test", endpoint="http://localhost:8002/v1")
        rr = Reranker(config)
        mock_resp = _mock_score_response([0.5, 0.8, 0.3, 0.9, 0.1])
        rr._client = MagicMock()
        rr._client.post.return_value = mock_resp

        result = rr.rerank("query", ["a", "b", "c", "d", "e"], top_n=2)
        assert len(result) == 2

    def test_rerank_sends_correct_payload(self):
        config = ModelConfig(name="my-model", endpoint="http://localhost:8002/v1")
        rr = Reranker(config)
        mock_resp = _mock_score_response([0.5])
        rr._client = MagicMock()
        rr._client.post.return_value = mock_resp

        rr.rerank("test query", ["doc1"], top_n=1)

        call_args = rr._client.post.call_args
        assert call_args[0][0] == "http://localhost:8002/v1/score"
        payload = call_args[1]["json"]
        assert payload["model"] == "my-model"
        assert payload["text_1"] == "test query"
        assert payload["text_2"] == ["doc1"]


class TestNetworkGateIntegration:

    def test_guard_called_before_request(self):
        gate = NetworkGate(mode="offline")
        config = ModelConfig(name="test", endpoint="http://localhost:8002/v1")
        rr = Reranker(config, gate=gate)

        with pytest.raises(PermissionError):
            rr.rerank("query", ["doc1"])

    def test_localhost_allowed(self):
        gate = NetworkGate(mode="localhost")
        config = ModelConfig(name="test", endpoint="http://localhost:8002/v1")
        rr = Reranker(config, gate=gate)
        mock_resp = _mock_score_response([0.5])
        rr._client = MagicMock()
        rr._client.post.return_value = mock_resp

        result = rr.rerank("query", ["doc1"])
        assert len(result) == 1


class TestContextManager:

    def test_close(self):
        config = ModelConfig(name="test", endpoint="http://localhost:8002/v1")
        rr = Reranker(config)
        rr._client = MagicMock()
        rr.close()
        rr._client.close.assert_called_once()

    def test_context_manager(self):
        config = ModelConfig(name="test", endpoint="http://localhost:8002/v1")
        with Reranker(config) as rr:
            rr._client = MagicMock()
