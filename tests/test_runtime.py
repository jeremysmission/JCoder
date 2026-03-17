"""Tests for core/runtime.py -- LLM generation pipeline."""

import math
from dataclasses import dataclass
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.runtime import Runtime, GenerationResult, DEFAULT_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@dataclass
class FakeModelConfig:
    endpoint: str = "http://localhost:8000/v1"
    name: str = "test-model"


@pytest.fixture
def mock_client():
    client = MagicMock()
    resp = MagicMock()
    resp.json.return_value = {
        "choices": [{"message": {"content": "Generated answer."}}]
    }
    resp.raise_for_status = MagicMock()
    client.post.return_value = resp
    return client


@pytest.fixture
def runtime(mock_client):
    with patch("core.runtime.make_client", return_value=mock_client):
        rt = Runtime(FakeModelConfig(), timeout=30)
    return rt


# ---------------------------------------------------------------------------
# GenerationResult
# ---------------------------------------------------------------------------

class TestGenerationResult:
    def test_basic(self):
        r = GenerationResult(text="hello", logprobs=[])
        assert r.text == "hello"
        assert r.self_certainty is None

    def test_self_certainty_with_logprobs(self):
        entries = [
            {"logprob": -0.1},
            {"logprob": -0.2},
            {"logprob": -0.05},
        ]
        r = GenerationResult(text="x", logprobs=entries)
        cert = r.self_certainty
        assert cert is not None
        expected = math.exp((-0.1 + -0.2 + -0.05) / 3)
        assert abs(cert - expected) < 0.01

    def test_self_certainty_from_top_logprobs(self):
        entries = [
            {"top_logprobs": [{"logprob": -0.3}]},
        ]
        r = GenerationResult(text="x", logprobs=entries)
        cert = r.self_certainty
        assert cert is not None
        assert abs(cert - math.exp(-0.3)) < 0.01

    def test_self_certainty_clamped_0_to_1(self):
        entries = [{"logprob": 0.0}]
        r = GenerationResult(text="x", logprobs=entries)
        assert r.self_certainty == 1.0

    def test_self_certainty_no_valid_entries(self):
        entries = [{"other": "data"}]
        r = GenerationResult(text="x", logprobs=entries)
        assert r.self_certainty is None


# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------

class TestRuntime:
    def test_generate_calls_endpoint(self, runtime, mock_client):
        result = runtime.generate("What is X?", ["chunk1", "chunk2"])
        assert result == "Generated answer."
        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args
        assert "/chat/completions" in call_args[0][0]

    def test_generate_sends_model_name(self, runtime, mock_client):
        runtime.generate("q", ["c"])
        payload = mock_client.post.call_args[1]["json"]
        assert payload["model"] == "test-model"

    def test_generate_includes_system_prompt(self, runtime, mock_client):
        runtime.generate("q", ["c"])
        payload = mock_client.post.call_args[1]["json"]
        system_msg = payload["messages"][0]
        assert system_msg["role"] == "system"
        assert "code assistant" in system_msg["content"].lower()

    def test_generate_includes_context_and_question(self, runtime, mock_client):
        runtime.generate("What is X?", ["chunk1"])
        payload = mock_client.post.call_args[1]["json"]
        user_msg = payload["messages"][1]
        assert "chunk1" in user_msg["content"]
        assert "What is X?" in user_msg["content"]

    def test_generate_custom_system_prompt(self, runtime, mock_client):
        runtime.generate("q", ["c"], system_prompt="Custom prompt")
        payload = mock_client.post.call_args[1]["json"]
        assert payload["messages"][0]["content"] == "Custom prompt"

    def test_generate_custom_temperature(self, runtime, mock_client):
        runtime.generate("q", ["c"], temperature=0.5)
        payload = mock_client.post.call_args[1]["json"]
        assert payload["temperature"] == 0.5

    def test_generate_custom_max_tokens(self, runtime, mock_client):
        runtime.generate("q", ["c"], max_tokens=100)
        payload = mock_client.post.call_args[1]["json"]
        assert payload["max_tokens"] == 100

    def test_generate_empty_context(self, runtime, mock_client):
        runtime.generate("q", [])
        payload = mock_client.post.call_args[1]["json"]
        assert "Context:" in payload["messages"][1]["content"]

    def test_generate_trims_context_to_budget(self, mock_client):
        with patch("core.runtime.make_client", return_value=mock_client):
            rt = Runtime(
                FakeModelConfig(),
                max_context_tokens=100,
                max_tokens=10,
            )
        big_chunks = ["x" * 5000, "y" * 5000]
        rt.generate("q", big_chunks)
        payload = mock_client.post.call_args[1]["json"]
        user_content = payload["messages"][1]["content"]
        assert len(user_content) < 5000

    def test_generate_bad_response_raises(self, mock_client):
        mock_client.post.return_value.json.return_value = {"bad": "data"}
        with patch("core.runtime.make_client", return_value=mock_client):
            rt = Runtime(FakeModelConfig())
        with pytest.raises(ValueError, match="missing expected fields"):
            rt.generate("q", ["c"])

    def test_network_gate_called(self, mock_client):
        gate = MagicMock()
        with patch("core.runtime.make_client", return_value=mock_client):
            rt = Runtime(FakeModelConfig(), gate=gate)
        rt.generate("q", ["c"])
        gate.guard.assert_called_once()

    def test_context_manager(self, mock_client):
        with patch("core.runtime.make_client", return_value=mock_client):
            with Runtime(FakeModelConfig()) as rt:
                rt.generate("q", ["c"])
        mock_client.close.assert_called_once()

    def test_default_system_prompt(self):
        assert "code assistant" in DEFAULT_SYSTEM_PROMPT.lower()
