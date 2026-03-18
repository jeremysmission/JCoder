"""Extended tests for model cascade router and LLM runtime."""

from __future__ import annotations

import math
from unittest.mock import MagicMock, patch, PropertyMock

import httpx
import pytest

from core.cascade import (
    CascadeLevel,
    CascadeResult,
    ModelCascade,
    estimate_answer_confidence,
    estimate_complexity,
)
from core.config import ModelConfig
from core.network_gate import NetworkGate
from core.runtime import DEFAULT_SYSTEM_PROMPT, GenerationResult, Runtime


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _model(name: str = "test-model", endpoint: str = "http://localhost:8000/v1") -> ModelConfig:
    return ModelConfig(name=name, endpoint=endpoint)


def _level(name: str, max_c: float, timeout: int = 30) -> CascadeLevel:
    return CascadeLevel(
        name=name,
        model_config=_model(name, "http://localhost:8000/v1"),
        max_complexity=max_c,
        timeout_s=timeout,
    )


def _chat_response(text: str = "hello") -> dict:
    return {"choices": [{"message": {"content": text}}]}


# ---------------------------------------------------------------------------
# Cascade -- complexity estimation
# ---------------------------------------------------------------------------

class TestEstimateComplexity:
    def test_simple_short_query(self):
        score = estimate_complexity("what is a list?")
        assert score < 0.2, f"Simple query scored too high: {score}"

    def test_medium_query(self):
        score = estimate_complexity(
            "How should I refactor the database schema to support concurrent writes?"
        )
        assert 0.3 <= score <= 0.8, f"Medium query out of range: {score}"

    def test_complex_query(self):
        score = estimate_complexity(
            "We need to redesign the distributed database migration strategy, "
            "architect a new schema that handles concurrent parallel writes "
            "and optimize performance for security-sensitive integration "
            "workloads across multiple services."
        )
        assert score >= 0.5, f"Complex query scored too low: {score}"

    def test_empty_query(self):
        assert estimate_complexity("") == 0.0

    def test_simple_keywords_reduce_score(self):
        base = estimate_complexity("tell me about files")
        simple = estimate_complexity("what is the syntax for import?")
        assert simple <= base + 0.05

    def test_code_references_increase_score(self):
        plain = estimate_complexity("explain the module")
        refs = estimate_complexity("explain core.cascade.ModelCascade and core.runtime.Runtime")
        assert refs > plain

    def test_multi_part_increases_score(self):
        single = estimate_complexity("explain lists")
        multi = estimate_complexity("explain lists, and also dicts, and sets?")
        assert multi > single

    def test_bounded_zero_one(self):
        for q in ["", "x", "a " * 200, "refactor " * 50]:
            s = estimate_complexity(q)
            assert 0.0 <= s <= 1.0


# ---------------------------------------------------------------------------
# Cascade -- answer confidence
# ---------------------------------------------------------------------------

class TestEstimateAnswerConfidence:
    def test_empty_answer(self):
        assert estimate_answer_confidence("") == 0.0
        assert estimate_answer_confidence("   ") == 0.0

    def test_short_answer_low(self):
        assert estimate_answer_confidence("yes") < 0.3

    def test_code_block_bonus(self):
        no_code = estimate_answer_confidence("Use the function to do X and Y and Z nicely.")
        with_code = estimate_answer_confidence(
            "Use the function:\n```python\ndef foo(): pass\n```"
        )
        assert with_code > no_code

    def test_hedge_words_penalty(self):
        confident = estimate_answer_confidence(
            "The function returns the sum of two integers."
        )
        hedging = estimate_answer_confidence(
            "I think maybe the function possibly returns the sum, not sure."
        )
        assert hedging < confident

    def test_bounded_zero_one(self):
        for a in ["", "x", "maybe " * 50, "```py\ncode\n```" * 5]:
            c = estimate_answer_confidence(a)
            assert 0.0 <= c <= 1.0

    def test_backtick_refs_bonus(self):
        plain = estimate_answer_confidence("Call the foo function with bar argument here.")
        refs = estimate_answer_confidence("Call `foo()` with `bar` argument here.")
        assert refs >= plain


# ---------------------------------------------------------------------------
# Cascade -- model selection and routing
# ---------------------------------------------------------------------------

class TestModelCascadeRouting:

    def _make_cascade(self, levels=None, confidence_threshold=0.4):
        lvls = levels or [
            _level("small", 0.3),
            _level("medium", 0.6),
            _level("large", 1.0),
        ]
        gate = NetworkGate(mode="localhost")
        return ModelCascade(
            lvls, gate=gate, confidence_threshold=confidence_threshold
        )

    @patch("core.cascade.Runtime")
    def test_simple_routes_to_smallest(self, MockRuntime):
        instance = MockRuntime.return_value
        instance.generate.return_value = "Simple answer"
        cascade = self._make_cascade()
        result = cascade.route("what is a list?", ["ctx"])
        assert result.model_used == "small"
        assert not result.escalated

    @patch("core.cascade.Runtime")
    def test_complex_routes_to_largest(self, MockRuntime):
        instance = MockRuntime.return_value
        instance.generate.return_value = "Complex answer"
        cascade = self._make_cascade()
        result = cascade.route(
            "We need to redesign and architect a distributed database migration "
            "strategy, optimize concurrent parallel performance, fix the security "
            "vulnerability in the integration layer, and also compare the tradeoff "
            "between schema.migration and core.runtime versus the new async approach?",
            ["ctx"],
        )
        assert result.model_used == "large"

    @patch("core.cascade.Runtime")
    def test_escalation_on_low_confidence(self, MockRuntime):
        instance = MockRuntime.return_value
        instance.generate.return_value = "maybe answer"
        cascade = self._make_cascade()
        # confidence_fn always returns low
        result = cascade.route(
            "what is x?", ["ctx"], confidence_fn=lambda _: 0.2
        )
        assert result.escalated

    @patch("core.cascade.Runtime")
    def test_no_escalation_on_high_confidence(self, MockRuntime):
        instance = MockRuntime.return_value
        instance.generate.return_value = "Definite answer"
        cascade = self._make_cascade()
        result = cascade.route(
            "what is x?", ["ctx"], confidence_fn=lambda _: 0.9
        )
        assert not result.escalated
        assert result.model_used == "small"

    @patch("core.cascade.Runtime")
    def test_skip_connection_very_low_confidence(self, MockRuntime):
        """Very low confidence should skip intermediate and jump to last."""
        instance = MockRuntime.return_value
        call_count = [0]

        def side_effect(q, ctx):
            call_count[0] += 1
            if call_count[0] == 1:
                return "bad"
            return "good final answer"

        instance.generate.side_effect = side_effect
        cascade = self._make_cascade()
        cascade._skip_threshold = 0.15

        calls = [0]
        def low_conf(ans):
            calls[0] += 1
            if calls[0] == 1:
                return 0.05  # below skip threshold
            return 0.9

        result = cascade.route("what is x?", ["ctx"], confidence_fn=low_conf)
        assert result.model_used == "large"
        assert result.escalated
        assert cascade.skip_count >= 1

    @patch("core.cascade.Runtime")
    def test_multi_level_cascade_chain(self, MockRuntime):
        """Four levels: query escalates through each one."""
        instance = MockRuntime.return_value
        attempts = [0]

        def gen(q, ctx):
            attempts[0] += 1
            return f"answer {attempts[0]}"

        instance.generate.side_effect = gen
        levels = [
            _level("tiny", 0.2),
            _level("small", 0.4),
            _level("medium", 0.7),
            _level("large", 1.0),
        ]
        cascade = self._make_cascade(levels)

        conf_calls = [0]
        def rising_conf(ans):
            conf_calls[0] += 1
            # Only accept on the last level
            return 0.1 if conf_calls[0] < 4 else 0.9

        result = cascade.route("what is x?", ["ctx"], confidence_fn=rising_conf)
        assert result.model_used == "large"
        assert result.escalated

    @patch("core.cascade.Runtime")
    def test_fallback_on_runtime_error(self, MockRuntime):
        """If a level throws, cascade moves to next."""
        instance = MockRuntime.return_value
        call_num = [0]

        def gen(q, ctx):
            call_num[0] += 1
            if call_num[0] == 1:
                raise ConnectionError("model down")
            return "fallback answer"

        instance.generate.side_effect = gen
        cascade = self._make_cascade()
        result = cascade.route("what is x?", ["ctx"])
        assert result.escalated
        assert result.answer == "fallback answer"

    @patch("core.cascade.Runtime")
    def test_all_levels_fail(self, MockRuntime):
        instance = MockRuntime.return_value
        instance.generate.side_effect = RuntimeError("down")
        cascade = self._make_cascade()
        result = cascade.route("what is x?", ["ctx"])
        assert "failed" in result.answer.lower()
        assert result.escalated


# ---------------------------------------------------------------------------
# Runtime -- request building
# ---------------------------------------------------------------------------

class TestRuntimeRequestBuilding:

    @patch("core.runtime.make_client")
    def test_payload_structure(self, mock_make):
        client = MagicMock()
        resp = MagicMock()
        resp.json.return_value = _chat_response("ok")
        resp.raise_for_status = MagicMock()
        client.post.return_value = resp
        mock_make.return_value = client

        rt = Runtime(_model(), timeout=10)
        rt.generate("question?", ["chunk1"])

        call_args = client.post.call_args
        payload = call_args.kwargs.get("json") or call_args[1].get("json")
        assert payload["model"] == "test-model"
        assert len(payload["messages"]) == 2
        assert payload["messages"][0]["role"] == "system"
        assert payload["messages"][1]["role"] == "user"
        assert "question?" in payload["messages"][1]["content"]
        assert "chunk1" in payload["messages"][1]["content"]

    @patch("core.runtime.make_client")
    def test_custom_system_prompt(self, mock_make):
        client = MagicMock()
        resp = MagicMock()
        resp.json.return_value = _chat_response("ok")
        resp.raise_for_status = MagicMock()
        client.post.return_value = resp
        mock_make.return_value = client

        rt = Runtime(_model(), system_prompt="Be brief.")
        rt.generate("q", [])
        payload = client.post.call_args.kwargs.get("json") or client.post.call_args[1]["json"]
        assert payload["messages"][0]["content"] == "Be brief."

    @patch("core.runtime.make_client")
    def test_api_key_header(self, mock_make):
        client = MagicMock()
        resp = MagicMock()
        resp.json.return_value = _chat_response("ok")
        resp.raise_for_status = MagicMock()
        client.post.return_value = resp
        mock_make.return_value = client

        rt = Runtime(_model(), api_key="sk-test-123")
        rt.generate("q", [])
        headers = client.post.call_args.kwargs.get("headers") or client.post.call_args[1]["headers"]
        assert headers["Authorization"] == "Bearer sk-test-123"

    @patch("core.runtime.make_client")
    def test_per_call_overrides(self, mock_make):
        client = MagicMock()
        resp = MagicMock()
        resp.json.return_value = _chat_response("ok")
        resp.raise_for_status = MagicMock()
        client.post.return_value = resp
        mock_make.return_value = client

        rt = Runtime(_model(), temperature=0.1, max_tokens=100)
        rt.generate("q", [], temperature=0.9, max_tokens=500)
        payload = client.post.call_args.kwargs.get("json") or client.post.call_args[1]["json"]
        assert payload["temperature"] == 0.9
        assert payload["max_tokens"] == 500


# ---------------------------------------------------------------------------
# Runtime -- response parsing
# ---------------------------------------------------------------------------

class TestRuntimeResponseParsing:

    @patch("core.runtime.make_client")
    def test_normal_response(self, mock_make):
        client = MagicMock()
        resp = MagicMock()
        resp.json.return_value = _chat_response("hello world")
        resp.raise_for_status = MagicMock()
        client.post.return_value = resp
        mock_make.return_value = client

        rt = Runtime(_model())
        assert rt.generate("q", []) == "hello world"

    @patch("core.runtime.make_client")
    def test_malformed_response_raises(self, mock_make):
        client = MagicMock()
        resp = MagicMock()
        resp.json.return_value = {"bad": "data"}
        resp.raise_for_status = MagicMock()
        client.post.return_value = resp
        mock_make.return_value = client

        rt = Runtime(_model())
        with pytest.raises(ValueError, match="missing expected fields"):
            rt.generate("q", [])

    @patch("core.runtime.make_client")
    def test_http_error_propagates(self, mock_make):
        client = MagicMock()
        resp = MagicMock()
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "500", request=MagicMock(), response=MagicMock()
        )
        client.post.return_value = resp
        mock_make.return_value = client

        rt = Runtime(_model())
        with pytest.raises(httpx.HTTPStatusError):
            rt.generate("q", [])


# ---------------------------------------------------------------------------
# Runtime -- context budget / token counting
# ---------------------------------------------------------------------------

class TestRuntimeTokenBudget:

    @patch("core.runtime.make_client")
    def test_context_trimming(self, mock_make):
        client = MagicMock()
        resp = MagicMock()
        resp.json.return_value = _chat_response("ok")
        resp.raise_for_status = MagicMock()
        client.post.return_value = resp
        mock_make.return_value = client

        # Very small context budget forces trimming
        rt = Runtime(_model(), max_tokens=2048, max_context_tokens=2200)
        big_chunks = ["x" * 5000, "y" * 5000]
        rt.generate("q", big_chunks)

        payload = client.post.call_args.kwargs.get("json") or client.post.call_args[1]["json"]
        user_content = payload["messages"][1]["content"]
        # Should not contain the full 10000 chars of chunks
        assert len(user_content) < 10000

    @patch("core.runtime.make_client")
    def test_empty_chunks(self, mock_make):
        client = MagicMock()
        resp = MagicMock()
        resp.json.return_value = _chat_response("ok")
        resp.raise_for_status = MagicMock()
        client.post.return_value = resp
        mock_make.return_value = client

        rt = Runtime(_model())
        result = rt.generate("q", [])
        assert result == "ok"


# ---------------------------------------------------------------------------
# Runtime -- timeout handling
# ---------------------------------------------------------------------------

class TestRuntimeTimeout:

    @patch("core.runtime.make_client")
    def test_timeout_error_propagates(self, mock_make):
        client = MagicMock()
        client.post.side_effect = httpx.TimeoutException("read timed out")
        mock_make.return_value = client

        rt = Runtime(_model(), timeout=1)
        with pytest.raises(httpx.TimeoutException):
            rt.generate("q", ["ctx"])

    @patch("core.runtime.make_client")
    def test_connect_error_propagates(self, mock_make):
        client = MagicMock()
        client.post.side_effect = httpx.ConnectError("refused")
        mock_make.return_value = client

        rt = Runtime(_model())
        with pytest.raises(httpx.ConnectError):
            rt.generate("q", [])


# ---------------------------------------------------------------------------
# Runtime -- GenerationResult / logprobs
# ---------------------------------------------------------------------------

class TestGenerationResult:

    def test_self_certainty_with_logprobs(self):
        lps = [{"logprob": -0.5}, {"logprob": -0.3}]
        gr = GenerationResult(text="hello", logprobs=lps)
        expected = math.exp((-0.5 + -0.3) / 2)
        assert gr.self_certainty == pytest.approx(expected, abs=1e-6)

    def test_self_certainty_empty(self):
        gr = GenerationResult(text="hello", logprobs=[])
        assert gr.self_certainty is None

    def test_self_certainty_top_logprobs(self):
        lps = [{"top_logprobs": [{"logprob": -0.1}]}]
        gr = GenerationResult(text="hi", logprobs=lps)
        assert gr.self_certainty is not None
        assert 0.0 <= gr.self_certainty <= 1.0

    def test_self_certainty_clamped(self):
        lps = [{"logprob": 0.0}]  # exp(0) = 1.0
        gr = GenerationResult(text="hi", logprobs=lps)
        assert gr.self_certainty == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Runtime -- close / context manager
# ---------------------------------------------------------------------------

class TestRuntimeLifecycle:

    @patch("core.runtime.make_client")
    def test_close_releases_client(self, mock_make):
        client = MagicMock()
        mock_make.return_value = client
        rt = Runtime(_model())
        rt.close()
        client.close.assert_called_once()

    @patch("core.runtime.make_client")
    def test_context_manager(self, mock_make):
        client = MagicMock()
        mock_make.return_value = client
        with Runtime(_model()) as rt:
            assert rt is not None
        client.close.assert_called_once()


# ---------------------------------------------------------------------------
# Runtime -- network gate integration
# ---------------------------------------------------------------------------

class TestRuntimeNetworkGate:

    @patch("core.runtime.make_client")
    def test_gate_guard_called(self, mock_make):
        client = MagicMock()
        resp = MagicMock()
        resp.json.return_value = _chat_response("ok")
        resp.raise_for_status = MagicMock()
        client.post.return_value = resp
        mock_make.return_value = client

        gate = MagicMock()
        rt = Runtime(_model(), gate=gate)
        rt.generate("q", [])
        gate.guard.assert_called_once()
