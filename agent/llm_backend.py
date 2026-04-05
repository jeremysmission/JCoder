"""
Dual LLM Backend
-----------------
Unified interface for sending tool-augmented chat to either:
- OpenAI-compatible API (Ollama, vLLM, OpenRouter, Azure)
- Anthropic API (direct Claude)

Both return the same ChatResponse so the agent loop doesn't care
which backend is active.
"""

from __future__ import annotations

import json
import math
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import httpx

from core.http_factory import make_client


# ---------------------------------------------------------------------------
# Response types
# ---------------------------------------------------------------------------

@dataclass
class ToolCall:
    """A single tool invocation requested by the LLM."""
    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass
class ChatResponse:
    """Unified response from any backend."""
    content: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)
    logprobs: List[Dict[str, Any]] = field(default_factory=list)
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    elapsed_s: float = 0.0
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0

    @property
    def self_certainty(self) -> Optional[float]:
        if not self.logprobs:
            return None

        values: list[float] = []
        for entry in self.logprobs:
            if not isinstance(entry, dict):
                continue
            if isinstance(entry.get("logprob"), (int, float)):
                values.append(float(entry["logprob"]))
                continue
            top = entry.get("top_logprobs")
            if isinstance(top, list):
                for candidate in top:
                    if isinstance(candidate, dict) and isinstance(
                        candidate.get("logprob"), (int, float)
                    ):
                        values.append(float(candidate["logprob"]))
                        break

        if not values:
            return None

        certainty = math.exp(sum(values) / len(values))
        return max(0.0, min(1.0, certainty))


def _decode_tool_arguments(args: Any, *, tool_name: str) -> Dict[str, Any]:
    """Normalize LLM tool arguments into a dict with clear errors."""
    if args in (None, ""):
        return {}

    parsed = args
    if isinstance(args, str):
        try:
            parsed = json.loads(args)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Malformed tool arguments for {tool_name or 'tool'}: {exc.msg}"
            ) from exc

    if not isinstance(parsed, dict):
        raise ValueError(
            f"Tool arguments for {tool_name or 'tool'} must decode to an object"
        )
    return parsed


# ---------------------------------------------------------------------------
# Abstract backend
# ---------------------------------------------------------------------------

class LLMBackend(ABC):
    """Base class for LLM backends."""

    @abstractmethod
    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ) -> ChatResponse:
        """Send a chat completion request, optionally with tool definitions."""

    @abstractmethod
    def close(self):
        """Release resources."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ---------------------------------------------------------------------------
# OpenAI-compatible backend (Ollama, vLLM, OpenRouter, Azure)
# ---------------------------------------------------------------------------

class OpenAIBackend(LLMBackend):
    """
    Works with any OpenAI-compatible /v1/chat/completions endpoint.
    Covers: Ollama, vLLM, OpenRouter (Claude via OpenAI format), Azure.
    """

    def __init__(
        self,
        endpoint: str,
        model: str,
        api_key: str = "",
        timeout_s: float = 300.0,
    ):
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self._api_key = api_key
        self._client = make_client(timeout_s=timeout_s)

    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ) -> ChatResponse:
        t0 = time.monotonic()
        headers = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        # Newer models (gpt-5+, gpt-4.1, o-series) require max_completion_tokens
        _new_style = any(self.model.startswith(p) for p in ("gpt-5", "gpt-4.1", "o1", "o3", "o4"))
        token_key = "max_completion_tokens" if _new_style else "max_tokens"
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            token_key: max_tokens,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        url = f"{self.endpoint}/chat/completions"
        resp = self._client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()

        msg = data["choices"][0]["message"]
        tool_calls = []
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function", {})
            args = fn.get("arguments", "{}")
            tool_calls.append(ToolCall(
                id=tc.get("id", ""),
                name=fn.get("name", ""),
                arguments=_decode_tool_arguments(
                    args,
                    tool_name=fn.get("name", ""),
                ),
            ))

        usage = data.get("usage", {})
        return ChatResponse(
            content=msg.get("content") or "",
            tool_calls=tool_calls,
            model=data.get("model", self.model),
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            elapsed_s=time.monotonic() - t0,
            raw=data,
        )

    def close(self):
        self._client.close()


# ---------------------------------------------------------------------------
# Anthropic backend (direct Claude API)
# ---------------------------------------------------------------------------

class AnthropicBackend(LLMBackend):
    """
    Direct Anthropic Messages API with native tool use.
    Endpoint: https://api.anthropic.com/v1/messages
    """

    API_URL = "https://api.anthropic.com/v1/messages"

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key: str = "",
        timeout_s: float = 300.0,
        max_retries: int = 2,
    ):
        self.model = model
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self._client = httpx.Client(
            timeout=httpx.Timeout(timeout_s),
            transport=httpx.HTTPTransport(retries=max_retries),
        )

    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ) -> ChatResponse:
        t0 = time.monotonic()

        # Separate system message from conversation
        system_text = ""
        conv_messages = []
        for m in messages:
            if m["role"] == "system":
                system_text = m["content"]
            else:
                conv_messages.append(m)

        headers = {
            "x-api-key": self._api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": self._convert_messages(conv_messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if system_text:
            payload["system"] = system_text
        if tools:
            payload["tools"] = self._convert_tools(tools)

        resp = self._client.post(self.API_URL, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()

        # Parse response
        content_text = ""
        tool_calls = []
        for block in data.get("content", []):
            if block["type"] == "text":
                content_text += block["text"]
            elif block["type"] == "tool_use":
                tool_calls.append(ToolCall(
                    id=block["id"],
                    name=block["name"],
                    arguments=block.get("input", {}),
                ))

        usage = data.get("usage", {})
        return ChatResponse(
            content=content_text,
            tool_calls=tool_calls,
            model=data.get("model", self.model),
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            elapsed_s=time.monotonic() - t0,
            raw=data,
        )

    def _convert_messages(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Convert OpenAI-style messages to Anthropic format."""
        result = []
        for m in messages:
            role = m["role"]
            if role == "tool":
                # Tool result in Anthropic format
                result.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": m.get("tool_call_id", ""),
                        "content": m.get("content", ""),
                    }],
                })
            elif role == "assistant" and m.get("tool_calls"):
                # Assistant message with tool calls
                content = []
                if m.get("content"):
                    content.append({"type": "text", "text": m["content"]})
                for tc in m["tool_calls"]:
                    fn = tc.get("function", tc)
                    args = fn.get("arguments", {})
                    content.append({
                        "type": "tool_use",
                        "id": tc.get("id", ""),
                        "name": fn.get("name", ""),
                        "input": _decode_tool_arguments(
                            args,
                            tool_name=fn.get("name", ""),
                        ),
                    })
                result.append({"role": "assistant", "content": content})
            else:
                result.append({"role": role, "content": m.get("content", "")})
        return result

    def _convert_tools(
        self, tools: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Convert OpenAI tool format to Anthropic tool format."""
        result = []
        for t in tools:
            fn = t.get("function", t)
            result.append({
                "name": fn["name"],
                "description": fn.get("description", ""),
                "input_schema": fn.get("parameters", {}),
            })
        return result

    def close(self):
        self._client.close()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_backend(
    backend_type: str = "openai",
    endpoint: str = "http://localhost:11434/v1",
    model: str = "phi4:14b-q4_K_M",
    api_key: str = "",
    timeout_s: float = 300.0,
) -> LLMBackend:
    """
    Create an LLM backend by type.

    backend_type:
        "openai"    -- OpenAI-compatible (Ollama, vLLM, OpenRouter)
        "anthropic" -- Direct Anthropic Claude API
        "ollama"    -- Alias for openai with Ollama default endpoint

    For OpenRouter (Claude via OpenAI format):
        create_backend("openai",
                       endpoint="https://openrouter.ai/api/v1",
                       model="anthropic/claude-sonnet-4-20250514",
                       api_key=os.environ["OPENROUTER_API_KEY"])

    For local Ollama:
        create_backend("ollama", model="phi4:14b-q4_K_M")

    For direct Anthropic:
        create_backend("anthropic",
                       model="claude-sonnet-4-20250514",
                       api_key=os.environ["ANTHROPIC_API_KEY"])
    """
    if backend_type in ("openai", "ollama"):
        if backend_type == "ollama" and endpoint == "http://localhost:11434/v1":
            pass  # default is already Ollama
        return OpenAIBackend(
            endpoint=endpoint,
            model=model,
            api_key=api_key,
            timeout_s=timeout_s,
        )
    elif backend_type == "anthropic":
        return AnthropicBackend(
            model=model,
            api_key=api_key,
            timeout_s=timeout_s,
        )
    else:
        raise ValueError(f"Unknown backend type: {backend_type!r}")
