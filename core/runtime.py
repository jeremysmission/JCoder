"""Runtime Module."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, List, Optional

import httpx

from .config import ModelConfig
from .http_factory import make_client
from .network_gate import NetworkGate

DEFAULT_SYSTEM_PROMPT = (
    "You are a code assistant. Answer the user's question using ONLY "
    "the provided context. If the context does not contain enough "
    "information, say so. Cite source files when possible."
)


@dataclass
class GenerationResult:
    """Structured runtime output with optional token logprob metadata."""

    text: str
    logprobs: List[dict[str, Any]] = field(default_factory=list)

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


class Runtime:
    """
    Responsible ONLY for talking to the LLM endpoint.
    Uses vLLM's OpenAI-compatible /v1/chat/completions API.
    """

    def __init__(
        self,
        config: ModelConfig,
        timeout: int = 120,
        gate: NetworkGate = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        system_prompt: str = None,
        max_context_tokens: int = 8192,
        api_key: str = "",
    ):
        self.endpoint = config.endpoint.rstrip("/")
        self.model_name = config.name
        self._client = make_client(timeout_s=timeout)
        self._gate = gate
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self._max_context_tokens = max_context_tokens
        self._api_key = api_key

    def generate(
        self,
        question: str,
        context_chunks: List[str],
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Send question + retrieved context to the model.
        Returns the model's text response.

        Per-call overrides for temperature/max_tokens/system_prompt
        take precedence over instance defaults.
        """
        prompt = system_prompt or self._system_prompt

        # Enforce context budget (~4 chars/token).
        # Reserve room for: system prompt, question, separators, output tokens.
        chars_per_token = 4
        overhead_tokens = (
            len(prompt) // chars_per_token       # system prompt
            + len(question) // chars_per_token   # user question
            + self._max_tokens                   # reserved for generation output
            + 100                                # separators + framing
        )
        budget_tokens = max(0, self._max_context_tokens - overhead_tokens)
        budget_chars = budget_tokens * chars_per_token
        separator = "\n\n---\n\n"
        sep_chars = len(separator)

        trimmed = []
        total = 0
        for chunk in context_chunks:
            needed = len(chunk) + (sep_chars if trimmed else 0)
            if total + needed > budget_chars:
                remaining = budget_chars - total - (sep_chars if trimmed else 0)
                if remaining > 200:
                    trimmed.append(chunk[:remaining])
                break
            trimmed.append(chunk)
            total += needed
        context_block = separator.join(trimmed)

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": f"Context:\n{context_block}\n\nQuestion:\n{question}",
                },
            ],
            "temperature": temperature if temperature is not None else self._temperature,
            "max_tokens": max_tokens if max_tokens is not None else self._max_tokens,
        }

        url = f"{self.endpoint}/chat/completions"
        if self._gate:
            self._gate.guard(url)
        headers = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        response = self._client.post(url, json=payload, headers=headers)
        response.raise_for_status()

        try:
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            import logging
            logging.getLogger(__name__).error(
                "Unexpected LLM response structure: %s -- raw keys: %s",
                exc,
                list(data.keys()) if isinstance(data, dict) else type(data).__name__,
            )
            raise ValueError(
                f"LLM response missing expected fields: {exc}"
            ) from exc

    def generate_with_logprobs(
        self,
        question: str,
        context_chunks: List[str],
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> GenerationResult:
        """Best-effort logprob-aware generation with graceful fallback."""
        text = self.generate(
            question,
            context_chunks,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return GenerationResult(text=text, logprobs=[])

    def close(self):
        """Release HTTP connection pool."""
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
