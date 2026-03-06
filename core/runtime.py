"""
Runtime Module
--------------
Handles communication with the vLLM LLM server.

Non-programmer explanation:
This is the part that sends the retrieved code and your question
to the AI model and gets back an answer. It talks to vLLM's
OpenAI-compatible chat API running on localhost.
"""

from typing import List, Optional

import httpx

from .config import ModelConfig
from .http_factory import make_client
from .network_gate import NetworkGate

DEFAULT_SYSTEM_PROMPT = (
    "You are a code assistant. Answer the user's question using ONLY "
    "the provided context. If the context does not contain enough "
    "information, say so. Cite source files when possible."
)


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
    ):
        self.endpoint = config.endpoint.rstrip("/")
        self.model_name = config.name
        self._client = make_client(timeout_s=timeout)
        self._gate = gate
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT

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
        context_block = "\n\n---\n\n".join(context_chunks)
        prompt = system_prompt or self._system_prompt

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
        response = self._client.post(url, json=payload)
        response.raise_for_status()

        return response.json()["choices"][0]["message"]["content"]

    def close(self):
        """Release HTTP connection pool."""
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
