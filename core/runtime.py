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
from .network_gate import NetworkGate


class Runtime:
    """
    Responsible ONLY for talking to the LLM endpoint.
    Uses vLLM's OpenAI-compatible /v1/chat/completions API.
    """

    SYSTEM_PROMPT = (
        "You are a code assistant. Answer the user's question using ONLY "
        "the provided context. If the context does not contain enough "
        "information, say so. Cite source files when possible."
    )

    def __init__(self, config: ModelConfig, timeout: int = 120,
                 gate: NetworkGate = None):
        self.endpoint = config.endpoint.rstrip("/")
        self.model_name = config.name
        self._client = httpx.Client(timeout=httpx.Timeout(timeout))
        self._gate = gate

    def generate(
        self,
        question: str,
        context_chunks: List[str],
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Send question + retrieved context to the model.
        Returns the model's text response.
        """
        context_block = "\n\n---\n\n".join(context_chunks)
        prompt = system_prompt or self.SYSTEM_PROMPT

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": f"Context:\n{context_block}\n\nQuestion:\n{question}",
                },
            ],
            "temperature": 0.1,
            "max_tokens": 2048,
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
