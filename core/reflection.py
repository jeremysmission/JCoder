"""
Reflection Engine (Self-RAG)
----------------------------
Implements Self-RAG style reflection tokens for self-assessment.

Three reflection signals:
  ISREL  -- Are the retrieved chunks relevant to the query?
  ISSUP  -- Does the retrieved evidence support the generated answer?
  ISUSE  -- Is the final answer useful and complete?

Each produces a 0.0-1.0 confidence score. These signals drive:
- Corrective retrieval (re-query on low ISREL)
- Hallucination detection (low ISSUP)
- Quality gating (low ISUSE triggers fallback)
- Telemetry logging for feedback loops

Uses the same LLM endpoint as the main runtime (no extra model needed).
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from core.runtime import Runtime


# Lightweight prompts that extract a 0-10 score from the LLM.
# Kept short to minimize latency; the LLM only needs to output a number.

_ISREL_PROMPT = (
    "Rate 0-10 how relevant these code chunks are to the question. "
    "Output ONLY a single integer 0-10.\n\n"
    "Question: {question}\n\n"
    "Chunks:\n{chunks}\n\n"
    "Relevance score:"
)

_ISSUP_PROMPT = (
    "Rate 0-10 how well the evidence supports this answer. "
    "Output ONLY a single integer 0-10.\n\n"
    "Answer: {answer}\n\n"
    "Evidence:\n{chunks}\n\n"
    "Support score:"
)

_ISUSE_PROMPT = (
    "Rate 0-10 the overall quality and usefulness of this answer "
    "to the question. Output ONLY a single integer 0-10.\n\n"
    "Question: {question}\n\n"
    "Answer: {answer}\n\n"
    "Usefulness score:"
)

_SCORE_RE = re.compile(r"\b(10|[0-9])\b")


def _extract_score(text: str) -> float:
    """Parse an integer 0-10 from LLM output, return as 0.0-1.0."""
    m = _SCORE_RE.search(text.strip())
    if m:
        return int(m.group(1)) / 10.0
    return 0.5  # default when parse fails


class ReflectionEngine:
    """
    Self-assessment engine using the project's existing LLM endpoint.
    Produces ISREL, ISSUP, ISUSE confidence scores (0.0-1.0).
    """

    def __init__(self, runtime: Runtime):
        self.runtime = runtime

    def score_relevance(self, question: str,
                        chunks: List[Dict]) -> float:
        """ISREL: Are chunks relevant to the query? Returns 0.0-1.0."""
        chunk_text = self._format_chunks(chunks)
        prompt = _ISREL_PROMPT.format(question=question, chunks=chunk_text)
        raw = self.runtime.generate(
            question=prompt, context_chunks=[],
            system_prompt="You are a relevance evaluator. Output only a number.",
            temperature=0.0, max_tokens=8,
        )
        return _extract_score(raw)

    def score_support(self, answer: str,
                      chunks: List[Dict]) -> float:
        """ISSUP: Does evidence support the answer? Returns 0.0-1.0."""
        chunk_text = self._format_chunks(chunks)
        prompt = _ISSUP_PROMPT.format(answer=answer[:300], chunks=chunk_text)
        raw = self.runtime.generate(
            question=prompt, context_chunks=[],
            system_prompt="You are a factual support evaluator. Output only a number.",
            temperature=0.0, max_tokens=8,
        )
        return _extract_score(raw)

    def score_usefulness(self, question: str, answer: str) -> float:
        """ISUSE: Is the answer useful? Returns 0.0-1.0."""
        prompt = _ISUSE_PROMPT.format(question=question, answer=answer[:300])
        raw = self.runtime.generate(
            question=prompt, context_chunks=[],
            system_prompt="You are a quality evaluator. Output only a number.",
            temperature=0.0, max_tokens=8,
        )
        return _extract_score(raw)

    def full_reflection(self, question: str, chunks: List[Dict],
                        answer: str) -> Dict[str, float]:
        """
        Run all three reflection checks. Returns dict with keys:
        relevant, supported, useful, confidence (aggregate).
        """
        rel = self.score_relevance(question, chunks)
        sup = self.score_support(answer, chunks)
        use = self.score_usefulness(question, answer)
        # Weighted aggregate: support matters most (hallucination guard)
        confidence = 0.3 * rel + 0.4 * sup + 0.3 * use
        return {
            "relevant": rel,
            "supported": sup,
            "useful": use,
            "confidence": round(confidence, 3),
        }

    @staticmethod
    def _format_chunks(chunks: List[Dict], max_chars: int = 2000) -> str:
        """Format chunks for reflection prompts, truncating to limit."""
        parts = []
        total = 0
        for c in chunks:
            text = c.get("content", "")[:500]
            src = c.get("source_path", "?")
            part = f"[{src}]\n{text}"
            if total + len(part) > max_chars:
                break
            parts.append(part)
            total += len(part)
        return "\n---\n".join(parts) if parts else "(no chunks)"
