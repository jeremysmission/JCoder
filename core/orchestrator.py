"""
Orchestrator
------------
Coordinates retrieval and generation into a single answer pipeline.

Non-programmer explanation:
This is the conductor of the orchestra.
It tells each component when to play:
1. Retrieval engine finds relevant code.
2. Runtime sends that code + your question to the LLM.
3. You get an answer grounded in real code.
"""

from dataclasses import dataclass
from typing import List, Optional

from .retrieval_engine import RetrievalEngine
from .runtime import Runtime


@dataclass
class AnswerResult:
    """Structured response from the orchestrator."""
    answer: str
    sources: List[str]
    chunk_count: int
    chunks: Optional[List[dict]] = None


class Orchestrator:
    """
    Single entry point for answering questions.
    Composes retrieval + generation with source tracking.
    """

    def __init__(self, retriever: RetrievalEngine, runtime: Runtime):
        self.retriever = retriever
        self.runtime = runtime

    def answer(self, question: str) -> AnswerResult:
        """
        End-to-end: retrieve context, generate answer, return with sources.
        """
        chunks = self.retriever.retrieve(question)

        if not chunks:
            return AnswerResult(
                answer="No relevant code found in the index.",
                sources=[],
                chunk_count=0,
            )

        chunk_texts = [c.get("content", "") for c in chunks]
        sources = list({c.get("source_path", "unknown") for c in chunks})

        response = self.runtime.generate(question, chunk_texts)

        return AnswerResult(
            answer=response,
            sources=sorted(sources),
            chunk_count=len(chunks),
            chunks=chunks,
        )
