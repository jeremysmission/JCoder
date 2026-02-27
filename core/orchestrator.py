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

import concurrent.futures
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

    def __init__(self, retriever: RetrievalEngine, runtime: Runtime,
                 timeout: float = 300.0):
        self.retriever = retriever
        self.runtime = runtime
        self._timeout = timeout

    def answer(self, question: str) -> AnswerResult:
        """
        End-to-end: retrieve context, generate answer, return with sources.
        Raises TimeoutError if the pipeline exceeds the configured timeout.
        """
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(self._answer_sync, question)
            try:
                return future.result(timeout=self._timeout)
            except concurrent.futures.TimeoutError:
                raise TimeoutError(
                    f"Pipeline exceeded {self._timeout}s timeout"
                )

    def _answer_sync(self, question: str) -> AnswerResult:
        """Synchronous answer pipeline (runs inside timeout wrapper)."""
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
