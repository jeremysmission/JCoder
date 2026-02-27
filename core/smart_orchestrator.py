"""
Smart Orchestrator (Self-Learning Pipeline)
--------------------------------------------
Drop-in replacement for Orchestrator that adds:

1. Corrective retrieval (CRAG) -- confidence-gated re-query
2. Self-RAG reflection -- ISREL/ISSUP/ISUSE scoring
3. Telemetry logging -- every query feeds the feedback loop
4. Confidence-based answer gating -- low confidence = honest "I don't know"

The standard Orchestrator is still available for production use.
SmartOrchestrator wraps it with self-learning capabilities that
can be enabled/disabled independently.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from core.orchestrator import AnswerResult, Orchestrator
from core.corrective_retrieval import CorrectiveRetriever
from core.reflection import ReflectionEngine
from core.retrieval_engine import RetrievalEngine
from core.runtime import Runtime
from core.telemetry import QueryEvent, TelemetryStore


@dataclass
class SmartAnswerResult(AnswerResult):
    """Extended answer result with self-assessment metadata."""
    confidence: float = 0.0
    reflection: Dict[str, float] = field(default_factory=dict)
    retrieval_strategy: str = "standard"
    retrieval_attempts: int = 1


class SmartOrchestrator:
    """
    Self-learning orchestrator. Wraps retrieval + generation with:
    - Corrective retrieval (CRAG pattern)
    - Self-RAG reflection scoring
    - Telemetry logging for feedback loops
    - Confidence gating

    Each feature can be independently disabled by passing None.
    """

    def __init__(
        self,
        retriever: RetrievalEngine,
        runtime: Runtime,
        timeout: float = 300.0,
        telemetry: Optional[TelemetryStore] = None,
        reflection: Optional[ReflectionEngine] = None,
        corrective: Optional[CorrectiveRetriever] = None,
        confidence_gate: float = 0.2,
    ):
        self.retriever = retriever
        self.runtime = runtime
        self._timeout = timeout
        self.telemetry = telemetry
        self.reflection = reflection
        self.corrective = corrective
        self.confidence_gate = confidence_gate

        # Fallback standard orchestrator for timeout wrapping
        self._base = Orchestrator(retriever, runtime, timeout)

    def answer(self, question: str) -> SmartAnswerResult:
        """
        Full self-learning answer pipeline:
        1. Corrective retrieval (if enabled)
        2. Generation
        3. Self-RAG reflection (if enabled)
        4. Confidence gating
        5. Telemetry logging
        """
        t0 = time.time()
        query_id = hashlib.sha256(
            f"{question}:{t0}".encode()).hexdigest()[:12]

        # --- Step 1: Retrieval (corrective or standard) ---
        t_ret = time.time()
        retrieval_meta = {}

        if self.corrective:
            chunks, retrieval_meta = self.corrective.retrieve(question)
        else:
            chunks = self.retriever.retrieve(question)
            retrieval_meta = {"strategy": "standard", "confidence": 1.0,
                              "attempts": 1}

        retrieval_ms = (time.time() - t_ret) * 1000

        if not chunks:
            result = SmartAnswerResult(
                answer="No relevant code found in the index.",
                sources=[], chunk_count=0,
                confidence=0.0,
                retrieval_strategy=retrieval_meta.get("strategy", "none"),
            )
            self._log_event(query_id, question, result, retrieval_ms, 0)
            return result

        # --- Step 2: Generation ---
        t_gen = time.time()
        chunk_texts = [c.get("content", "") for c in chunks]
        sources = sorted({c.get("source_path", "unknown") for c in chunks})
        response = self.runtime.generate(question, chunk_texts)
        gen_ms = (time.time() - t_gen) * 1000

        # --- Step 3: Self-RAG reflection ---
        reflection_scores = {}
        confidence = retrieval_meta.get("confidence", 0.5)

        if self.reflection:
            try:
                reflection_scores = self.reflection.full_reflection(
                    question, chunks, response)
                confidence = reflection_scores.get("confidence", confidence)
            except Exception:
                pass  # reflection failure should not block the answer

        # --- Step 4: Confidence gating ---
        if confidence < self.confidence_gate:
            response = (
                f"I found some code but I'm not confident it answers your "
                f"question well (confidence: {confidence:.0%}). Here's my "
                f"best attempt:\n\n{response}"
            )

        result = SmartAnswerResult(
            answer=response,
            sources=sources,
            chunk_count=len(chunks),
            chunks=chunks,
            confidence=confidence,
            reflection=reflection_scores,
            retrieval_strategy=retrieval_meta.get("strategy", "standard"),
            retrieval_attempts=retrieval_meta.get("attempts", 1),
        )

        # --- Step 5: Telemetry ---
        self._log_event(
            query_id, question, result, retrieval_ms, gen_ms)

        return result

    def _log_event(self, query_id: str, question: str,
                    result: SmartAnswerResult,
                    retrieval_ms: float, gen_ms: float) -> None:
        """Log to telemetry store if available."""
        if not self.telemetry:
            return

        try:
            event = QueryEvent(
                query_id=query_id,
                query_text=question,
                timestamp=time.time(),
                retrieval_latency_ms=retrieval_ms,
                generation_latency_ms=gen_ms,
                chunk_ids=[c.get("id", "") for c in (result.chunks or [])],
                chunk_scores=[],
                source_files=result.sources,
                answer_snippet=result.answer[:500],
                confidence=result.confidence,
                reflection_relevant=result.reflection.get("relevant", 0),
                reflection_supported=result.reflection.get("supported", 0),
                reflection_useful=result.reflection.get("useful", 0),
            )
            self.telemetry.log(event)
        except Exception:
            pass  # telemetry failure should never block answers
