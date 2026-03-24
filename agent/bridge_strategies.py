"""
Strategy, routing, and reasoning-enhancement methods for AgentBridge.

Extracted from agent/bridge.py to stay under the 500-line module limit.
All self-learning modules are optional -- degrades gracefully.
"""
from __future__ import annotations
import hashlib, logging, time
from typing import Any, Callable, Dict, List, Optional

log = logging.getLogger(__name__)


class BridgeStrategyMixin:
    """Mixin providing strategy selection, cascade routing, and reasoning
    enhancement methods.  Mixed into AgentBridge in agent/bridge.py.

    Attributes used from the host class (AgentBridge):
        meta, cascade, pipeline, smart_orchestrator, qd_archive,
        experience, telemetry
    """

    # --- Strategy selection ---

    def select_strategy(self, task: str) -> Dict[str, Any]:
        """Ask meta_cognitive which approach to use for this task.

        Also consults QD archive for niche-specific hints.
        """
        # Import classify_query from bridge.py's module namespace so
        # tests that patch agent.bridge.classify_query still work.
        import agent.bridge as _bridge
        _classify_query = _bridge.classify_query

        defaults: Dict[str, Any] = {"temperature": 0.1, "max_iterations": 50,
                                     "strategy": "standard"}
        if not self.meta or not _classify_query:
            return self._apply_qd_hints(task, defaults)
        try:
            name, sig = self.meta.select_strategy(task)
            result = {"temperature": 0.1 if sig.complexity < 0.5 else 0.3,
                      "max_iterations": 30 if sig.query_type == "lookup" else 50,
                      "strategy": name, "query_type": sig.query_type,
                      "complexity": sig.complexity, "has_code": sig.has_code}
            return self._apply_qd_hints(task, result)
        except Exception as exc:
            log.warning("Strategy selection failed: %s", exc)
            return self._apply_qd_hints(task, defaults)

    def _apply_qd_hints(self, task: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Overlay quality-diversity archive hints if available."""
        import agent.bridge as _bridge
        _compute_behavior = _bridge.compute_behavior
        _classify_query = _bridge.classify_query

        if not self.qd_archive or _compute_behavior is None or _classify_query is None:
            return params
        try:
            sig = _classify_query(task)
            behavior = _compute_behavior(
                query_complexity=sig.complexity,
                answer_type=sig.query_type,
                retrieval_confidence=0.5,
            )
            niche = self.qd_archive.lookup(behavior)
            if niche and niche.get("fitness", 0) > 0.5:
                params["qd_niche"] = niche.get("niche", "")
                params["qd_fitness"] = niche.get("fitness", 0)
        except Exception as exc:
            log.warning("QD lookup: %s", exc)
        return params

    # --- Routing methods ---

    def cascade_route(
        self, question: str, context_chunks: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Route a question through the ModelCascade (complexity-based routing).

        Returns dict with answer, model_used, complexity, escalation info,
        and latency, or None if the cascade is unavailable.
        """
        casc = self.cascade
        if casc is None and self.pipeline and hasattr(self.pipeline, "cascade"):
            casc = self.pipeline.cascade
        if casc is None:
            return None
        try:
            from core.cascade import estimate_answer_confidence
            result = casc.route(
                question,
                context_chunks or [],
                confidence_fn=estimate_answer_confidence,
            )
            return {
                "answer": result.answer,
                "model_used": result.model_used,
                "level_index": result.level_index,
                "complexity_score": result.complexity_score,
                "escalated": result.escalated,
                "latency_ms": result.latency_ms,
            }
        except Exception as exc:
            log.warning("ModelCascade route: %s", exc)
            return None

    def smart_answer(self, question: str) -> Optional[Dict[str, Any]]:
        """Route a question through the SmartOrchestrator (CRAG + Self-RAG).

        Returns dict with answer, confidence, reflection scores, and retrieval
        metadata, or None if the orchestrator is unavailable.
        """
        orch = self.smart_orchestrator
        if orch is None and self.pipeline and hasattr(self.pipeline, "smart_orchestrator"):
            orch = self.pipeline.smart_orchestrator
        if orch is None:
            return None
        try:
            result = orch.answer(question)
            return {
                "answer": result.answer,
                "confidence": result.confidence,
                "reflection": result.reflection,
                "sources": result.sources,
                "chunk_count": result.chunk_count,
                "retrieval_strategy": result.retrieval_strategy,
                "retrieval_attempts": result.retrieval_attempts,
            }
        except Exception as exc:
            log.warning("SmartOrchestrator answer: %s", exc)
            return None

    # --- Reasoning enhancement methods ---

    def enhance_with_reasoning(
        self, task: str, result,
    ) -> Optional[Dict[str, Any]]:
        """Post-process a task result with STaR + Reflection.

        If STaR is available and confidence is borderline, generates a
        rationalization trace. If Reflection is available, produces a
        self-critique and optional revision. Returns enhancement dict or None.
        """
        enhancements: Dict[str, Any] = {}

        # STaR: generate reasoning trace for borderline results
        if self.pipeline and hasattr(self.pipeline, "star") and self.pipeline.star:
            try:
                conf = 1.0 if result.success else 0.0
                if result.success and result.iterations > 5:
                    conf = max(0.3, 1.0 - (result.iterations / 50.0))
                if conf < 0.7:  # borderline -- worth rationalizing
                    reasoning, _ = self.pipeline.star.answer_with_reasoning(
                        task, [result.summary[:500]],
                    )
                    enhancements["star_reasoning"] = reasoning[:500]
                    # Store trace in experience replay
                    if self.experience:
                        try:
                            tid = hashlib.sha256(
                                f"star:{task}:{time.time()}".encode()
                            ).hexdigest()[:12]
                            self.experience.store(
                                exp_id=tid, query=task,
                                answer=f"[RATIONALIZED] {reasoning[:400]}",
                                source_files=[], confidence=conf,
                            )
                        except Exception as exc:
                            log.debug("STaR experience store: %s", exc)
            except Exception as exc:
                log.debug("STaR enhancement: %s", exc)

        # Reflection: self-critique
        if self.pipeline and hasattr(self.pipeline, "reflection") and self.pipeline.reflection:
            try:
                chunks = [{"content": result.summary[:500], "source_path": "agent_result"}]
                scores = self.pipeline.reflection.full_reflection(
                    task, chunks, result.summary,
                )
                enhancements["reflection"] = scores
            except Exception as exc:
                log.debug("Reflection enhancement: %s", exc)

        return enhancements if enhancements else None

    def best_of_n_query(
        self, question: str, context_chunks: List[str],
    ) -> Optional[Dict[str, Any]]:
        """Run a Best-of-N generation for a thorough answer.

        Returns dict with answer, score, and candidate count, or None.
        """
        if not self.pipeline or not hasattr(self.pipeline, "best_of_n") or not self.pipeline.best_of_n:
            return None
        try:
            result = self.pipeline.best_of_n.generate(question, context_chunks)
            return {
                "answer": result.answer,
                "score": result.selected_score,
                "candidates": result.candidates_generated,
                "all_scores": result.all_scores,
            }
        except Exception as exc:
            log.debug("Best-of-N query: %s", exc)
            return None
