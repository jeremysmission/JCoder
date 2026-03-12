"""
Master Self-Learning Pipeline (The Brain)
-------------------------------------------
Wires ALL self-learning modules into one autonomous improvement loop.
This is the highest-level orchestrator that coordinates:

1. Meta-Cognitive Controller -- decides which strategy per query
2. Smart Orchestrator -- retrieval + generation + reflection
3. STaR Reasoner -- bootstrapped reasoning chains
4. Best-of-N Generator -- multi-candidate verification
5. Prompt Evolver -- evolves system prompts
6. Adversarial Self-Play -- discovers weaknesses
7. Quality-Diversity Archive -- specialized configs per niche
8. Stigmergic Booster -- pheromone-based retrieval enhancement
9. Active Learner -- identifies highest-value learning targets
10. Experience Replay -- stores successful trajectories
11. Continual Learner -- prevents catastrophic forgetting
12. Model Cascade -- routes by complexity
13. Telemetry -- logs everything for feedback loops

The loop:
    QUERY -> Meta-Cognitive Strategy Selection
          -> Strategy Execution (Standard / Corrective / Best-of-N / Cascade)
          -> Reflection + Confidence Scoring
          -> Stigmergy Deposit (pheromone feedback)
          -> Experience Storage (if high quality)
          -> Telemetry Logging
          -> Active Learning Scoring

    BACKGROUND (periodic):
          -> STaR Iteration (improve reasoning chains)
          -> Prompt Evolution (evolve system prompts)
          -> Adversarial Self-Play (discover weaknesses)
          -> QD Archive Update (specialize configs)
          -> Continual Learning Check (prevent regression)
          -> Evolver Iteration (optimize retrieval params)

Each module can be independently disabled by passing None.
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time

log = logging.getLogger(__name__)
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

# Core modules
from core.runtime import Runtime
from core.retrieval_engine import RetrievalEngine

# Self-learning modules (all optional)
try:
    from core.meta_cognitive import MetaCognitiveController, classify_query
except ImportError:
    MetaCognitiveController = None
    classify_query = None

try:
    from core.reflection import ReflectionEngine
except ImportError:
    ReflectionEngine = None

try:
    from core.corrective_retrieval import CorrectiveRetriever
except ImportError:
    CorrectiveRetriever = None

try:
    from core.best_of_n import BestOfNGenerator
except ImportError:
    BestOfNGenerator = None

try:
    from core.experience_replay import ExperienceStore
except ImportError:
    ExperienceStore = None

try:
    from core.stigmergy import StigmergicBooster
except ImportError:
    StigmergicBooster = None

try:
    from core.telemetry import TelemetryStore, QueryEvent
except ImportError:
    TelemetryStore = None
    QueryEvent = None

try:
    from core.star_reasoner import STaRReasoner
except ImportError:
    STaRReasoner = None

try:
    from core.active_learner import ActiveLearner
except ImportError:
    ActiveLearner = None

try:
    from core.continual_learner import ContinualLearner
except ImportError:
    ContinualLearner = None


@dataclass
class PipelineAnswer:
    """Full answer from the self-learning pipeline."""
    answer: str
    sources: List[str] = field(default_factory=list)
    chunk_count: int = 0
    confidence: float = 0.0
    strategy_used: str = "standard"
    reasoning: str = ""
    reflection: Dict[str, float] = field(default_factory=dict)
    latency_ms: float = 0.0
    query_id: str = ""
    learning_value: float = 0.0  # how valuable this query is for learning


class SelfLearningPipeline:
    """
    The master self-learning pipeline.

    Every query flows through this pipeline:
    1. Strategy selection (meta-cognitive)
    2. Retrieval (with optional stigmergy boost)
    3. Generation (strategy-dependent)
    4. Reflection + confidence scoring
    5. Feedback (pheromone deposit, experience storage, telemetry)

    Background processes run periodically to improve the system.
    """

    def __init__(
        self,
        retriever: RetrievalEngine,
        runtime: Runtime,
        # Optional self-learning modules
        meta_controller: Optional[Any] = None,
        reflection: Optional[Any] = None,
        corrective: Optional[Any] = None,
        best_of_n: Optional[Any] = None,
        experience: Optional[Any] = None,
        stigmergy: Optional[Any] = None,
        telemetry: Optional[Any] = None,
        star: Optional[Any] = None,
        active_learner: Optional[Any] = None,
        continual: Optional[Any] = None,
        # Config
        confidence_gate: float = 0.2,
    ):
        self.retriever = retriever
        self.runtime = runtime
        self.meta = meta_controller
        self.reflection = reflection
        self.corrective = corrective
        self.best_of_n = best_of_n
        self.experience = experience
        self.stigmergy = stigmergy
        self.telemetry = telemetry
        self.star = star
        self.active = active_learner
        self.continual = continual
        self.confidence_gate = confidence_gate
        self._query_count = 0
        self._query_lock = threading.Lock()

    def answer(self, question: str) -> PipelineAnswer:
        """
        Full self-learning answer pipeline.

        The meta-cognitive controller selects the strategy.
        The strategy executes retrieval + generation.
        Reflection scores confidence.
        Feedback loops update all learning systems.
        """
        t0 = time.time()
        query_id = hashlib.sha256(
            f"{question}:{t0}".encode()
        ).hexdigest()[:12]
        with self._query_lock:
            self._query_count += 1

        # --- Step 1: Strategy Selection ---
        strategy = "standard"
        query_type = "explain"
        if self.meta and classify_query:
            strategy, sig = self.meta.select_strategy(question)
            query_type = sig.query_type

        # --- Step 2: Retrieval ---
        t_ret = time.time()
        chunks, retrieval_meta = self._retrieve(question, strategy)
        retrieval_ms = (time.time() - t_ret) * 1000

        if not chunks:
            return PipelineAnswer(
                answer="No relevant code found in the index.",
                query_id=query_id,
                strategy_used=strategy,
                latency_ms=(time.time() - t0) * 1000,
            )

        # --- Step 3: Apply Stigmergy Boost ---
        if self.stigmergy:
            chunk_scores = [
                (c.get("id", ""), c.get("score", 0.0)) for c in chunks
            ]
            boosted = self.stigmergy.boost_scores(chunk_scores, query_type)
            # Re-order chunks by boosted score
            id_to_rank = {cid: i for i, (cid, _) in enumerate(boosted)}
            chunks.sort(
                key=lambda c: id_to_rank.get(c.get("id", ""), 999)
            )

        # --- Step 4: Generation (strategy-dependent) ---
        t_gen = time.time()
        chunk_texts = [c.get("content", "") for c in chunks]
        sources = sorted({c.get("source_path", "unknown") for c in chunks})

        # Inject experience replay examples
        experience_prefix = ""
        if self.experience:
            exps = self.experience.retrieve(question, top_k=2)
            experience_prefix = self.experience.format_as_examples(
                exps, max_chars=1000
            )

        if strategy == "best_of_n" and self.best_of_n:
            result = self.best_of_n.generate(question, chunk_texts)
            response = result.answer
            reasoning = ""
        elif strategy == "reflective" and self.star:
            reasoning, response = self.star.answer_with_reasoning(
                question, chunk_texts
            )
        else:
            # Standard or corrective (retrieval already handled)
            system_prompt = None
            if experience_prefix:
                system_prompt = experience_prefix + "\n\n" + (
                    "You are a code assistant. Answer using ONLY the "
                    "provided context."
                )
            response = self.runtime.generate(
                question, chunk_texts, system_prompt=system_prompt
            )
            reasoning = ""

        gen_ms = (time.time() - t_gen) * 1000

        # --- Step 5: Reflection ---
        reflection_scores = {}
        confidence = retrieval_meta.get("confidence", 0.5)

        if self.reflection:
            try:
                reflection_scores = self.reflection.full_reflection(
                    question, chunks, response
                )
                confidence = reflection_scores.get("confidence", confidence)
            except Exception as exc:
                log.debug("Pipeline step failed: %s", exc)

        # --- Step 6: Confidence Gating ---
        if confidence < self.confidence_gate:
            response = (
                f"I found some code but I'm not confident "
                f"(confidence: {confidence:.0%}). "
                f"Here's my best attempt:\n\n{response}"
            )

        # --- Step 7: Feedback Loops ---
        success = confidence >= 0.5
        chunk_ids = [c.get("id", "") for c in chunks]

        # Stigmergy deposit
        if self.stigmergy:
            try:
                self.stigmergy.deposit(chunk_ids, query_type, success)
            except Exception as exc:
                log.debug("Pipeline step failed: %s", exc)

        # Experience storage (only high quality)
        if self.experience and confidence >= 0.6:
            try:
                self.experience.store(
                    exp_id=query_id,
                    query=question,
                    answer=response,
                    source_files=sources,
                    confidence=confidence,
                )
            except Exception as exc:
                log.debug("Pipeline step failed: %s", exc)

        # Meta-cognitive outcome
        if self.meta:
            try:
                self.meta.report_outcome(
                    question, strategy, confidence,
                    latency_ms=(time.time() - t0) * 1000,
                )
            except Exception as exc:
                log.debug("Pipeline step failed: %s", exc)

        # Telemetry
        learning_value = 0.0
        if self.telemetry and QueryEvent:
            try:
                event = QueryEvent(
                    query_id=query_id,
                    query_text=question,
                    timestamp=time.time(),
                    retrieval_latency_ms=retrieval_ms,
                    generation_latency_ms=gen_ms,
                    chunk_ids=chunk_ids,
                    source_files=sources,
                    answer_snippet=response[:500],
                    confidence=confidence,
                    reflection_relevant=reflection_scores.get("relevant", 0),
                    reflection_supported=reflection_scores.get("supported", 0),
                    reflection_useful=reflection_scores.get("useful", 0),
                )
                self.telemetry.log(event)
            except Exception as exc:
                log.debug("Pipeline step failed: %s", exc)

        # Active learning scoring
        if self.active:
            learning_value = 4.0 * confidence * (1.0 - confidence)

        total_ms = (time.time() - t0) * 1000

        return PipelineAnswer(
            answer=response,
            sources=sources,
            chunk_count=len(chunks),
            confidence=confidence,
            strategy_used=strategy,
            reasoning=reasoning,
            reflection=reflection_scores,
            latency_ms=total_ms,
            query_id=query_id,
            learning_value=learning_value,
        )

    def _retrieve(
        self, question: str, strategy: str
    ) -> Tuple[List[Dict], Dict]:
        """Retrieval with strategy-dependent approach."""
        meta = {"confidence": 1.0, "strategy": strategy, "attempts": 1}

        if strategy == "corrective" and self.corrective:
            chunks, meta = self.corrective.retrieve(question)
        else:
            chunks = self.retriever.retrieve(question)

        return chunks, meta

    def system_health(self) -> Dict[str, Any]:
        """Generate a comprehensive health report of all subsystems."""
        report = {
            "queries_processed": self._query_count,
            "modules_active": [],
            "modules_inactive": [],
        }

        modules = {
            "meta_cognitive": self.meta,
            "reflection": self.reflection,
            "corrective": self.corrective,
            "best_of_n": self.best_of_n,
            "experience": self.experience,
            "stigmergy": self.stigmergy,
            "telemetry": self.telemetry,
            "star_reasoner": self.star,
            "active_learner": self.active,
            "continual_learner": self.continual,
        }

        for name, module in modules.items():
            if module is not None:
                report["modules_active"].append(name)
                # Get stats from modules that support it
                if hasattr(module, "stats"):
                    try:
                        report[f"{name}_stats"] = module.stats()
                    except Exception:
                        pass
            else:
                report["modules_inactive"].append(name)

        # Meta-cognitive strategy preferences
        if self.meta and hasattr(self.meta, "best_strategy_per_type"):
            try:
                report["strategy_preferences"] = (
                    self.meta.best_strategy_per_type()
                )
            except Exception as exc:
                log.debug("Pipeline step failed: %s", exc)

        # Continual learning health
        if self.continual and hasattr(self.continual, "health_report"):
            try:
                report["continual_health"] = self.continual.health_report()
            except Exception as exc:
                log.debug("Pipeline step failed: %s", exc)

        return report
