"""
Agent <-> Self-Learning Bridge: connects the autonomous agent to JCoder's
self-learning pipeline. Completed tasks feed back into experience replay,
telemetry, meta-cognitive strategy selection, and active learning.
Every self-learning module is optional -- degrades gracefully.
"""
from __future__ import annotations
import hashlib, logging, time
from typing import Any, Callable, Dict, List, Optional, Tuple
from agent.core import Agent, AgentResult

log = logging.getLogger(__name__)

# Self-learning imports (all optional)
try: from core.telemetry import TelemetryStore, QueryEvent
except ImportError: TelemetryStore = QueryEvent = None  # type: ignore[assignment,misc]
try: from core.experience_replay import ExperienceStore
except ImportError: ExperienceStore = None  # type: ignore[assignment,misc]
try: from core.active_learner import ActiveLearner
except ImportError: ActiveLearner = None  # type: ignore[assignment,misc]
try: from core.meta_cognitive import MetaCognitiveController, classify_query
except ImportError: MetaCognitiveController = classify_query = None  # type: ignore[assignment,misc]
try: from core.smart_orchestrator import SmartOrchestrator
except ImportError: SmartOrchestrator = None  # type: ignore[assignment,misc]
try: from core.corrective_retrieval import CorrectiveRetriever
except ImportError: CorrectiveRetriever = None  # type: ignore[assignment,misc]
try: from core.knowledge_graph import CodeKnowledgeGraph
except ImportError: CodeKnowledgeGraph = None  # type: ignore[assignment,misc]
try: from core.prompt_evolver import PromptEvolver
except ImportError: PromptEvolver = None  # type: ignore[assignment,misc]
try: from core.adversarial_self_play import AdversarialSelfPlay
except ImportError: AdversarialSelfPlay = None  # type: ignore[assignment,misc]
try: from core.rapid_digest import RapidDigester
except ImportError: RapidDigester = None  # type: ignore[assignment,misc]
try: from core.stigmergy import StigmergicBooster
except ImportError: StigmergicBooster = None  # type: ignore[assignment,misc]
try: from core.cascade import ModelCascade
except ImportError: ModelCascade = None  # type: ignore[assignment,misc]
try: from core.config import load_config
except ImportError: load_config = None  # type: ignore[assignment,misc]
try: from agent.llm_backend import create_backend
except ImportError: create_backend = None  # type: ignore[assignment,misc]
try: from agent.memory import AgentMemory
except ImportError: AgentMemory = None  # type: ignore[assignment,misc]
try: from core.procedural_memory import ProceduralMemory, ProceduralExperience
except ImportError: ProceduralMemory = ProceduralExperience = None  # type: ignore[assignment,misc]
try: from core.quality_diversity import QualityDiversityArchive, QDSolution, compute_behavior
except ImportError: QualityDiversityArchive = QDSolution = compute_behavior = None  # type: ignore[assignment,misc]
try: from core.continual_learner import ContinualLearner
except ImportError: ContinualLearner = None  # type: ignore[assignment,misc]
try: from core.self_learning_pipeline import SelfLearningPipeline
except ImportError: SelfLearningPipeline = None  # type: ignore[assignment,misc]
try: from core.star_reasoner import STaRReasoner
except ImportError: STaRReasoner = None  # type: ignore[assignment,misc]
try: from core.best_of_n import BestOfNGenerator
except ImportError: BestOfNGenerator = None  # type: ignore[assignment,misc]
try: from core.reflection import ReflectionEngine
except ImportError: ReflectionEngine = None  # type: ignore[assignment,misc]

from agent.bridge_strategies import BridgeStrategyMixin


class AgentBridge(BridgeStrategyMixin):
    """Connects the agent to JCoder's self-learning pipeline."""

    def __init__(self, agent: Agent, telemetry: Optional[Any] = None,
                 experience_store: Optional[Any] = None,
                 active_learner: Optional[Any] = None,
                 meta_cognitive: Optional[Any] = None,
                 memory: Optional[Any] = None,
                 procedural_memory: Optional[ProceduralMemory] = None,
                 qd_archive: Optional[Any] = None,
                 continual_learner: Optional[Any] = None,
                 pipeline: Optional[Any] = None,
                 smart_orchestrator: Optional[Any] = None,
                 cascade: Optional[Any] = None):
        self.agent = agent
        self.telemetry = telemetry
        self.experience = experience_store
        self.active = active_learner
        self.meta = meta_cognitive
        self.memory = memory
        self.procedural_memory = procedural_memory
        self.qd_archive = qd_archive
        self.continual_learner = continual_learner
        self.pipeline = pipeline
        self.smart_orchestrator = smart_orchestrator
        self.cascade = cascade
        active = [n for n, m in [("telemetry", telemetry), ("experience", experience_store),
                  ("active", active_learner), ("meta_cog", meta_cognitive),
                  ("memory", memory), ("procedural_memory", procedural_memory),
                  ("qd_archive", qd_archive), ("continual_learner", continual_learner),
                  ("pipeline", pipeline), ("smart_orchestrator", smart_orchestrator),
                  ("cascade", cascade)] if m]
        log.info("AgentBridge: modules active: %s", active or "(none)")

    def on_task_complete(self, task: str, result: AgentResult) -> None:
        """Called after agent finishes. Feeds data back to self-learning."""
        tid = hashlib.sha256(f"{task}:{time.time()}".encode()).hexdigest()[:12]
        # Telemetry
        if self.telemetry and QueryEvent:
            try:
                self.telemetry.log(QueryEvent(
                    query_id=tid, query_text=task[:500], timestamp=time.time(),
                    generation_latency_ms=result.total_elapsed_s * 1000,
                    chunk_ids=[s.tool_name for s in result.steps],
                    answer_snippet=result.summary[:500],
                    confidence=1.0 if result.success else 0.0))
            except Exception as exc: log.warning("Telemetry: %s", exc)
        # Experience replay (successes and failures)
        if self.experience:
            try:
                confidence = 1.0 if result.success else 0.0
                if result.success and result.iterations > 5:
                    confidence = max(0.3, 1.0 - (result.iterations / 50.0))
                prefix = "[FAIL] " if not result.success else ""
                self.experience.store(
                    exp_id=tid, query=task,
                    answer=f"{prefix}{result.summary}",
                    source_files=self._extract_source_files(result),
                    confidence=confidence)
            except Exception as exc: log.warning("Experience: %s", exc)
        # Meta-cognitive outcome
        if self.meta and hasattr(self.meta, "report_outcome"):
            try:
                self.meta.report_outcome(task, "agent",
                    1.0 if result.success else 0.0,
                    latency_ms=result.total_elapsed_s * 1000)
            except Exception as exc: log.warning("Meta-cog: %s", exc)
        # Active learner (borderline tasks get frontier scoring)
        if self.active and hasattr(self.active, "score_queries"):
            try:
                conf = 1.0 if result.success else 0.0
                if result.success and result.iterations > 5:
                    conf = max(0.3, 1.0 - (result.iterations / 50.0))
                self.active.score_queries([task], [conf])
            except Exception as exc: log.warning("Active learner: %s", exc)
        # Agent memory (store successful task results for future recall)
        if self.memory and hasattr(self.memory, "ingest_task_result"):
            try:
                self.memory.ingest_task_result(task=task, result=result)
            except Exception as exc: log.warning("Agent memory: %s", exc)
        if self.procedural_memory and ProceduralExperience is not None:
            try:
                summary_hash = hashlib.sha256(
                    (task + result.summary).encode()).hexdigest()
                self.procedural_memory.store(
                    ProceduralExperience(
                        state_hash=summary_hash,
                        action=task,
                        outcome=result.summary,
                        success=result.success,
                        metadata={"iterations": result.iterations,
                                  "tokens": result.total_input_tokens + result.total_output_tokens},
                    )
                )
            except Exception as exc:
                log.warning("Procedural memory: %s", exc)
        # Quality-diversity archive
        if self.qd_archive and QDSolution is not None and compute_behavior is not None:
            try:
                sig = classify_query(task) if classify_query else None
                if sig:
                    behavior = compute_behavior(
                        query_complexity=sig.complexity,
                        answer_type=sig.query_type,
                        retrieval_confidence=1.0 if result.success else 0.0,
                    )
                    fitness = 1.0 if result.success else 0.0
                    if result.success and result.iterations > 5:
                        fitness = max(0.3, 1.0 - (result.iterations / 50.0))
                    self.qd_archive.add(QDSolution(
                        config={"strategy": "agent",
                                "iterations": result.iterations},
                        fitness=fitness, behavior=behavior))
            except Exception as exc:
                log.warning("QD archive: %s", exc)

    def suggest_next_study(self) -> Optional[str]:
        """Ask active_learner what the agent should study next.
        Filters out topics the agent already has strong memory of."""
        if not self.active or not hasattr(self.active, "top_learning_opportunities"):
            return None
        try:
            opps = self.active.top_learning_opportunities(limit=5)
            if not opps:
                return None
            for t in opps:
                query = t.get("query", "")
                # Skip topics the memory already covers well
                if self.memory and query:
                    try:
                        hits = self.memory.search(query, top_k=1)
                        if hits and hits[0].get("score", 0) > 0.85:
                            log.debug("Skipping known topic: %s", query[:80])
                            continue
                    except Exception:
                        log.debug("Memory search unavailable for study suggestion", exc_info=True)
                return (f"Study topic (value={t.get('learning_value', 0):.2f}, "
                        f"uncertainty={t.get('uncertainty', 0):.2f}): {query}")
            return None  # all suggestions already known
        except Exception as exc:
            log.warning("Study suggestion failed: %s", exc)
            return None

    def get_memory_stats(self) -> Dict[str, Any]:
        """Return statistics from the agent's memory, or empty dict if unavailable."""
        if not self.memory or not hasattr(self.memory, "stats"):
            return {}
        try:
            return self.memory.stats()
        except Exception as exc:
            log.warning("Memory stats failed: %s", exc)
            return {}

    def check_regression(self, config: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Run continual learner regression check. Returns check result or None."""
        if not self.continual_learner or not hasattr(self.continual_learner, "check_regression"):
            return None
        try:
            check = self.continual_learner.check_regression(config=config)
            if not check.passed:
                log.warning("Regression detected in: %s", check.failed_capabilities)
            return {
                "passed": check.passed,
                "failed": check.failed_capabilities,
                "scores": check.scores,
                "baselines": check.baselines,
            }
        except Exception as exc:
            log.warning("Regression check failed: %s", exc)
            return None

    def pipeline_health(self) -> Dict[str, Any]:
        """Get self-learning pipeline health report."""
        if not self.pipeline or not hasattr(self.pipeline, "system_health"):
            return {}
        try:
            return self.pipeline.system_health()
        except Exception as exc:
            log.warning("Pipeline health failed: %s", exc)
            return {}

    def wrap_rag_callback(self, base_cb: Callable[[str], str]) -> Callable[[str], str]:
        """Wrap a RAG callback with telemetry logging."""
        def wrapped(query: str) -> str:
            t0 = time.time()
            answer = base_cb(query)
            if self.telemetry and QueryEvent:
                try:
                    qid = hashlib.sha256(f"rag:{query}:{t0}".encode()).hexdigest()[:12]
                    self.telemetry.log(QueryEvent(
                        query_id=qid, query_text=f"[rag_tool] {query[:450]}",
                        timestamp=t0, retrieval_latency_ms=(time.time() - t0) * 1000,
                        answer_snippet=(answer or "")[:500], confidence=0.5))
                except Exception as exc: log.debug("RAG telemetry: %s", exc)
            return answer
        return wrapped

    def get_agent_trajectory(self, result: AgentResult) -> Dict[str, Any]:
        """Convert AgentResult into a trajectory dict for experience replay."""
        steps = [{"action": f"{s.tool_name}({_summarize_args(s.tool_args)})",
                  "observation": s.tool_result[:300], "success": s.tool_success}
                 for s in result.steps]
        return {"task": "",  # caller fills
                "steps": steps,
                "outcome": "success" if result.success else "failure",
                "summary": result.summary[:500],
                "tokens": {"input": result.total_input_tokens,
                           "output": result.total_output_tokens,
                           "total": result.total_input_tokens + result.total_output_tokens},
                "iterations": result.iterations,
                "elapsed_s": round(result.total_elapsed_s, 2)}

    @staticmethod
    def _extract_source_files(result: AgentResult) -> List[str]:
        """Pull file paths touched by file tools."""
        paths = set()
        for s in result.steps:
            if s.tool_name in ("read_file", "write_file", "edit_file") and s.tool_success:
                p = s.tool_args.get("path", "")
                if p: paths.add(p)
        return sorted(paths)


# ---------------------------------------------------------------------------
# Adapters (Finding 1 fix: bridge stack shape mismatch)
# ---------------------------------------------------------------------------

class _BackendRuntimeAdapter:
    """Adapts an agent LLM backend to the Runtime.generate() interface.

    SelfLearningPipeline calls runtime.generate(question, context_chunks).
    The agent backend exposes backend.chat(messages, tools, ...).
    """

    def __init__(self, backend: Any):
        self._backend = backend

    def generate(
        self,
        question: str,
        context_chunks: List[str],
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        ctx = "\n\n".join(context_chunks[:10]) if context_chunks else ""
        user_content = f"{question}\n\nContext:\n{ctx}" if ctx else question
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_content})
        kwargs: Dict[str, Any] = {"tools": []}
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        resp = self._backend.chat(messages, **kwargs)
        return resp.content

    def close(self) -> None:
        pass


class _FederatedRetrieverAdapter:
    """Adapts FederatedSearch to the RetrievalEngine.retrieve() interface.

    SelfLearningPipeline calls retriever.retrieve(query) -> List[Dict].
    FederatedSearch exposes federated.search(query, top_k) -> results.
    """

    def __init__(self, federated: Any, embedder: Any = None, top_k: int = 10):
        self._federated = federated
        self._embedder = embedder
        self._top_k = top_k

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        try:
            query_vec = None
            if self._embedder and hasattr(self._embedder, "embed_query"):
                try:
                    query_vec = self._embedder.embed_query(query)
                except Exception:
                    log.debug("Embedding query failed for federated retrieval", exc_info=True)
            results = self._federated.search(
                query, top_k=self._top_k, query_vec=query_vec,
            )
            return [
                {
                    "id": getattr(r, "chunk_id", "") or f"{getattr(r, 'index_name', '')}:{i}",
                    "content": getattr(r, "content", ""),
                    "source_path": getattr(r, "source", ""),
                    "score": getattr(r, "score", 0.0),
                }
                for i, r in enumerate(results)
            ]
        except Exception as exc:
            log.debug("Federated retriever: %s", exc)
            return []

# ---------------------------------------------------------------------------
# Re-export factory from bridge_factory (keeps existing imports working)
# ---------------------------------------------------------------------------
from agent.bridge_factory import create_wired_agent  # noqa: F401
from agent.bridge_factory import _try_init_continual  # noqa: F401


def _summarize_args(args: Dict[str, Any], max_len: int = 80) -> str:
    parts = []
    for k, v in args.items():
        s = str(v)
        parts.append(f"{k}={s[:37]}..." if len(s) > 40 else f"{k}={s}")
    joined = ", ".join(parts)
    return joined[:max_len - 3] + "..." if len(joined) > max_len else joined
