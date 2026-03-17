"""
Agent <-> Self-Learning Bridge: connects the autonomous agent to JCoder's
self-learning pipeline. Completed tasks feed back into experience replay,
telemetry, meta-cognitive strategy selection, and active learning.
Every self-learning module is optional -- degrades gracefully.
"""
from __future__ import annotations
import hashlib, logging, time
from typing import Any, Callable, Dict, List, Optional, Tuple
from agent.core import Agent, AgentResult, AgentStep
from agent.tools import ToolRegistry

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


class AgentBridge:
    """Connects the agent to JCoder's self-learning pipeline."""

    def __init__(self, agent: Agent, telemetry: Optional[Any] = None,
                 experience_store: Optional[Any] = None,
                 active_learner: Optional[Any] = None,
                 meta_cognitive: Optional[Any] = None,
                 memory: Optional[Any] = None,
                 procedural_memory: Optional[ProceduralMemory] = None,
                 qd_archive: Optional[Any] = None,
                 continual_learner: Optional[Any] = None,
                 pipeline: Optional[Any] = None):
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
        active = [n for n, m in [("telemetry", telemetry), ("experience", experience_store),
                  ("active", active_learner), ("meta_cog", meta_cognitive),
                  ("memory", memory), ("procedural_memory", procedural_memory),
                  ("qd_archive", qd_archive), ("continual_learner", continual_learner),
                  ("pipeline", pipeline)] if m]
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
                        pass  # memory unavailable, don't block suggestion
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

    # --- Sprint 9: Reasoning module methods ---

    def enhance_with_reasoning(
        self, task: str, result: AgentResult,
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

    def select_strategy(self, task: str) -> Dict[str, Any]:
        """Ask meta_cognitive which approach to use for this task.

        Also consults QD archive for niche-specific hints.
        """
        defaults: Dict[str, Any] = {"temperature": 0.1, "max_iterations": 50,
                                     "strategy": "standard"}
        if not self.meta or not classify_query:
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
        if not self.qd_archive or compute_behavior is None or classify_query is None:
            return params
        try:
            sig = classify_query(task)
            behavior = compute_behavior(
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
                    pass
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
# Factory
# ---------------------------------------------------------------------------

def create_wired_agent(config: Optional[Any] = None, backend: Optional[Any] = None,
                       working_dir: str = ".") -> Tuple[Agent, AgentBridge]:
    """Create an Agent fully wired to self-learning. Returns (agent, bridge).

    Delegates to config_loader.build_agent_from_config() for the canonical
    stack (federated search, RAG callback, memory, sessions), then layers
    on bridge-specific self-learning modules (telemetry, experience replay,
    meta-cognitive, active learner).

    Parameters
    ----------
    config : optional
        Pre-built AgentConfig. If None, loads from YAML.
    backend : optional
        Pre-built LLM backend. If provided, overrides config-driven backend.
    working_dir : str
        Working directory for file tools.
    """
    from agent.config_loader import load_agent_config, build_agent_from_config

    # Use caller's config or load from YAML
    agent_config = config
    if agent_config is None:
        try:
            agent_config = load_agent_config()
        except Exception as exc:
            log.warning("Agent config load failed: %s", exc)
    if agent_config is not None and working_dir != ".":
        agent_config.working_dir = working_dir

    stack = build_agent_from_config(agent_config)

    # Honor caller-provided backend by replacing the one from config
    if backend is not None:
        stack["backend"] = backend
        # Rebuild agent with caller's backend
        from agent.core import Agent
        agent = Agent(
            backend=backend,
            tools=stack["tools"],
            system_prompt=getattr(stack["agent"], "_system_prompt", None),
            max_iterations=agent_config.max_iterations if agent_config else 50,
            max_tokens_budget=agent_config.max_tokens_budget if agent_config else 500_000,
            session_store=stack.get("session_store"),
            logger=stack.get("logger"),
        )
    else:
        agent = stack["agent"]

    memory = stack.get("memory")

    # Layer on bridge-specific self-learning modules
    telemetry = _try_init(TelemetryStore, "_telemetry/agent_events.db")
    experience = _try_init(ExperienceStore, "_experience/agent_replay.db")
    meta = _try_init(MetaCognitiveController, "_meta_cog/agent_controller.db")
    active = _try_init_active(backend=stack.get("backend"))
    procedural_memory = _try_init(ProceduralMemory, "_procedural_memory/memory.db")
    qd_archive = _try_init(QualityDiversityArchive, "_qd_archive/agent_archive.db")

    # Sprint 8: Self-learning pipeline + continual learner
    sl_config = _get_self_learning_config()
    continual = _try_init_continual(sl_config) if sl_config.get("continual_learner_enabled") else None
    pipeline = _try_init_pipeline(
        sl_config, stack, telemetry, experience, meta, active, continual,
    ) if sl_config.get("pipeline_enabled") else None

    bridge = AgentBridge(agent=agent, telemetry=telemetry, experience_store=experience,
                         active_learner=active, meta_cognitive=meta,
                         memory=memory, procedural_memory=procedural_memory,
                         qd_archive=qd_archive, continual_learner=continual,
                         pipeline=pipeline)

    # Wrap RAG callback with telemetry if both are available
    rag_callback = getattr(stack["tools"], "_rag_callback", None)
    if rag_callback and telemetry:
        stack["tools"]._rag_callback = bridge.wrap_rag_callback(rag_callback)

    # Register committee strategies for active learner
    if active and stack.get("backend"):
        llm = stack["backend"]
        # Strategy 1: direct LLM (low temp)
        active.register_strategy(
            lambda q, _b=llm: _b.chat(
                [{"role": "user", "content": q}],
                tools=[], temperature=0.1, max_tokens=512).content)
        # Strategy 2: direct LLM (high temp)
        active.register_strategy(
            lambda q, _b=llm: _b.chat(
                [{"role": "user", "content": q}],
                tools=[], temperature=0.7, max_tokens=512).content)
        # Strategy 3: RAG-augmented (if available)
        if rag_callback:
            active.register_strategy(
                lambda q, _cb=rag_callback: _cb(q))

    return agent, bridge


def _try_init(cls: Optional[type], db_path: str) -> Optional[Any]:
    if cls is None: return None
    try: return cls(db_path=db_path)
    except Exception as exc:
        log.warning("%s init failed: %s", cls.__name__, exc); return None


def _try_init_active(backend: Optional[Any] = None) -> Optional[Any]:
    if ActiveLearner is None: return None
    def _generate(query: str, temperature: float) -> str:
        if backend is None: return ""
        try:
            resp = backend.chat(
                [{"role": "user", "content": query}],
                tools=[], temperature=temperature, max_tokens=512)
            return resp.content
        except Exception: return ""
    try: return ActiveLearner(generate_fn=_generate,
                              db_path="_active_learn/agent_learner.db")
    except Exception as exc:
        log.warning("ActiveLearner init failed: %s", exc); return None


def _try_init_memory() -> Optional[Any]:
    if AgentMemory is None: return None
    try: return AgentMemory(
        index_dir="data/indexes", index_name="agent_memory",
        dimension=768, knowledge_dir="data/agent_knowledge")
    except Exception as exc:
        log.warning("AgentMemory init failed: %s", exc); return None


def _get_self_learning_config() -> Dict[str, Any]:
    """Load self_learning + reasoning config from agent.yaml, with safe defaults."""
    defaults = {
        "pipeline_enabled": False,
        "continual_learner_enabled": False,
        "regression_margin": 0.05,
        "confidence_gate": 0.2,
        "star_enabled": False,
        "best_of_n_enabled": False,
        "reflection_enabled": False,
        "best_of_n_candidates": 3,
        "corrective_retrieval_enabled": False,
        "smart_orchestrator_enabled": False,
        "corrective_confidence_threshold": 0.5,
        "knowledge_graph_enabled": False,
        "prompt_evolver_enabled": False,
        "adversarial_self_play_enabled": False,
        "rapid_digest_enabled": False,
        "stigmergy_enabled": False,
        "cascade_enabled": False,
    }
    try:
        import yaml
        from pathlib import Path
        # Resolve relative to repo root, not cwd
        _repo_root = Path(__file__).resolve().parent.parent
        cfg_path = _repo_root / "config" / "agent.yaml"
        if cfg_path.exists():
            with open(cfg_path) as f:
                raw = yaml.safe_load(f) or {}
            sl = raw.get("agent", {}).get("self_learning", {})
            defaults.update(sl)
            reasoning = raw.get("agent", {}).get("reasoning", {})
            defaults.update(reasoning)
            corrective = raw.get("agent", {}).get("corrective", {})
            defaults.update(corrective)
            evolution = raw.get("agent", {}).get("evolution", {})
            defaults.update(evolution)
            cascade_cfg = raw.get("agent", {}).get("cascade", {})
            defaults.update(cascade_cfg)
    except Exception as exc:
        log.debug("Self-learning config load: %s", exc)
    return defaults


def _try_init_continual(sl_config: Dict[str, Any]) -> Optional[Any]:
    """Initialize ContinualLearner if available."""
    if ContinualLearner is None:
        return None
    try:
        def _eval_fn(capability: str, test_queries: List[str]) -> float:
            log.warning(
                "[PLACEHOLDER] ContinualLearner eval_fn returning 1.0 "
                "for capability=%s -- regression checks are non-functional "
                "until a real evaluator is wired",
                capability,
            )
            return 1.0
        return ContinualLearner(
            eval_fn=_eval_fn,
            db_path="_continual/agent_learner.db",
            regression_margin=sl_config.get("regression_margin", 0.05),
        )
    except Exception as exc:
        log.warning("ContinualLearner init failed: %s", exc)
        return None


def _try_init_pipeline(
    sl_config: Dict[str, Any],
    stack: Dict[str, Any],
    telemetry: Optional[Any],
    experience: Optional[Any],
    meta: Optional[Any],
    active: Optional[Any],
    continual: Optional[Any],
) -> Optional[Any]:
    """Initialize SelfLearningPipeline from stack components.

    config_loader provides 'backend' and 'federated' but not 'retriever'
    or 'runtime' directly. We build lightweight adapters here.
    """
    if SelfLearningPipeline is None:
        return None

    # Try explicit keys first, then build adapters from stack components
    retriever = stack.get("retriever")
    runtime = stack.get("runtime")

    if runtime is None and stack.get("backend") is not None:
        runtime = _BackendRuntimeAdapter(stack["backend"])
        log.info("Pipeline: built runtime adapter from backend")

    if retriever is None and stack.get("federated") is not None:
        retriever = _FederatedRetrieverAdapter(
            stack["federated"], stack.get("embedder"),
        )
        log.info("Pipeline: built retriever adapter from federated search")

    if retriever is None or runtime is None:
        log.info("Pipeline skipped: cannot build retriever or runtime from stack")
        return None
    # Sprint 9: reasoning modules (config-gated)
    star = None
    if sl_config.get("star_enabled") and STaRReasoner is not None:
        try:
            star = STaRReasoner(runtime=runtime, db_path="_star/agent_reasoner.db")
        except Exception as exc:
            log.warning("STaRReasoner init failed: %s", exc)

    reflection = None
    if sl_config.get("reflection_enabled") and ReflectionEngine is not None:
        try:
            reflection = ReflectionEngine(runtime=runtime)
        except Exception as exc:
            log.warning("ReflectionEngine init failed: %s", exc)

    best_of_n = None
    if sl_config.get("best_of_n_enabled") and BestOfNGenerator is not None:
        try:
            n = sl_config.get("best_of_n_candidates", 3)
            best_of_n = BestOfNGenerator(runtime=runtime, n=n)
        except Exception as exc:
            log.warning("BestOfNGenerator init failed: %s", exc)

    # Sprint 10: Corrective retrieval (CRAG)
    corrective = None
    if sl_config.get("corrective_retrieval_enabled") and CorrectiveRetriever is not None:
        try:
            corrective = CorrectiveRetriever(
                retriever=retriever,
                reflection=reflection,
                confidence_threshold=sl_config.get("corrective_confidence_threshold", 0.5),
            )
        except Exception as exc:
            log.warning("CorrectiveRetriever init failed: %s", exc)

    # Sprint 10: Knowledge graph
    knowledge_graph = None
    if sl_config.get("knowledge_graph_enabled") and CodeKnowledgeGraph is not None:
        try:
            knowledge_graph = CodeKnowledgeGraph(db_path="_kg/agent_knowledge_graph.db")
        except Exception as exc:
            log.warning("CodeKnowledgeGraph init failed: %s", exc)

    # Sprint 11: Prompt evolver
    prompt_evolver = None
    if sl_config.get("prompt_evolver_enabled") and PromptEvolver is not None:
        try:
            def _eval_fn(prompt_text, query):
                log.warning(
                    "[PLACEHOLDER] PromptEvolver eval_fn returning 0.5 "
                    "-- fitness scores are synthetic until a real "
                    "evaluator is wired",
                )
                return 0.5
            prompt_evolver = PromptEvolver(
                runtime=runtime, eval_fn=_eval_fn,
                db_path="_prompt_evo/agent_lineage.db",
            )
        except Exception as exc:
            log.warning("PromptEvolver init failed: %s", exc)

    # Sprint 11: Adversarial self-play
    adversarial = None
    if sl_config.get("adversarial_self_play_enabled") and AdversarialSelfPlay is not None:
        try:
            def _answer_fn(question):
                return runtime.generate(question, [])
            adversarial = AdversarialSelfPlay(
                runtime=runtime, answer_fn=_answer_fn,
                db_path="_self_play/agent_games.db",
            )
        except Exception as exc:
            log.warning("AdversarialSelfPlay init failed: %s", exc)

    # Sprint 11: Rapid digester
    rapid_digest = None
    if sl_config.get("rapid_digest_enabled") and RapidDigester is not None:
        try:
            rapid_digest = RapidDigester(
                runtime=runtime, db_path="_digests/agent_digest.db",
            )
        except Exception as exc:
            log.warning("RapidDigester init failed: %s", exc)

    # Sprint 11: Stigmergy booster
    stigmergy = None
    if sl_config.get("stigmergy_enabled") and StigmergicBooster is not None:
        try:
            stigmergy = StigmergicBooster(db_path="_stigmergy/agent_pheromones.db")
        except Exception as exc:
            log.warning("StigmergicBooster init failed: %s", exc)

    # Sprint 10: Smart orchestrator (Self-RAG + CRAG + confidence gating)
    smart_orch = None
    if sl_config.get("smart_orchestrator_enabled") and SmartOrchestrator is not None:
        try:
            smart_orch = SmartOrchestrator(
                retriever=retriever,
                runtime=runtime,
                telemetry=telemetry,
                reflection=reflection,
                corrective=corrective,
                confidence_gate=sl_config.get("confidence_gate", 0.2),
            )
            log.info("SmartOrchestrator: initialized")
        except Exception as exc:
            log.warning("SmartOrchestrator init failed: %s", exc)

    # Sprint 12: Model cascade router (complexity-based routing)
    cascade = None
    if sl_config.get("cascade_enabled") and ModelCascade is not None:
        try:
            from core.cascade import CascadeLevel
            from core.config import ModelConfig as _MC
            cascade_levels_raw = sl_config.get("cascade_levels", [])
            if cascade_levels_raw:
                levels = [
                    CascadeLevel(
                        name=lv.get("name", f"level_{i}"),
                        model_config=_MC(
                            name=lv.get("model_name", "phi4-mini"),
                            endpoint=lv.get("endpoint", "http://localhost:11434/v1"),
                        ),
                        max_complexity=lv.get("max_complexity", (i + 1) * 0.3),
                    )
                    for i, lv in enumerate(cascade_levels_raw)
                ]
            else:
                levels = [
                    CascadeLevel(
                        name="default",
                        model_config=_MC(
                            name=sl_config.get("cascade_model", "phi4-mini"),
                            endpoint=sl_config.get("cascade_endpoint",
                                                   "http://localhost:11434/v1"),
                        ),
                        max_complexity=1.0,
                    )
                ]
            cascade = ModelCascade(
                levels=levels,
                confidence_threshold=sl_config.get("cascade_confidence", 0.4),
            )
            log.info("ModelCascade: initialized with %d levels", len(levels))
        except Exception as exc:
            log.warning("ModelCascade init failed: %s", exc)

    try:
        pipeline = SelfLearningPipeline(
            retriever=retriever,
            runtime=runtime,
            meta_controller=meta,
            reflection=reflection,
            best_of_n=best_of_n,
            experience=experience,
            telemetry=telemetry,
            star=star,
            active_learner=active,
            continual=continual,
            confidence_gate=sl_config.get("confidence_gate", 0.2),
        )
        # Attach Sprint 10-12 modules as attributes for bridge access
        pipeline.corrective = corrective
        pipeline.knowledge_graph = knowledge_graph
        pipeline.prompt_evolver = prompt_evolver
        pipeline.adversarial = adversarial
        pipeline.rapid_digest = rapid_digest
        pipeline.stigmergy_booster = stigmergy
        pipeline.smart_orchestrator = smart_orch
        pipeline.cascade = cascade
        return pipeline
    except Exception as exc:
        log.warning("SelfLearningPipeline init failed: %s", exc)
        return None


def _summarize_args(args: Dict[str, Any], max_len: int = 80) -> str:
    parts = []
    for k, v in args.items():
        s = str(v)
        parts.append(f"{k}={s[:37]}..." if len(s) > 40 else f"{k}={s}")
    joined = ", ".join(parts)
    return joined[:max_len - 3] + "..." if len(joined) > max_len else joined
