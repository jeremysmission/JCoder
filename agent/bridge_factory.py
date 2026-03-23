"""
Agent Bridge Factory: creates a fully-wired Agent + AgentBridge.

Split from bridge.py to keep each module under 500 LOC.
The canonical entry point is create_wired_agent().
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from agent.bridge import AgentBridge, _BackendRuntimeAdapter, _FederatedRetrieverAdapter

log = logging.getLogger(__name__)

# Self-learning imports (all optional) -- needed by factory helpers
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
    from agent.core import Agent

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

    # Extract SmartOrchestrator and ModelCascade from pipeline (if created)
    smart_orch = getattr(pipeline, "smart_orchestrator", None) if pipeline else None
    cascade = getattr(pipeline, "cascade", None) if pipeline else None

    bridge = AgentBridge(agent=agent, telemetry=telemetry, experience_store=experience,
                         active_learner=active, meta_cognitive=meta,
                         memory=memory, procedural_memory=procedural_memory,
                         qd_archive=qd_archive, continual_learner=continual,
                         pipeline=pipeline, smart_orchestrator=smart_orch,
                         cascade=cascade)

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
