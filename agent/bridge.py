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
try: from core.config import load_config
except ImportError: load_config = None  # type: ignore[assignment,misc]
try: from agent.llm_backend import create_backend
except ImportError: create_backend = None  # type: ignore[assignment,misc]
try: from agent.memory import AgentMemory
except ImportError: AgentMemory = None  # type: ignore[assignment,misc]


class AgentBridge:
    """Connects the agent to JCoder's self-learning pipeline."""

    def __init__(self, agent: Agent, telemetry: Optional[Any] = None,
                 experience_store: Optional[Any] = None,
                 active_learner: Optional[Any] = None,
                 meta_cognitive: Optional[Any] = None,
                 memory: Optional[Any] = None):
        self.agent = agent
        self.telemetry = telemetry
        self.experience = experience_store
        self.active = active_learner
        self.meta = meta_cognitive
        self.memory = memory
        active = [n for n, m in [("telemetry", telemetry), ("experience", experience_store),
                  ("active", active_learner), ("meta_cog", meta_cognitive),
                  ("memory", memory)] if m]
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
        # Experience replay (successful only)
        if self.experience and result.success:
            try:
                self.experience.store(
                    exp_id=tid, query=task, answer=result.summary,
                    source_files=self._extract_source_files(result), confidence=1.0)
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

    def select_strategy(self, task: str) -> Dict[str, Any]:
        """Ask meta_cognitive which approach to use for this task."""
        defaults: Dict[str, Any] = {"temperature": 0.1, "max_iterations": 50,
                                     "strategy": "standard"}
        if not self.meta or not classify_query:
            return defaults
        try:
            name, sig = self.meta.select_strategy(task)
            return {"temperature": 0.1 if sig.complexity < 0.5 else 0.3,
                    "max_iterations": 30 if sig.query_type == "lookup" else 50,
                    "strategy": name, "query_type": sig.query_type,
                    "complexity": sig.complexity, "has_code": sig.has_code}
        except Exception as exc:
            log.warning("Strategy selection failed: %s", exc)
            return defaults

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
                except Exception: pass
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
    bridge = AgentBridge(agent=agent, telemetry=telemetry, experience_store=experience,
                         active_learner=active, meta_cognitive=meta, memory=memory)

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


def _summarize_args(args: Dict[str, Any], max_len: int = 80) -> str:
    parts = []
    for k, v in args.items():
        s = str(v)
        parts.append(f"{k}={s[:37]}..." if len(s) > 40 else f"{k}={s}")
    joined = ", ".join(parts)
    return joined[:max_len - 3] + "..." if len(joined) > max_len else joined
