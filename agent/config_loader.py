"""
Agent Configuration Loader
---------------------------
Reads agent.yaml and memory.yaml, merges into a typed AgentConfig,
then provides a one-call factory to build the complete agent stack.

Entry points:
  load_agent_config() -- parse YAML into AgentConfig dataclass
  build_agent_from_config() -- create Agent + all subsystems
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from agent.config_yaml_helpers import (
    _discover_fts5_indexes,
    _find_config_dir,
    _load_fts5_index,
    _load_yaml,
    _resolve_backend_defaults,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

@dataclass
class AgentConfig:
    """Typed agent configuration -- merged from agent.yaml + memory.yaml."""

    # Core
    backend: str = "openai"
    model: str = ""
    endpoint: str = ""
    api_key_env: str = ""  # env var name for API key
    max_iterations: int = 50
    max_tokens_budget: int = 500_000
    working_dir: str = "."
    mode: str = "agent"
    temperature: float = 0.1
    top_k: int = 10

    # Memory
    memory_enabled: bool = True
    memory_index_name: str = "agent_memory"
    memory_index_dir: str = "data/indexes"
    memory_knowledge_dir: str = "data/agent_knowledge"
    auto_ingest: bool = True
    dedup_threshold: float = 0.95

    # Persistence
    goals_path: str = "data/agent_goals.json"
    session_enabled: bool = True
    session_dir: str = "data/agent_sessions"

    # Logging
    logging_enabled: bool = True
    log_dir: str = "logs/agent"

    # Embedder (optional -- enables dense vector search alongside FTS5)
    embedder_enabled: bool = False
    embedder_endpoint: str = "http://localhost:11434/v1"
    embedder_model: str = "nomic-embed-text-v2-moe"
    embedder_code_model: str = ""   # e.g. nomic-embed-code (empty = use primary)
    embedder_text_model: str = ""   # e.g. nomic-embed-text-v2-moe (empty = use primary)
    embedder_dimension: int = 768
    embedder_timeout: int = 120

    # Federated search
    federated_enabled: bool = True
    federated_rrf_k: int = 60
    federated_index_weights: Dict[str, float] = field(default_factory=dict)
    federated_data_dir: str = ""  # directory containing FTS5 .fts5.db files

    # Safety
    allowed_dirs: List[str] = field(default_factory=list)
    max_command_timeout_s: int = 120


def load_agent_config(config_dir: Optional[str] = None) -> AgentConfig:
    """Load agent configuration from YAML files.

    Reads config/agent.yaml and config/memory.yaml, merges into a
    single AgentConfig. Environment variables override YAML values:
      JCODER_AGENT_BACKEND, JCODER_AGENT_MODEL, JCODER_AGENT_ENDPOINT
    """
    d = _find_config_dir(config_dir)

    agent_raw = _load_yaml(d / "agent.yaml").get("agent", {})
    memory_raw = _load_yaml(d / "memory.yaml").get("memory", {})

    backend, model, endpoint, api_key_env = _resolve_backend_defaults(agent_raw)
    safety = agent_raw.get("safety", {})

    # Federated search (from memory.yaml)
    mem_yaml = _load_yaml(d / "memory.yaml")
    fed_raw = mem_yaml.get("federated_search", {})
    fed_indexes = fed_raw.get("indexes", {})
    fed_weights = {
        name: float(idx_cfg.get("weight", 1.0))
        for name, idx_cfg in fed_indexes.items()
    } if isinstance(fed_indexes, dict) else {}

    # Embedder (from memory.yaml -- collocated with memory/federated settings)
    emb_raw = mem_yaml.get("embedder", {})

    cfg = AgentConfig(
        backend=backend,
        model=model,
        endpoint=endpoint,
        api_key_env=api_key_env,
        max_iterations=agent_raw.get("max_iterations", 50),
        max_tokens_budget=agent_raw.get("max_tokens_budget", 500_000),
        working_dir=agent_raw.get("working_dir", "."),
        mode="agent",
        # Memory (from memory.yaml)
        memory_enabled=True,
        memory_index_name=memory_raw.get("index_name", "agent_memory"),
        memory_index_dir=memory_raw.get("index_dir", "data/indexes"),
        memory_knowledge_dir=memory_raw.get("knowledge_dir", "data/agent_knowledge"),
        auto_ingest=memory_raw.get("auto_ingest", True),
        dedup_threshold=memory_raw.get("dedup_threshold", 0.95),
        # Embedder (from memory.yaml)
        embedder_enabled=emb_raw.get("enabled", False),
        embedder_endpoint=emb_raw.get("endpoint", "http://localhost:11434/v1"),
        embedder_model=emb_raw.get("model", "nomic-embed-text-v2-moe"),
        embedder_code_model=emb_raw.get("code_model", ""),
        embedder_text_model=emb_raw.get("text_model", ""),
        embedder_dimension=emb_raw.get("dimension", 768),
        embedder_timeout=emb_raw.get("timeout", 120),
        # Federated search (from memory.yaml)
        federated_enabled=bool(fed_raw),
        federated_rrf_k=fed_raw.get("rrf_k", 60),
        federated_index_weights=fed_weights,
        federated_data_dir=fed_raw.get("data_dir", ""),
        # Persistence (from agent.yaml)
        goals_path=agent_raw.get("goals_path", "data/agent_goals.json"),
        session_dir=agent_raw.get("session_dir", "data/agent_sessions"),
        # Safety (from agent.yaml)
        allowed_dirs=safety.get("allowed_dirs", []),
        max_command_timeout_s=safety.get("max_command_timeout_s", 120),
    )

    # Environment variable overrides (deployment flexibility)
    env_backend = os.environ.get("JCODER_AGENT_BACKEND")
    if env_backend:
        cfg.backend = env_backend
    env_model = os.environ.get("JCODER_AGENT_MODEL")
    if env_model:
        cfg.model = env_model
    env_endpoint = os.environ.get("JCODER_AGENT_ENDPOINT")
    if env_endpoint:
        cfg.endpoint = env_endpoint

    return cfg


# ---------------------------------------------------------------------------
# Profile loader
# ---------------------------------------------------------------------------

def load_profiles(config_dir: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """Load query profiles from config/profiles.yaml.

    Returns a dict of {profile_name: {mode, temperature, max_iterations, top_k, description}}.
    Returns empty dict if the file is missing.
    """
    d = _find_config_dir(config_dir)
    raw = _load_yaml(d / "profiles.yaml")
    return raw.get("profiles", {})


def apply_profile(config: AgentConfig, profile_name: str,
                  config_dir: Optional[str] = None) -> AgentConfig:
    """Apply a named profile's settings to an AgentConfig.

    Overwrites mode, temperature, max_iterations, and top_k.
    Returns the modified config (same object, mutated in place).
    Raises KeyError if the profile name is not found.
    """
    profiles = load_profiles(config_dir)
    if profile_name not in profiles:
        available = ", ".join(sorted(profiles.keys())) or "(none)"
        raise KeyError(
            f"Unknown profile {profile_name!r}. Available: {available}"
        )
    p = profiles[profile_name]
    config.mode = p.get("mode", config.mode)
    config.temperature = float(p.get("temperature", config.temperature))
    config.max_iterations = int(p.get("max_iterations", config.max_iterations))
    config.top_k = int(p.get("top_k", config.top_k))
    return config


# ---------------------------------------------------------------------------
# One-call factory
# ---------------------------------------------------------------------------

def build_agent_from_config(
    config: Optional[AgentConfig] = None,
) -> Dict[str, Any]:
    """Build the complete agent stack from config.

    Returns a dict with all wired components::

        {
            "agent": Agent,
            "backend": LLMBackend,
            "tools": ToolRegistry,
            "memory": AgentMemory or None,
            "session_store": SessionStore or None,
            "logger": AgentLogger or None,
            "prompt_builder": PromptBuilder,
            "config": AgentConfig,
        }

    Graceful degradation: if memory, session, or logging fail to
    initialise, they are set to None with a warning instead of
    crashing the whole stack.
    """
    if config is None:
        config = load_agent_config()

    # -- Resolve API key from env var name ---------------------------------
    api_key = ""
    if config.api_key_env:
        api_key = os.environ.get(config.api_key_env, "")

    # -- LLM backend -------------------------------------------------------
    from agent.llm_backend import create_backend

    backend = create_backend(
        backend_type=config.backend,
        endpoint=config.endpoint,
        model=config.model,
        api_key=api_key,
    )

    # -- Embedder (optional) -----------------------------------------------
    embedder = _build_embedder(config)

    # -- Memory (optional) -------------------------------------------------
    memory = None
    if config.memory_enabled:
        try:
            from agent.memory import AgentMemory
            memory = AgentMemory(
                embedding_engine=embedder,
                index_dir=config.memory_index_dir,
                index_name=config.memory_index_name,
                knowledge_dir=config.memory_knowledge_dir,
            )
            mode = "dual-embedding" if embedder else "FTS5-only"
            log.info("[OK] Agent memory initialised (%s)", mode)
        except Exception as exc:
            log.warning("[WARN] Agent memory unavailable: %s", exc)

    # -- Federated search (optional) ---------------------------------------
    federated = None
    if config.federated_enabled:
        try:
            federated = _build_federated(config, embedder)
            if federated:
                log.info("[OK] Federated search: %d indexes", len(federated.list_indexes()))
        except Exception as exc:
            log.warning("[WARN] Federated search unavailable: %s", exc)

    # -- Session store (optional) ------------------------------------------
    session_store = None
    if config.session_enabled:
        try:
            from agent.session import SessionStore
            session_store = SessionStore(store_dir=config.session_dir)
            log.info("[OK] Session store: %s", config.session_dir)
        except Exception as exc:
            log.warning("[WARN] Session store unavailable: %s", exc)

    # -- Logger (optional) -------------------------------------------------
    agent_logger = None
    if config.logging_enabled:
        try:
            from agent.logger import AgentLogger
            agent_logger = AgentLogger(log_dir=config.log_dir)
            log.info("[OK] Agent logger: %s", config.log_dir)
        except Exception as exc:
            log.warning("[WARN] Agent logger unavailable: %s", exc)

    # -- Prompt builder ----------------------------------------------------
    from agent.prompts import PromptBuilder
    prompt_builder = PromptBuilder(mode=config.mode)

    # -- RAG callback (connects federated search to agent's rag_query tool) --
    rag_callback = None
    if federated:
        def rag_callback(query: str) -> str:
            results = federated.search(query, top_k=10)
            if not results:
                return "[No relevant chunks found in knowledge base]"
            lines = []
            for i, r in enumerate(results, 1):
                lines.append(
                    f"[{i}] ({r.index_name}, score={r.score:.3f})\n"
                    f"{r.content[:800]}\n"
                    f"Source: {r.source}"
                )
            return "\n\n".join(lines)
        log.info("[OK] RAG callback wired to federated search")

    # -- Tool registry -----------------------------------------------------
    from agent.tools import ToolRegistry
    tools = ToolRegistry(
        working_dir=config.working_dir,
        allowed_dirs=config.allowed_dirs or [],
        memory=memory,
        rag_callback=rag_callback,
    )

    # -- Agent core --------------------------------------------------------
    from agent.core import Agent
    agent = Agent(
        backend=backend,
        tools=tools,
        system_prompt=prompt_builder.build_messages("_bootstrap")[0]["content"],
        max_iterations=config.max_iterations,
        max_tokens_budget=config.max_tokens_budget,
        session_store=session_store,
        logger=agent_logger,
    )

    return {
        "agent": agent,
        "backend": backend,
        "tools": tools,
        "memory": memory,
        "embedder": embedder,
        "federated": federated,
        "session_store": session_store,
        "logger": agent_logger,
        "prompt_builder": prompt_builder,
        "config": config,
    }


# ---------------------------------------------------------------------------
# Federated search builder
# ---------------------------------------------------------------------------

def _build_embedder(config: AgentConfig):
    """Build a DualEmbeddingEngine from config, or return None.

    Returns None if embedder is disabled or the embedding server is
    unreachable.  The probe inside DualEmbeddingEngine sets internal
    flags so it degrades gracefully even if one model is missing.
    """
    if not config.embedder_enabled:
        return None

    try:
        from core.config import ModelConfig
        from core.embedding_engine import DualEmbeddingEngine

        model_cfg = ModelConfig(
            name=config.embedder_model,
            endpoint=config.embedder_endpoint,
            dimension=config.embedder_dimension,
            code_model=config.embedder_code_model or None,
            text_model=config.embedder_text_model or None,
        )
        engine = DualEmbeddingEngine(model_cfg, timeout=config.embedder_timeout)

        if not engine._code_ok and not engine._text_ok:
            log.warning("[WARN] Embedder configured but no models responded; "
                        "falling back to FTS5-only")
            engine.close()
            return None

        models = []
        if engine._code_ok:
            models.append(f"code={model_cfg.code_model or model_cfg.name}")
        if engine._text_ok:
            models.append(f"text={model_cfg.text_model or model_cfg.name}")
        log.info("[OK] DualEmbeddingEngine: %s", ", ".join(models))
        return engine
    except Exception as exc:
        log.warning("[WARN] Embedder init failed: %s", exc)
        return None


def _build_federated(config: AgentConfig, embedding_engine=None):
    """Build a FederatedSearch from discovered FTS5 indexes.

    Returns None if no indexes are found or FederatedSearch cannot be created.
    When embedding_engine is provided, FederatedSearch can use dense vectors
    alongside FTS5 for hybrid retrieval.
    """
    from core.federated_search import FederatedSearch

    # Determine data directories to scan for FTS5 files
    scan_dirs: List[str] = []
    if config.federated_data_dir:
        scan_dirs.append(config.federated_data_dir)
    # Also scan the memory index_dir (agent_memory lives here)
    if config.memory_index_dir:
        scan_dirs.append(config.memory_index_dir)

    # Discover all FTS5 databases across scan directories
    all_indexes: Dict[str, str] = {}
    for d in scan_dirs:
        if os.path.isdir(d):
            all_indexes.update(_discover_fts5_indexes(d))

    if not all_indexes:
        log.info("[WARN] No FTS5 indexes found; federated search disabled")
        return None

    fed = FederatedSearch(embedding_engine=embedding_engine, rrf_k=config.federated_rrf_k)

    for name, db_path in sorted(all_indexes.items()):
        try:
            idx = _load_fts5_index(db_path, name)
            chunk_count = getattr(idx, "_fts5_chunk_count", 0) or len(idx.metadata)
            if chunk_count == 0:
                log.warning("[WARN] Skipping empty index: %s", name)
                continue
            weight = config.federated_index_weights.get(name, 1.0)
            fed.add_index(name, idx, weight=weight)
            log.info("[OK] Federated index: %s (%d chunks, weight=%.1f)",
                     name, chunk_count, weight)
        except Exception as exc:
            log.warning("[WARN] Failed to load index %s: %s", name, exc)

    if not fed.list_indexes():
        return None
    return fed
