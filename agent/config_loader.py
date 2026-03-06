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

    # Memory
    memory_enabled: bool = True
    memory_index_name: str = "agent_memory"
    memory_index_dir: str = "data/indexes"
    memory_knowledge_dir: str = "data/agent_knowledge"
    auto_ingest: bool = True
    dedup_threshold: float = 0.95

    # Session
    session_enabled: bool = True
    session_dir: str = "data/agent_sessions"

    # Logging
    logging_enabled: bool = True
    log_dir: str = "logs/agent"

    # Safety
    allowed_dirs: List[str] = field(default_factory=list)
    max_command_timeout_s: int = 120


# ---------------------------------------------------------------------------
# YAML loader
# ---------------------------------------------------------------------------

def _load_yaml(path: Path) -> dict:
    """Load a YAML file, return empty dict if missing."""
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _find_config_dir(config_dir: Optional[str] = None) -> Path:
    """Locate the config directory."""
    if config_dir:
        return Path(config_dir)
    env = os.environ.get("JCODER_CONFIG_DIR")
    if env:
        return Path(env)
    return Path(__file__).resolve().parent.parent / "config"


def _resolve_backend_defaults(agent_raw: dict) -> tuple:
    """Resolve model/endpoint/api_key_env from backend type and YAML fields.

    Returns (backend, model, endpoint, api_key_env).
    """
    backend = agent_raw.get("backend", "openai")
    model = agent_raw.get("model", "")
    endpoint = agent_raw.get("endpoint", "")

    # Auto-fill from backend-specific defaults when not explicitly set
    if backend == "ollama" or (backend == "openai" and not endpoint and not model):
        model = model or agent_raw.get("ollama_model", "devstral-small-2:24b")
        endpoint = endpoint or agent_raw.get("ollama_endpoint", "http://localhost:11434/v1")
        api_key_env = ""
    elif backend == "anthropic":
        model = model or agent_raw.get("api_model", "claude-sonnet-4-20250514")
        endpoint = ""  # Anthropic backend uses hardcoded URL
        api_key_env = "ANTHROPIC_API_KEY"
    else:
        # openai with explicit endpoint (OpenRouter, Azure, vLLM)
        model = model or agent_raw.get("api_model", "")
        endpoint = endpoint or agent_raw.get("api_endpoint", "")
        api_key_env = "OPENROUTER_API_KEY" if "openrouter" in endpoint.lower() else "OPENAI_API_KEY"

    return backend, model, endpoint, api_key_env


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

    # -- Memory (optional) -------------------------------------------------
    memory = None
    if config.memory_enabled:
        try:
            from agent.memory import AgentMemory
            memory = AgentMemory(
                embedding_engine=None,  # FTS5-only (no embedding server)
                index_dir=config.memory_index_dir,
                index_name=config.memory_index_name,
                knowledge_dir=config.memory_knowledge_dir,
            )
            log.info("[OK] Agent memory initialised (FTS5-only)")
        except Exception as exc:
            log.warning("[WARN] Agent memory unavailable: %s", exc)

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

    # -- Tool registry -----------------------------------------------------
    from agent.tools import ToolRegistry
    tools = ToolRegistry(
        working_dir=config.working_dir,
        allowed_dirs=config.allowed_dirs or [],
        memory=memory,
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
        "session_store": session_store,
        "logger": agent_logger,
        "prompt_builder": prompt_builder,
        "config": config,
    }
