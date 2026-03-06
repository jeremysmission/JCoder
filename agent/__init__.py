"""JCoder Agent -- autonomous coding agent with dual LLM backend."""

from agent.core import Agent
from agent.llm_backend import create_backend
from agent.session import SessionInfo, SessionStore
from agent.tools import ToolRegistry
from agent.config_loader import AgentConfig, load_agent_config, build_agent_from_config

# Optional modules (may not be built yet)
try:
    from agent.memory import AgentMemory
except ImportError:
    AgentMemory = None  # type: ignore[assignment,misc]

try:
    from core.federated_search import FederatedSearch
except ImportError:
    FederatedSearch = None  # type: ignore[assignment,misc]

__all__ = ["Agent", "create_backend", "ToolRegistry",
           "SessionStore", "SessionInfo",
           "AgentConfig", "load_agent_config", "build_agent_from_config",
           "AgentMemory", "FederatedSearch"]
