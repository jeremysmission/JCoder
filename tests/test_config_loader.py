"""
Tests for agent.config_loader
-----------------------------
Covers: AgentConfig defaults, YAML loading, config directory resolution,
backend defaults, full load_agent_config pipeline, env var overrides,
and build_agent_from_config (mocked subsystems).
"""

from __future__ import annotations

import os
import sqlite3
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from agent.config_loader import (
    AgentConfig,
    _find_config_dir,
    _load_fts5_index,
    _load_yaml,
    _resolve_backend_defaults,
    apply_profile,
    build_agent_from_config,
    load_agent_config,
    load_profiles,
)


# ===================================================================
# 1. AgentConfig dataclass defaults
# ===================================================================


class TestAgentConfigDefaults:
    """Verify every default value on a bare AgentConfig()."""

    def test_core_defaults(self):
        cfg = AgentConfig()
        assert cfg.backend == "openai"
        assert cfg.model == ""
        assert cfg.endpoint == ""
        assert cfg.api_key_env == ""
        assert cfg.max_iterations == 50
        assert cfg.max_tokens_budget == 500_000
        assert cfg.working_dir == "."
        assert cfg.mode == "agent"

    def test_memory_defaults(self):
        cfg = AgentConfig()
        assert cfg.memory_enabled is True
        assert cfg.memory_index_name == "agent_memory"
        assert cfg.memory_index_dir == "data/indexes"
        assert cfg.memory_knowledge_dir == "data/agent_knowledge"
        assert cfg.auto_ingest is True
        assert cfg.dedup_threshold == 0.95

    def test_session_defaults(self):
        cfg = AgentConfig()
        assert cfg.session_enabled is True
        assert cfg.session_dir == "data/agent_sessions"

    def test_logging_defaults(self):
        cfg = AgentConfig()
        assert cfg.logging_enabled is True
        assert cfg.log_dir == "logs/agent"

    def test_safety_defaults(self):
        cfg = AgentConfig()
        assert cfg.allowed_dirs == []
        assert cfg.max_command_timeout_s == 120

    def test_allowed_dirs_is_independent_list(self):
        """Each instance gets its own list (field default_factory)."""
        a = AgentConfig()
        b = AgentConfig()
        a.allowed_dirs.append("/tmp")
        assert b.allowed_dirs == []


# ===================================================================
# 2. _load_yaml -- missing file returns empty dict
# ===================================================================


class TestLoadYaml:
    def test_missing_file_returns_empty_dict(self, tmp_path):
        result = _load_yaml(tmp_path / "nonexistent.yaml")
        assert result == {}

    def test_valid_yaml(self, tmp_path):
        p = tmp_path / "test.yaml"
        p.write_text(
            textwrap.dedent("""\
                agent:
                  backend: ollama
                  model: phi4:14b
            """),
            encoding="utf-8",
        )
        result = _load_yaml(p)
        assert result == {"agent": {"backend": "ollama", "model": "phi4:14b"}}

    def test_empty_yaml_returns_empty_dict(self, tmp_path):
        p = tmp_path / "empty.yaml"
        p.write_text("", encoding="utf-8")
        result = _load_yaml(p)
        assert result == {}

    def test_yaml_with_only_null(self, tmp_path):
        """A YAML file containing just `null` or `~` should return {}."""
        p = tmp_path / "null.yaml"
        p.write_text("~\n", encoding="utf-8")
        result = _load_yaml(p)
        assert result == {}


# ===================================================================
# 4. _find_config_dir
# ===================================================================


class TestFindConfigDir:
    def test_explicit_path(self, tmp_path):
        result = _find_config_dir(str(tmp_path))
        assert result == Path(tmp_path)

    def test_env_var(self, tmp_path, monkeypatch):
        monkeypatch.setenv("JCODER_CONFIG_DIR", str(tmp_path / "custom"))
        result = _find_config_dir()
        assert result == tmp_path / "custom"

    def test_explicit_overrides_env(self, tmp_path, monkeypatch):
        """Explicit config_dir takes priority over env var."""
        monkeypatch.setenv("JCODER_CONFIG_DIR", str(tmp_path / "env"))
        result = _find_config_dir(str(tmp_path / "explicit"))
        assert result == tmp_path / "explicit"

    def test_default_falls_back_to_project_config(self, monkeypatch):
        monkeypatch.delenv("JCODER_CONFIG_DIR", raising=False)
        result = _find_config_dir()
        # Should resolve to <project_root>/config
        assert result.name == "config"
        assert result.is_absolute()


# ===================================================================
# 6-8. _resolve_backend_defaults
# ===================================================================


class TestResolveBackendDefaults:
    def test_ollama_backend(self):
        backend, model, endpoint, api_key_env = _resolve_backend_defaults(
            {"backend": "ollama"}
        )
        assert backend == "ollama"
        assert model == "phi4:14b-q4_K_M"
        assert endpoint == "http://localhost:11434/v1"
        assert api_key_env == ""

    def test_ollama_with_custom_model(self):
        backend, model, endpoint, api_key_env = _resolve_backend_defaults(
            {"backend": "ollama", "ollama_model": "phi4:14b"}
        )
        assert model == "phi4:14b"

    def test_ollama_explicit_model_overrides_ollama_model(self):
        """Top-level 'model' key takes priority over 'ollama_model'."""
        _, model, _, _ = _resolve_backend_defaults(
            {"backend": "ollama", "model": "mistral:7b", "ollama_model": "phi4:14b"}
        )
        assert model == "mistral:7b"

    def test_anthropic_backend(self):
        backend, model, endpoint, api_key_env = _resolve_backend_defaults(
            {"backend": "anthropic"}
        )
        assert backend == "anthropic"
        assert model == "claude-sonnet-4-20250514"
        assert endpoint == ""
        assert api_key_env == "ANTHROPIC_API_KEY"

    def test_anthropic_custom_model(self):
        _, model, _, _ = _resolve_backend_defaults(
            {"backend": "anthropic", "api_model": "claude-opus-4-20250514"}
        )
        assert model == "claude-opus-4-20250514"

    def test_openai_with_openrouter_endpoint(self):
        backend, model, endpoint, api_key_env = _resolve_backend_defaults(
            {
                "backend": "openai",
                "endpoint": "https://openrouter.ai/api/v1",
                "model": "meta-llama/llama-3-70b",
            }
        )
        assert backend == "openai"
        assert model == "meta-llama/llama-3-70b"
        assert endpoint == "https://openrouter.ai/api/v1"
        assert api_key_env == "OPENROUTER_API_KEY"

    def test_openai_with_generic_endpoint(self):
        """Non-openrouter endpoint falls back to OPENAI_API_KEY."""
        _, _, _, api_key_env = _resolve_backend_defaults(
            {"backend": "openai", "endpoint": "https://my-vllm.local/v1", "model": "x"}
        )
        assert api_key_env == "OPENAI_API_KEY"

    def test_openai_no_model_no_endpoint_uses_ollama_defaults(self):
        """openai backend with nothing set falls through to ollama defaults."""
        backend, model, endpoint, _ = _resolve_backend_defaults(
            {"backend": "openai"}
        )
        assert backend == "openai"
        assert model == "phi4:14b-q4_K_M"
        assert endpoint == "http://localhost:11434/v1"

    def test_missing_backend_key_defaults_to_openai(self):
        backend, _, _, _ = _resolve_backend_defaults({})
        assert backend == "openai"


# ===================================================================
# 9. load_agent_config -- reads from YAML files
# ===================================================================


@pytest.fixture
def config_dir(tmp_path):
    """Create a temporary config directory with agent.yaml + memory.yaml."""
    agent_yaml = tmp_path / "agent.yaml"
    agent_yaml.write_text(
        textwrap.dedent("""\
            agent:
              backend: ollama
              model: phi4:14b
              endpoint: http://localhost:11434/v1
              max_iterations: 30
              max_tokens_budget: 200000
              working_dir: /projects/test
              safety:
                allowed_dirs:
                  - /projects/test
                  - /tmp
                max_command_timeout_s: 60
        """),
        encoding="utf-8",
    )
    memory_yaml = tmp_path / "memory.yaml"
    memory_yaml.write_text(
        textwrap.dedent("""\
            memory:
              index_name: test_memory
              index_dir: /data/idx
              knowledge_dir: /data/knowledge
              auto_ingest: false
              dedup_threshold: 0.85
        """),
        encoding="utf-8",
    )
    return tmp_path


class TestLoadAgentConfig:
    def test_reads_agent_yaml(self, config_dir, monkeypatch):
        monkeypatch.delenv("JCODER_AGENT_BACKEND", raising=False)
        monkeypatch.delenv("JCODER_AGENT_MODEL", raising=False)
        monkeypatch.delenv("JCODER_AGENT_ENDPOINT", raising=False)

        cfg = load_agent_config(config_dir=str(config_dir))
        assert cfg.backend == "ollama"
        assert cfg.model == "phi4:14b"
        assert cfg.endpoint == "http://localhost:11434/v1"
        assert cfg.max_iterations == 30
        assert cfg.max_tokens_budget == 200_000
        assert cfg.working_dir == "/projects/test"

    def test_reads_memory_yaml(self, config_dir, monkeypatch):
        monkeypatch.delenv("JCODER_AGENT_BACKEND", raising=False)
        monkeypatch.delenv("JCODER_AGENT_MODEL", raising=False)
        monkeypatch.delenv("JCODER_AGENT_ENDPOINT", raising=False)

        cfg = load_agent_config(config_dir=str(config_dir))
        assert cfg.memory_index_name == "test_memory"
        assert cfg.memory_index_dir == "/data/idx"
        assert cfg.memory_knowledge_dir == "/data/knowledge"
        assert cfg.auto_ingest is False
        assert cfg.dedup_threshold == 0.85

    def test_reads_safety_fields(self, config_dir, monkeypatch):
        monkeypatch.delenv("JCODER_AGENT_BACKEND", raising=False)
        monkeypatch.delenv("JCODER_AGENT_MODEL", raising=False)
        monkeypatch.delenv("JCODER_AGENT_ENDPOINT", raising=False)

        cfg = load_agent_config(config_dir=str(config_dir))
        assert cfg.allowed_dirs == ["/projects/test", "/tmp"]
        assert cfg.max_command_timeout_s == 60

    def test_missing_yaml_uses_defaults(self, tmp_path, monkeypatch):
        """Empty config dir -- all fields fall to dataclass defaults."""
        monkeypatch.delenv("JCODER_AGENT_BACKEND", raising=False)
        monkeypatch.delenv("JCODER_AGENT_MODEL", raising=False)
        monkeypatch.delenv("JCODER_AGENT_ENDPOINT", raising=False)

        cfg = load_agent_config(config_dir=str(tmp_path))
        # openai with no model/endpoint falls through to ollama defaults
        assert cfg.model == "phi4:14b-q4_K_M"
        assert cfg.max_iterations == 50
        assert cfg.memory_index_name == "agent_memory"

    def test_mode_is_always_agent(self, config_dir, monkeypatch):
        monkeypatch.delenv("JCODER_AGENT_BACKEND", raising=False)
        monkeypatch.delenv("JCODER_AGENT_MODEL", raising=False)
        monkeypatch.delenv("JCODER_AGENT_ENDPOINT", raising=False)

        cfg = load_agent_config(config_dir=str(config_dir))
        assert cfg.mode == "agent"


# ===================================================================
# 10. load_agent_config -- env var overrides
# ===================================================================


class TestLoadAgentConfigEnvOverrides:
    def test_backend_override(self, config_dir, monkeypatch):
        monkeypatch.setenv("JCODER_AGENT_BACKEND", "anthropic")
        monkeypatch.delenv("JCODER_AGENT_MODEL", raising=False)
        monkeypatch.delenv("JCODER_AGENT_ENDPOINT", raising=False)

        cfg = load_agent_config(config_dir=str(config_dir))
        assert cfg.backend == "anthropic"

    def test_model_override(self, config_dir, monkeypatch):
        monkeypatch.delenv("JCODER_AGENT_BACKEND", raising=False)
        monkeypatch.setenv("JCODER_AGENT_MODEL", "mistral-nemo:12b")
        monkeypatch.delenv("JCODER_AGENT_ENDPOINT", raising=False)

        cfg = load_agent_config(config_dir=str(config_dir))
        assert cfg.model == "mistral-nemo:12b"

    def test_endpoint_override(self, config_dir, monkeypatch):
        monkeypatch.delenv("JCODER_AGENT_BACKEND", raising=False)
        monkeypatch.delenv("JCODER_AGENT_MODEL", raising=False)
        monkeypatch.setenv("JCODER_AGENT_ENDPOINT", "http://custom:9999/v1")

        cfg = load_agent_config(config_dir=str(config_dir))
        assert cfg.endpoint == "http://custom:9999/v1"

    def test_all_three_overrides(self, config_dir, monkeypatch):
        monkeypatch.setenv("JCODER_AGENT_BACKEND", "openai")
        monkeypatch.setenv("JCODER_AGENT_MODEL", "gpt-4o")
        monkeypatch.setenv("JCODER_AGENT_ENDPOINT", "https://api.openai.com/v1")

        cfg = load_agent_config(config_dir=str(config_dir))
        assert cfg.backend == "openai"
        assert cfg.model == "gpt-4o"
        assert cfg.endpoint == "https://api.openai.com/v1"

    def test_env_overrides_only_set_fields(self, config_dir, monkeypatch):
        """Env override for model should not change backend or endpoint."""
        monkeypatch.delenv("JCODER_AGENT_BACKEND", raising=False)
        monkeypatch.setenv("JCODER_AGENT_MODEL", "override-model")
        monkeypatch.delenv("JCODER_AGENT_ENDPOINT", raising=False)

        cfg = load_agent_config(config_dir=str(config_dir))
        # backend and endpoint should come from YAML, not defaults
        assert cfg.backend == "ollama"
        assert cfg.endpoint == "http://localhost:11434/v1"
        assert cfg.model == "override-model"


# ===================================================================
# 11. build_agent_from_config -- returns dict with expected keys
# ===================================================================


@pytest.fixture
def mock_subsystems():
    """Patch all subsystem imports inside build_agent_from_config."""
    mock_backend = MagicMock(name="MockBackend")
    mock_memory = MagicMock(name="MockMemory")
    mock_session = MagicMock(name="MockSessionStore")
    mock_logger = MagicMock(name="MockAgentLogger")
    mock_prompt_builder = MagicMock(name="MockPromptBuilder")
    mock_prompt_builder.build_messages.return_value = [{"content": "system prompt"}]
    mock_tools = MagicMock(name="MockToolRegistry")
    mock_agent = MagicMock(name="MockAgent")

    patches = {
        "create_backend": patch(
            "agent.config_loader.create_backend",
            create=True,
        ),
        "AgentMemory": patch("agent.memory.AgentMemory", create=True),
        "SessionStore": patch("agent.session.SessionStore", create=True),
        "AgentLogger": patch("agent.logger.AgentLogger", create=True),
        "PromptBuilder": patch("agent.prompts.PromptBuilder", create=True),
        "ToolRegistry": patch("agent.tools.ToolRegistry", create=True),
        "Agent": patch("agent.core.Agent", create=True),
    }

    return mock_backend, patches


class TestBuildAgentFromConfig:
    EXPECTED_KEYS = {
        "agent",
        "backend",
        "tools",
        "memory",
        "embedder",
        "federated",
        "session_store",
        "logger",
        "prompt_builder",
        "config",
    }

    @patch("agent.config_loader.create_backend", create=True)
    def test_returns_all_expected_keys(self, mock_create_backend, monkeypatch):
        """build_agent_from_config returns a dict with the documented keys."""
        mock_create_backend.return_value = MagicMock(name="backend")
        monkeypatch.delenv("JCODER_AGENT_BACKEND", raising=False)
        monkeypatch.delenv("JCODER_AGENT_MODEL", raising=False)
        monkeypatch.delenv("JCODER_AGENT_ENDPOINT", raising=False)

        cfg = AgentConfig(
            backend="ollama",
            model="phi4:14b",
            endpoint="http://localhost:11434/v1",
            memory_enabled=False,
            session_enabled=False,
            logging_enabled=False,
        )

        mock_prompt = MagicMock()
        mock_prompt.build_messages.return_value = [{"content": "sys"}]

        with (
            patch.dict("sys.modules", {
                "agent.llm_backend": MagicMock(
                    create_backend=mock_create_backend
                ),
                "agent.memory": MagicMock(),
                "agent.session": MagicMock(),
                "agent.logger": MagicMock(),
                "agent.prompts": MagicMock(PromptBuilder=MagicMock(return_value=mock_prompt)),
                "agent.tools": MagicMock(ToolRegistry=MagicMock(return_value=MagicMock())),
                "agent.core": MagicMock(Agent=MagicMock(return_value=MagicMock())),
            }),
        ):
            result = build_agent_from_config(config=cfg)

        assert set(result.keys()) == self.EXPECTED_KEYS
        assert result["config"] is cfg

    @patch("agent.config_loader.create_backend", create=True)
    def test_disabled_subsystems_are_none(self, mock_create_backend, monkeypatch):
        """When memory/session/logging disabled, those keys are None."""
        mock_create_backend.return_value = MagicMock()
        monkeypatch.delenv("JCODER_AGENT_BACKEND", raising=False)
        monkeypatch.delenv("JCODER_AGENT_MODEL", raising=False)
        monkeypatch.delenv("JCODER_AGENT_ENDPOINT", raising=False)

        cfg = AgentConfig(
            backend="ollama",
            model="phi4:14b",
            endpoint="http://localhost:11434/v1",
            memory_enabled=False,
            session_enabled=False,
            logging_enabled=False,
        )

        mock_prompt = MagicMock()
        mock_prompt.build_messages.return_value = [{"content": "sys"}]

        with patch.dict("sys.modules", {
            "agent.llm_backend": MagicMock(create_backend=mock_create_backend),
            "agent.memory": MagicMock(),
            "agent.session": MagicMock(),
            "agent.logger": MagicMock(),
            "agent.prompts": MagicMock(PromptBuilder=MagicMock(return_value=mock_prompt)),
            "agent.tools": MagicMock(ToolRegistry=MagicMock(return_value=MagicMock())),
            "agent.core": MagicMock(Agent=MagicMock(return_value=MagicMock())),
        }):
            result = build_agent_from_config(config=cfg)

        assert result["memory"] is None
        assert result["session_store"] is None
        assert result["logger"] is None


# ===================================================================
# 12. build_agent_from_config -- graceful degradation
# ===================================================================


class TestBuildAgentGracefulDegradation:
    """Memory, session, and logger failures should not crash the build."""

    @patch("agent.config_loader.create_backend", create=True)
    def test_memory_failure_sets_none(self, mock_create_backend, monkeypatch):
        mock_create_backend.return_value = MagicMock()
        monkeypatch.delenv("JCODER_AGENT_BACKEND", raising=False)
        monkeypatch.delenv("JCODER_AGENT_MODEL", raising=False)
        monkeypatch.delenv("JCODER_AGENT_ENDPOINT", raising=False)

        cfg = AgentConfig(
            backend="ollama",
            model="phi4:14b",
            endpoint="http://localhost:11434/v1",
            memory_enabled=True,
            session_enabled=False,
            logging_enabled=False,
            federated_enabled=False,
        )

        mock_prompt = MagicMock()
        mock_prompt.build_messages.return_value = [{"content": "sys"}]

        # AgentMemory constructor raises
        bad_memory_mod = MagicMock()
        bad_memory_mod.AgentMemory.side_effect = RuntimeError("disk full")

        with patch.dict("sys.modules", {
            "agent.llm_backend": MagicMock(create_backend=mock_create_backend),
            "agent.memory": bad_memory_mod,
            "agent.session": MagicMock(),
            "agent.logger": MagicMock(),
            "agent.prompts": MagicMock(PromptBuilder=MagicMock(return_value=mock_prompt)),
            "agent.tools": MagicMock(ToolRegistry=MagicMock(return_value=MagicMock())),
            "agent.core": MagicMock(Agent=MagicMock(return_value=MagicMock())),
        }):
            result = build_agent_from_config(config=cfg)

        # Memory failed gracefully -- None, not an exception
        assert result["memory"] is None

    @patch("agent.config_loader.create_backend", create=True)
    def test_session_failure_sets_none(self, mock_create_backend, monkeypatch):
        mock_create_backend.return_value = MagicMock()
        monkeypatch.delenv("JCODER_AGENT_BACKEND", raising=False)
        monkeypatch.delenv("JCODER_AGENT_MODEL", raising=False)
        monkeypatch.delenv("JCODER_AGENT_ENDPOINT", raising=False)

        cfg = AgentConfig(
            backend="ollama",
            model="phi4:14b",
            endpoint="http://localhost:11434/v1",
            memory_enabled=False,
            session_enabled=True,
            logging_enabled=False,
            federated_enabled=False,
        )

        mock_prompt = MagicMock()
        mock_prompt.build_messages.return_value = [{"content": "sys"}]

        bad_session_mod = MagicMock()
        bad_session_mod.SessionStore.side_effect = OSError("permission denied")

        with patch.dict("sys.modules", {
            "agent.llm_backend": MagicMock(create_backend=mock_create_backend),
            "agent.memory": MagicMock(),
            "agent.session": bad_session_mod,
            "agent.logger": MagicMock(),
            "agent.prompts": MagicMock(PromptBuilder=MagicMock(return_value=mock_prompt)),
            "agent.tools": MagicMock(ToolRegistry=MagicMock(return_value=MagicMock())),
            "agent.core": MagicMock(Agent=MagicMock(return_value=MagicMock())),
        }):
            result = build_agent_from_config(config=cfg)

        assert result["session_store"] is None

    @patch("agent.config_loader.create_backend", create=True)
    def test_logger_failure_sets_none(self, mock_create_backend, monkeypatch):
        mock_create_backend.return_value = MagicMock()
        monkeypatch.delenv("JCODER_AGENT_BACKEND", raising=False)
        monkeypatch.delenv("JCODER_AGENT_MODEL", raising=False)
        monkeypatch.delenv("JCODER_AGENT_ENDPOINT", raising=False)

        cfg = AgentConfig(
            backend="ollama",
            model="phi4:14b",
            endpoint="http://localhost:11434/v1",
            memory_enabled=False,
            session_enabled=False,
            logging_enabled=True,
            federated_enabled=False,
        )

        mock_prompt = MagicMock()
        mock_prompt.build_messages.return_value = [{"content": "sys"}]

        bad_logger_mod = MagicMock()
        bad_logger_mod.AgentLogger.side_effect = ValueError("bad log dir")

        with patch.dict("sys.modules", {
            "agent.llm_backend": MagicMock(create_backend=mock_create_backend),
            "agent.memory": MagicMock(),
            "agent.session": MagicMock(),
            "agent.logger": bad_logger_mod,
            "agent.prompts": MagicMock(PromptBuilder=MagicMock(return_value=mock_prompt)),
            "agent.tools": MagicMock(ToolRegistry=MagicMock(return_value=MagicMock())),
            "agent.core": MagicMock(Agent=MagicMock(return_value=MagicMock())),
        }):
            result = build_agent_from_config(config=cfg)

        assert result["logger"] is None


# ===================================================================
# 13. build_agent_from_config -- rag_callback wiring
# ===================================================================


class TestRagCallbackWiring:
    """Federated search should be wired as rag_callback to ToolRegistry."""

    @patch("agent.config_loader.create_backend", create=True)
    @patch("agent.config_loader._build_federated")
    def test_rag_callback_passed_when_federated_exists(
        self, mock_build_fed, mock_create_backend, monkeypatch,
    ):
        mock_create_backend.return_value = MagicMock()
        mock_build_fed.return_value = MagicMock(
            list_indexes=MagicMock(return_value=[{"name": "test"}]),
        )
        monkeypatch.delenv("JCODER_AGENT_BACKEND", raising=False)
        monkeypatch.delenv("JCODER_AGENT_MODEL", raising=False)
        monkeypatch.delenv("JCODER_AGENT_ENDPOINT", raising=False)

        cfg = AgentConfig(
            backend="ollama",
            model="phi4:14b",
            endpoint="http://localhost:11434/v1",
            memory_enabled=False,
            session_enabled=False,
            logging_enabled=False,
            federated_enabled=True,
        )

        mock_prompt = MagicMock()
        mock_prompt.build_messages.return_value = [{"content": "sys"}]
        mock_tool_reg_cls = MagicMock(return_value=MagicMock())

        with patch.dict("sys.modules", {
            "agent.llm_backend": MagicMock(create_backend=mock_create_backend),
            "agent.memory": MagicMock(),
            "agent.session": MagicMock(),
            "agent.logger": MagicMock(),
            "agent.prompts": MagicMock(PromptBuilder=MagicMock(return_value=mock_prompt)),
            "agent.tools": MagicMock(ToolRegistry=mock_tool_reg_cls),
            "agent.core": MagicMock(Agent=MagicMock(return_value=MagicMock())),
        }):
            build_agent_from_config(config=cfg)

        # ToolRegistry should have received a callable rag_callback
        call_kwargs = mock_tool_reg_cls.call_args
        assert call_kwargs is not None
        assert callable(call_kwargs.kwargs.get("rag_callback")), \
            "rag_callback should be a callable when federated search is available"

    @patch("agent.config_loader.create_backend", create=True)
    @patch("agent.config_loader._build_federated")
    def test_rag_callback_none_when_no_federated(
        self, mock_build_fed, mock_create_backend, monkeypatch,
    ):
        mock_create_backend.return_value = MagicMock()
        mock_build_fed.return_value = None
        monkeypatch.delenv("JCODER_AGENT_BACKEND", raising=False)
        monkeypatch.delenv("JCODER_AGENT_MODEL", raising=False)
        monkeypatch.delenv("JCODER_AGENT_ENDPOINT", raising=False)

        cfg = AgentConfig(
            backend="ollama",
            model="phi4:14b",
            endpoint="http://localhost:11434/v1",
            memory_enabled=False,
            session_enabled=False,
            logging_enabled=False,
            federated_enabled=True,
        )

        mock_prompt = MagicMock()
        mock_prompt.build_messages.return_value = [{"content": "sys"}]
        mock_tool_reg_cls = MagicMock(return_value=MagicMock())

        with patch.dict("sys.modules", {
            "agent.llm_backend": MagicMock(create_backend=mock_create_backend),
            "agent.memory": MagicMock(),
            "agent.session": MagicMock(),
            "agent.logger": MagicMock(),
            "agent.prompts": MagicMock(PromptBuilder=MagicMock(return_value=mock_prompt)),
            "agent.tools": MagicMock(ToolRegistry=mock_tool_reg_cls),
            "agent.core": MagicMock(Agent=MagicMock(return_value=MagicMock())),
        }):
            build_agent_from_config(config=cfg)

        call_kwargs = mock_tool_reg_cls.call_args
        assert call_kwargs is not None
        assert call_kwargs.kwargs.get("rag_callback") is None, \
            "rag_callback should be None when federated search is unavailable"


# ===================================================================
# 10. Query Profiles
# ===================================================================

class TestLoadProfiles:
    """Tests for load_profiles() and apply_profile()."""

    def test_load_profiles_from_project(self):
        """Profiles load from the actual config/profiles.yaml."""
        profiles = load_profiles()
        assert "code" in profiles
        assert "debug" in profiles
        assert "agent" in profiles
        assert profiles["code"]["mode"] == "code"

    def test_load_profiles_empty_when_missing(self, tmp_path):
        """Missing profiles.yaml returns empty dict."""
        profiles = load_profiles(config_dir=str(tmp_path))
        assert profiles == {}

    def test_apply_profile_sets_fields(self):
        """apply_profile overwrites mode, temperature, max_iterations, top_k."""
        cfg = AgentConfig()
        apply_profile(cfg, "code")
        assert cfg.mode == "code"
        assert cfg.temperature == 0.1
        assert cfg.max_iterations == 30
        assert cfg.top_k == 10

    def test_apply_profile_deep(self):
        cfg = AgentConfig()
        apply_profile(cfg, "deep")
        assert cfg.max_iterations == 100
        assert cfg.top_k == 20

    def test_apply_profile_quick(self):
        cfg = AgentConfig()
        apply_profile(cfg, "quick")
        assert cfg.mode == "qa"
        assert cfg.max_iterations == 5

    def test_apply_unknown_profile_raises(self):
        cfg = AgentConfig()
        with pytest.raises(KeyError, match="Unknown profile"):
            apply_profile(cfg, "nonexistent_profile_xyz")

    def test_apply_profile_preserves_unrelated_fields(self):
        """Profile application should not change backend, model, etc."""
        cfg = AgentConfig(backend="anthropic", model="claude-test")
        apply_profile(cfg, "code")
        assert cfg.backend == "anthropic"
        assert cfg.model == "claude-test"

    def test_all_profiles_have_required_fields(self):
        """Every profile must have mode, temperature, max_iterations, top_k."""
        profiles = load_profiles()
        required = {"mode", "temperature", "max_iterations", "top_k"}
        for name, p in profiles.items():
            for fld in required:
                assert fld in p, f"Profile {name!r} missing {fld!r}"

    def test_config_defaults_include_new_fields(self):
        """AgentConfig has temperature and top_k with sensible defaults."""
        cfg = AgentConfig()
        assert cfg.temperature == 0.1
        assert cfg.top_k == 10


# ===================================================================
# 14. Embedder config
# ===================================================================


class TestEmbedderConfig:
    """Tests for embedder configuration fields and parsing."""

    def test_embedder_defaults_disabled(self):
        """Embedder is disabled by default."""
        cfg = AgentConfig()
        assert cfg.embedder_enabled is False
        assert cfg.embedder_endpoint == "http://localhost:11434/v1"
        assert cfg.embedder_model == "nomic-embed-text-v2-moe"
        assert cfg.embedder_code_model == ""
        assert cfg.embedder_text_model == ""
        assert cfg.embedder_dimension == 768
        assert cfg.embedder_timeout == 120

    def test_embedder_parsed_from_memory_yaml(self, tmp_path, monkeypatch):
        """Embedder section in memory.yaml is parsed into AgentConfig."""
        monkeypatch.delenv("JCODER_AGENT_BACKEND", raising=False)
        monkeypatch.delenv("JCODER_AGENT_MODEL", raising=False)
        monkeypatch.delenv("JCODER_AGENT_ENDPOINT", raising=False)

        (tmp_path / "agent.yaml").write_text("agent:\n  backend: ollama\n",
                                              encoding="utf-8")
        (tmp_path / "memory.yaml").write_text(textwrap.dedent("""\
            memory:
              index_name: test
            embedder:
              enabled: true
              endpoint: "http://my-server:9000/v1"
              model: "custom-embed"
              code_model: "code-embed-v2"
              text_model: "text-embed-v2"
              dimension: 1024
              timeout: 60
        """), encoding="utf-8")

        cfg = load_agent_config(config_dir=str(tmp_path))
        assert cfg.embedder_enabled is True
        assert cfg.embedder_endpoint == "http://my-server:9000/v1"
        assert cfg.embedder_model == "custom-embed"
        assert cfg.embedder_code_model == "code-embed-v2"
        assert cfg.embedder_text_model == "text-embed-v2"
        assert cfg.embedder_dimension == 1024
        assert cfg.embedder_timeout == 60

    def test_missing_embedder_section_uses_defaults(self, tmp_path, monkeypatch):
        """No embedder section in memory.yaml means disabled with defaults."""
        monkeypatch.delenv("JCODER_AGENT_BACKEND", raising=False)
        monkeypatch.delenv("JCODER_AGENT_MODEL", raising=False)
        monkeypatch.delenv("JCODER_AGENT_ENDPOINT", raising=False)

        (tmp_path / "agent.yaml").write_text("agent:\n  backend: ollama\n",
                                              encoding="utf-8")
        (tmp_path / "memory.yaml").write_text("memory:\n  index_name: test\n",
                                               encoding="utf-8")

        cfg = load_agent_config(config_dir=str(tmp_path))
        assert cfg.embedder_enabled is False
        assert cfg.embedder_model == "nomic-embed-text-v2-moe"


class TestBuildEmbedder:
    """Tests for _build_embedder() factory function."""

    def test_returns_none_when_disabled(self):
        from agent.config_loader import _build_embedder
        cfg = AgentConfig(embedder_enabled=False)
        assert _build_embedder(cfg) is None

    def test_returns_none_when_no_models_respond(self):
        from agent.config_loader import _build_embedder
        cfg = AgentConfig(
            embedder_enabled=True,
            embedder_endpoint="http://localhost:99999/v1",
        )
        # DualEmbeddingEngine probes will fail on bad port
        mock_engine = MagicMock()
        mock_engine._code_ok = False
        mock_engine._text_ok = False
        mock_engine.close = MagicMock()

        with patch("agent.config_loader.DualEmbeddingEngine",
                   create=True, return_value=mock_engine) as mock_cls:
            with patch.dict("sys.modules", {
                "core.config": MagicMock(ModelConfig=MagicMock()),
                "core.embedding_engine": MagicMock(
                    DualEmbeddingEngine=mock_cls
                ),
            }):
                result = _build_embedder(cfg)

        assert result is None
        mock_engine.close.assert_called_once()

    def test_returns_engine_when_models_respond(self):
        from agent.config_loader import _build_embedder
        cfg = AgentConfig(
            embedder_enabled=True,
            embedder_endpoint="http://localhost:11434/v1",
            embedder_model="nomic-embed-text-v2-moe",
            embedder_code_model="nomic-embed-code",
        )

        mock_engine = MagicMock()
        mock_engine._code_ok = True
        mock_engine._text_ok = True

        with patch("agent.config_loader.DualEmbeddingEngine",
                   create=True, return_value=mock_engine) as mock_cls:
            with patch.dict("sys.modules", {
                "core.config": MagicMock(ModelConfig=MagicMock()),
                "core.embedding_engine": MagicMock(
                    DualEmbeddingEngine=mock_cls
                ),
            }):
                result = _build_embedder(cfg)

        assert result is mock_engine

    def test_returns_engine_with_partial_models(self):
        """Only code model responding still returns a working engine."""
        from agent.config_loader import _build_embedder
        cfg = AgentConfig(
            embedder_enabled=True,
            embedder_endpoint="http://localhost:11434/v1",
        )

        mock_engine = MagicMock()
        mock_engine._code_ok = True
        mock_engine._text_ok = False

        with patch("agent.config_loader.DualEmbeddingEngine",
                   create=True, return_value=mock_engine) as mock_cls:
            with patch.dict("sys.modules", {
                "core.config": MagicMock(ModelConfig=MagicMock()),
                "core.embedding_engine": MagicMock(
                    DualEmbeddingEngine=mock_cls
                ),
            }):
                result = _build_embedder(cfg)

        assert result is mock_engine

    def test_import_failure_returns_none(self):
        """If core.config or core.embedding_engine can't import, returns None."""
        from agent.config_loader import _build_embedder
        cfg = AgentConfig(embedder_enabled=True)

        with patch.dict("sys.modules", {
            "core.config": None,  # simulate ImportError
        }):
            # The import inside _build_embedder will fail
            result = _build_embedder(cfg)

        assert result is None


# ===================================================================
# 15. Persistence paths from YAML (goals_path, session_dir)
# ===================================================================


class TestPersistencePathsFromYaml:
    """goals_path and session_dir should be read from agent.yaml."""

    def test_goals_path_default(self):
        cfg = AgentConfig()
        assert cfg.goals_path == "data/agent_goals.json"

    def test_goals_path_from_yaml(self, tmp_path, monkeypatch):
        monkeypatch.delenv("JCODER_AGENT_BACKEND", raising=False)
        monkeypatch.delenv("JCODER_AGENT_MODEL", raising=False)
        monkeypatch.delenv("JCODER_AGENT_ENDPOINT", raising=False)

        (tmp_path / "agent.yaml").write_text(textwrap.dedent("""\
            agent:
              backend: ollama
              goals_path: "custom/goals.json"
              session_dir: "custom/sessions"
        """), encoding="utf-8")
        (tmp_path / "memory.yaml").write_text("", encoding="utf-8")

        cfg = load_agent_config(config_dir=str(tmp_path))
        assert cfg.goals_path == "custom/goals.json"
        assert cfg.session_dir == "custom/sessions"

    def test_missing_goals_path_uses_default(self, tmp_path, monkeypatch):
        monkeypatch.delenv("JCODER_AGENT_BACKEND", raising=False)
        monkeypatch.delenv("JCODER_AGENT_MODEL", raising=False)
        monkeypatch.delenv("JCODER_AGENT_ENDPOINT", raising=False)

        (tmp_path / "agent.yaml").write_text(textwrap.dedent("""\
            agent:
              backend: ollama
        """), encoding="utf-8")
        (tmp_path / "memory.yaml").write_text("", encoding="utf-8")

        cfg = load_agent_config(config_dir=str(tmp_path))
        assert cfg.goals_path == "data/agent_goals.json"
        assert cfg.session_dir == "data/agent_sessions"

    def test_session_dir_from_yaml_reaches_session_store(self, tmp_path, monkeypatch):
        """session_dir parsed from YAML is used by build_agent_from_config."""
        monkeypatch.delenv("JCODER_AGENT_BACKEND", raising=False)
        monkeypatch.delenv("JCODER_AGENT_MODEL", raising=False)
        monkeypatch.delenv("JCODER_AGENT_ENDPOINT", raising=False)

        cfg = AgentConfig(
            backend="ollama", model="phi4:14b",
            endpoint="http://localhost:11434/v1",
            memory_enabled=False, session_enabled=True,
            logging_enabled=False, federated_enabled=False,
            session_dir="/my/custom/sessions",
        )

        mock_prompt = MagicMock()
        mock_prompt.build_messages.return_value = [{"content": "sys"}]
        mock_session_cls = MagicMock(return_value=MagicMock())
        mock_create_backend = MagicMock(return_value=MagicMock())

        with patch.dict("sys.modules", {
            "agent.llm_backend": MagicMock(create_backend=mock_create_backend),
            "agent.memory": MagicMock(),
            "agent.session": MagicMock(SessionStore=mock_session_cls),
            "agent.logger": MagicMock(),
            "agent.prompts": MagicMock(PromptBuilder=MagicMock(return_value=mock_prompt)),
            "agent.tools": MagicMock(ToolRegistry=MagicMock(return_value=MagicMock())),
            "agent.core": MagicMock(Agent=MagicMock(return_value=MagicMock())),
        }):
            with patch("agent.config_loader.create_backend",
                       mock_create_backend, create=True):
                build_agent_from_config(config=cfg)

        mock_session_cls.assert_called_once_with(store_dir="/my/custom/sessions")


class TestFederatedFts5Loader:
    """Regression coverage for threaded federated FTS5 loading."""

    @staticmethod
    def _create_fts5_db(path: Path, content: str) -> None:
        conn = sqlite3.connect(path)
        conn.execute(
            "CREATE VIRTUAL TABLE chunks USING fts5(search_content, source_path, chunk_id)"
        )
        conn.execute(
            "INSERT INTO chunks(search_content, source_path, chunk_id) VALUES (?, ?, ?)",
            (content, str(path.with_suffix('.py')), path.stem),
        )
        conn.commit()
        conn.close()

    @staticmethod
    def _create_legacy_fts5_db(path: Path, content: str) -> None:
        conn = sqlite3.connect(path)
        conn.execute(
            "CREATE VIRTUAL TABLE chunks USING fts5(content, source, category)"
        )
        conn.execute(
            "INSERT INTO chunks(content, source, category) VALUES (?, ?, ?)",
            (content, "research:absolute_zero_reasoner.md", "self_learning_research"),
        )
        conn.commit()
        conn.close()

    def test_load_fts5_index_keeps_connection_lazy_for_threaded_federation(self, tmp_path):
        from core.federated_search import FederatedSearch

        alpha = tmp_path / "alpha.fts5.db"
        beta = tmp_path / "beta.fts5.db"
        self._create_fts5_db(alpha, "def alpha_ready(): return 'ready'")
        self._create_fts5_db(beta, "def beta_ready(): return 'ready'")

        alpha_index = _load_fts5_index(str(alpha), "alpha")
        beta_index = _load_fts5_index(str(beta), "beta")

        assert alpha_index._fts_conn is None
        assert beta_index._fts_conn is None

        fed = FederatedSearch(embedding_engine=None, max_workers=2)
        fed.add_index("alpha", alpha_index)
        fed.add_index("beta", beta_index)
        try:
            results = fed.search("ready", top_k=5)
        finally:
            fed.close()
            alpha_index.close()
            beta_index.close()

        found_indexes = {result.index_name for result in results}
        assert "alpha" in found_indexes
        assert "beta" in found_indexes

    def test_load_fts5_index_supports_legacy_content_source_schema(self, tmp_path):
        from core.federated_search import FederatedSearch

        legacy = tmp_path / "research_papers.fts5.db"
        self._create_legacy_fts5_db(
            legacy,
            "# Absolute Zero Reasoner\n\nAZR demonstrates reasoning gains with zero labels.",
        )

        legacy_index = _load_fts5_index(str(legacy), "research_papers")
        fed = FederatedSearch(embedding_engine=None, max_workers=2)
        fed.add_index("research_papers", legacy_index)
        try:
            results = fed.search("absolute zero reasoning", top_k=5)
        finally:
            fed.close()
            legacy_index.close()

        assert legacy_index._fts5_error_logged is False
        assert len(results) == 1
        assert results[0].index_name == "research_papers"
        assert "Absolute Zero Reasoner" in results[0].content
        assert results[0].source == "research:absolute_zero_reasoner.md"
        assert results[0].metadata["id"].startswith("research:absolute_zero_reasoner.md:")
