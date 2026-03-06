"""
Tests for agent.config_loader
-----------------------------
Covers: AgentConfig defaults, YAML loading, config directory resolution,
backend defaults, full load_agent_config pipeline, env var overrides,
and build_agent_from_config (mocked subsystems).
"""

from __future__ import annotations

import os
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from agent.config_loader import (
    AgentConfig,
    _find_config_dir,
    _load_yaml,
    _resolve_backend_defaults,
    build_agent_from_config,
    load_agent_config,
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
        assert model == "devstral-small-2:24b"
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
        assert model == "devstral-small-2:24b"
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
        assert cfg.model == "devstral-small-2:24b"
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
