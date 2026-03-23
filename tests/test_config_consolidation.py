"""Tests for R17: Config consolidation and entry point polish.

Verifies configs are consistent, no dead references, and env vars work.
"""

from __future__ import annotations

import os
import pytest
from pathlib import Path
from unittest.mock import patch

import yaml


_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_DIR = _ROOT / "config"


# ---------------------------------------------------------------------------
# Config file existence
# ---------------------------------------------------------------------------

class TestConfigFilesExist:

    @pytest.mark.parametrize("filename", [
        "agent.yaml",
        "memory.yaml",
        "models.yaml",
        "ports.yaml",
        "policies.yaml",
    ])
    def test_config_exists(self, filename):
        assert (_CONFIG_DIR / filename).is_file(), f"Missing config: {filename}"


# ---------------------------------------------------------------------------
# No hardcoded D: drive in runtime configs
# ---------------------------------------------------------------------------

class TestNoHardcodedPaths:

    def test_memory_yaml_no_hardcoded_d_drive(self):
        content = (_CONFIG_DIR / "memory.yaml").read_text(encoding="utf-8")
        cfg = yaml.safe_load(content)
        fed = cfg.get("federated_search", {})
        data_dir = fed.get("data_dir", "")
        # Should use env var reference, not bare D: path
        assert "D:" not in data_dir or "${" in data_dir, \
            f"Hardcoded D: path in memory.yaml: {data_dir}"

    def test_agent_yaml_no_hardcoded_d_drive(self):
        content = (_CONFIG_DIR / "agent.yaml").read_text(encoding="utf-8")
        # agent.yaml shouldn't contain D: drive paths
        assert "D:\\" not in content and "D:/" not in content, \
            "Hardcoded D: path found in agent.yaml"


# ---------------------------------------------------------------------------
# Embedder config consistency
# ---------------------------------------------------------------------------

class TestEmbedderConsistency:

    def test_models_yaml_has_embedder(self):
        cfg = yaml.safe_load((_CONFIG_DIR / "models.yaml").read_text(encoding="utf-8"))
        embedder = cfg.get("embedder", {})
        assert "name" in embedder
        assert "dimension" in embedder
        assert embedder["dimension"] == 768

    def test_memory_yaml_embedder_matches_models(self):
        models_cfg = yaml.safe_load((_CONFIG_DIR / "models.yaml").read_text(encoding="utf-8"))
        memory_cfg = yaml.safe_load((_CONFIG_DIR / "memory.yaml").read_text(encoding="utf-8"))

        models_dim = models_cfg.get("embedder", {}).get("dimension", 0)
        memory_dim = memory_cfg.get("embedder", {}).get("dimension", 0)
        assert models_dim == memory_dim, \
            f"Dimension mismatch: models.yaml={models_dim}, memory.yaml={memory_dim}"

    def test_nomic_embed_text_configured(self):
        """Embedder should be nomic-embed-text (available in Ollama)."""
        cfg = yaml.safe_load((_CONFIG_DIR / "models.yaml").read_text(encoding="utf-8"))
        name = cfg.get("embedder", {}).get("name", "")
        assert "nomic-embed-text" in name, f"Expected nomic-embed-text, got {name}"


# ---------------------------------------------------------------------------
# CLI entry points
# ---------------------------------------------------------------------------

class TestCLIEntryPoints:

    def test_main_py_exists(self):
        assert (_ROOT / "main.py").is_file()

    def test_cli_importable(self):
        from cli.commands import cli
        assert cli is not None

    def test_doctor_cmd_importable(self):
        from cli.doctor_cmd import doctor_cmd
        assert doctor_cmd is not None

    def test_cli_has_ask_command(self):
        from cli.commands import cli
        assert "ask" in [c.name for c in cli.commands.values()]

    def test_cli_has_doctor_command(self):
        from cli.commands import cli
        assert "doctor" in [c.name for c in cli.commands.values()]


# ---------------------------------------------------------------------------
# Agent config
# ---------------------------------------------------------------------------

class TestAgentConfig:

    def test_open_knowledge_fallback_configured(self):
        """R13: open_knowledge_fallback should be in agent.yaml."""
        cfg = yaml.safe_load((_CONFIG_DIR / "agent.yaml").read_text(encoding="utf-8"))
        retrieval = cfg.get("agent", {}).get("retrieval", {})
        assert "open_knowledge_fallback" in retrieval

    def test_self_learning_enabled(self):
        cfg = yaml.safe_load((_CONFIG_DIR / "agent.yaml").read_text(encoding="utf-8"))
        sl = cfg.get("agent", {}).get("self_learning", {})
        assert sl.get("pipeline_enabled") is True

    def test_cascade_enabled(self):
        cfg = yaml.safe_load((_CONFIG_DIR / "agent.yaml").read_text(encoding="utf-8"))
        cascade = cfg.get("agent", {}).get("cascade", {})
        assert cascade.get("cascade_enabled") is True
