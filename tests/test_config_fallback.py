"""Config loader must produce valid defaults when files are missing."""

import os
import tempfile

from core.config import load_config, JCoderConfig


def test_load_config_with_empty_dir():
    """Config from empty directory should return all defaults."""
    with tempfile.TemporaryDirectory() as d:
        config = load_config(config_dir=d)

    assert isinstance(config, JCoderConfig)
    assert config.llm.name == ""  # No models.yaml -> empty name
    assert config.policies.max_parallel_requests == 4  # Default policy


def test_load_config_with_partial_files(tmp_path):
    """Config with only ports.yaml should still load without error."""
    ports = tmp_path / "ports.yaml"
    ports.write_text("llm: 9000\nembedder: 9001\n", encoding="utf-8")

    config = load_config(config_dir=str(tmp_path))

    assert isinstance(config, JCoderConfig)
    assert "9000" in config.llm.endpoint
    assert "9001" in config.embedder.endpoint


def test_load_config_policies_override(tmp_path):
    """Policies from file should override defaults."""
    policies = tmp_path / "policies.yaml"
    policies.write_text(
        "concurrency:\n  max_parallel_requests: 8\n",
        encoding="utf-8",
    )

    config = load_config(config_dir=str(tmp_path))
    assert config.policies.max_parallel_requests == 8


def test_ms_to_seconds_minimum():
    """_ms_to_seconds must never return 0."""
    from core.config import _ms_to_seconds

    assert _ms_to_seconds(1) == 1
    assert _ms_to_seconds(999) == 1
    assert _ms_to_seconds(1000) == 1
    assert _ms_to_seconds(1001) == 2
    assert _ms_to_seconds(0) == 1  # Edge case: 0ms -> 1s minimum
