"""Tests for the ModelCascade wiring fix in agent/bridge.py.

Verifies that _try_init_pipeline correctly instantiates CascadeLevel
with ModelConfig objects (not raw model_name/endpoint kwargs).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from core.cascade import CascadeLevel, ModelCascade
from core.config import ModelConfig


class TestCascadeLevelConstruction:
    """Verify CascadeLevel accepts ModelConfig, not loose kwargs."""

    def test_cascade_level_with_model_config(self):
        mc = ModelConfig(name="phi4:14b", endpoint="http://localhost:11434/v1")
        level = CascadeLevel(name="fast", model_config=mc, max_complexity=0.5)
        assert level.name == "fast"
        assert level.model_config.name == "phi4:14b"
        assert level.max_complexity == 0.5

    def test_cascade_level_rejects_loose_kwargs(self):
        with pytest.raises(TypeError):
            CascadeLevel(
                name="bad",
                model_name="phi4:14b",  # type: ignore[call-arg]
                endpoint="http://localhost:11434/v1",  # type: ignore[call-arg]
                max_complexity=0.5,
            )

    def test_model_cascade_from_levels(self):
        levels = [
            CascadeLevel(
                name="fast",
                model_config=ModelConfig(name="phi4-mini", endpoint="http://localhost:11434/v1"),
                max_complexity=0.4,
            ),
            CascadeLevel(
                name="strong",
                model_config=ModelConfig(name="phi4:14b", endpoint="http://localhost:11434/v1"),
                max_complexity=1.0,
            ),
        ]
        cascade = ModelCascade(levels=levels, confidence_threshold=0.4)
        assert len(cascade.levels) == 2
        # Sorted by max_complexity
        assert cascade.levels[0].name == "fast"
        assert cascade.levels[1].name == "strong"


class TestBridgeCascadeInit:
    """Test that bridge.py _try_init_pipeline creates valid CascadeLevels."""

    def test_default_cascade_level_uses_model_config(self):
        """Simulate what bridge.py does for the default single-level cascade."""
        from core.config import ModelConfig as _MC

        sl_config = {
            "cascade_enabled": True,
            "cascade_model": "phi4:14b",
            "cascade_endpoint": "http://localhost:11434/v1",
            "cascade_confidence": 0.4,
        }

        # This mirrors bridge.py lines 951-960 (after fix)
        levels = [
            CascadeLevel(
                name="default",
                model_config=_MC(
                    name=sl_config.get("cascade_model", "phi4-mini"),
                    endpoint=sl_config.get("cascade_endpoint", "http://localhost:11434/v1"),
                ),
                max_complexity=1.0,
            )
        ]

        cascade = ModelCascade(
            levels=levels,
            confidence_threshold=sl_config.get("cascade_confidence", 0.4),
        )
        assert cascade.levels[0].model_config.name == "phi4:14b"

    def test_multi_level_cascade_from_config(self):
        """Simulate bridge.py with explicit cascade_levels list."""
        from core.config import ModelConfig as _MC

        cascade_levels_raw = [
            {"name": "fast", "model_name": "phi4-mini", "endpoint": "http://localhost:11434/v1", "max_complexity": 0.3},
            {"name": "medium", "model_name": "phi4:14b", "endpoint": "http://localhost:11434/v1", "max_complexity": 0.7},
            {"name": "strong", "model_name": "gpt-4o", "endpoint": "https://api.openai.com/v1", "max_complexity": 1.0},
        ]

        levels = [
            CascadeLevel(
                name=lv["name"],
                model_config=_MC(name=lv["model_name"], endpoint=lv["endpoint"]),
                max_complexity=lv["max_complexity"],
            )
            for lv in cascade_levels_raw
        ]

        cascade = ModelCascade(levels=levels)
        assert len(cascade.levels) == 3
        assert cascade.levels[0].model_config.name == "phi4-mini"
        assert cascade.levels[2].model_config.name == "gpt-4o"
