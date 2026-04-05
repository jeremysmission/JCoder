"""
Tests for DualEmbeddingEngine
-----------------------------
Covers: auto-detection routing, explicit content_type, single-model
fallback (graceful degradation), dimension consistency, edge cases.
"""

from __future__ import annotations

from typing import List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from core.config import ModelConfig
from core.embedding_engine import (
    DualEmbeddingEngine,
    EmbeddingEngine,
    detect_content_type,
)


@pytest.fixture(autouse=True)
def _force_mocked_embedding_path(monkeypatch):
    """Keep these tests on the mocked HTTP path instead of live CUDA backends."""
    monkeypatch.setenv("JCODER_FORCE_OLLAMA_EMBED", "1")
    monkeypatch.delenv("JCODER_EMBED_ONNX", raising=False)


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def _fake_embed(model_name: str, dimension: int = 768):
    """Return a closure that produces deterministic vectors seeded by model_name."""
    seed = hash(model_name) & 0xFFFFFFFF

    def _embed(texts: List[str]) -> np.ndarray:
        rng = np.random.RandomState(seed)
        vecs = rng.randn(len(texts), dimension).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vecs / norms

    return _embed


def _make_config(
    code_model: str = "nomic-embed-code",
    text_model: str = "nomic-embed-text",
    dimension: int = 768,
) -> ModelConfig:
    return ModelConfig(
        name="nomic-embed-code",
        endpoint="http://localhost:8001/v1",
        dimension=dimension,
        code_model=code_model,
        text_model=text_model,
    )


# -------------------------------------------------------------------
# 1. detect_content_type heuristic
# -------------------------------------------------------------------


class TestDetectContentType:

    def test_python_function_detected_as_code(self):
        snippet = (
            "def calculate_sum(a, b):\n"
            "    return a + b\n"
        )
        assert detect_content_type(snippet) == "code"

    def test_javascript_detected_as_code(self):
        snippet = (
            "const greet = (name) => {\n"
            "    console.log(`Hello ${name}`);\n"
            "};\n"
        )
        assert detect_content_type(snippet) == "code"

    def test_class_with_imports_detected_as_code(self):
        snippet = (
            "import os\n"
            "import sys\n"
            "\n"
            "class Foo:\n"
            "    def bar(self):\n"
            "        return 42\n"
        )
        assert detect_content_type(snippet) == "code"

    def test_plain_english_detected_as_text(self):
        snippet = (
            "This document describes the architecture of the system.\n"
            "It consists of three main components that work together.\n"
            "The first component handles user input validation.\n"
        )
        assert detect_content_type(snippet) == "text"

    def test_empty_string_returns_text(self):
        assert detect_content_type("") == "text"

    def test_markdown_docs_detected_as_text(self):
        snippet = (
            "# Architecture Overview\n"
            "\n"
            "The retrieval engine combines vector search with keyword matching.\n"
            "Results are fused using Reciprocal Rank Fusion (RRF).\n"
            "\n"
            "## Components\n"
            "\n"
            "- Embedding engine: converts text to vectors\n"
            "- Index engine: stores and searches vectors\n"
        )
        assert detect_content_type(snippet) == "text"


# -------------------------------------------------------------------
# 2. DualEmbeddingEngine routing
# -------------------------------------------------------------------


class TestDualRouting:
    """Verify that code and text content route to the correct engine."""

    @pytest.fixture()
    def dual(self):
        """Build a DualEmbeddingEngine with mocked HTTP calls."""
        cfg = _make_config()

        # Patch make_client so no real HTTP happens
        with patch("core.embedding_engine.make_client") as mock_client_factory:
            mock_http = MagicMock()
            mock_client_factory.return_value = mock_http

            # We need to track which model_name is in the request
            call_log = []

            def _mock_post(url, json=None):
                model = json["model"]
                texts = json["input"]
                call_log.append(model)
                dim = cfg.dimension or 768
                embeddings = [
                    {"index": i, "embedding": list(np.random.randn(dim).astype(float))}
                    for i in range(len(texts))
                ]
                resp = MagicMock()
                resp.json.return_value = {"data": embeddings}
                resp.raise_for_status = MagicMock()
                return resp

            mock_http.post = _mock_post
            mock_http.close = MagicMock()

            engine = DualEmbeddingEngine(cfg)
            engine._call_log = call_log
            yield engine
            engine.close()

    def test_explicit_code_routes_to_code_model(self, dual):
        dual.embed(["def foo(): pass"], content_type="code")
        # The last real embed call should use the code model
        assert "nomic-embed-code" in dual._call_log

    def test_explicit_text_routes_to_text_model(self, dual):
        dual.embed(["This is documentation."], content_type="text")
        assert "nomic-embed-text" in dual._call_log

    def test_auto_routes_code_snippet(self, dual):
        dual._call_log.clear()
        snippet = "import os\ndef main():\n    return os.getcwd()\n"
        dual.embed([snippet], content_type="auto")
        assert "nomic-embed-code" in dual._call_log

    def test_auto_routes_doc_paragraph(self, dual):
        dual._call_log.clear()
        doc = (
            "The system architecture is based on microservices.\n"
            "Each service communicates over REST endpoints.\n"
            "Deployments are managed through containers.\n"
        )
        dual.embed([doc], content_type="auto")
        assert "nomic-embed-text" in dual._call_log


# -------------------------------------------------------------------
# 3. Graceful degradation (single-model fallback)
# -------------------------------------------------------------------


class TestGracefulDegradation:

    def _build_with_failures(self, code_fails: bool, text_fails: bool):
        """Create a DualEmbeddingEngine where probe calls may fail."""
        cfg = _make_config()
        dim = cfg.dimension or 768

        call_count = {"n": 0}

        with patch("core.embedding_engine.make_client") as mock_factory:
            mock_http = MagicMock()
            mock_factory.return_value = mock_http

            def _mock_post(url, json=None):
                model = json["model"]
                texts = json["input"]

                if code_fails and model == "nomic-embed-code":
                    raise ConnectionError("code model down")
                if text_fails and model == "nomic-embed-text":
                    raise ConnectionError("text model down")

                call_count["n"] += 1
                embeddings = [
                    {"index": i, "embedding": list(np.random.randn(dim).astype(float))}
                    for i in range(len(texts))
                ]
                resp = MagicMock()
                resp.json.return_value = {"data": embeddings}
                resp.raise_for_status = MagicMock()
                return resp

            mock_http.post = _mock_post
            mock_http.close = MagicMock()

            engine = DualEmbeddingEngine(cfg)
            return engine

    def test_code_model_down_falls_back_to_text(self):
        engine = self._build_with_failures(code_fails=True, text_fails=False)
        assert engine._code_ok is False
        assert engine._text_ok is True
        # Requesting code content should still work via text model
        result = engine.embed_single("def foo(): pass", content_type="code")
        assert result.shape == (768,)
        engine.close()

    def test_text_model_down_falls_back_to_code(self):
        engine = self._build_with_failures(code_fails=False, text_fails=True)
        assert engine._code_ok is True
        assert engine._text_ok is False
        result = engine.embed_single("This is plain text.", content_type="text")
        assert result.shape == (768,)
        engine.close()

    def test_both_models_down_raises_on_embed(self):
        engine = self._build_with_failures(code_fails=True, text_fails=True)
        assert engine._code_ok is False
        assert engine._text_ok is False
        with pytest.raises(ConnectionError):
            engine.embed(["anything"])
        engine.close()


# -------------------------------------------------------------------
# 4. Dimension consistency
# -------------------------------------------------------------------


class TestDimensionConsistency:

    def test_output_matches_configured_dimension(self):
        dim = 768
        cfg = _make_config(dimension=dim)

        with patch("core.embedding_engine.make_client") as mock_factory:
            mock_http = MagicMock()
            mock_factory.return_value = mock_http

            def _mock_post(url, json=None):
                texts = json["input"]
                embeddings = [
                    {"index": i, "embedding": list(np.random.randn(dim).astype(float))}
                    for i in range(len(texts))
                ]
                resp = MagicMock()
                resp.json.return_value = {"data": embeddings}
                resp.raise_for_status = MagicMock()
                return resp

            mock_http.post = _mock_post
            mock_http.close = MagicMock()

            engine = DualEmbeddingEngine(cfg)
            code_vec = engine.embed_single("def x(): pass", content_type="code")
            text_vec = engine.embed_single("Some documentation.", content_type="text")

            assert code_vec.shape == (dim,)
            assert text_vec.shape == (dim,)
            # Both must be unit-normalized (L2 norm ~ 1.0)
            assert abs(np.linalg.norm(code_vec) - 1.0) < 1e-5
            assert abs(np.linalg.norm(text_vec) - 1.0) < 1e-5
            engine.close()

    def test_empty_input_returns_empty_array(self):
        cfg = _make_config()

        with patch("core.embedding_engine.make_client") as mock_factory:
            mock_http = MagicMock()
            mock_factory.return_value = mock_http

            def _mock_post(url, json=None):
                resp = MagicMock()
                resp.json.return_value = {"data": []}
                resp.raise_for_status = MagicMock()
                return resp

            mock_http.post = _mock_post
            mock_http.close = MagicMock()

            engine = DualEmbeddingEngine(cfg)
            result = engine.embed([], content_type="auto")
            assert result.shape == (0, 768)
            engine.close()


# -------------------------------------------------------------------
# 5. Config backward compatibility
# -------------------------------------------------------------------


class TestConfigBackwardCompat:

    def test_no_code_text_model_uses_primary(self):
        """When code_model/text_model are None, both engines use primary name."""
        cfg = ModelConfig(
            name="nomic-embed-code",
            endpoint="http://localhost:8001/v1",
            dimension=768,
        )
        assert cfg.code_model is None
        assert cfg.text_model is None

        with patch("core.embedding_engine.make_client") as mock_factory:
            mock_http = MagicMock()
            mock_factory.return_value = mock_http

            used_models = []

            def _mock_post(url, json=None):
                used_models.append(json["model"])
                texts = json["input"]
                embeddings = [
                    {"index": i, "embedding": list(np.random.randn(768).astype(float))}
                    for i in range(len(texts))
                ]
                resp = MagicMock()
                resp.json.return_value = {"data": embeddings}
                resp.raise_for_status = MagicMock()
                return resp

            mock_http.post = _mock_post
            mock_http.close = MagicMock()

            engine = DualEmbeddingEngine(cfg)
            engine.embed(["def f(): pass"], content_type="code")
            engine.embed(["plain doc"], content_type="text")
            # Both calls should use the primary model name
            non_probe = [m for m in used_models if used_models.index(m) >= 2]
            for m in non_probe:
                assert m == "nomic-embed-code"
            engine.close()

    def test_existing_embedding_engine_unchanged(self):
        """EmbeddingEngine still works exactly as before."""
        cfg = ModelConfig(
            name="test-model",
            endpoint="http://localhost:8001/v1",
            dimension=768,
        )

        with patch("core.embedding_engine.make_client") as mock_factory:
            mock_http = MagicMock()
            mock_factory.return_value = mock_http

            def _mock_post(url, json=None):
                texts = json["input"]
                embeddings = [
                    {"index": i, "embedding": list(np.random.randn(768).astype(float))}
                    for i in range(len(texts))
                ]
                resp = MagicMock()
                resp.json.return_value = {"data": embeddings}
                resp.raise_for_status = MagicMock()
                return resp

            mock_http.post = _mock_post

            engine = EmbeddingEngine(cfg)
            result = engine.embed(["hello world"])
            assert result.shape == (1, 768)
            assert abs(np.linalg.norm(result[0]) - 1.0) < 1e-5
            engine.close()
