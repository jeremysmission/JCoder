"""Unit tests for core/embedding_engine.py (R18).

Tests EmbeddingEngine, DualEmbeddingEngine, and detect_content_type
without a live vLLM server.
"""

from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from core.embedding_engine import (
    EmbeddingEngine,
    DualEmbeddingEngine,
    detect_content_type,
)
from core.config import ModelConfig


@pytest.fixture(autouse=True)
def _force_mocked_embedding_path(monkeypatch):
    """Keep unit tests off direct CUDA/ONNX paths for deterministic behavior."""
    monkeypatch.setenv("JCODER_FORCE_OLLAMA_EMBED", "1")
    monkeypatch.delenv("JCODER_EMBED_ONNX", raising=False)


def _model_config(**kwargs):
    defaults = {
        "name": "nomic-embed-text",
        "endpoint": "http://localhost:8000/v1",
        "dimension": 768,
    }
    defaults.update(kwargs)
    return ModelConfig(**defaults)


def _mock_embed_response(dim=768, n=1):
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        "data": [{"embedding": [0.1] * dim, "index": i} for i in range(n)]
    }
    resp.raise_for_status = MagicMock()
    return resp


def _context_length_error():
    request = __import__("httpx").Request("POST", "http://localhost:8000/v1/embeddings")
    response = __import__("httpx").Response(
        400,
        request=request,
        json={
            "error": {
                "message": "the input length exceeds the context length",
                "type": "invalid_request_error",
            }
        },
    )
    return __import__("httpx").HTTPStatusError(
        "400 context length",
        request=request,
        response=response,
    )


# ---------------------------------------------------------------------------
# detect_content_type
# ---------------------------------------------------------------------------

class TestDetectContentType:

    def test_code_with_def(self):
        assert detect_content_type("def foo():\n    return 42") == "code"

    def test_code_with_import(self):
        assert detect_content_type("import os\nprint(os.getcwd())") == "code"

    def test_code_with_class(self):
        assert detect_content_type("class Foo:\n    def bar(self):\n        pass") == "code"

    def test_text_plain_english(self):
        assert detect_content_type("How to sort a list in Python?") == "text"

    def test_text_long_prose(self):
        assert detect_content_type("The weather is nice today and I enjoy programming.") == "text"

    def test_empty_string(self):
        result = detect_content_type("")
        assert result in ("code", "text")


# ---------------------------------------------------------------------------
# EmbeddingEngine
# ---------------------------------------------------------------------------

class TestEmbeddingEngine:

    def test_init_stores_config(self):
        eng = EmbeddingEngine(config=_model_config())
        assert eng.model_name == "nomic-embed-text"
        assert eng.dimension == 768

    def test_embed_single(self):
        eng = EmbeddingEngine(config=_model_config())
        eng._client = MagicMock()
        eng._client.post.return_value = _mock_embed_response()
        vec = eng.embed_single("test")
        assert isinstance(vec, np.ndarray)
        assert vec.shape == (768,)

    def test_embed_batch(self):
        eng = EmbeddingEngine(config=_model_config())
        eng._client = MagicMock()
        eng._client.post.return_value = _mock_embed_response(n=3)
        vecs = eng.embed(["a", "b", "c"])
        assert vecs.shape == (3, 768)

    def test_close_is_safe(self):
        eng = EmbeddingEngine(config=_model_config())
        eng.close()

    def test_embed_batch_falls_back_to_individual_on_context_error(self):
        eng = EmbeddingEngine(config=_model_config())
        eng._use_direct_cuda = False
        eng._client = MagicMock()

        def _post(_url, json):
            inputs = json["input"]
            if len(inputs) > 1:
                raise _context_length_error()
            if len(inputs[0]) > 3500:
                raise _context_length_error()
            return _mock_embed_response(n=1)

        eng._client.post.side_effect = _post

        vecs = eng.embed(["short", "x" * 4000, "tiny"])
        assert vecs.shape == (3, 768)
        assert eng._client.post.call_count >= 5

    def test_embed_single_truncates_on_context_error(self):
        eng = EmbeddingEngine(config=_model_config())
        eng._use_direct_cuda = False
        eng._client = MagicMock()

        def _post(_url, json):
            text = json["input"][0]
            if len(text) > 3500:
                raise _context_length_error()
            return _mock_embed_response()

        eng._client.post.side_effect = _post

        vecs = eng.embed(["x" * 4000])
        assert vecs.shape == (1, 768)
        assert eng._client.post.call_count == 3


# ---------------------------------------------------------------------------
# DualEmbeddingEngine
# ---------------------------------------------------------------------------

class TestDualEmbeddingEngine:

    @patch("core.embedding_engine.DualEmbeddingEngine._probe", return_value=False)
    def test_creates_two_engines(self, _):
        cfg = _model_config(code_model="nomic-embed-code", text_model="nomic-embed-text")
        eng = DualEmbeddingEngine(config=cfg)
        assert eng._code_engine.model_name == "nomic-embed-code"
        assert eng._text_engine.model_name == "nomic-embed-text"

    @patch("core.embedding_engine.DualEmbeddingEngine._probe", return_value=False)
    def test_dimension_from_config(self, _):
        eng = DualEmbeddingEngine(config=_model_config(dimension=512))
        assert eng.dimension == 512

    @patch("core.embedding_engine.DualEmbeddingEngine._probe", return_value=False)
    def test_fallback_to_primary_name(self, _):
        eng = DualEmbeddingEngine(config=_model_config())
        assert eng._code_engine.model_name == "nomic-embed-text"

    @patch("core.embedding_engine.DualEmbeddingEngine._probe", return_value=False)
    def test_select_engine_code(self, _):
        cfg = _model_config(code_model="nomic-embed-code")
        eng = DualEmbeddingEngine(config=cfg)
        eng._code_ok = True
        selected = eng._select_engine("code")
        assert selected.model_name == "nomic-embed-code"

    @patch("core.embedding_engine.DualEmbeddingEngine._probe", return_value=False)
    def test_select_engine_text(self, _):
        cfg = _model_config(text_model="nomic-embed-text")
        eng = DualEmbeddingEngine(config=cfg)
        eng._text_ok = True
        selected = eng._select_engine("text")
        assert selected.model_name == "nomic-embed-text"

    @patch("core.embedding_engine.DualEmbeddingEngine._probe", return_value=False)
    def test_close_is_safe(self, _):
        eng = DualEmbeddingEngine(config=_model_config())
        eng.close()
