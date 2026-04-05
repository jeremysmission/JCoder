"""
Embedding Engine
----------------
Converts text into numerical vectors by calling the vLLM embedding server.

Non-programmer explanation:
Think of this like turning text into a fingerprint.
Similar code snippets get similar fingerprints.
We send text to a local AI server (vLLM) that returns
a list of numbers representing the meaning of that text.

DualEmbeddingEngine wraps two EmbeddingEngine instances -- one tuned
for source code (nomic-embed-code) and one for documentation
(nomic-embed-text).  When only one model is available, it degrades
gracefully and uses a single model for everything.
"""

import logging
import os
from typing import List, Optional

import httpx
import numpy as np

from .config import ModelConfig
from .http_factory import make_client
from .embedding_support import detect_content_type, pack_token_budget_batches
from .network_gate import NetworkGate
log = logging.getLogger(__name__)


class EmbeddingEngine:
    """
    Responsible ONLY for converting text into vectors.

    Tries direct CUDA via sentence-transformers first (45x faster).
    Falls back to Ollama/vLLM OpenAI-compatible /v1/embeddings endpoint.

    No indexing logic. No retrieval logic. Just embedding.
    """

    # Map Ollama model names to HuggingFace IDs for direct CUDA loading.
    # Same mapping used by HybridRAG3_Educational (vetted + QA'd).
    _OLLAMA_TO_HF = {
        "nomic-embed-text": "nomic-ai/nomic-embed-text-v1.5",
        "nomic-embed-text:latest": "nomic-ai/nomic-embed-text-v1.5",
        "nomic-embed-text-v2-moe": "nomic-ai/nomic-embed-text-v2-moe",
        "nomic-embed-text-v2-moe:latest": "nomic-ai/nomic-embed-text-v2-moe",
    }

    def __init__(self, config: ModelConfig, timeout: int = 120,
                 gate: NetworkGate = None):
        self.endpoint = config.endpoint.rstrip("/")
        self.model_name = config.name
        self.dimension = config.dimension or 768
        self._gate = gate
        self._direct_model = None
        self._use_direct_cuda = False
        self._use_onnx_cpu = False
        self._cuda_batch_backoff = None

        # Tier 1: Direct CUDA (45x faster than Ollama HTTP)
        if os.getenv("JCODER_FORCE_OLLAMA_EMBED", "").lower() not in ("1", "true"):
            self._try_init_direct_cuda()

        # Tier 2: ONNX CPU (3-12x faster than Ollama HTTP, no GPU needed)
        if not self._use_direct_cuda:
            if os.getenv("JCODER_EMBED_ONNX", "").lower() in ("1", "true"):
                self._try_init_onnx_cpu()

        # Tier 3: Ollama HTTP fallback
        self._client = make_client(timeout_s=timeout)

        # Ollama bug #6262: batch embedding quality degrades when
        # OLLAMA_NUM_PARALLEL > 1. Set to 1 for embedding jobs.
        if not self._use_direct_cuda and not self._use_onnx_cpu:
            if "OLLAMA_NUM_PARALLEL" not in os.environ:
                os.environ["OLLAMA_NUM_PARALLEL"] = "1"

    def _try_init_direct_cuda(self) -> None:
        """Try to load embedding model directly on GPU via sentence-transformers.

        Bypasses Ollama HTTP layer entirely, eliminating ~150ms per-request
        overhead. If torch or sentence-transformers is not installed, or no
        CUDA GPU is available, silently falls back to Ollama HTTP path.
        Same pattern as HybridRAG3_Educational embedder (vetted).
        """
        try:
            import torch
            if not torch.cuda.is_available():
                log.debug("direct_cuda_skip: no CUDA GPU available")
                return

            from sentence_transformers import SentenceTransformer

            hf_model = self._OLLAMA_TO_HF.get(self.model_name)
            if not hf_model:
                log.debug("direct_cuda_skip: no HF mapping for %s", self.model_name)
                return

            # Pin to CUDA:1 per JCoder GPU assignment (CUDA:0 = HybridRAG)
            cuda_device = os.getenv("JCODER_EMBED_DEVICE", "cuda:1")
            self._direct_model = SentenceTransformer(
                hf_model, device=cuda_device, trust_remote_code=True,
            )
            self._use_direct_cuda = True
            log.info(
                "Direct CUDA embedding ready: model=%s device=%s gpu=%s",
                hf_model, cuda_device, torch.cuda.get_device_name(
                    int(cuda_device.split(":")[-1]) if ":" in cuda_device else 0
                ),
            )
        except ImportError:
            log.debug("direct_cuda_skip: torch or sentence-transformers not installed")
        except Exception as exc:
            log.warning("direct_cuda_init_failed: %s", exc)

    def _try_init_onnx_cpu(self) -> None:
        """Try ONNX CPU backend for embedding (3-12x faster than Ollama HTTP).

        Ported from HybridRAG3 (commit df8563e). Uses sentence-transformers
        with backend="onnx" and INT8 quantization. Requires onnxruntime and
        optimum to be installed (never via sentence-transformers[onnx] extra).
        """
        try:
            from sentence_transformers import SentenceTransformer

            hf_model = self._OLLAMA_TO_HF.get(self.model_name)
            if not hf_model:
                log.debug("onnx_cpu_skip: no HF mapping for %s", self.model_name)
                return

            self._direct_model = SentenceTransformer(
                hf_model, backend="onnx", trust_remote_code=True,
                model_kwargs={"provider": "CPUExecutionProvider"},
            )
            self._use_onnx_cpu = True
            log.info("ONNX CPU embedding ready: model=%s", hf_model)
        except ImportError:
            log.debug("onnx_cpu_skip: onnxruntime or sentence-transformers not installed")
        except Exception as exc:
            log.warning("onnx_cpu_init_failed: %s", exc)

    def _post_embeddings(self, texts: List[str]) -> np.ndarray:
        """Call the embedding endpoint once and normalize the returned vectors."""
        url = f"{self.endpoint}/embeddings"
        if self._gate:
            self._gate.guard(url)
        response = self._client.post(
            url,
            json={
                "model": self.model_name,
                "input": texts,
            },
        )
        response.raise_for_status()

        body = response.json()
        if not isinstance(body, dict) or "data" not in body:
            raise ValueError(
                f"Embedding response missing 'data' key; got keys: "
                f"{list(body.keys()) if isinstance(body, dict) else type(body).__name__}"
            )
        data = body["data"]
        vectors = [item["embedding"] for item in sorted(data, key=lambda x: x["index"])]
        result = np.array(vectors, dtype=np.float32)

        norms = np.linalg.norm(result, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return result / norms

    @staticmethod
    def _is_context_length_error(exc: Exception) -> bool:
        """Return True for Ollama/vLLM 400s caused by oversized embedding input."""
        if not isinstance(exc, httpx.HTTPStatusError):
            return False
        response = exc.response
        if response is None or response.status_code != 400:
            return False
        body = ""
        try:
            body = response.text.lower()
        except Exception:
            return False
        return "context length" in body and "input length" in body

    def _embed_single_with_fallback(self, text: str) -> np.ndarray:
        """Retry a single oversized chunk with progressively shorter prefixes."""
        last_exc: Exception | None = None
        try:
            return self._post_embeddings([text])[0]
        except Exception as exc:
            if not self._is_context_length_error(exc):
                raise
            last_exc = exc
            log.warning(
                "Embedding input exceeded context window; retrying with truncation",
            )

        cuts = [3500, 3000, 2500, 2000, 1500, 1000, 512]
        for cut in cuts:
            if len(text) <= cut:
                continue
            try:
                return self._post_embeddings([text[:cut]])[0]
            except Exception as exc:
                if not self._is_context_length_error(exc):
                    raise
                last_exc = exc

        if last_exc is not None:
            raise last_exc
        raise RuntimeError("Embedding truncation fallback failed without an exception")

    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Convert multiple text chunks into vector form.
        Returns an (N, dimension) numpy array of normalized embeddings.

        Uses token-budget dynamic batching and OOM backoff when on
        direct CUDA path (ported from HybridRAG3).
        """
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        # --- Tier 1: Direct CUDA with token-budget batching ---
        if self._use_direct_cuda and self._direct_model is not None:
            return self._direct_cuda_encode(texts)

        # --- Tier 2: ONNX CPU (3-12x faster than Ollama HTTP) ---
        if self._use_onnx_cpu and self._direct_model is not None:
            return self._onnx_cpu_encode(texts)

        # --- Tier 3: Ollama HTTP ---
        try:
            return self._post_embeddings(texts)
        except Exception as exc:
            if not self._is_context_length_error(exc):
                raise

        log.warning(
            "Embedding batch exceeded context window; retrying items individually",
        )
        vectors = [self._embed_single_with_fallback(text) for text in texts]
        return np.vstack(vectors)

    def _direct_cuda_encode(self, texts: List[str]) -> np.ndarray:
        """Encode via direct CUDA with token-budget batching and OOM backoff.

        Ported from HybridRAG3 (commits 0750633, afe7877). Packs texts
        into variable-size batches to maximize GPU utilization on short
        sequences while preventing OOM on long ones.
        """
        max_batch = max(32, int(os.getenv("JCODER_EMBED_BATCH", "256")))
        token_budget = int(os.getenv("JCODER_TOKEN_BUDGET", "49152"))
        batches = pack_token_budget_batches(texts, token_budget, max_batch)

        all_results = []
        for start, end in batches:
            batch_texts = texts[start:end]
            batch_size = max_batch
            while True:
                try:
                    result = self._direct_model.encode(
                        batch_texts,
                        batch_size=batch_size,
                        show_progress_bar=len(batch_texts) > 1000,
                        convert_to_numpy=True,
                        normalize_embeddings=True,
                    )
                    all_results.append(np.asarray(result, dtype=np.float32))
                    if batch_size != max_batch:
                        self._cuda_batch_backoff = batch_size
                    break
                except RuntimeError as exc:
                    message = str(exc).lower()
                    if "out of memory" not in message or batch_size <= 32:
                        raise
                    next_batch_size = max(32, batch_size // 2)
                    log.warning(
                        "CUDA OOM backoff: %d -> %d", batch_size, next_batch_size,
                    )
                    try:
                        import torch
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                    if next_batch_size == batch_size:
                        raise
                    batch_size = next_batch_size

        return np.vstack(all_results) if all_results else np.empty(
            (0, self.dimension), dtype=np.float32,
        )

    def _onnx_cpu_encode(self, texts: List[str]) -> np.ndarray:
        """Encode via ONNX CPU backend with token-budget batching.

        Ported from HybridRAG3 (commit df8563e). 3-12x faster than
        Ollama HTTP on CPU-only machines.
        """
        max_batch = int(os.getenv("JCODER_EMBED_BATCH", "256"))
        token_budget = int(os.getenv("JCODER_TOKEN_BUDGET", "49152"))
        batches = pack_token_budget_batches(texts, token_budget, max_batch)

        all_results = []
        for start, end in batches:
            batch_texts = texts[start:end]
            result = self._direct_model.encode(
                batch_texts,
                batch_size=max_batch,
                show_progress_bar=len(batch_texts) > 1000,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            all_results.append(np.asarray(result, dtype=np.float32))

        return np.vstack(all_results) if all_results else np.empty(
            (0, self.dimension), dtype=np.float32,
        )

    def embed_single(self, text: str) -> np.ndarray:
        """Convert a single query into a vector."""
        return self.embed([text])[0]

    def close(self):
        """Release HTTP connection pool."""
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# -----------------------------------------------------------------------
# Dual Embedding Engine
# -----------------------------------------------------------------------

class DualEmbeddingEngine:
    """Routes embedding requests to a code-specific or text-specific model.

    Satisfies the same IEmbedder protocol as EmbeddingEngine so it is a
    drop-in replacement everywhere the pipeline expects an embedder.

    Graceful degradation:
      - If both models are available, routes by content_type.
      - If only one model responds, uses it for everything.
      - If neither responds, raises on first call (same as EmbeddingEngine).
    """

    def __init__(
        self,
        config: ModelConfig,
        timeout: int = 120,
        gate: Optional[NetworkGate] = None,
    ):
        self.dimension = config.dimension or 768

        # Build per-model configs. Fall back to the primary model name
        # when a specialized model name is not configured.
        code_name = config.code_model or config.name
        text_name = config.text_model or config.name

        code_cfg = ModelConfig(
            name=code_name,
            endpoint=config.endpoint,
            dimension=config.dimension,
        )
        text_cfg = ModelConfig(
            name=text_name,
            endpoint=config.endpoint,
            dimension=config.dimension,
        )

        self._code_engine = EmbeddingEngine(code_cfg, timeout=timeout, gate=gate)
        self._text_engine = EmbeddingEngine(text_cfg, timeout=timeout, gate=gate)

        # Probe availability -- set flags so we know what works.
        self._code_ok = self._probe(self._code_engine)
        self._text_ok = self._probe(self._text_engine)

        # If neither model responds, keep both engines alive so the
        # first real call surfaces the actual HTTP error to the caller.

    # -- probing --------------------------------------------------------

    @staticmethod
    def _probe(engine: EmbeddingEngine) -> bool:
        """Send a tiny request to check model availability."""
        try:
            engine.embed(["probe"])
            return True
        except Exception:
            log.debug("Embedding engine probe failed", exc_info=True)
            return False

    # -- routing --------------------------------------------------------

    def _select_engine(self, content_type: str) -> EmbeddingEngine:
        """Pick the best available engine for the requested content type."""
        if content_type == "code":
            if self._code_ok:
                return self._code_engine
            if self._text_ok:
                return self._text_engine
            # Neither confirmed -- let the code engine try (surfaces error)
            return self._code_engine

        if content_type == "text":
            if self._text_ok:
                return self._text_engine
            if self._code_ok:
                return self._code_engine
            return self._text_engine

        # content_type == "auto" -- should not reach here, resolved earlier
        return self._code_engine if self._code_ok else self._text_engine

    def _resolve_type(self, text: str, content_type: str) -> str:
        """Resolve 'auto' to 'code' or 'text'."""
        if content_type in ("code", "text"):
            return content_type
        return detect_content_type(text)

    # -- public API (IEmbedder compatible) ------------------------------

    def embed(
        self,
        texts: List[str],
        content_type: str = "auto",
    ) -> np.ndarray:
        """Embed multiple texts, routing each to the appropriate model.

        Parameters
        ----------
        texts : list[str]
            Chunks to embed.
        content_type : str
            'code', 'text', or 'auto' (per-chunk detection).
        """
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        # Fast path: all same type -- single batch call
        if content_type in ("code", "text"):
            engine = self._select_engine(content_type)
            return engine.embed(texts)

        # Auto mode: partition, embed separately, reassemble in order
        code_idxs: List[int] = []
        text_idxs: List[int] = []
        for i, t in enumerate(texts):
            if detect_content_type(t) == "code":
                code_idxs.append(i)
            else:
                text_idxs.append(i)

        result = np.empty((len(texts), self.dimension), dtype=np.float32)

        if code_idxs:
            code_texts = [texts[i] for i in code_idxs]
            code_vecs = self._select_engine("code").embed(code_texts)
            for pos, idx in enumerate(code_idxs):
                result[idx] = code_vecs[pos]

        if text_idxs:
            text_texts = [texts[i] for i in text_idxs]
            text_vecs = self._select_engine("text").embed(text_texts)
            for pos, idx in enumerate(text_idxs):
                result[idx] = text_vecs[pos]

        return result

    def embed_single(self, text: str, content_type: str = "auto") -> np.ndarray:
        """Embed a single text with content-type routing."""
        resolved = self._resolve_type(text, content_type)
        engine = self._select_engine(resolved)
        return engine.embed_single(text)

    # -- lifecycle ------------------------------------------------------

    def close(self):
        """Release both HTTP connection pools."""
        self._code_engine.close()
        self._text_engine.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
