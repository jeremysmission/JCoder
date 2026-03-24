"""Configuration loader -- reads split YAML files, exposes typed dataclasses."""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """Settings for a single model served by vLLM."""
    name: str
    endpoint: str = ""
    dimension: Optional[int] = None
    enabled: bool = True
    quantization: str = ""
    tensor_parallel: int = 1
    max_model_len: int = 32768
    # Dual-embedding support: optional specialized models for code vs text.
    # When set, DualEmbeddingEngine uses these instead of the primary `name`.
    # If unset, the primary model handles both content types.
    code_model: Optional[str] = None
    text_model: Optional[str] = None


@dataclass
class RetrievalConfig:
    top_k: int = 50
    rerank_top_n: int = 10
    rrf_k: int = 60


@dataclass
class ChunkingConfig:
    max_chars: int = 4000
    strategy: str = "tree_sitter"


@dataclass
class StorageConfig:
    data_dir: str = "data"
    index_dir: str = "data/indexes"


@dataclass
class NetworkConfig:
    """Network access control policy."""
    mode: str = "localhost"
    allowlist: List[str] = field(default_factory=list)


@dataclass
class HarvesterConfig:
    """Research harvester settings."""
    out_dir: str = "_harvested"
    cache_dir: str = ".harvest_cache"
    user_agent: str = "JCoder-Harvester/1.0"
    timeout_s: int = 30
    max_bytes: int = 10_000_000
    min_interval_s: float = 1.0
    retries: int = 2
    backoff_s: float = 2.0
    github_repos: List[str] = field(default_factory=list)
    pypi_packages: List[str] = field(default_factory=list)
    hn_queries: List[str] = field(default_factory=list)
    arxiv_queries: List[str] = field(default_factory=list)


@dataclass
class ResearchConfig:
    """Advanced research pipeline settings."""
    prisma_enabled: bool = True
    credibility_scoring: bool = True
    devils_advocate: bool = False
    claim_verification: bool = False
    satellite_cutoff: float = 0.3
    drone_cutoff: float = 0.5
    max_deep_dive: int = 5
    max_counter_queries: int = 3
    max_verify_claims: int = 5
    synthesis_max_themes: int = 8
    output_dir: str = "_research"


@dataclass
class PoliciesConfig:
    """Hard caps enforced at runtime."""
    max_parallel_requests: int = 4
    max_context_tokens: int = 8192
    max_generation_tokens: int = 800
    max_chunks_retrieved: int = 30
    max_rerank_n: int = 50
    embed_batch_size: int = 16
    rerank_batch_size: int = 32
    gpu_memory_safety_margin_mb: int = 2048
    gpu_memory_utilization: float = 0.85
    timeout_embed_ms: int = 30000
    timeout_retrieve_ms: int = 10000
    timeout_rerank_ms: int = 30000
    timeout_generate_ms: int = 120000
    eval_temperature: float = 0.0
    benchmark_hash_verify: bool = True
    benchmark_hash_manifest: str = "config/benchmark_hashes.json"


@dataclass
class JCoderConfig:
    """Top-level config aggregating all subsystems."""
    llm: ModelConfig = field(default_factory=lambda: ModelConfig(
        name="Qwen/Qwen3-Coder-Next-80B",
        endpoint="http://localhost:8000/v1",
    ))
    embedder: ModelConfig = field(default_factory=lambda: ModelConfig(
        name="nomic-ai/nomic-embed-code-v1",
        endpoint="http://localhost:8001/v1",
        dimension=768,
    ))
    reranker: ModelConfig = field(default_factory=lambda: ModelConfig(
        name="Qwen/Qwen3-Reranker-4B",
        endpoint="http://localhost:8002/v1",
    ))
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    policies: PoliciesConfig = field(default_factory=PoliciesConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    harvester: HarvesterConfig = field(default_factory=HarvesterConfig)
    research: ResearchConfig = field(default_factory=ResearchConfig)
    evolver_seed: int = 1337


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def _find_config_dir(explicit_dir: Optional[str] = None) -> Path:
    """Locate the config directory."""
    if explicit_dir:
        return Path(explicit_dir)
    env = os.environ.get("JCODER_CONFIG_DIR")
    if env:
        return Path(env)
    return Path(__file__).resolve().parent.parent / "config"


def _load_yaml(path: Path) -> dict:
    """Load a YAML file, return empty dict if missing."""
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _ms_to_seconds(ms: int) -> float:
    """Convert milliseconds to seconds. Minimum 1.0 to avoid zero-timeout bugs."""
    import math
    return float(max(1, math.ceil(ms / 1000)))


def _build_model_config(model_raw: dict, port: int) -> ModelConfig:
    """Merge model definition with its port to produce a ModelConfig."""
    endpoint = f"http://localhost:{port}/v1"
    return ModelConfig(
        name=model_raw.get("name", ""),
        endpoint=endpoint,
        dimension=model_raw.get("dimension"),
        enabled=model_raw.get("enabled", True),
        quantization=model_raw.get("quantization", ""),
        tensor_parallel=model_raw.get("tensor_parallel", 1),
        max_model_len=model_raw.get("max_model_len", 32768),
        code_model=model_raw.get("code_model"),
        text_model=model_raw.get("text_model"),
    )


def _build_policies(raw: dict) -> PoliciesConfig:
    """Flatten nested policies.yaml into PoliciesConfig."""
    conc = raw.get("concurrency", {})
    ctx = raw.get("context", {})
    batch = raw.get("batching", {})
    gpu = raw.get("gpu", {})
    timeouts = raw.get("timeouts_ms", {})
    ev = raw.get("eval", {})
    return PoliciesConfig(
        max_parallel_requests=conc.get("max_parallel_requests", 4),
        max_context_tokens=ctx.get("max_context_tokens", 8192),
        max_generation_tokens=ctx.get("max_generation_tokens", 800),
        max_chunks_retrieved=ctx.get("max_chunks_retrieved", 30),
        max_rerank_n=ctx.get("max_rerank_n", 50),
        embed_batch_size=batch.get("embed_batch_size", 16),
        rerank_batch_size=batch.get("rerank_batch_size", 32),
        gpu_memory_safety_margin_mb=gpu.get("memory_safety_margin_mb", 2048),
        gpu_memory_utilization=gpu.get("memory_utilization", 0.85),
        timeout_embed_ms=timeouts.get("embed", 30000),
        timeout_retrieve_ms=timeouts.get("retrieve", 10000),
        timeout_rerank_ms=timeouts.get("rerank", 30000),
        timeout_generate_ms=timeouts.get("generate", 120000),
        eval_temperature=ev.get("temperature", 0.0),
        benchmark_hash_verify=ev.get("benchmark_hash_verify", True),
        benchmark_hash_manifest=ev.get("benchmark_hash_manifest", "config/benchmark_hashes.json"),
    )


def load_config(config_dir: Optional[str] = None) -> JCoderConfig:
    """
    Load config from split YAML files:
      ports.yaml   -- port assignments
      models.yaml  -- model names, dimensions, quantization
      policies.yaml -- hard caps and timeouts
      default.yaml -- storage, chunking, retrieval (legacy single-file support)
    """
    # Performance: enable Ollama flash attention if not explicitly set.
    # Free 15-20% speed boost + 30% VRAM reduction on supported GPUs.
    if "OLLAMA_FLASH_ATTENTION" not in os.environ:
        os.environ["OLLAMA_FLASH_ATTENTION"] = "1"

    d = _find_config_dir(config_dir)

    # Report which config files are present
    expected_files = ["ports.yaml", "models.yaml", "policies.yaml", "default.yaml"]
    missing = [f for f in expected_files if not (d / f).exists()]
    if missing:
        _log.warning(
            "Config files not found in %s: %s — using defaults",
            d, ", ".join(missing),
        )

    ports = _load_yaml(d / "ports.yaml")
    models = _load_yaml(d / "models.yaml")
    policies_raw = _load_yaml(d / "policies.yaml")
    default = _load_yaml(d / "default.yaml")

    # Build model configs by merging model definitions with ports
    llm = _build_model_config(
        models.get("llm", {}),
        ports.get("llm", 8000),
    )
    embedder = _build_model_config(
        models.get("embedder", {}),
        ports.get("embedder", 8001),
    )
    reranker = _build_model_config(
        models.get("reranker", {}),
        ports.get("reranker", 8002),
    )

    # Retrieval, chunking, storage, network, harvester from default.yaml
    retrieval_raw = default.get("retrieval", {})
    chunking_raw = default.get("chunking", {})
    storage_raw = default.get("storage", {})
    network_raw = default.get("network", {})
    harvester_raw = default.get("harvester", {})
    research_raw = default.get("research", {})
    evolver_raw = default.get("evolver", {})

    return JCoderConfig(
        llm=llm,
        embedder=embedder,
        reranker=reranker,
        retrieval=RetrievalConfig(**retrieval_raw) if retrieval_raw else RetrievalConfig(),
        chunking=ChunkingConfig(**chunking_raw) if chunking_raw else ChunkingConfig(),
        storage=StorageConfig(**storage_raw) if storage_raw else StorageConfig(),
        policies=_build_policies(policies_raw),
        network=NetworkConfig(**network_raw) if network_raw else NetworkConfig(),
        harvester=HarvesterConfig(**harvester_raw) if harvester_raw else HarvesterConfig(),
        research=ResearchConfig(**research_raw) if research_raw else ResearchConfig(),
        evolver_seed=evolver_raw.get("seed", 1337) if evolver_raw else 1337,
    )
