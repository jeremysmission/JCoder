import random
from typing import Dict

Config = Dict[str, object]


class GEPAConfigMutator:
    """Mutates RAG configuration parameters within defined choices."""

    TOP_K_OPTIONS = [4, 6, 8, 10, 12]
    CHUNK_SIZES = [256, 384, 512, 768]
    TEMPERATURES = [0.1, 0.2, 0.3, 0.4]
    PROMPTS = ["cot_prompt_v3", "self_reflect_v2", "chain_check"]
    RERANKERS = ["none", "bge-reranker", "qwen-reranker"]

    def __init__(self, seed: int = 42):
        self.random = random.Random(seed)

    def mutate(self, config: Config) -> Config:
        config["top_k"] = self.random.choice(self.TOP_K_OPTIONS)
        config["chunk_size"] = self.random.choice(self.CHUNK_SIZES)
        config["temperature"] = self.random.choice(self.TEMPERATURES)
        config["prompt_template"] = self.random.choice(self.PROMPTS)
        config["retriever"] = config.get("retriever", "hybrid")  # keep default
        config["reranker"] = self.random.choice(self.RERANKERS)
        config["min_score"] = round(self.random.uniform(0.08, 0.22), 2)
        return config
