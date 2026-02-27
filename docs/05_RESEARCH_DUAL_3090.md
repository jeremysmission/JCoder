# JCoder: Dual RTX 3090 Optimization Research

Research date: 2026-02-26

---

## Inference Engines -- Ranked for Dual 3090

| Engine | Tensor Parallel | Speculative Decode | Best For | Dual 3090 Speed |
|--------|----------------|-------------------|----------|----------------|
| **ExLlamaV2/V3** | YES (gpu_split auto) | YES (draft models) | Single-user, max tok/s | Fastest |
| **vLLM** | YES (NVLink-aware) | YES (Eagle 3) | Multi-user, batching | Near-fastest |
| **SGLang** | YES | YES | Structured generation | Fast |
| **TensorRT-LLM** | YES | YES | NVIDIA-optimized | 62% faster than llama.cpp |
| **llama.cpp** | NO (layer split only) | Experimental | CPU+GPU hybrid, GGUF | Slow multi-GPU |
| **Ollama** | NO (wraps llama.cpp) | NO | Easy prototyping | Slow multi-GPU |

**Critical**: llama.cpp and Ollama do NOT do true tensor parallelism.
They split layers across GPUs (pipeline parallelism), which is significantly
less efficient. Use ExLlamaV2 or vLLM for dual 3090.

Source: https://www.ahmadosman.com/blog/do-not-use-llama-cpp-or-ollama-on-multi-gpus-setups-use-vllm-or-exllamav2/

---

## Best Code LLMs That Fit in 48GB

### No Restrictions (JCoder -- personal use)

| Model | Params | Quant | VRAM | SWE-bench | HumanEval | License |
|-------|--------|-------|------|-----------|-----------|---------|
| Qwen2.5-Coder-32B | 32B dense | Q4_K_M | ~20 GB | ~41% | **91.0%** | Apache 2.0 |
| Qwen3-Coder-30B-A3B | 30B MoE (3B active) | Q8_0 | ~32.5 GB | 50.3% | N/A | Apache 2.0 |
| Devstral Small 2 | 24B dense | Q4_K_M | ~14-16 GB | **68.0%** | 86.6% | Apache 2.0 |
| Qwen3-Coder-Next-80B | 80B MoE (3B active) | Q4_K_M | ~46 GB | N/A | N/A | Apache 2.0 |

### With CPU Offload (48GB GPU + 128GB RAM)

| Model | Params | VRAM + RAM | Speed | SWE-bench |
|-------|--------|-----------|-------|-----------|
| Devstral 2 | 123B dense | 48GB + ~27GB RAM | 5-10 tok/s | **72.2%** |
| Qwen3-Coder-480B | 480B MoE (35B active) | Won't fit 128GB | N/A | 69.1% |

---

## Quantization -- Sweet Spots

| Quant | Bits/Weight | Max Model in 48GB | Quality | Recommendation |
|-------|-------------|-------------------|---------|---------------|
| Q3_K_M | ~3.5 | ~109B | Degraded for code | Avoid for code gen |
| **Q4_K_M** | ~4.5 | **~85B** | **92% quality** | **Gold standard** |
| Q5_K_M | ~5.5 | ~70B | 94% quality | Worth it if fits |
| Q8_0 | ~8 | ~48B | Near-lossless | Best if fits |
| EXL2 4-bit | ~4.5 | ~85B | 95-97% quality | Fastest on GPU |

FP8 does NOT work natively on RTX 3090 (needs Ada Lovelace 4090+).
Use INT8 or Q4_K_M instead.

---

## Memory Optimization

- **KV cache quantization**: FP8 or INT8 KV cache cuts memory ~50%. Supported by vLLM on 3090.
- **PagedAttention** (vLLM): Reduces KV cache waste to <4%, 2-4x throughput improvement.
- **Flash Attention 2**: Fully supported on 3090 (Ampere sm_86). Flash Attention 3 requires Hopper.
- **KTransformers**: Specialized CPU/GPU hybrid for MoE models. 4.6-19.7x prefill speedup.

### CPU Offloading Speed Penalty

| Config | Speed | Notes |
|--------|-------|-------|
| Fully in 48GB VRAM | 20-50 tok/s | Full speed |
| Split VRAM + RAM (dense) | 2-5 tok/s | 5-30x slower |
| Split VRAM + RAM (MoE) | 5-15 tok/s | Much better (only active params need GPU) |
| Fully CPU (MoE Q4) | 12-15 tok/s | Viable for MoE models |

---

## Code Embedders -- Ranked

| Model | Params | Dims | CodeSearchNet | Local? | License | Origin |
|-------|--------|------|---------------|--------|---------|--------|
| **Nomic Embed Code** | 7B | 768 | **SOTA** | YES (GGUF/Ollama) | Apache 2.0 | USA |
| **CodeRankEmbed** | 137M | Flexible | Strong | YES | Apache 2.0 | USA |
| Codestral Embed | ? | up to 3072 | SOTA | API only | Proprietary | France |
| Voyage Code 3 | ? | 256-2048 | Strong | API only | Proprietary | USA |
| Jina v2 Code | ~110M | 768 | Good | YES | Apache 2.0 | Germany |
| CodeSage V2 | 130M-1.3B | Flexible | Strong | YES | Apache 2.0 | USA |

**Best local stack**: CodeRankEmbed (137M, fast first-pass) + Nomic Embed Code (7B, precise).

---

## Code Rerankers -- Ranked

| Model | Params | MTEB-Code | Type | License | Origin |
|-------|--------|-----------|------|---------|--------|
| Qwen3-Reranker-8B | 8B | Highest | Cross-encoder | Apache 2.0 | China |
| Qwen3-Reranker-4B | 4B | Very strong | Cross-encoder | Apache 2.0 | China |
| **CodeRankLLM** | 7B | Strong | **Listwise** | Apache 2.0 | USA |
| Jina Reranker v2 | ~140M | Good | Cross-encoder | Apache 2.0 | Germany |

**Best for JCoder**: Qwen3-Reranker-8B (no restrictions) or CodeRankLLM (if preferring USA origin).

---

## RAG Optimizations for Code

- **AST chunking** (tree-sitter): Chunk at function/class boundaries, not arbitrary character limits.
- **GraphRAG**: Build dependency/call graphs, traverse edges during retrieval.
- **HyDE**: LLM generates hypothetical code snippet, embed that, search for similar real code.
- **Parent-child chunks**: Match on function (child), retrieve surrounding class (parent) for context.
- **ColBERT**: Token-level late interaction scoring via RAGatouille. More precise than single-vector.
- **Multi-representation**: Store both raw code AND natural language description per chunk.

---

## Fine-Tuning on Dual 3090

| Model Size | Dual 3090 QLoRA | Framework | Notes |
|-----------|----------------|-----------|-------|
| 7B-13B | Easy | Unsloth (single GPU) | Overkill for dual |
| 24B-32B | **Comfortable** | Axolotl (FSDP+QLoRA) | **Recommended target** |
| 70B | Tight, painful | Axolotl (DeepSpeed Zero-2) | OOM risk, very slow |

**Unsloth does NOT support multi-GPU.** Use Axolotl for dual 3090 training.

Post-training pipeline: SFT -> DPO -> optional KTO (safety).

---

## Recommended GPU Layout

### Config A: Maximum Speed (LLM on one GPU)
```
GPU 0 (24 GB): Qwen2.5-Coder-32B Q4_K_M (~20 GB) + KV cache
GPU 1 (24 GB): Nomic Embed Code 7B (~4.5 GB) + CodeRankLLM 7B (~4.5 GB) + spare
```

### Config B: Maximum Quality (LLM across both GPUs)
```
GPU 0+1 (48 GB): Devstral 2 123B Q4 via tensor parallel
System RAM: Embedder + Reranker on CPU (128 GB available)
```

### Config C: Best Balance
```
GPU 0+1 (48 GB): Qwen3-Coder-Next-80B Q4_K_M (~46 GB) via tensor parallel
System RAM: Embedder + Reranker on CPU
```

---

## Optimal RAG Pipeline

```
User Query
  -> HyDE (LLM generates hypothetical code)
  -> CodeRankEmbed-137M (fast retrieval, top-100)
  -> Nomic Embed Code 7B (precise re-embed, top-20)
  -> CodeRankLLM 7B (listwise rerank, top-5)
  -> AST-aware context assembly (tree-sitter parent-child)
  -> Qwen2.5-Coder-32B or Devstral Small 2 (generation)
```

## Sources

- vLLM 4x3090 benchmarks: http://himeshp.blogspot.com/2025/03/vllm-performance-benchmarks-4x-rtx-3090.html
- Stop using llama.cpp for multi-GPU: https://www.ahmadosman.com/blog/do-not-use-llama-cpp-or-ollama-on-multi-gpus-setups-use-vllm-or-exllamav2/
- TensorRT-LLM benchmarks: https://www.jan.ai/post/benchmarking-nvidia-tensorrt-llm
- ExLlamaV2: https://github.com/turboderp-org/exllamav2
- ExLlamaV3: https://github.com/turboderp-org/exllamav3
- Devstral 2 announcement: https://mistral.ai/news/devstral-2-vibe-cli
- Qwen3-Coder: https://github.com/QwenLM/Qwen3-Coder
- Nomic Embed Code: https://www.nomic.ai/blog/posts/introducing-state-of-the-art-nomic-embed-code
- CodeRankLLM: https://huggingface.co/nomic-ai/CodeRankLLM
- KTransformers: https://github.com/kvcache-ai/ktransformers
- cAST paper: https://arxiv.org/html/2506.15655v1
- SWE-bench leaderboard: https://www.swebench.com/viewer.html
