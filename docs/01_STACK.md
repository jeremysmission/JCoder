# JCoder: Hardware and Software Stack

## Hardware

| Component | Spec |
|-----------|------|
| CPU | Consumer desktop (exact TBD) |
| RAM | 128 GB DDR4/DDR5 |
| GPU | 2x NVIDIA RTX 3090 (24 GB each, NVLink) |
| Storage | 2 TB NVMe SSD |
| Network | 2 Gbps symmetric |
| OS | Windows 11 |

**Key capacity**: 48 GB pooled VRAM via tensor parallelism, 128 GB system RAM
for large batch processing, 2 TB for models + data dumps + indexes.

---

## Code LLM: Qwen3-Coder-Next 80B (MoE)

| Property | Value |
|----------|-------|
| Parameters | 80B total, ~3B active (MoE) |
| SWE-bench | 74.2% (state of art for open models) |
| License | Apache 2.0 |
| Quant | Q3_K_M (~37 GB) |
| Context | 256K tokens |
| Speed (est.) | ~71 tok/s on dual 3090 TP=2 |
| Serving | vLLM with tensor_parallel_size=2 |

**Why this model**: Highest SWE-bench score among open models. MoE architecture
means only 3B params active per token despite 80B total, giving excellent
speed. Q3 quant fits in 48 GB with room for embedder + reranker.

**Alternatives considered**:
- Devstral Small 2 24B: 65.4% SWE-bench, smaller but weaker
- DeepSeek-R1 671B: Too large even at Q2
- Qwen3-Coder 32B (dense): Slower than 80B MoE, lower SWE-bench

### vLLM Configuration

```bash
# Launch command (dual 3090 tensor parallel)
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-Coder-Next-80B \
    --tensor-parallel-size 2 \
    --quantization awq \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.85 \
    --port 8100
```

---

## Code Embedder: Nomic Embed Code 7B

| Property | Value |
|----------|-------|
| Parameters | 7B |
| CodeSearchNet MRR | Best in class (per Nomic benchmarks) |
| License | Apache 2.0 |
| Quant | Q4_K_M (~4.4 GB) |
| Dimensions | 768 (Matryoshka, truncatable to 256/512) |
| Max tokens | 8192 |
| Serving | vLLM /v1/embeddings or Ollama |

**Why this model**: Purpose-built for code. Understands syntax structure,
variable names, and code semantics better than general-purpose embedders.
7B size gives much better quality than smaller models while still fitting
alongside the 80B LLM.

**Alternatives considered**:
- CodeSage-Large-v2 (1.3B): Smaller, weaker on code search
- Nomic Embed Text v1.5 (137M): General purpose, not code-optimized
- voyage-code-3: API only, not local

---

## Code Reranker: Qwen3-Reranker-4B

| Property | Value |
|----------|-------|
| Parameters | 4B |
| MTEB-Code | 81.20 (state of art) |
| License | Apache 2.0 |
| VRAM | ~3 GB (Q4) |
| Max tokens | 8192 |
| Serving | vLLM or transformers |

**Why this model**: Highest MTEB-Code score available. Small enough to
coexist with LLM + embedder. Code-specific reranking dramatically improves
retrieval precision.

**Alternatives considered**:
- Qwen3-Reranker-0.6B: Smaller, scores 75.49 (6 points lower)
- bge-reranker-v2-m3: BAAI/China concern resolved (no restrictions on JCoder)
- Cohere Rerank 3.5: API only

---

## Inference Framework: vLLM

| Property | Value |
|----------|-------|
| Version | Latest stable |
| License | Apache 2.0 |
| Features | Tensor parallelism, PagedAttention, continuous batching |
| API | OpenAI-compatible /v1/chat/completions |

**Why vLLM**: Only framework that supports tensor parallelism across dual
3090 for the 80B model. PagedAttention gives near-optimal KV cache usage.
OpenAI-compatible API means existing code from HybridRAG3 works with
minimal changes.

**Alternative**: Ollama is simpler but lacks tensor parallelism for multi-GPU.

---

## Code Parsing: tree-sitter

| Property | Value |
|----------|-------|
| Purpose | AST-based code chunking |
| License | MIT |
| Languages | Python, JS, TS, Rust, Go, C, C++, Java, Ruby, PHP, C# |

**Why tree-sitter**: Chunks code at function/class boundaries instead of
arbitrary character limits. Preserves semantic units. Used by GitHub Copilot,
Sourcegraph, and most code-aware tools.

---

## Vector Store: FAISS + SQLite

| Component | Role |
|-----------|------|
| FAISS | Dense vector similarity search (GPU-accelerated) |
| SQLite FTS5 | BM25 keyword search |
| Hybrid fusion | Reciprocal Rank Fusion (RRF) combining both |

**Port from HybridRAG3**: vector_store.py (990 lines), already implements
FAISS + SQLite hybrid. Needs dimension upgrade (384 -> 768) and AST-aware
metadata fields.

---

## CLI Framework: Click + Rich

| Component | Role |
|-----------|------|
| Click | Command parsing, argument handling |
| Rich | Syntax highlighting, progress bars, panels, markdown |
| prompt_toolkit | REPL with history, autocomplete |

---

## Full VRAM Layout

```
GPU 0 (24 GB):                    GPU 1 (24 GB):
+---------------------------+     +---------------------------+
| Qwen3-Coder 80B (shard 1)|     | Qwen3-Coder 80B (shard 2)|
| ~18.5 GB                  |     | ~18.5 GB                  |
+---------------------------+     +---------------------------+
| Nomic Embed Code 7B       |     | KV Cache                  |
| ~4.4 GB                   |     | ~3.6 GB                   |
+---------------------------+     +---------------------------+
| Qwen3-Reranker-4B         |     | Free                      |
| ~3 GB                     |     | ~1.9 GB                   |
+---------------------------+     +---------------------------+
| Free: ~0.6 GB (safety)    |
+---------------------------+
```

---

## Python Dependencies (Core)

```
vllm>=0.8.0          # Inference engine, tensor parallel
tree-sitter>=0.24    # AST parsing
faiss-gpu>=1.9       # Vector similarity (GPU)
click>=8.0           # CLI framework
rich>=13.0           # Terminal formatting
prompt-toolkit>=3.0  # REPL
httpx>=0.28          # HTTP client (vLLM API)
pyyaml>=6.0          # Config
sqlite3              # Built-in, FTS5
beautifulsoup4>=4.12 # Web scraper
schedule>=1.2        # Weekly task scheduler
peft>=0.14           # QLoRA fine-tuning
transformers>=4.47   # Model loading for fine-tuning
datasets>=3.0        # Data processing
```

---

## Data Dumps: Priority Download Order

### Tier 1 -- Immediate (~10 GB, <1 min at 2 Gbps)

| Source | Size | Format | Content |
|--------|------|--------|---------|
| tldr-pages | 5 MB | Markdown | Concise CLI command examples |
| Python 3.12 docs | 30 MB | HTML | Complete stdlib reference |
| MDN Web Docs | 200 MB | HTML | JS/CSS/HTML reference |
| CodeSearchNet | 2 GB | JSON | 6M function-docstring pairs |
| DevDocs.io dump | 1 GB | HTML | 400+ API docs offline |
| Magicoder-OSS-Instruct | 1 GB | JSON | 75K coding instructions |
| Glaive Code Assistant v3 | 2 GB | JSON | 950K code Q&A pairs |
| OctoPack/CommitPackFT | 3 GB | JSON | High-quality commit-code pairs |

### Tier 2 -- Overnight (~27 GB, ~2 min download)

| Source | Size | Format | Content |
|--------|------|--------|---------|
| Stack Overflow Posts.7z | 20 GB | XML | All questions + answers |
| LeetCode solutions | 2 GB | JSON | Algorithm implementations |
| Awesome Lists dump | 500 MB | Markdown | Curated project lists |
| arXiv CS papers (2020-2026) | 5 GB | PDF/text | Latest research |

### Tier 3 -- Weekend (~500 GB, ~30 min download)

| Source | Size | Format | Content |
|--------|------|--------|---------|
| StarCoderData (permissive) | 250 GB | Parquet | GitHub code, permissive licenses |
| The Stack v2 (filtered) | 200 GB | Parquet | Deduplicated code, 600+ langs |
| SlimPajama | 50 GB | JSON | Clean web text for general knowledge |

### Download URLs

```
# Tier 1
git clone https://github.com/tldr-pages/tldr
wget https://docs.python.org/3/archives/python-3.12-docs-html.tar.bz2
huggingface-cli download code_search_net/code_search_net
huggingface-cli download ise-uiuc/Magicoder-OSS-Instruct-75K
huggingface-cli download glaiveai/glaive-code-assistant-v3
huggingface-cli download bigcode/commitpackft

# Tier 2
wget https://archive.org/download/stackexchange/stackoverflow.com-Posts.7z

# Tier 3
huggingface-cli download bigcode/starcoderdata --include "*.parquet"
huggingface-cli download bigcode/the-stack-v2 --include "*.parquet"
```
