# Dual GPU Setup Guide — JCoder on GPU 1

**Date:** 2026-03-18
**Purpose:** Dedicate GPU 1 (RTX 3090 #2) to JCoder while HybridRAG3 uses GPU 0

## Current State (Before)
- Ollama runs on default port 11434, uses GPU 0 primarily
- GPU 1 only used for VRAM overflow (no compute)
- JCoder and HybridRAG3 share the same Ollama instance

## Setup: Dedicated GPU 1 for JCoder

### Option A: Second Ollama Instance (Simple, Recommended for Tonight)

**Start a second Ollama on GPU 1, port 11435:**

```powershell
# In a new PowerShell terminal:
$env:CUDA_VISIBLE_DEVICES = "1"
$env:OLLAMA_HOST = "127.0.0.1:11435"
$env:OLLAMA_MODELS = "C:\Users\jerem\.ollama\models"
& "C:\Users\jerem\AppData\Local\Programs\Ollama\ollama.exe" serve
```

**Pull models on the second instance (one-time):**
```powershell
# In another terminal:
$env:OLLAMA_HOST = "127.0.0.1:11435"
& "C:\Users\jerem\AppData\Local\Programs\Ollama\ollama.exe" pull phi4:14b-q4_K_M
& "C:\Users\jerem\AppData\Local\Programs\Ollama\ollama.exe" pull nomic-embed-text
```

**Update JCoder config to use port 11435:**

Edit `C:\Users\jerem\JCoder\config\ports.yaml`:
```yaml
llm: 11435
embedder: 11435
reranker: 8002
```

### Option B: vLLM with Tensor Parallel (Future — Max Performance)

Requires WSL2 + vLLM install. Splits compute across BOTH GPUs for single-model inference.

```bash
# In WSL2:
pip install vllm
vllm serve microsoft/phi-4 --tensor-parallel-size 2 --port 8000 --max-model-len 32768
```

Then update JCoder `config/ports.yaml`:
```yaml
llm: 8000
```

## How to Undo (Revert to Shared GPU)

1. Stop the second Ollama instance (close the terminal)
2. Edit `C:\Users\jerem\JCoder\config\ports.yaml` back to:
```yaml
llm: 11434
embedder: 11434
reranker: 8002
```
3. Everything goes back through the single default Ollama on GPU 0

## How to Run Both GPUs for Everything (Option B Full)

For maximum performance where BOTH GPUs compute together:

1. Stop Ollama entirely
2. Start vLLM with `--tensor-parallel-size 2` (uses both GPUs)
3. Point ALL services (HybridRAG3 + JCoder) at vLLM endpoint
4. Every inference uses both GPUs = ~30-50% faster per query
5. Trade-off: only one model loaded at a time

## Verification

Check which GPU each process is using:
```powershell
nvidia-smi
```

Live monitoring (1-second updates):
```powershell
nvidia-smi dmon -s u -d 1
```

Both GPUs should show compute utilization when running queries.

## Hardware Reference
- GPU 0: RTX 3090 24GB (Bus 01:00.0) — HybridRAG3/Coder
- GPU 1: RTX 3090 24GB (Bus 03:00.0) — JCoder (dedicated)
- Total VRAM: 48 GB
- No NVLink bridge (independent operation)
