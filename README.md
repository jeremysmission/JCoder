# JCoder

Fully local, offline CLI AI coding assistant. Ingests codebases, builds hybrid vector+keyword indexes (FAISS + SQLite FTS5), and answers questions using a local LLM served via vLLM.

## Requirements

- Python 3.10+
- CUDA-capable GPU (dual RTX 3090 recommended for full model stack)
- vLLM for model serving

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

## Configuration

Config files live in `config/`:

| File | Purpose |
|------|---------|
| `models.yaml` | Model names, quantization, tensor parallelism |
| `ports.yaml` | Port assignments for vLLM servers |
| `policies.yaml` | Hard caps, timeouts, GPU margins, eval settings |
| `default.yaml` | Storage paths, retrieval params, chunking strategy |

Override config location: `--config-dir <path>` or `JCODER_CONFIG_DIR` env var.

## Usage

```bash
# Check environment readiness
python main.py doctor

# Ingest a repository
python main.py ingest /path/to/repo --index-name myproject

# Ask a question
python main.py ask "How does authentication work?" --index-name myproject

# Run evaluation benchmark
python main.py eval --benchmark evaluation/golden_questions_v1.json

# CPU-only testing (no vLLM needed)
python main.py --mock ask "What does the chunker do?"
```

## Architecture

```
CLI (Click) -> Orchestrator -> RetrievalEngine -> EmbeddingEngine (vLLM)
                    |                |
                    |          IndexEngine (FAISS + FTS5 + RRF)
                    |                |
                    |           Reranker (vLLM cross-encoder)
                    |
               Runtime (vLLM chat completions)
```

## Testing

```bash
pytest tests/ -v
```
