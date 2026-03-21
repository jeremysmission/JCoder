# JCoder Operational Runbook

## Quick Start

```bash
cd C:\Users\jerem\JCoder

# 1. Check environment
python main.py doctor check

# 2. Run demo (no GPU needed)
python scripts/demo.py

# 3. Ask a question (mock mode, no Ollama needed)
python main.py --mock ask "How do I read a file in Python?" --index-name jcoder_demo

# 4. Ask a question (live, requires Ollama)
python main.py ask "How do I parse JSON in Python?" --index-name all_sources

# 5. Interactive chat
python main.py interactive
```

## Environment Setup

### Toaster (laptop, CPU-only)
- Python 3.12+, no GPU, FTS5-only search
- Ollama with phi4-mini for generation (optional)
- All indexes in `%JCODER_DATA%\indexes` or `C:\Users\jerem\JCoder\data\indexes` when unset

### Workstation (12 GB GPU)
- Ollama with phi4:14b-q4_K_M (primary), context_window=4096
- FAISS for dense vector search (optional, FTS5 fallback always available)

### BEAST (dual RTX 3090, 48 GB VRAM)
- Devstral Small 2 24B (primary coding model)
- nomic-embed-code for code embeddings
- Full FAISS + FTS5 hybrid search

## Daily Operations

### Run Tests
```bash
python -m pytest tests/ -q            # full suite (~5 min)
python -m pytest tests/ -x -q         # stop on first failure
python -m pytest tests/test_federated_search.py -v  # specific file
```

### Ingest a Repository
```bash
python main.py ingest /path/to/repo --index-name my_project
python main.py --mock ingest /path/to/repo --index-name my_project  # no embedding server
```

### Run Evaluation
```bash
# Validate eval set structure (no model needed)
python evaluation/agent_eval_runner.py --validate-only --eval-set evaluation/agent_eval_set_200.json

# Mock-mode eval (no GPU/Ollama)
python main.py --mock eval --benchmark evaluation/agent_eval_set_200.json

# Live eval (requires Ollama)
python main.py eval --benchmark evaluation/agent_eval_set.json
```

### Run Download Backlog
```bash
python scripts/run_download_queue.py --list
python scripts/run_download_queue.py
python scripts/run_download_queue.py --status
python scripts/run_download_queue.py --only learn_rust
```

Use the inclusive downloader for new acquisition work:
- `core/download_manager.py`
- `docs/INCLUSIVE_DOWNLOADER.md`
- Multi-job queue runs continue past individual item failures by default so one missing dependency does not block the backlog.

### Benchmark Search Latency
```bash
python main.py bench-search --queries 10 --top-k 10
python main.py bench-search --index-dir C:\Users\jerem\JCoder\data\indexes
```

### Agent Operations
```bash
python main.py agent run "Explain how async/await works in Python"
python main.py agent study --topic "design patterns"
python main.py agent goals                     # list goals
python main.py agent sessions                  # list sessions
python main.py agent resume <session_id>       # resume session
```

## Index Management

### Discover Indexes
```bash
python main.py doctor check   # shows FTS5 indexes in step 9
python scripts/demo.py        # shows full inventory in step 2
```

### Build StackExchange Indexes
```bash
python scripts/build_se_indexes.py
# Safe to interrupt and resume -- skips completed indexes
```

### Build Code Corpus Indexes
```bash
python scripts/download_code_corpora.py
# Downloads + indexes: CodeSearchNet, codeparrot, commitpack, etc.
```

### Index Locations
| Path | Contents |
|------|----------|
| C:\Users\jerem\JCoder\data\indexes | Default indexes when `JCODER_DATA` is unset |
| %JCODER_DATA%\indexes | Optional override location when you point data outside the repo |

## Troubleshooting

### Common Issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| `UnicodeEncodeError` on Windows | cp1252 stdout | Wrap with `io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')` |
| `ModuleNotFoundError: faiss` | FAISS not installed | Expected on CPU-only -- FTS5 fallback is automatic |
| `sqlite3.ProgrammingError: same thread` | Old index_engine.py | Ensure `check_same_thread=False` in `_get_fts_conn()` |
| Slow first query (>10s) | Lazy FTS5 connection open | Normal on first query per index; subsequent queries use cached connection |
| `PermissionError` on index file | Another process has it open | Close other JCoder instances or skip the locked index |
| Test `test_prisma_concurrent_chaos` fails | SQLite write contention timing | Known flaky -- rerun passes |

### Health Check
```bash
python main.py doctor check    # full environment check
python main.py doctor fix      # auto-create missing directories
```

## Configuration

| File | Purpose |
|------|---------|
| config/agent.yaml | Agent model, temperature, max_steps, tool permissions |
| config/memory.yaml | Memory + federated search (index weights, FTS5 settings) |
| config/default_config.yaml | Default model, retrieval params |

## Key Metrics

- **Test suite**: 780 passed, 2 skipped, 48 test files
- **Indexes**: 59 FTS5 databases, ~18.9 GB total
- **Search latency**: P50 ~200ms warm (toaster, 10 indexes)
- **Eval set**: 200 questions across 9 categories
