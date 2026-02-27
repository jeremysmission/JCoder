# JCoder: Concept-to-Production Roadmap

## Vision

Fully local, offline, free CLI AI coding assistant. Runs on beast hardware
(128 GB RAM, 48 GB dual-3090, 2 TB NVMe, 2 Gbps internet). No license
restrictions, no cloud dependency, no cost per query. Leverages existing
HybridRAG3 RAG architecture and LimitlessApp self-evolution code.

---

## Phase 0: Research (Week 1) -- COMPLETE

- [x] Best offline code LLMs (Qwen3-Coder-Next 80B selected)
- [x] Best code embedders (Nomic Embed Code 7B selected)
- [x] Best rerankers (Qwen3-Reranker-4B selected)
- [x] Best data dumps (tiered download plan built)
- [x] Reusable code audit (12,612 lines identified)
- [x] Hardware capacity validation (48 GB VRAM fits full stack)

## Phase 1: Foundation (Week 2)

**Goal**: Repo structure, venv, model downloads, basic CLI shell.

| Task | Details |
|------|---------|
| Repo scaffold | src/, tests/, config/, tools/, scripts/, data/ |
| Python venv | 3.12, vLLM, tree-sitter, httpx, click, rich |
| Model downloads | Qwen3-Coder-Next 80B Q3 (~37 GB), Nomic Embed Code 7B Q4 (~4.4 GB), Qwen3-Reranker-4B (~3 GB) |
| vLLM tensor parallel | Dual 3090 NVLink, TP=2, verify 80B loads |
| Basic CLI | `jcoder` command, REPL loop, syntax highlighting |
| Config system | YAML-based, port from HybridRAG3 config.py |

**Exit criteria**: `jcoder "hello world"` returns a response from the 80B model.

## Phase 2: RAG Engine (Weeks 3-4)

**Goal**: Code-aware retrieval pipeline producing grounded answers.

| Task | Details |
|------|---------|
| AST chunker | tree-sitter parser for Python/JS/TS/Rust/Go/C/Java |
| Embedder service | Nomic Embed Code 7B via vLLM /v1/embeddings |
| Vector store | SQLite + FAISS hybrid (port from HybridRAG3 vector_store.py) |
| BM25 keyword index | FTS5 (port from HybridRAG3) |
| Hybrid search | Dense + sparse fusion with RRF scoring |
| Reranker | Qwen3-Reranker-4B as second-pass scorer |
| Query engine | RAG pipeline: parse -> retrieve -> rerank -> generate |

**Exit criteria**: Ask a question about indexed codebase, get a grounded answer with citations.

## Phase 3: Data Ingestion (Weeks 4-5)

**Goal**: Ingest high-value code data dumps into RAG.

| Tier | Data | Size | Time |
|------|------|------|------|
| 1 (immediate) | tldr-pages, Python docs, MDN, CodeSearchNet, DevDocs, Magicoder | ~10 GB | <1 min download |
| 2 (overnight) | Stack Overflow Posts.7z, LeetCode solutions | ~27 GB | ~2 min download |
| 3 (weekend) | StarCoderData (permissive subset), The Stack v2 (filtered) | ~500 GB | ~30 min download |

| Task | Details |
|------|---------|
| Download scripts | Automated tiered download with resume support |
| Format converters | XML (SO), JSON (CodeSearchNet), parquet (Stack) |
| Dedup pipeline | MinHash + exact hash deduplication |
| Incremental indexing | Chunk -> embed -> insert without full rebuild |
| Storage monitor | Track disk usage, warn at 80% capacity |

**Exit criteria**: 500+ GB of code knowledge indexed and searchable.

## Phase 4: CLI Agent (Weeks 5-7)

**Goal**: Full agentic coding assistant with file operations.

| Task | Details |
|------|---------|
| Tool framework | Read, Write, Edit, Bash, Glob, Grep tools |
| Context management | Sliding window, file tracking, conversation history |
| Multi-turn reasoning | Chain-of-thought with tool use planning |
| Code generation | Generate, apply, and verify code changes |
| Test runner | Detect test framework, run tests, parse results |
| Git integration | Status, diff, commit, branch management |
| Error recovery | Retry with different strategy on failure |

**Exit criteria**: `jcoder "add a fibonacci function with tests"` creates the file, writes tests, runs them, all pass.

## Phase 5: Weekly Scraper (Week 7-8)

**Goal**: Autonomous agent that keeps the knowledge base current.

| Task | Details |
|------|---------|
| Source registry | GitHub trending, HN, Reddit, arXiv, release notes |
| Scraper engine | httpx + BeautifulSoup, respect robots.txt |
| Summarizer | Qwen3-Coder summarizes each article/repo |
| RAG ingest | Auto-embed and index new content |
| Changelog | Weekly digest of what was learned |
| Scheduler | Windows Task Scheduler or cron, runs Sunday 2 AM |
| Dedup guard | Don't re-ingest content already in the index |

**Exit criteria**: After one weekly cycle, new content appears in RAG searches.

## Phase 6: 10x Code Evolver (Weeks 8-10)

**Goal**: Self-improving code through iterative simulation.

| Task | Details |
|------|---------|
| Simulation harness | Run full test suite, collect metrics |
| Analyzer | Parse test results, identify weak areas |
| Improver | Generate code improvements targeting weak areas |
| Verifier | Confirm improvements don't regress other tests |
| Evolution loop | 10 iterations per cycle, keep best version |
| Distillation | Use Claude Max ($200/mo) to generate training pairs |
| Fine-tuner | QLoRA on dual 3090 (24 GB each), merge to base |
| Weekly trigger | Evolver runs after scraper, uses new knowledge |

**Exit criteria**: After one evolution cycle, measurable improvement on coding benchmarks.

## Phase 7: Production Hardening (Weeks 10-12)

| Task | Details |
|------|---------|
| IBIT/CBIT | Health monitoring (port from HybridRAG3) |
| Crash recovery | Checkpoint conversation state, resume on restart |
| Performance tuning | KV cache optimization, batch inference, speculative decode |
| Memory management | GPU memory monitoring, graceful OOM handling |
| User preferences | Persistent settings, project-specific configs |
| Documentation | User guide, architecture doc, API reference |
| Packaging | Single install script, model auto-download |

**Exit criteria**: Stable daily-driver tool that survives 8-hour coding sessions.

---

## VRAM Budget

| Component | VRAM | Notes |
|-----------|------|-------|
| Qwen3-Coder-Next 80B (Q3) | ~37 GB | TP=2 across dual 3090 |
| Nomic Embed Code 7B (Q4) | ~4.4 GB | Shared GPU 1 |
| Qwen3-Reranker-4B | ~3 GB | Shared GPU 1 |
| KV cache + overhead | ~3.6 GB | Remaining headroom |
| **Total** | **~48 GB** | Fits dual 3090 (24+24) |

## Disk Budget

| Component | Size | Notes |
|-----------|------|-------|
| Models (GGUF/safetensors) | ~45 GB | 80B + 7B + 4B |
| Tier 1-2 data dumps | ~37 GB | Core knowledge |
| Tier 3 data dumps | ~500 GB | Extended knowledge |
| Vector indexes | ~100 GB | FAISS + SQLite |
| Working space | ~200 GB | Temp, cache, checkpoints |
| **Total** | **~882 GB** | Fits 2 TB NVMe |

## Timeline Summary

| Phase | Duration | Milestone |
|-------|----------|-----------|
| 0. Research | Week 1 | Stack selected |
| 1. Foundation | Week 2 | CLI responds |
| 2. RAG Engine | Weeks 3-4 | Grounded answers |
| 3. Data Ingestion | Weeks 4-5 | 500 GB indexed |
| 4. CLI Agent | Weeks 5-7 | Agentic coding |
| 5. Weekly Scraper | Weeks 7-8 | Auto-updating knowledge |
| 6. 10x Evolver | Weeks 8-10 | Self-improving |
| 7. Hardening | Weeks 10-12 | Production ready |
