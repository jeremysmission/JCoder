# JCoder Sprint Plan -- 2026-03-10

## Baseline

| Metric | Value |
|--------|-------|
| Tests | 813 pass, 0 fail, 3 skip |
| FTS5 indexes | 85 (24.6 GB) |
| Eval set | 200 questions, 9 categories |
| Eval pass rate | 97.5% (phi4-mini, CPU-only) |
| Agent tools | 12 (read, write, edit, bash, glob, grep, git, web, memory, think, ask, complete) |
| Self-learning | bridge, telemetry, experience replay, meta-cog, active learner -- all wired |
| FIM support | 5 model formats (Devstral, StarCoder, CodeLlama, Deepseek, generic) |
| Download manager | Staging, SQLite ledger, resume, SHA256 verify |
| Federated search | 85 weighted indexes, RRF fusion, lazy FTS5 connections |

### Completed Sprints (1-6)

| Sprint | Scope | Status |
|--------|-------|--------|
| 1 | Foundation: model research, data planning, sprint design | DONE |
| 2 | Data acquisition: 85 FTS5 indexes across SE, CSN, code corpora, docs | DONE |
| 3 | Ingestion pipeline: parsers, PII scanner, MinHash dedup, checkpoints | DONE |
| 4 | RAG infrastructure: federated search, agent memory, FTS5 fallback | DONE |
| 4.5 | Agent framework: 12 tools, goals, bridge, sessions, logging, CLI | DONE |
| 5 | Integration: dual embedder, index discovery, expanded eval, query profiles | DONE |
| 6 | Production basics: search optimization, demo script, operational runbook | DONE |

---

## Sprint 7: Online API Integration and Eval Baseline

**Hardware**: Toaster (no GPU required)
**Dependency**: OpenRouter API key or Anthropic API key
**Duration**: 1-2 sessions

### Goal
Establish quality ceiling by running the full eval with a frontier model,
then use distillation to enrich the weakest knowledge areas.

### Tasks

- [ ] Configure online API backend in config/agent.yaml (OpenRouter or direct Anthropic)
- [ ] Run 200q eval with online API model (target: >99% pass)
- [ ] A/B comparison report: online API vs phi4-mini per category
  - Identify categories where offline model is weakest
  - Rank categories by quality gap (online - offline)
- [ ] Distillation pilot: enrich top-20 weakest topics via Claude API
  - Send weak-scoring questions + retrieved context to Claude
  - Get expert explanations, index them alongside original chunks
  - Re-run eval on those 20 questions, measure improvement
- [ ] Track API cost (integrate with cost_tracker.py pattern)
- [ ] Establish budget guard: daily/weekly token limit for distillation

### Exit Criteria
- 200q eval report comparing online vs offline models
- 20 enriched topics indexed, measurable score improvement
- Cost per enriched topic documented

---

## Sprint 8: Hybrid Search Activation

**Hardware**: Toaster for tuning (CPU embeddings), BEAST for full-scale
**Dependency**: Ollama running with nomic-embed-text
**Duration**: 2-3 sessions

### Goal
Move from FTS5-only search to FAISS+FTS5 hybrid retrieval with RRF fusion.
Start with the 5 highest-value corpuses, prove the quality lift, then scale.

### Tasks

- [ ] Embed top-5 highest-value FTS5 corpuses (CPU-mode Ollama):
  1. python_docs (22 MB FTS5 -- authoritative, small)
  2. best_practices (0.1 MB FTS5 -- highest weight, tiny)
  3. python_instructions (82 MB FTS5 -- Q&A pairs)
  4. code_alpaca (13 MB FTS5 -- instruction-tuned)
  5. agent_memory (personal RAG)
- [ ] Build FAISS indexes for each (IndexFlatIP, 768-dim)
- [ ] Enable hybrid retrieval path in agent config (FAISS + FTS5 + RRF)
- [ ] Build retrieval eval: precision@5, recall@5, MRR on 50 test queries
- [ ] A/B test: FTS5-only vs hybrid on 50 queries, measure quality delta
- [ ] Tune RRF k parameter (test k=20, 40, 60, 80) on retrieval eval
- [ ] Document optimal RRF weights per corpus type (code vs docs vs Q&A)

### Exit Criteria
- 5 FAISS indexes built, hybrid search operational
- Retrieval eval baselines documented
- Measurable precision improvement over FTS5-only

---

## Sprint 9: Self-Learning Pipeline Activation

**Hardware**: Toaster (CPU-only, online API for study)
**Dependency**: Sprint 7 (online API configured)
**Duration**: 2-3 sessions

### Goal
Activate the full self-learning pipeline so JCoder improves from its own
experience. Validate that learning components work end-to-end, not just
in isolation.

### Tasks

- [ ] End-to-end learning validation:
  - Agent runs a task -> bridge captures experience -> telemetry records
  - Experience replay selects high-value past results
  - Meta-cog analyzes patterns and updates strategy
  - Active learner identifies knowledge gaps
- [ ] Run online API self-study on 10 weakest eval categories
  - goals.py generates study topics per category
  - Agent researches each topic using RAG + online API
  - Extracted insights auto-ingested into agent_memory
- [ ] Implement PRAXIS pattern: experience-driven prompt refinement
  - Track which prompts produce best results per query type
  - Auto-select prompt template based on category match
- [ ] Run 10-cycle autopilot validation (agent autopilot --cycles 10)
  - Measure: goal completion rate per cycle
  - Measure: strategy selection accuracy
  - Measure: memory growth rate and dedup effectiveness
- [ ] Re-run 200q eval after learning cycles, compare to Sprint 7 baseline

### Exit Criteria
- 10 autopilot cycles complete with metrics
- Measurable eval improvement from learning
- Strategy rankings show convergence (not random)

---

## Sprint 10: BEAST Hardware Migration

**Hardware**: BEAST (128 GB RAM, 48 GB VRAM, 2 TB NVMe) REQUIRED
**Dependency**: BEAST hardware built and operational
**Duration**: 2-3 sessions

### Goal
Migrate JCoder to the full hardware stack. Install the primary coding model,
embed all corpuses with GPU acceleration, and establish production baselines.

### Tasks

- [ ] Install primary models:
  - `ollama pull devstral-small-2:24b` (14 GB, primary coding model)
  - `ollama pull manutic/nomic-embed-code` (code embeddings)
  - Verify both models respond correctly
- [ ] Benchmark Devstral vs phi4-mini vs online API on 200q eval
  - Category-by-category comparison
  - Latency measurement (target: <5s query-to-answer)
- [ ] Full-corpus embedding with GPU acceleration:
  - Embed all 85 FTS5 corpuses using nomic-embed-code (code) and nomic-embed-text (docs)
  - Build FAISS indexes for each
  - Estimate: ~24 GB FAISS total at 768-dim
- [ ] GPU stress test: sustained load for 1 hour
  - Monitor VRAM usage, temperature, throughput
  - Verify no OOM under concurrent query + embed workloads
- [ ] VRAM profiling: map actual usage per model combination
  - Devstral 24B + KV cache at ctx=4096, 8192, 16384, 32768
  - Dual-model: Devstral (GPU 1) + embedder (GPU 2)
- [ ] Migrate data: copy D:\JCoder_Data to BEAST NVMe
- [ ] Re-run full test suite on BEAST, confirm 813+ pass

### Exit Criteria
- Devstral 24B operational, <5s query latency
- All 85 corpuses embedded with FAISS indexes
- VRAM budget validated under sustained load
- Full test regression green on BEAST

---

## Sprint 11: Weekly Knowledge Scraper

**Hardware**: BEAST preferred (for LLM summarization), Toaster viable (online API)
**Dependency**: Sprint 10 (model installed) or Sprint 7 (online API)
**Duration**: 2-3 sessions

### Goal
Build an autonomous agent that keeps JCoder's knowledge base current.
Runs weekly, scrapes high-signal programming sources, summarizes,
and auto-ingests into RAG.

### Tasks

- [ ] Source registry (config/scraper_sources.yaml):
  - GitHub Trending (daily, top-50 repos by language)
  - Hacker News (top stories, filtered for programming)
  - Reddit (r/programming, r/python, r/rust, r/golang)
  - arXiv (cs.AI, cs.SE, cs.PL -- new papers)
  - Release notes (Python, Node, Rust, Go -- major releases)
- [ ] Scraper engine:
  - httpx-based with rate limiting and robots.txt respect
  - HTML -> markdown conversion (reuse from download_python_docs.py)
  - Dedup guard: check SHA256 against existing index before ingest
  - Error handling: skip unavailable sources, log failures
- [ ] Summarizer pipeline:
  - Send scraped content to local LLM (Devstral) or online API
  - Extract: key concepts, code patterns, notable changes
  - Tag with source, date, language, topic
- [ ] Auto-ingest:
  - Chunk summarized content
  - Embed and index into dedicated scraper FTS5 + FAISS indexes
  - Update federated search config to include new indexes
- [ ] Weekly digest generator:
  - Markdown report of what was learned this week
  - Stats: sources scraped, articles ingested, topics covered
  - Output to logs/weekly_digest_YYYY-MM-DD.md
- [ ] Scheduler integration:
  - Windows Task Scheduler (BEAST) or cron (Linux)
  - Runs Sunday 02:00, completes before Monday morning
  - Lock file to prevent concurrent runs

### Exit Criteria
- One full weekly scrape cycle completes successfully
- New content appears in RAG search results
- Weekly digest report generated
- Dedup guard prevents re-ingestion on second run

---

## Sprint 12: Code Evolution Engine

**Hardware**: BEAST REQUIRED (GPU for model inference + fine-tuning)
**Dependency**: Sprint 10 (Devstral installed), Sprint 9 (learning pipeline active)
**Duration**: 3-4 sessions

### Goal
Build the self-improvement loop: JCoder generates code, tests it,
analyzes failures, improves, and optionally fine-tunes itself.

### Tasks

- [ ] Simulation harness:
  - Select coding challenges from eval set + HumanEval + code_contests
  - Run JCoder agent on each challenge (generate solution + tests)
  - Collect metrics: pass rate, latency, code quality score
- [ ] Weakness analyzer:
  - Parse test results to identify failure patterns
  - Categorize: syntax errors, logic errors, missing edge cases, wrong algorithm
  - Rank categories by frequency and severity
- [ ] Improvement generator:
  - For each weak category, generate targeted practice problems
  - Run agent self-study with RAG-grounded improvement attempts
  - Capture successful strategies into procedural memory
- [ ] Verification loop:
  - Re-run challenges after improvement cycle
  - Confirm gains without regressions
  - Best-of-N selection (keep only improvements that pass all tests)
- [ ] Distillation at scale:
  - Send top-100 hardest questions to Claude API
  - Get expert solutions + explanations
  - Index enriched content, re-evaluate
  - Budget: track cost per improvement point
- [ ] QLoRA fine-tuning (stretch goal):
  - Collect high-quality (prompt, response) pairs from distillation
  - Fine-tune Devstral 24B on dual RTX 3090 using QLoRA
  - Evaluate fine-tuned model vs base on 200q eval
  - Keep fine-tuned weights only if measurably better

### Exit Criteria
- One full evolution cycle (generate -> test -> analyze -> improve -> verify)
- Measurable improvement on coding benchmarks after cycle
- Distilled content indexed and searchable
- QLoRA experiment results documented (if attempted)

---

## Sprint 13: Production Hardening

**Hardware**: BEAST primary, Toaster for portability testing
**Dependency**: Sprint 10-12 features stabilized
**Duration**: 2-3 sessions

### Goal
Make JCoder a reliable daily-driver tool that survives 8-hour coding
sessions without crashes, memory leaks, or degraded performance.

### Tasks

- [ ] Crash recovery:
  - Checkpoint full agent state every N steps (configurable)
  - Resume from last checkpoint on restart
  - Conversation history persistence across crashes
- [ ] Memory management:
  - GPU VRAM monitoring with configurable thresholds
  - Graceful OOM handling: offload KV cache, reduce context, warn user
  - FTS5 connection pooling (limit open handles for 85+ indexes)
- [ ] Performance tuning:
  - KV cache optimization (quantized cache for longer context)
  - Speculative decoding (if model supports it)
  - Batch embedding requests (group multiple chunks per API call)
  - Index warm-up on startup (preload hot indexes)
- [ ] User preferences:
  - Persistent settings per project (language, style, test framework)
  - Custom tool permissions per working directory
  - Default mode selection (code, debug, review, explain)
- [ ] Packaging:
  - Single install script (setup.py or pyproject.toml entry point)
  - Model auto-download on first run (with progress bar)
  - Health check command: `jcoder doctor` (verify all dependencies)
- [ ] Documentation:
  - User guide with examples for each command
  - Architecture doc with component diagrams
  - API reference for all public modules
  - Troubleshooting guide (common errors + fixes)
- [ ] Portability validation:
  - Run full test suite on Toaster (FTS5-only mode)
  - Run full test suite on BEAST (full FAISS + FTS5)
  - Verify graceful degradation: missing GPU, missing Ollama, no internet

### Exit Criteria
- 8-hour stress test without crashes or memory leaks
- Doctor command reports all-green on BEAST
- Graceful degradation verified on Toaster
- Full documentation suite published

---

## Sprint 14: LimitlessApp V2 Integration (Stretch)

**Hardware**: BEAST preferred
**Dependency**: Sprint 9 (self-learning active), Sprint 12 (evolution engine)
**Duration**: 2-3 sessions

### Goal
Connect LimitlessApp V2's advanced learning capabilities to JCoder:
Thompson sampling for strategy selection, semantic distillation for
memory compression, and evolution loop for continuous self-improvement.

### Tasks

- [ ] Thompson sampling integration:
  - Track success/failure per retrieval strategy (FTS5-only, hybrid, federated)
  - Thompson sampling selects strategy per query type
  - Bayesian update after each query result
- [ ] Semantic distillation:
  - Periodically compress agent_memory (merge similar entries)
  - Extract generalizable patterns from accumulated experience
  - Prune low-confidence or stale memories
- [ ] Evolution loop:
  - Self-scoring: rate own answer quality before user feedback
  - Self-refine: if score < threshold, retry with different strategy
  - Adaptive prompting: evolve prompt templates based on success rates
- [ ] Cross-session learning:
  - Transfer insights between projects
  - Build user-specific coding style model
  - Track long-term improvement trends

### Exit Criteria
- Thompson sampling demonstrably selects better strategies over time
- Memory size stabilizes (distillation prevents unbounded growth)
- Self-scoring correlates with actual answer quality

---

## Timeline Summary

| Sprint | Scope | Hardware | Status |
|--------|-------|----------|--------|
| 1-6 | Foundation through production basics | Toaster | DONE |
| 7 | Online API eval + distillation pilot | Toaster | NEXT |
| 8 | Hybrid search activation | Toaster/BEAST | -- |
| 9 | Self-learning pipeline activation | Toaster | -- |
| 10 | BEAST hardware migration | BEAST | BLOCKED on hardware |
| 11 | Weekly knowledge scraper | BEAST/Toaster | -- |
| 12 | Code evolution engine | BEAST | -- |
| 13 | Production hardening | BEAST | -- |
| 14 | LimitlessApp V2 integration (stretch) | BEAST | -- |

### Critical Path

```
Sprint 7 (API eval) -----> Sprint 9 (self-learning)
                    \
                     ----> Sprint 8 (hybrid search) -----> Sprint 10 (BEAST)
                                                              |
                                                              v
                                          Sprint 11 (scraper) + Sprint 12 (evolution)
                                                              |
                                                              v
                                                    Sprint 13 (hardening)
                                                              |
                                                              v
                                                    Sprint 14 (LimitlessApp)
```

Sprints 7, 8, 9 can run in parallel on the Toaster.
Sprint 10 unblocks 11-14 (all need BEAST hardware).
Sprint 14 is stretch -- defer if other priorities emerge.
