# JCoder Sprint Plan -- Toaster + Online API Track
# Created: 2026-03-08
# Context: BEAST not built yet. Toaster (16GB RAM, Intel Iris Xe) is dev machine.
# Strategy: Use online API for eval/testing + nomic-embed-text for vector indexing.
# Do NOT wait for BEAST -- maximize progress with what's available.

---

## STATUS ENTERING THIS PLAN

- **799 tests pass, 0 fail** (Sprint 5 code verified 2026-03-08)
- **59 FTS5 indexes, ~19 GB** across JCoder_Data and JCoder/data
- **200q eval: 97.5% pass** (phi4-mini CPU-only)
- **Vector infra ready**: EmbeddingEngine, IndexEngine (FAISS+FTS5), FederatedSearch all coded and tested
- **Online API supported**: agent.yaml already has OpenRouter/Anthropic backend config
- **nomic-embed-text installed** on Ollama (768-dim, localhost:11434)

### Data Gaps Found (2026-03-08 audit)

| Issue | Dataset | Action |
|-------|---------|--------|
| Empty FTS5 (0 rows) | strandset_rust (191K rows parquet) | Re-index |
| Empty FTS5 (0 rows) | capybara (16K rows parquet) | Re-index |
| Partial FTS5 (5K/20K) | python_23k_sharegpt (unclosed WAL) | Delete WAL, re-index |
| Download failed (404) | CodeQA | Drop or find alt source |
| Download failed (empty) | OctoPack, react_code_instruct, instruction_fusion_code | Drop or retry |
| Never downloaded | learn_rust, tiny_codes | Download if still valuable |
| Redundant raw data | 12 GB CSN JSONL in raw_downloads | Delete after confirming indexes |

---

## SPRINT 7: Data Cleanup & Vector Bootstrap
**Target: Toaster, no external deps, 1-2 sessions**

### 7.1 Commit & Push
- [x] Verify Sprint 5 agent code (799 pass, 0 fail -- DONE 2026-03-08)
- [ ] Commit all verified Sprint 5 changes + untracked docs
- [ ] Push to GitHub master

### 7.2 Fix Broken FTS5 Indexes
- [ ] Delete strandset_rust.fts5.db (0-row shell), rebuild from parquet
- [ ] Delete capybara.fts5.db (0-row shell), rebuild from parquet
- [ ] Delete python_23k WAL/SHM files, rebuild from parquet
- [ ] Verify row counts match source data after rebuild
- [ ] Command: `python tools/ingest_all_pending.py` or manual `build_fts5_indexes.py`

### 7.3 Clean Up Dead Data
- [ ] Delete empty download dirs (OctoPack, react_code_instruct, instruction_fusion_code)
- [ ] Delete D:\JCoder_Data\raw_downloads\codesearchnet\ (12 GB, redundant)
- [ ] Delete D:\JCoder_Data\prep_stage\ (3 MB stale reports)
- [ ] Delete HF .cache and .lock files in downloads dirs
- [ ] Decision: drop CodeQA (404) or find alternative source

### 7.4 Vector Index Bootstrap (nomic-embed-text)
- [ ] Verify Ollama serving nomic-embed-text: `ollama list`
- [ ] Pick top-5 highest-value FTS5 indexes for initial embedding:
  1. csn_python (919 MB FTS5, highest code relevance)
  2. python_docs (22 MB, authoritative reference)
  3. python_instructions (78 MB, Q&A pairs)
  4. best_practices (0.1 MB, high weight in memory.yaml)
  5. agent_memory (personal RAG, weight 1.5)
- [ ] Run Phase 2 embedding on each:
  ```bash
  python scripts/bulk_ingest_se.py --phase2-embed --index-name csn_python
  ```
- [ ] Build FAISS indexes from embeddings
- [ ] Smoke test: hybrid search (FAISS+FTS5) vs FTS5-only on 10 queries
- [ ] NOTE: Large indexes (codeparrot 6.2GB, glaive 3GB) defer to BEAST

### 7.5 Regression Gate
- [ ] Run full test suite: `python -m pytest tests/ -v --tb=short`
- [ ] Commit + push

---

## SPRINT 8: Online API Integration & Eval
**Target: Toaster + Online API key, 1-2 sessions**
**Prereq: User provides API key/endpoint**

### 8.1 Configure Online API Backend
- [ ] Update config/agent.yaml with provided API credentials
- [ ] Options: OpenRouter (Claude Sonnet/Opus), direct Anthropic, Azure OpenAI
- [ ] Test: `jcoder agent run "Write a Python fibonacci function"` with online model
- [ ] Verify: tool calling works with online model (structured JSON responses)

### 8.2 Online Eval Campaign
- [ ] Run 200q eval with online API model (expect significant quality jump vs phi4-mini)
- [ ] Compare: online model vs phi4-mini per-category scores
- [ ] Identify: which categories benefit most from larger model
- [ ] Document: quality delta, latency delta, cost per query
- [ ] Target: >99% pass rate with online model

### 8.3 Agentic Loop Testing
- [ ] Test multi-step coding tasks (file read -> edit -> test -> fix cycle)
- [ ] Test study mode with online model (richer synthesis than phi4-mini)
- [ ] Test FIM (fill-in-middle) if online model supports it
- [ ] Benchmark: agent iterations needed to solve task (online vs offline)

### 8.4 Distillation (Pilot)
- [ ] Select top-20 most-queried/weakest topics from eval results
- [ ] Send to online API with structured prompt for expert analysis
- [ ] Index enriched explanations alongside original code
- [ ] Re-run affected eval questions -- measure improvement
- [ ] Track cost (tokens used, USD spent)
- [ ] Budget guard: set daily limit before starting

### 8.5 Regression Gate
- [ ] Full test suite pass
- [ ] Commit + push

---

## SPRINT 9: Hybrid Search Activation
**Target: Toaster (embedding runs slow but works), 2-3 sessions**

### 9.1 Expand Vector Coverage
- [ ] Embed next tier of FTS5 indexes (medium-priority):
  - code_290k_sharegpt (928 MB)
  - evol_instruct_code (195 MB)
  - code_feedback (605 MB)
  - commitpack (457 MB)
  - rfc (5 MB)
- [ ] Build FAISS indexes for each
- [ ] Update memory.yaml with FAISS paths

### 9.2 Enable Hybrid Search in Agent
- [ ] Switch retrieval_engine from FTS5-only to hybrid (FAISS+FTS5+RRF)
- [ ] Tune RRF weights: start with defaults (rrf_k=60), measure precision
- [ ] A/B test: 50 queries through FTS5-only vs hybrid, compare relevance
- [ ] Measure latency impact (hybrid adds embedding + FAISS lookup)

### 9.3 Federated Hybrid Search
- [ ] Enable FAISS in federated_search.py for multi-index queries
- [ ] Test: cross-corpus query ("Python async error handling" hits csn_python + python_docs + stackoverflow)
- [ ] Tune per-corpus weights in memory.yaml based on relevance results

### 9.4 Retrieval Quality Metrics
- [ ] Build retrieval eval: 50 queries with expected source chunks
- [ ] Measure: precision@5, recall@5, MRR
- [ ] Compare: FTS5-only vs hybrid vs hybrid+rerank
- [ ] Document baseline for BEAST comparison later

---

## SPRINT 10: Self-Learning Pipeline Activation
**Target: Toaster + Online API, 2-3 sessions**

### 10.1 Validate Learning Components
- [ ] Test experience replay store (bridge.py -> telemetry -> experience)
- [ ] Test active learner with real queries (not just mocks)
- [ ] Test meta-cognitive loop (agent reflects on its own performance)
- [ ] Test procedural memory persistence across sessions

### 10.2 Online API Self-Study
- [ ] Run `jcoder agent study` with online API model
- [ ] Topics: top-10 weakest categories from eval results
- [ ] Measure: learning delta (pre-study vs post-study eval scores)
- [ ] Compare: online model study quality vs phi4-mini study quality

### 10.3 PRAXIS/RISE Pattern Implementation
- [ ] PRAXIS: experience-driven prompt refinement (Tier 1, immediately implementable)
- [ ] RISE: reflective self-improvement (Tier 1)
- [ ] Wire into agent autopilot loop
- [ ] Measure: does self-improvement actually improve eval scores?

### 10.4 Autopilot Validation
- [ ] Run `jcoder agent autopilot` for 10 cycles
- [ ] Monitor: goal completion rate, strategy selection, memory growth
- [ ] Check: no runaway loops, budget respected, quality maintained

---

## SPRINT 11: BEAST Migration (WHEN HARDWARE READY)
**Target: BEAST (128GB RAM, 48GB VRAM), 2-3 sessions**

### 11.1 Model Installation
- [ ] `ollama pull devstral-small-2:24b` (14 GB, primary coding model)
- [ ] `ollama pull manutic/nomic-embed-code` (code-specific embeddings)
- [ ] Benchmark: Devstral vs phi4-mini vs online API on 200q eval
- [ ] Benchmark: nomic-embed-code vs nomic-embed-text on retrieval precision

### 11.2 Full-Corpus Embedding
- [ ] Embed ALL 59 FTS5 indexes with GPU acceleration
- [ ] Priority: codeparrot (6.2 GB), glaive (3 GB), code_exercises (1.8 GB)
- [ ] Estimate: ~4-8 hours total on RTX 3090
- [ ] Build consolidated FAISS indexes

### 11.3 Large-Scale Data Acquisition
- [ ] CSN Python FTS5 rebuild (326K files)
- [ ] StackOverflow full FTS5 (953K files)
- [ ] The Stack v2 (gated, needs HF token + license)
- [ ] Ultra-large dumps (100+ GB, 2 TB budget)

### 11.4 Stress & VRAM Profiling
- [ ] Concurrent query load test (10 parallel queries)
- [ ] Memory profiling under sustained use
- [ ] VRAM monitoring: Devstral 24B + nomic-embed-code simultaneously
- [ ] Latency target: <5s query-to-answer on BEAST

---

## SPRINT 12: Autonomy (BEAST, Future Phases)
**Target: BEAST, 4-8 weeks**

### 12.1 Weekly Scraper (Subsystem 3)
- [ ] Autonomous knowledge ingestion from top coding sites
- [ ] Scheduled weekly runs (cron/task scheduler)
- [ ] Auto-sanitize, chunk, embed, index new content
- [ ] Quality gate: only index content scoring above threshold

### 12.2 10x Code Evolver (Subsystem 4)
- [ ] Self-improvement loop: agent generates code -> evaluates -> refines
- [ ] Prompt evolution via GEPA/DEEVO patterns
- [ ] QLoRA fine-tuning pipeline (dual RTX 3090)
- [ ] vLLM tensor parallelism for 70B models

---

## KEY DECISIONS

| Decision | Choice | Reason |
|----------|--------|--------|
| Don't wait for BEAST | Use online API + toaster now | Maximize velocity; hardware ETA unknown |
| Embed with nomic-embed-text | Already installed, 768-dim, works | Code-specific model (nomic-embed-code) can upgrade later |
| Start with top-5 indexes | csn_python, python_docs, python_instructions, best_practices, agent_memory | Highest value per compute cost |
| Online API for eval | Quality jump over phi4-mini | Proves system works before BEAST arrives |
| Defer large-corpus embedding | codeparrot 6.2GB, glaive 3GB | Too slow on toaster CPU; worth GPU time |
| Drop failed downloads | CodeQA (404), OctoPack (empty) | Don't chase dead links; plenty of data already |

---

## QUICK REFERENCE

```bash
# Fix broken indexes
python tools/ingest_all_pending.py

# Phase 2 embedding (per index)
python scripts/bulk_ingest_se.py --phase2-embed --index-name INDEX_NAME

# Run eval with online API
python evaluation/agent_eval_runner.py --model claude-sonnet --endpoint OPENROUTER_URL

# Agent with online API
jcoder agent run "your query" --model claude-sonnet

# Self-study
jcoder agent study --topic "async Python patterns"

# Full regression
python -m pytest tests/ -v --tb=short
```
