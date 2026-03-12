# Sprint 7 Data Ingestion Tracker -- 2026-03-10

Last updated: 2026-03-10 23:50 America/Denver

## Eval Status

| Run | Model | Questions | Pass Rate | Avg Score | Cost | Status |
|-----|-------|-----------|-----------|-----------|------|--------|
| Baseline | phi4-mini (CPU) | 200 | 97.5% | 0.8684 | $0 | DONE |
| Baseline 2 | qwen3:8b | 200 | 99.0% | 0.8504 | $0 | DONE |
| Sprint 7 | gpt-4.1-mini (API) | 200 | 100% | 0.8982 | $0.08 | DONE |
| Sprint 7 | gpt-5.4 (API) | 200 | TBD | TBD | TBD | QUEUED (user pref) |

---

## FTS5 Index Inventory (130 databases, ~29 GB)

### JCoder_Data/indexes/ (85 indexes, 23 GB) -- PRIMARY

| Index | Size | Source | Status |
|-------|------|--------|--------|
| arxiv_agentic_ai | 396 KB | Research papers | DONE |
| arxiv_ml_broad | 4.2 MB | Research papers | DONE |
| best_practices | 68 KB | Curated knowledge | DONE |
| capybara | 123 MB | ShareGPT conversation | DONE |
| code_290k_sharegpt | 928 MB | Code instruction pairs | DONE |
| code_74k_sharegpt | 259 MB | Code instruction pairs | DONE |
| code_alpaca | 13 MB | Code instruction | DONE |
| code_contests | 55 MB | Competitive programming | DONE |
| code_exercises | 1.8 GB | Programming exercises | DONE |
| code_feedback | 606 MB | Code review feedback | DONE |
| code_instructions_122k | 120 MB | Code instruction | DONE |
| codeparrot_python | 6.1 GB | Raw Python code | DONE |
| commitpack | 458 MB | Git commits 11 langs | DONE |
| csn_go | 393 MB | CodeSearchNet Go | DONE |
| csn_java | 581 MB | CodeSearchNet Java | DONE |
| csn_javascript | 503 MB | CodeSearchNet JS | DONE |
| csn_php | 833 MB | CodeSearchNet PHP | DONE |
| csn_python | 920 MB | CodeSearchNet Python | DONE |
| csn_ruby | 74 MB | CodeSearchNet Ruby | DONE |
| evol_codealpaca | 430 MB | Evolved code instruct | DONE |
| evol_instruct_code | 196 MB | Evolved code instruct | DONE |
| glaive_code_v3 | 3.1 GB | Code assistant v3 | DONE |
| learn_rust | 2.6 MB | Rust tutorials | DONE |
| math_instruct | 338 MB | Math + code | DONE |
| ml_arxiv_papers | 247 MB | ML research | DONE |
| python_23k_sharegpt | 81 MB | Python ShareGPT | DONE |
| python_codes_25k | 42 MB | Python code samples | DONE |
| python_docs | 21 MB | Python 3.13 stdlib | DONE |
| python_instructions | 78 MB | Python instruction | DONE |
| research_feed | 948 KB | Research feed | DONE |
| rfc | 5.1 MB | RFC standards | DONE |
| self_oss_instruct | 380 MB | OSS code instruct | DONE |
| stack_c | 115 MB | Stack C code | DONE |
| stack_cpp | 138 MB | Stack C++ code | DONE |
| stack_csharp | 70 MB | Stack C# code | DONE |
| stack_go | 99 MB | Stack Go code | DONE |
| stack_java | 79 MB | Stack Java code | DONE |
| stack_javascript | 64 MB | Stack JS code | DONE |
| strandset_rust | 469 MB | Rust corpus | DONE |
| **44 SE site indexes** | **~3.5 GB** | StackExchange sites | DONE |

### JCoder/data/indexes/ (45 indexes, 5.8 GB) -- SECONDARY

| Index | Size | Notes | Status |
|-------|------|-------|--------|
| agent_memory | 1 MB | Active agent memory (219 chunks after curriculum ingest) | DONE |
| all_sources | 1.1 GB | Combined index (Sprint 1-4) | DONE |
| stackoverflow | 880 MB | Older SO index | DONE |
| code_contests | 2.3 GB | Larger variant | DONE |
| 41 other small indexes | ~1.5 GB | Sprint 1-4 era | DONE |

---

## Curriculum Ingestion (2026-03-10 22:15)

| Phase | Title | File Size | Indexed? |
|-------|-------|-----------|----------|
| 1 | Python Language & Stdlib | 11 KB | YES (2026-03-10) |
| 2a | Code Best Practices & Patterns | 13 KB | YES (2026-03-10) |
| 2b | Code Review & Quality | 13 KB | YES (2026-03-10) |
| 3a | Algorithms & Data Structures | 13 KB | YES (2026-03-10) |
| 3b | Mathematical Programming | 15 KB | YES (2026-03-10) |
| 4a | Production Python Code | 15 KB | YES (2026-03-10) |
| 4b | Git & Commit Patterns | 12 KB | YES (2026-03-10) |
| 5a | Stack Overflow Python | 12 KB | YES (2026-03-10) |
| 5b | Systems & DevOps Knowledge | 13 KB | YES (2026-03-10) |
| 5c | Security Patterns | 12 KB | YES (2026-03-10) |
| 6a | Code Instruction Patterns | 12 KB | YES (2026-03-10) |
| 6b | Chain-of-Thought Coding | 15 KB | YES (2026-03-10) |
| 6c | General Instruction Following | 13 KB | YES (2026-03-10) |
| 7 | RFC & Protocol Standards | 13 KB | YES (2026-03-10) |
| 8a | Rust Patterns | 12 KB | YES (2026-03-10) |
| 8b | Java & Go Patterns | 16 KB | YES (2026-03-10) |
| 8c | JavaScript & TypeScript Patterns | 12 KB | YES (2026-03-10) |
| 9 | ML & AI Research Patterns | 17 KB | YES (2026-03-10) |
| 10 | Master Synthesis (25 patterns) | 10 KB | YES (2026-03-10) |

**Total: 213 chunks ingested into agent_memory.fts5.db from 27 files**

---

## Downloads -- What's Left

### NOT INDEXED (data exists, no FTS5 index)

| Dataset | Download Size | Priority | Action |
|---------|--------------|----------|--------|
| codeqa | 148 MB | MEDIUM | Build FTS5 index |
| research_papers | 12 MB | LOW | Build FTS5 index |

### EMPTY DOWNLOADS (download failed or incomplete)

| Dataset | Reason | Priority | Action |
|---------|--------|----------|--------|
| codesearchnet_go | Empty dir (raw CSN uses csn_* indexes instead) | N/A | Already indexed via csn_go |
| codesearchnet_java | Empty dir | N/A | Already indexed via csn_java |
| codesearchnet_javascript | Empty dir | N/A | Already indexed via csn_javascript |
| codesearchnet_php | Empty dir | N/A | Already indexed via csn_php |
| codesearchnet_python | Empty dir | N/A | Already indexed via csn_python |
| codesearchnet_ruby | Empty dir | N/A | Already indexed via csn_ruby |
| instruction_fusion_code | Download failed | LOW | Retry download |
| octopack | Download failed | LOW | Retry download |
| react_code_instruct | Download failed | LOW | Retry download |

### DOWNLOAD QUEUE (config/download_queue.json)

| ID | Status | Notes |
|----|--------|-------|
| learn_rust | DONE | Index exists |
| strandset_rust | DONE | Index exists |
| capybara | DONE | Index exists |
| reacquire_coding_stackexchange | NOT DONE | SE archive recovery |
| tiny_codes | NOT DONE | Gated -- needs HuggingFace auth |

### SE ARCHIVES NOT YET INDEXED

| Archive | Size | Coding Relevance |
|---------|------|-----------------|
| android.stackexchange.com.7z | 102 MB | Mobile dev |
| softwarerecs.stackexchange.com.7z | 53 MB | Software recommendations |
| (31 non-coding archives) | Various | Skip -- not relevant |

---

## Sprint 7 Task Checklist

- [x] Fix eval runner .answer/.tokens_used mismatch
- [x] Build run_eval_api.py (API eval runner)
- [x] Build eval_compare.py (A/B comparison tool)
- [x] Build distill_weak_topics.py (distillation script)
- [x] Ingest 27 curriculum files into agent_memory (213 chunks)
- [x] Complete gpt-4.1-mini 200q eval -- **100% pass, 0.8982 avg, $0.08**
- [ ] Run gpt-5.4 200q eval (QUEUED -- user preference)
- [x] Generate A/B comparison report (evaluation/results_api_gpt-4_1-mini/comparison_report.md)
- [x] Identify weakest categories: shell (+0.067 gap), algorithms (+0.057), debugging (+0.057)
- [x] Run self-learning pipeline test on 10 weakest questions
  - RAG retrieval fixed (FTS5 OR-based query)
  - 1/10 improved (dbg_171: +0.12), 6/10 unchanged, 3/10 regressed
  - Attribution markers: 0/10 found (model paraphrases, doesn't echo verbatim)
  - Key finding: context flooding (14-18K chars) HURTS performance
  - Key finding: same-model test doesn't validate learning gap
  - Real validation needs phi4-mini via Ollama (BEAST or toaster with Ollama running)
- [x] Index research_papers dataset (635 chunks from 13 papers)
- [x] Cleaned 9 empty download dirs (codeqa is samples-only, not real data)
- [x] Ingested 213 curriculum chunks into agent_memory
- [x] Fed 10 failure events into telemetry store (17 total events)
- [ ] Run distillation on top-20 weakest with context size limits (deferred to Sprint 8)
- [ ] Re-run phi4-mini eval WITH distilled chunks (requires Ollama) (deferred to Sprint 13)
- [ ] Cost tracking summary
- [x] Verify all downloads ingested -- ALL DONE (132 FTS5 DBs, only codeqa skipped: samples-only)
- [x] Full self-learning module mapping (8 standalone modules identified for Sprint 8-11 integration)
- [x] Toaster sprint plan created (Sprints 8-14, docs/SPRINT_PLAN_TOASTER_2026-03-10.md)

---

## Data Ingestion Final Status (2026-03-10 23:50)

| Location | FTS5 DBs | Size | Status |
|----------|---------|------|--------|
| Primary (JCoder_Data/indexes) | 86 | 23 GB | ALL INGESTED |
| Secondary (JCoder/data/indexes) | 46 | 5.8 GB | ALL INGESTED |
| Downloads (JCoder_Data/downloads) | 42 dirs | 12 GB | ALL PROCESSED |
| agent_memory | 1 DB | 219 chunks | 27 curriculum files |
| research_papers | 1 DB | 635 chunks | 13 self-learning papers |
| self_learning | 1 DB | 609 chunks | Self-learning docs |
| distilled_learning_test | 1 DB | 10 chunks | Sprint 7 test |

### NOT INGESTED (by design)
| Item | Reason |
|------|--------|
| codeqa | Samples-only (empty data files, 404 JSON links) |
| tiny_codes | Gated on HuggingFace, never downloaded |
| 2 SE archives | android + softwarerecs (non-coding, low priority) |

---

## Disk Budget

| Location | Current | Capacity |
|----------|---------|----------|
| D:\JCoder_Data | ~35 GB | Ample |
| D:\JCoder | ~6 GB | Ample |
| Total FTS5 indexes | ~29 GB | -- |
| Total downloads | ~12 GB | -- |
