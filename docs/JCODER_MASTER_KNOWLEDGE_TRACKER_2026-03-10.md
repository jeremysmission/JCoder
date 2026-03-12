# JCoder Master Knowledge Tracker

Last updated: 2026-03-10 23:45 America/Denver

---

## MISSION

Make JCoder the best coding assistant, agent, and agent coordinator on Earth.
Not "good enough" -- the best. Every dataset, every expert document, every lesson
plan serves this goal.

---

## COMPLETE INGESTION INVENTORY

Everything JCoder has ever learned, organized by source.

### 1. Code Corpora (Primary) -- 86 FTS5 Indexes, ~8.95M chunks, 23 GB

These are the original code data dumps from Stack Exchange, GitHub code dumps,
and programming tutorial sites. Ingested in Sprints 1-6.

| # | Source Category | Indexes | Est Chunks | Notes |
|---|----------------|---------|------------|-------|
| 1 | Stack Exchange (44 sites) | 44 | ~3.5M | stackoverflow, codereview, softwareengineering, unix, etc |
| 2 | GitHub code archives | 20 | ~3M | Multiple language repos, open-source projects |
| 3 | Programming tutorials | 10 | ~1M | Python, JS, Rust, Go, Java tutorials |
| 4 | Documentation dumps | 12 | ~1.45M | Man pages, RFC summaries, API docs |

### 2. Code Corpora (Secondary) -- 46 FTS5 Indexes, ~2M chunks, 5.8 GB

Additional code corpora from specialized sources.

| # | Source Category | Indexes | Est Chunks | Notes |
|---|----------------|---------|------------|-------|
| 1 | CommitPackFT | 5 | ~23K | Commit messages + diffs (5 languages) |
| 2 | DS-1000 | 1 | ~1.2K | Data science code problems |
| 3 | HumanEvalPack | 1 | ~828 | Code generation benchmarks |
| 4 | Other specialized | 39 | ~1.97M | Various code quality, style, and reference data |

### 3. New Datasets (Sessions 13-14) -- 9 FTS5 Indexes, 659K chunks, 2.3 GB

Downloaded and indexed 2026-03-10.

| # | Dataset | HuggingFace ID | Entries | Chunks | FTS5 Size | Status |
|---|---------|---------------|---------|--------|-----------|--------|
| 1 | LeetCode Problems | newfacade/LeetCodeDataset | 2,641 | 2,671 | small | DONE |
| 2 | CRUXEval | cruxeval-org/cruxeval | 800 | 800 | small | DONE |
| 3 | BigCodeBench | bigcode/bigcodebench | 1,140 | 1,142 | small | DONE |
| 4 | DebugBench | Rtian/DebugBench | 4,253 | 4,729 | small | DONE |
| 5 | Codeforces | open-r1/codeforces | 9,552 | 10,392 | small | DONE |
| 6 | Software Architecture | ajibawa-2023/Software-Architecture | 448,100 | 451,143 | 1,472 MB | DONE |
| 7 | OpenCodeReasoning-2 | nvidia/OpenCodeReasoning-2 | 120,000 | 120,000 | 545 MB | DONE |
| 8 | SWE-smith | SWE-bench/SWE-smith | 59,136 | 59,747 | 210 MB | DONE |
| 9 | XLCoST | codeparrot/xlcost-text-to-code | 9,263 | 9,299 | 20 MB | DONE |
| 10 | OpenCodeInstruct | nvidia/OpenCodeInstruct | 600K/5M | partial | - | DEFERRED (disk) |

### 4. Agent Memory (Curriculum + Expert Docs) -- 353 chunks

Distilled knowledge stored in agent_memory.fts5.db for direct retrieval.

#### Phase 1: Curriculum Foundations (219 chunks, 27 files)
| # | Topic | Source File | Chunks | Status |
|---|-------|-----------|--------|--------|
| 1 | Python Language & Stdlib | phase1_python | ~12 | DONE |
| 2 | Code Best Practices & Patterns | phase2a | ~12 | DONE |
| 3 | Code Review & Quality | phase2b | ~12 | DONE |
| 4 | Algorithms & Data Structures | phase3a | ~12 | DONE |
| 5 | Mathematical Programming | phase3b | ~12 | DONE |
| 6 | Production Python Code | phase4a | ~12 | DONE |
| 7 | Git & Commit Patterns | phase4b | ~12 | DONE |
| 8 | Stack Overflow Python | phase5a | ~12 | DONE |
| 9 | Systems & DevOps Knowledge | phase5b | ~12 | DONE |
| 10 | Security Patterns | phase5c | ~12 | DONE |
| 11 | Code Instruction Patterns | phase6a | ~12 | DONE |
| 12 | Chain-of-Thought Coding | phase6b | ~12 | DONE |
| 13 | General Instruction Following | phase6c | ~12 | DONE |
| 14 | RFC & Protocol Standards | phase7 | ~12 | DONE |
| 15 | Rust Patterns | phase8a | ~12 | DONE |
| 16 | Java & Go Patterns | phase8b | ~12 | DONE |
| 17 | JavaScript & TypeScript | phase8c | ~12 | DONE |
| 18 | ML & AI Research Patterns | phase9 | ~12 | DONE |
| 19 | Master Synthesis | phase10 | ~9 | DONE |

#### Phase 2: Expert Knowledge (36 chunks, 5 files)
| # | Topic | File | Chunks | Status |
|---|-------|------|--------|--------|
| 1 | Autonomous Software Agent Blueprint | autonomous_software_agent_blueprint.md | 9 | DONE |
| 2 | RAG Tuning Expert Knowledge | rag_tuning_expert_knowledge.md | 6 | DONE |
| 3 | Multi-Agent Team Coordination | multi_agent_team_coordination.md | 8 | DONE |
| 4 | AI Distillation Techniques | ai_distillation_techniques_2026.md | 6 | DONE |
| 5 | Agent Framework Patterns | agent_framework_patterns_2026.md | 7 | DONE |

#### Phase 5: Expert Knowledge Documents (98 chunks, 8 files)
| # | Topic | File | Chunks | Status |
|---|-------|------|--------|--------|
| 1 | Python AI/ML Engineering | python_ai_ml_engineering.md | 14 | DONE |
| 2 | API Design Best Practices | api_design_best_practices.md | 10 | DONE |
| 3 | Database Engineering | database_engineering.md | 16 | DONE |
| 4 | Testing Strategy Expert | testing_strategy_expert.md | 11 | DONE |
| 5 | DevOps & CI/CD Patterns | devops_cicd_patterns.md | 8 | DONE |
| 6 | Concurrency & Async Python | concurrency_async_python.md | 11 | DONE |
| 7 | Error Handling & Resilience | error_handling_resilience.md | 10 | DONE |
| 8 | Documentation Engineering | documentation_engineering.md | 18 | DONE |

### 5. Research & Self-Learning -- 3 FTS5 Indexes, 1,254 chunks

| # | Source | Index | Chunks | Status |
|---|--------|-------|--------|--------|
| 1 | 13 self-learning papers | research_papers.fts5.db | 635 | DONE |
| 2 | Self-learning docs | self_learning.fts5.db | 609 | DONE |
| 3 | Sprint 7 distillation test | distilled_learning_test.fts5.db | 10 | DONE |

### 6. Other Agent Memory Entries -- 6 entries
| # | Source | Chunks |
|---|--------|--------|
| 1 | Self-learning task results | 3 |
| 2 | Agent tool store patterns | 2 |
| 3 | Research query results | 1 |

---

## GRAND TOTALS

| Category | Indexes | Chunks | Size |
|----------|---------|--------|------|
| Code corpora (primary) | 86 | ~8.95M | 23 GB |
| Code corpora (secondary) | 46 | ~2M | 5.8 GB |
| New datasets (session 13-14) | 9 | ~660K | 2.3 GB |
| Agent memory | 1 | 353 | <1 MB |
| Research/self-learning | 3 | 1,254 | <1 MB |
| **TOTAL INGESTED** | **145** | **~11.6M** | **~31.1 GB** |

---

## PHASE 6: WORLD-CLASS AGENT CURRICULUM

### Sprint 16: Agent Trajectories & Reasoning (HIGHEST PRIORITY) -- DOWNLOADING
**Goal**: Teach JCoder HOW agents solve real bugs step by step.
**Script**: `scripts/download_phase6_datasets.py --category A`

| Task | Dataset | HuggingFace ID | Entries | Status |
|------|---------|---------------|---------|--------|
| 16.1 | HumanEval+ | evalplus/humanevalplus | 164 | DOWNLOADING |
| 16.2 | MBPP+ | evalplus/mbppplus | 378 | DOWNLOADING |
| 16.3 | SWE-smith Trajectories | SWE-bench/SWE-smith-trajectories | 5,017 | DOWNLOADING |
| 16.4 | OpenHands Trajectories | nebius/SWE-rebench-openhands-trajectories | 67,074 | DOWNLOADING |
| 16.5 | SWE-agent Trajectories | nebius/SWE-agent-trajectories | 80,036 | DOWNLOADING |
| 16.6 | MEnvData Trajectories | ernie-research/MEnvData-SWE-Trajectory | 3,872 | DOWNLOADING |
| 16.7 | CoderForge Preview | togethercomputer/CoderForge-Preview | 51,000 | DOWNLOADING |
| 16.8 | Code Review Python | Dahoas/code-review-instruct-critique-revision-python | varies | DOWNLOADING |
| 16.9 | Code Review General | VatsaDev/code-review | varies | DOWNLOADING |
| 16.10 | Expert doc: Agent Reasoning | create new | ~15 chunks | QUEUED |
| 16.11 | Expert doc: Code Review Mastery | create new | ~15 chunks | QUEUED |

### Sprint 17: Coding Instruction & Mastery (HIGH PRIORITY)
**Goal**: Massive instruction-following datasets to cover every coding pattern.
**Script**: `scripts/download_phase6_datasets.py --category B`

| Task | Dataset | HuggingFace ID | Entries | Status |
|------|---------|---------------|---------|--------|
| 17.1 | Magicoder 110K | ise-uiuc/Magicoder-Evol-Instruct-110K | 110,000 | QUEUED |
| 17.2 | Evol-CodeAlpaca | theblackcat102/evol-codealpaca-v1 | varies | QUEUED |
| 17.3 | OpenCodeInstruct (retry) | nvidia/OpenCodeInstruct | 5M | QUEUED (BEAST) |
| 17.4 | Expert doc: Library API Mastery | create new | ~15 chunks | QUEUED |

### Sprint 18: Security & Vulnerability Intelligence (MEDIUM-HIGH PRIORITY)
**Goal**: JCoder writes secure code by default, knows every CWE and CVE pattern.
**Script**: `scripts/download_phase6_datasets.py --category C`

| Task | Dataset | HuggingFace ID | Entries | Status |
|------|---------|---------------|---------|--------|
| 18.1 | CIRCL Vuln+CWE+Patch | CIRCL/vulnerability-cwe-patch | 39,260 | QUEUED |
| 18.2 | CVE+CWE 1999-2025 | stasvinokur/cve-and-cwe-dataset-1999-2025 | all CVEs | QUEUED |
| 18.3 | Security DPO | CyberNative/Code_Vulnerability_Security_DPO | varies | QUEUED |
| 18.4 | SecureCode Web | scthornton/securecode-web | 1,378 | QUEUED |
| 18.5 | CVE Training | AlicanKiraz0/All-CVE-Records-Training-Dataset | 300,000 | QUEUED |
| 18.6 | Expert doc: Security-First Coding | create new | ~15 chunks | QUEUED |

### Sprint 19: Large-Scale Code & Project Structure (MEDIUM PRIORITY)
**Goal**: Latest real-world code from 2025 + project scaffolding knowledge.
**Script**: `scripts/download_phase6_datasets.py --category D`

| Task | Dataset | HuggingFace ID | Entries | Status |
|------|---------|---------------|---------|--------|
| 19.1 | GitHub Code 2025 | nick007x/github-code-2025 | 1.5M+ | QUEUED |
| 19.2 | Expert doc: Project Scaffolding | create new | ~15 chunks | QUEUED |

### Sprint 20: API Docs & Reference (MANUAL SCRAPE)
**Goal**: JCoder knows every Python stdlib function and popular library API cold.

| Task | Source | Method | Status |
|------|--------|--------|--------|
| 20.1 | Python stdlib docs | scrape docs.python.org | PLANNED |
| 20.2 | FastAPI docs | scrape fastapi.tiangolo.com | PLANNED |
| 20.3 | Pydantic docs | scrape docs.pydantic.dev | PLANNED |
| 20.4 | pytest docs | scrape docs.pytest.org | PLANNED |
| 20.5 | httpx docs | scrape www.python-httpx.org | PLANNED |

---

## EXISTING SPRINT HISTORY (Complete)

| Sprint | Focus | Status | Key Deliverables |
|--------|-------|--------|-----------------|
| 1 | Core RAG pipeline | DONE | FTS5 search, chunking, prompt builder |
| 2 | Embedding & indexing | DONE | nomic-embed-code, batch indexer |
| 3 | Query engine & CLI | DONE | Query pipeline, agent CLI |
| 4 | Self-learning foundation | DONE | Telemetry, experience replay, Thompson sampling |
| 5 | Agent framework | DONE | Bridge, tools, subagent spawning |
| 6 | Data ingestion pipeline | DONE | Download manager, SE indexing, 85 FTS5 DBs |
| 7 | Knowledge curriculum | DONE | 9 new datasets, 13 expert docs, eval baselines |
| 8 | Self-learning activation | PLANNED | Wire SelfLearningPipeline, feedback loop |
| 9 | Reasoning & reflection | PLANNED | STaR, Best-of-N, Reflection wiring |
| 10 | Corrective retrieval | PLANNED | Smart Orchestrator, Knowledge Graph |
| 11 | Prompt evolution | PLANNED | Adversarial self-play |
| 12 | Cascade router | PLANNED | Multi-model routing |
| 13 | E2E validation | PLANNED | Self-learning validation |
| 14 | Knowledge scraper | PLANNED | Weekly auto-update pipeline |
| 15 | Multi-agent spawning | PLANNED | Subagent coordination system |
| **16** | **Code reasoning** | **NEW** | **Agent trajectories, code review, refactoring** |
| **17** | **API mastery** | **NEW** | **Python docs, library docs** |
| **18** | **Security & reliability** | **NEW** | **CWE/OWASP, postmortems** |
| **19** | **Project & DevOps** | **NEW** | **Scaffolding, CI/CD, schemas** |
| **20** | **Architecture & perf** | **NEW** | **ADRs, OpenAPI, benchmarks** |

---

## WHAT'S LEFT (Priority Order)

### Immediate (Can do now on toaster)
1. Download and index Phase 6 small datasets (< 500 MB each)
2. Create Phase 6 expert knowledge documents (5 new docs)
3. Sprint 8-9: Wire self-learning pipeline and reasoning modules

### Needs BEAST Hardware
1. OpenCodeInstruct full index (5M entries, ~10 GB)
2. Any dataset > 1 GB raw download
3. Sprint 12: Cascade router (needs multiple models loaded)
4. Full 400q eval with phi4:14b

### Needs API Access
1. Distillation from Claude/GPT-5/Gemini (generate expert explanations)
2. EnrichIndex (pre-generate summaries for all FTS5 chunks)
3. Sprint 13: E2E self-learning validation with real LLM

---

## DISK BUDGET

| Location | Used | Purpose |
|----------|------|---------|
| D:\JCoder_Data\indexes | 23 GB | Primary FTS5 indexes |
| D:\JCoder\data\indexes | 5.8 GB | Secondary + new indexes |
| D:\JCoder_Data\downloads | 12 GB | Download cache (can clean) |
| **Total** | **~41 GB** | |

**Strategy**: Clean download cache after indexing. Prioritize small high-value datasets.
Defer large datasets to BEAST hardware.
