# JCoder Knowledge Curriculum & Study Plan

Last updated: 2026-03-11 00:30 America/Denver

## Vision

Transform JCoder from a coding Q&A assistant into an **autonomous software-writing agent**
that can:
1. Write complete programs from requirements
2. Manage and coordinate subagent teams
3. Act as project manager, programming lead, and systems engineer
4. Self-improve through experience and distillation
5. Be expert in AI systems, RAG tuning, and Python engineering

---

## Curriculum Structure (Basics -> Advanced)

### Phase 1: Foundations (COMPLETE)
Knowledge already ingested into agent_memory.

| Topic | Source | Chunks | Status |
|-------|--------|--------|--------|
| Python Language & Stdlib | curriculum phase 1 | ~12 | DONE |
| Code Best Practices & Patterns | curriculum phase 2a | ~12 | DONE |
| Code Review & Quality | curriculum phase 2b | ~12 | DONE |
| Algorithms & Data Structures | curriculum phase 3a | ~12 | DONE |
| Mathematical Programming | curriculum phase 3b | ~12 | DONE |
| Production Python Code | curriculum phase 4a | ~12 | DONE |
| Git & Commit Patterns | curriculum phase 4b | ~12 | DONE |
| Stack Overflow Python | curriculum phase 5a | ~12 | DONE |
| Systems & DevOps Knowledge | curriculum phase 5b | ~12 | DONE |
| Security Patterns | curriculum phase 5c | ~12 | DONE |
| Code Instruction Patterns | curriculum phase 6a | ~12 | DONE |
| Chain-of-Thought Coding | curriculum phase 6b | ~12 | DONE |
| General Instruction Following | curriculum phase 6c | ~12 | DONE |
| RFC & Protocol Standards | curriculum phase 7 | ~12 | DONE |
| Rust Patterns | curriculum phase 8a | ~12 | DONE |
| Java & Go Patterns | curriculum phase 8b | ~12 | DONE |
| JavaScript & TypeScript | curriculum phase 8c | ~12 | DONE |
| ML & AI Research Patterns | curriculum phase 9 | ~12 | DONE |
| Master Synthesis | curriculum phase 10 | ~12 | DONE |

**Subtotal: 219 chunks (27 curriculum files)**

### Phase 2: Expert Knowledge (IN PROGRESS)
New knowledge documents created 2026-03-10.

| Topic | File | Chunks | Status |
|-------|------|--------|--------|
| Autonomous Software Agent Blueprint | autonomous_software_agent_blueprint.md | 9 | DONE |
| RAG Tuning Expert Knowledge | rag_tuning_expert_knowledge.md | 6 | DONE |
| Multi-Agent Team Coordination | multi_agent_team_coordination.md | 8 | DONE |
| AI Distillation Techniques | ai_distillation_techniques_2026.md | 6 | DONE |
| Agent Framework Patterns | agent_framework_patterns_2026.md | 7 | DONE |

**Subtotal: 36 chunks ingested (5 knowledge documents)**

### Phase 3: Domain Datasets (IN PROGRESS)
FTS5 indexes available for retrieval.

| Dataset | Index | Chunks | Status |
|---------|-------|--------|--------|
| LeetCode Problems | leetcode.fts5.db | 2,671 | DONE |
| CRUXEval Self-Assessment | cruxeval.fts5.db | 800 | DONE |
| BigCodeBench API Patterns | bigcodebench.fts5.db | 1,142 | DONE |
| DebugBench Bug Explanations | debugbench.fts5.db | 4,729 | DONE |
| Codeforces + Editorials | codeforces.fts5.db | 10,392 | DONE |
| Software Architecture | software_architecture.fts5.db | 451,143 | DONE (1,472 MB) |
| OpenCodeReasoning-2 (Python) | opencodereasoning2.fts5.db | 120,000 | DONE (545 MB) |
| SWE-smith Bug Fixes | swesmith.fts5.db | 59,747 | DONE (210 MB) |
| XLCoST Python Programs | xlcost.fts5.db | 9,299 | DONE (20 MB) |
| OpenCodeInstruct | opencodeinstruct.fts5.db | 600K/5M | DEFERRED (disk I/O error at 600K -- needs BEAST) |

### Phase 4: Self-Learning Research (IN PROGRESS)
Research papers and techniques indexed.

| Source | Index | Chunks | Status |
|--------|-------|--------|--------|
| 13 self-learning papers | research_papers.fts5.db | 635 | DONE |
| Self-learning docs | self_learning.fts5.db | 609 | DONE |
| Sprint 7 distillation test | distilled_learning_test.fts5.db | 10 | DONE |

### Phase 5: Expert Knowledge Documents (COMPLETE)
All 10 expert knowledge documents created and ingested.

| Priority | Topic | File | Chunks | Status |
|----------|-------|------|--------|--------|
| 1 | AI Distillation Techniques | ai_distillation_techniques_2026.md | 6 | DONE (Phase 2) |
| 2 | Autonomous Agent Architectures | agent_framework_patterns_2026.md | 7 | DONE (Phase 2) |
| 3 | Python AI/ML Engineering | python_ai_ml_engineering.md | 14 | DONE |
| 4 | API Design Best Practices | api_design_best_practices.md | 10 | DONE |
| 5 | Database Engineering | database_engineering.md | 16 | DONE |
| 6 | Testing Strategy Expert | testing_strategy_expert.md | 11 | DONE |
| 7 | DevOps & CI/CD Patterns | devops_cicd_patterns.md | 8 | DONE |
| 8 | Concurrency & Async Python | concurrency_async_python.md | 11 | DONE |
| 9 | Error Handling & Resilience | error_handling_resilience.md | 10 | DONE |
| 10 | Documentation Engineering | documentation_engineering.md | 18 | DONE |

**Subtotal: 111 chunks ingested (10 expert documents)**

---

## Download Queue (New Datasets)

| Priority | Dataset | HuggingFace ID | Est Size | Status |
|----------|---------|---------------|----------|--------|
| 1 | LeetCode | newfacade/LeetCodeDataset | 25 MB | DONE |
| 2 | CRUXEval | cruxeval-org/cruxeval | <1 MB | DONE |
| 3 | BigCodeBench | bigcode/bigcodebench | 2.4 MB | DONE |
| 4 | DebugBench | Rtian/DebugBench | 3.1 MB | DONE |
| 5 | Codeforces | open-r1/codeforces | 2.6 GB | DONE |
| 6 | Software Architecture | ajibawa-2023/Software-Architecture | 819 MB | DONE (451K chunks) |
| 7 | OpenCodeReasoning-2 | nvidia/OpenCodeReasoning-2 | 2.2 GB | DONE (120K Python) |
| 8 | SWE-smith | SWE-bench/SWE-smith | 288 MB | DONE (59K entries) |
| 9 | XLCoST | codeparrot/xlcost-text-to-code | 3.9 MB | DONE (9.3K entries) |
| 10 | OpenCodeInstruct | nvidia/OpenCodeInstruct | ~10 GB | DEFERRED (disk full on toaster) |

---

## Knowledge Ingestion Progress

### Total Knowledge Base (as of 2026-03-10 session 14)

| Category | Indexes | Chunks | Size |
|----------|---------|--------|------|
| Code corpora (primary) | 86 | ~8.95M | 23 GB |
| Code corpora (secondary) | 46 | ~2M est | 5.8 GB |
| New datasets (session 13-14) | 9 | 659,922 | 2,315 MB |
| Agent memory (curriculum + expert) | 1 | 353 | <1 MB |
| Research papers | 1 | 635 | <1 MB |
| Self-learning docs | 1 | 609 | <1 MB |
| **TOTAL** | **144+** | **~11.6M** | **~31.1 GB** |

### What's Accomplished (2026-03-10, Sessions 13-14)
- [x] Verified all existing data dumps ingested (132 FTS5 DBs)
- [x] gpt-4.1-mini 200q eval: 100% pass, 0.8982 avg, $0.08
- [x] A/B comparison report generated
- [x] Self-learning pipeline tested (10 questions, attribution study)
- [x] 9 new datasets downloaded and indexed (CRUXEval, BigCodeBench, LeetCode, DebugBench, Codeforces, Software Architecture, OpenCodeReasoning-2, SWE-smith, XLCoST)
- [x] 5 Phase 2 expert knowledge documents (agent blueprint, RAG tuning, multi-agent coordination, AI distillation, agent framework patterns)
- [x] 8 Phase 5 expert knowledge documents (Python AI/ML, API design, database, testing, DevOps, concurrency, error handling, documentation)
- [x] All 13 expert docs ingested into agent_memory (353 total chunks)
- [x] Toaster sprint plan written (Sprints 8-14, zero BEAST dependency)
- [x] 10 new dataset candidates identified via web search
- [x] Self-learning research survey completed (9 CPU-friendly techniques)
- [x] Module mapping: 8 dormant self-learning modules identified for activation

### What's Still Running
- (nothing)

### What's Ahead
- [ ] OpenCodeInstruct full index (deferred to BEAST -- disk I/O error on toaster at 600K/5M entries)
- [ ] Sprint 8: Wire SelfLearningPipeline, close feedback loop
- [ ] Sprint 9: Wire STaR, Best-of-N, Reflection into agent
- [ ] Sprint 10: Corrective retrieval + Smart Orchestrator
- [ ] Sprint 11: Prompt evolution + adversarial self-play
- [ ] Sprint 12: Cascade router (multi-model routing)
- [ ] Sprint 13: End-to-end self-learning validation
- [ ] Sprint 14: Weekly knowledge scraper
- [ ] Sprint 15: Build multi-agent subagent spawning system
