# JCoder Toaster-Viable Sprint Plan (2026-03-10)

Last updated: 2026-03-10 23:45 America/Denver

## Context

Sprints 1-7 COMPLETE. This plan covers Sprints 8-14 -- all runnable on the
Toaster (16 GB RAM, Intel Iris Xe, CPU-only Ollama, no discrete GPU).

BEAST hardware (128 GB RAM, 48 GB VRAM) is NOT required for any sprint below.
Hardware migration is Sprint 15+ (separate plan).

## Module Status Summary (as of Sprint 7 completion)

### WIRED & ACTIVE (via agent/bridge.py)
- Experience Replay (2 entries)
- Meta-Cognitive Controller (2 strategy arms, 0 decisions)
- Active Learner (5 candidates)
- Procedural Memory (5 experiences)
- Quality-Diversity Archive (0/0 -- EMPTY)
- Telemetry Store (17 events)
- Agent Memory (219 curriculum chunks + FTS5)

### STANDALONE -- NOT INSTANTIATED (code exists, never called)
- STaR Reasoner (core/star_reasoner.py)
- Prompt Evolver (core/prompt_evolver.py)
- Adversarial Self-Play (core/adversarial_self_play.py)
- Best-of-N Generator (core/best_of_n.py)
- Cascade Router (core/cascade.py)
- Reflection Engine (core/reflection.py)
- Corrective Retrieval (core/corrective_retrieval.py)
- Smart Orchestrator (core/smart_orchestrator.py)
- Continual Learner (core/continual_learner.py)
- SelfLearningPipeline (core/self_learning_pipeline.py) -- NEVER instantiated

### UNTESTED (no dedicated test files)
- Knowledge Graph (core/knowledge_graph.py)
- Rapid Digest (core/rapid_digest.py)
- Stigmergy Booster (core/stigmergy.py)

---

## Sprint 8: Self-Learning Pipeline Activation (Toaster)

**Goal**: Wire the SelfLearningPipeline orchestrator and close the
telemetry-to-curriculum feedback loop.

**Rationale**: The biggest gap -- 8 standalone modules exist but the pipeline
class that orchestrates them is never instantiated. This sprint connects them.

### Tasks

- [ ] 8.1: Write unit tests for SelfLearningPipeline class
  - Test pipeline construction with mocked sub-modules
  - Test pipeline.run() orchestration flow
  - Test graceful degradation when sub-modules fail
  - Test config-driven module enable/disable
  - Target: 15+ tests in tests/test_self_learning_pipeline.py

- [ ] 8.2: Instantiate SelfLearningPipeline in bridge.py
  - Add to create_wired_agent() alongside existing 5 modules
  - Config gate: config/agent.yaml `self_learning.pipeline_enabled: true`
  - Wire telemetry store output -> pipeline input
  - Wire pipeline output -> agent_memory ingest

- [ ] 8.3: Close telemetry feedback loop
  - Telemetry low_confidence_queries() -> active_learner candidates
  - Active learner selected candidates -> study engine topics
  - Study engine results -> agent_memory chunks
  - Build: scripts/close_feedback_loop.py (one-shot or cron-able)

- [ ] 8.4: Write unit tests for Continual Learner
  - Test baseline registration
  - Test forgetting detection
  - Test knowledge consolidation
  - Target: 10+ tests in tests/test_continual_learner.py

- [ ] 8.5: Wire Continual Learner into bridge
  - Register eval baselines from evaluation/results_local/
  - Periodic check: has any category score dropped?
  - If regression detected, flag for re-distillation

- [ ] 8.6: Populate QD Archive with initial solutions
  - Feed existing eval results as solution candidates
  - Build diversity metrics based on category + approach
  - Target: 50+ entries from existing eval data

**Exit criteria**: SelfLearningPipeline instantiated, telemetry -> learning
loop closed, continual learner tracking baselines, QD archive populated.

**Estimated effort**: 2-3 sessions

---

## Sprint 9: Reasoning & Reflection Integration (Toaster)

**Goal**: Wire STaR Reasoner, Best-of-N Generator, and Reflection Engine
into the agent runtime so it can self-improve at inference time.

**Rationale**: These modules provide inference-time reasoning upgrades that
work on CPU-only hardware. No fine-tuning needed -- they use prompting and
scoring to select better outputs.

### Tasks

- [ ] 9.1: Write unit tests for STaR Reasoner
  - Test rationalization generation (explain correct answer)
  - Test chain-of-thought extraction
  - Test integration with eval scorer
  - Target: 12+ tests in tests/test_star_reasoner.py

- [ ] 9.2: Write unit tests for Best-of-N Generator
  - Test N-sample generation with temperature sweep
  - Test scoring/selection of best candidate
  - Test budget guard (max N, max tokens)
  - Target: 10+ tests in tests/test_best_of_n.py

- [ ] 9.3: Write unit tests for Reflection Engine
  - Test self-critique generation
  - Test revision based on critique
  - Test max-iterations guard
  - Target: 10+ tests in tests/test_reflection.py

- [ ] 9.4: Wire STaR into agent query path
  - When confidence < threshold, generate rationalization
  - Store rationalization in experience replay
  - Config gate: agent.yaml `reasoning.star_enabled: true`

- [ ] 9.5: Wire Best-of-N as query strategy option
  - New query profile: "thorough" (generates N=3-5 candidates)
  - Score candidates against retrieval context
  - Return highest-scoring candidate
  - Budget guard: max 5 candidates, abort if first scores > 0.9

- [ ] 9.6: Wire Reflection into agent post-processing
  - After initial answer, generate self-critique
  - If critique identifies issues, revise once
  - Store critique + revision in experience replay
  - Config gate: agent.yaml `reasoning.reflection_enabled: true`

**Exit criteria**: All 3 modules tested, wired into agent with config gates,
experience replay captures reasoning traces.

**Estimated effort**: 2-3 sessions

---

## Sprint 10: Corrective Retrieval & Smart Orchestrator (Toaster)

**Goal**: Activate Corrective Retrieval (CRAG pattern) and Smart Orchestrator
to make retrieval self-correcting and strategy-adaptive.

**Rationale**: Current retrieval is fire-and-forget. CRAG pattern detects
when retrieved context is irrelevant and triggers corrective action. Smart
Orchestrator picks the best retrieval strategy per query type.

### Tasks

- [ ] 10.1: Write unit tests for Corrective Retrieval
  - Test relevance scoring of retrieved chunks
  - Test corrective action (re-query with refined terms)
  - Test fallback to broader search on low relevance
  - Target: 12+ tests in tests/test_corrective_retrieval.py

- [ ] 10.2: Write unit tests for Smart Orchestrator
  - Test strategy selection based on query features
  - Test strategy ranking updates from outcomes
  - Test fallback chain (primary -> secondary -> baseline)
  - Target: 10+ tests in tests/test_smart_orchestrator.py

- [ ] 10.3: Wire Corrective Retrieval into retrieval path
  - After initial FTS5 retrieval, score chunk relevance
  - If avg relevance < threshold, re-query with extracted keywords
  - Track corrective actions in telemetry
  - Config gate: agent.yaml `retrieval.corrective_enabled: true`

- [ ] 10.4: Wire Smart Orchestrator as retrieval front-end
  - Query classification: code/debug/explain/review/algorithm
  - Per-class strategy preferences (learned from telemetry)
  - Thompson sampling for strategy exploration/exploitation
  - Connects to meta-cognitive controller for arm selection

- [ ] 10.5: Write unit tests for Knowledge Graph
  - Test node/edge creation
  - Test traversal and query
  - Test persistence
  - Target: 10+ tests in tests/test_knowledge_graph.py

- [ ] 10.6: Build knowledge graph from eval data
  - Nodes: concepts, categories, question patterns
  - Edges: concept co-occurrence, prerequisite relationships
  - Use to improve retrieval: expand queries with related concepts

**Exit criteria**: CRAG pattern active, orchestrator selecting strategies,
knowledge graph populated and queryable.

**Estimated effort**: 2-3 sessions

---

## Sprint 11: Prompt Evolution & Adversarial Hardening (Toaster)

**Goal**: Activate Prompt Evolver and Adversarial Self-Play to continuously
improve prompt templates and harden against failure modes.

**Rationale**: Static prompts leave performance on the table. Prompt evolution
tests variations and keeps winners. Adversarial self-play generates hard
questions that expose weaknesses.

### Tasks

- [ ] 11.1: Write unit tests for Prompt Evolver
  - Test prompt variation generation (mutation, crossover)
  - Test fitness evaluation against eval set
  - Test population management (elite retention, diversity)
  - Target: 12+ tests in tests/test_prompt_evolver.py

- [ ] 11.2: Write unit tests for Adversarial Self-Play
  - Test adversarial question generation
  - Test difficulty calibration
  - Test weakness detection from failure patterns
  - Target: 10+ tests in tests/test_adversarial_self_play.py

- [ ] 11.3: Wire Prompt Evolver with eval runner
  - Seed population: current 6 prompt modes from prompts.py
  - Fitness function: eval score on 50-question subset
  - Evolution: mutate top performers, cross best features
  - Store winning prompts in agent_memory with [EVOLVED] tag
  - Run offline: scripts/evolve_prompts.py

- [ ] 11.4: Wire Adversarial Self-Play with telemetry
  - Generate adversarial questions targeting weakest categories
  - Run against agent, score results
  - Feed failures into telemetry -> distillation pipeline
  - scripts/adversarial_training.py

- [ ] 11.5: Test Rapid Digest module
  - Write 8+ tests in tests/test_rapid_digest.py
  - Verify summarization and key-point extraction
  - Wire into study engine for faster knowledge absorption

- [ ] 11.6: Test Stigmergy Booster module
  - Write 8+ tests in tests/test_stigmergy.py
  - Verify pheromone trail logic for retrieval boosting
  - Wire into federated search as scoring signal

**Exit criteria**: Prompt evolution running offline, adversarial questions
generated, Rapid Digest and Stigmergy tested and wired.

**Estimated effort**: 2-3 sessions

---

## Sprint 12: Cascade Router & Model Orchestration (Toaster)

**Goal**: Activate the Cascade Router for multi-model query routing and
build model-switching logic that works on CPU-only hardware.

**Rationale**: Different query types benefit from different models. The
cascade router sends easy queries to smaller/faster models and escalates
complex ones. On toaster, this means routing between phi4-mini (fast) and
phi4:14b (thorough) based on query complexity.

### Tasks

- [ ] 12.1: Write unit tests for Cascade Router
  - Test complexity estimation
  - Test model selection based on complexity
  - Test escalation when confidence is low
  - Test budget tracking (tokens/latency per tier)
  - Target: 12+ tests in tests/test_cascade.py

- [ ] 12.2: Wire Cascade into agent core
  - Tier 1 (fast): phi4-mini for simple lookups
  - Tier 2 (balanced): phi4:14b for coding tasks
  - Tier 3 (API): gpt-4.1-mini for hard questions (online mode)
  - Config: agent.yaml `cascade.tiers` with model + threshold

- [ ] 12.3: Build complexity estimator
  - Features: query length, keyword density, code presence
  - Categories: simple (lookup), moderate (coding), complex (architecture)
  - Train from existing eval data (200 scored questions)

- [ ] 12.4: Build latency-aware routing
  - Track per-model latency in telemetry
  - Auto-adjust tier thresholds based on observed latency
  - Degrade gracefully: if tier-2 model not loaded, skip to tier-3

- [ ] 12.5: Integration test: full cascade end-to-end
  - Run 50-question eval through cascade router
  - Compare: cascade vs single-model (quality and latency)
  - Document results in evaluation/cascade_comparison.md

**Exit criteria**: Cascade router tested, wired, routing queries to
appropriate model tier, latency tracking active.

**Estimated effort**: 1-2 sessions

---

## Sprint 13: End-to-End Self-Learning Validation (Toaster)

**Goal**: Validate the complete self-learning pipeline end-to-end with
real queries, real telemetry, and measurable improvement.

**Rationale**: Individual modules are tested but the full loop has never
run end-to-end. This sprint proves the system learns.

### Tasks

- [ ] 13.1: Design learning validation protocol
  - Baseline: run 200q eval, record scores per category
  - Learning phase: run 100 diverse queries through agent
  - Each query generates telemetry, experience, procedural memory
  - Close feedback loop: telemetry -> distillation -> re-index
  - Re-eval: run same 200q eval, compare scores

- [ ] 13.2: Build automated learning cycle script
  - scripts/learning_cycle.py
  - Phase 1: baseline eval (record scores)
  - Phase 2: generate 50 study queries from weak categories
  - Phase 3: run study engine on each query
  - Phase 4: close feedback loop (distill weak topics)
  - Phase 5: re-eval (record new scores)
  - Phase 6: compare and report delta

- [ ] 13.3: Run 3 learning cycles
  - Cycle 1: algorithms + debugging (weakest in Sprint 7 eval)
  - Cycle 2: shell + systems (second weakest)
  - Cycle 3: general Python + JS (validate no regression)
  - Document improvement per cycle

- [ ] 13.4: Validate experience replay contribution
  - Run eval WITH experience replay context vs WITHOUT
  - Measure: does accumulated experience help or hurt?
  - If context flooding detected, tune max_context_chars

- [ ] 13.5: Validate meta-cognitive strategy selection
  - After 3 cycles, check: does Thompson sampling prefer winning strategies?
  - Review strategy arm pulls and rewards
  - Document in evaluation/meta_cognitive_analysis.md

- [ ] 13.6: Write Sprint 8-13 summary report
  - Before/after scores for each category
  - Which modules contributed most to improvement
  - Remaining gaps requiring BEAST hardware
  - docs/SELF_LEARNING_VALIDATION_REPORT.md

**Exit criteria**: Measurable score improvement across at least 2 categories,
documented end-to-end pipeline execution, regression analysis complete.

**Estimated effort**: 2-3 sessions

---

## Sprint 14: Weekly Knowledge Scraper (Toaster)

**Goal**: Build automated pipeline to pull fresh coding knowledge from
public sources on a weekly schedule.

**Rationale**: Static knowledge gets stale. Python releases, new libraries,
security advisories, and best practices change. JCoder needs a way to
stay current without manual intervention.

### Tasks

- [ ] 14.1: Build RSS/Atom feed scraper
  - Sources: Python blog, Real Python, PEP index, HN top stories
  - Filter: coding-relevant content only
  - Output: markdown chunks for FTS5 indexing
  - scripts/weekly_scraper.py

- [ ] 14.2: Build changelog monitor
  - Track: Python releases, pip package updates (top 50 packages)
  - Detect: new features, breaking changes, security fixes
  - Generate: update summaries for agent_memory

- [ ] 14.3: Build SE recent answers scraper
  - Pull highest-voted recent answers from tracked SE sites
  - Filter: score >= 10, within last 30 days
  - Ingest into rotating fresh_knowledge.fts5.db (max 10K chunks)

- [ ] 14.4: Build knowledge freshness tracker
  - Tag each chunk with ingest timestamp
  - Decay old chunks in retrieval scoring (prefer recent)
  - Configurable: freshness_weight in memory.yaml

- [ ] 14.5: Build weekly automation script
  - scripts/weekly_knowledge_update.py
  - Runs all scrapers, ingests results, logs stats
  - Can be cron'd or run manually
  - Budget guard: max 50 MB new data per week

**Exit criteria**: Scraper pipeline functional, tested, and documented.
Can be scheduled as weekly task.

**Estimated effort**: 1-2 sessions

---

## Dependency Graph

```
Sprint 7 (DONE) --> Sprint 8 (pipeline activation)
                      |
                      v
                    Sprint 9 (reasoning & reflection)
                      |
                      v
                    Sprint 10 (corrective retrieval & orchestrator)
                      |
                      v
                    Sprint 11 (prompt evolution & adversarial)
                      |
                      v
                    Sprint 12 (cascade router)
                      |
                      v
                    Sprint 13 (end-to-end validation)
                      |
                      v
                    Sprint 14 (weekly scraper -- can start anytime after 8)
```

Sprints 8-12 are sequential (each builds on prior wiring).
Sprint 13 requires all prior sprints.
Sprint 14 is independent -- can run in parallel with 9-12.

## Total Estimated Effort

| Sprint | Sessions | Focus |
|--------|----------|-------|
| 8 | 2-3 | Pipeline activation + feedback loop |
| 9 | 2-3 | Reasoning modules |
| 10 | 2-3 | Retrieval intelligence |
| 11 | 2-3 | Prompt evolution + adversarial |
| 12 | 1-2 | Model cascading |
| 13 | 2-3 | End-to-end validation |
| 14 | 1-2 | Knowledge freshness |
| **Total** | **12-19** | -- |

All sprints run on Toaster. No BEAST hardware required.

---

## Appendix A: New Dataset Acquisition Queue (2026-03-10)

Datasets verified via web search. All freely downloadable, no gating.

| Priority | Dataset | HuggingFace ID | Size | License |
|----------|---------|---------------|------|---------|
| 1 | OpenCodeReasoning-2 | nvidia/OpenCodeReasoning-2 | 2.5M | CC-BY-4.0 |
| 2 | Codeforces + Editorials | open-r1/codeforces | 10K problems | CC-BY-4.0 |
| 3 | DebugBench | Rtian/DebugBench | 4,253 | Apache 2.0 |
| 4 | SWE-smith | SWE-bench/SWE-smith | 59K | MIT |
| 5 | OpenCodeInstruct | nvidia/OpenCodeInstruct | 5M | CC-BY-4.0 |
| 6 | Software-Architecture | ajibawa-2023/Software-Architecture | 450K | Apache 2.0 |
| 7 | XLCoST | codeparrot/xlcost-text-to-code | 460K | CC-BY-SA 4.0 |
| 8 | BigCodeBench | bigcode/bigcodebench | 1,140 | Apache 2.0 |
| 9 | LeetCodeDataset | newfacade/LeetCodeDataset | 2,870 | Apache 2.0 |
| 10 | CRUXEval | cruxeval-org/cruxeval | 800 | MIT |

## Appendix B: Self-Learning Research References (2026-03-10)

Techniques validated as CPU-only, no fine-tuning required:

| Technique | Paper | Key Insight |
|-----------|-------|-------------|
| CER | ACL 2025 (arxiv:2506.06698) | Store experiences as text, retrieve as few-shot context |
| RISE | NeurIPS 2024 (arxiv:2407.18219) | Multi-turn self-critique via prompting |
| V-STaR | 2024 | Best-of-N + verifier scoring at inference |
| Sol-Ver | Feb 2025 (arxiv:2502.14948) | Generate code + generate tests + verify loop |
| ITSI Survey | Dec 2024 (arxiv:2412.14352) | Taxonomy of inference-time improvement methods |
| DRAG | ACL 2025 (arxiv:2506.01954) | RAG distillation for small models |
| Contextual MAB | Oct 2025 (arxiv:2510.00841) | Thompson sampling conditioned on query features |
| QDAIF | ICLR 2024 | MAP-Elites + LLM judge for diverse archives |
| START | Mar 2025 (arxiv:2503.04625) | STaR + tool use for self-verification |
