# JCoder Sprint Roadmap -- 2026-03-11
# From Coding Assistant to Discovery Engine

## PHASE A: Foundation Hardening (Weeks 1-2)

### Sprint R1-R7: DPI Repair Sprints (IN PROGRESS -- Claude DPI)
Deep Packet Inspector found 3 CRITICAL, 9 HIGH, 26 MEDIUM across 188 files.
Being executed by separate Claude instance in parallel.
- R1: Crash and security (runtime guard, exec sandbox, SQLite leaks)
- R2: Race conditions (ThreadPool, atomic counters, Agent state)
- R3: Silent failures (bare except:pass in 8 files)
- R4: Config portability (hardcoded paths, YAML guard, dead code)
- R5: SQLite connections (threading.local() cache, 14 files)
- R6: Pipeline hardening (HF API, FTS5 query cap, NetworkGate)
- R7: Test infrastructure (conftest.py, timeouts, resource leaks)

### Sprint 8: Wire Self-Learning Pipeline
- Connect all 13 self-learning modules into live query flow
- Meta-cognitive controller selects strategy per query
- Stigmergy deposits pheromones on success/failure
- Experience replay injects past successes as few-shot
- Reflection scores every response (ISREL/ISSUP/ISUSE)
- Telemetry logs everything
- Gate: Run HumanEval+ and MBPP+ baseline before/after

### Sprint 9: Reasoning and Reflection Activation
- Wire STaR reasoner for chain-of-thought on hard queries
- Wire Best-of-N for code generation queries
- Wire corrective retrieval for low-confidence retrievals
- Gate: 5% improvement on hardest 25% of benchmark queries


## PHASE B: Intelligence Amplification (Weeks 3-5)

### Sprint 10: Federated Search Across All Indexes
- Wire federated_search.py to search ALL 68 FTS5 indexes
- RRF fusion of results from agent_memory + code corpora + trajectories + security
- Priority weighting: agent_memory > recent datasets > older corpora
- Recency boost: indexes from 2025-2026 weighted higher than 2023 data
- Gate: Retrieval relevance (ISREL) improves on 50+ test queries

### Sprint 11: Active Learning + Adversarial Self-Play Activation
- Wire active learner to identify highest-value queries
- Wire adversarial self-play (3 games: hardness escalation, trick questions, ambiguity)
- Use active learning scores to guide which queries to evolve prompts on
- Gate: Active learner identifies top 50 learning opportunities, self-play generates 100 challenges

### Sprint 12: Prompt Evolution Live Run
- Run prompt_evolver.py on live system with real eval data
- 5 generations, population of 8, tournament selection
- Use adversarial self-play failures as evolution pressure
- Gate: Evolved prompt outperforms hand-written prompt on golden set

### Sprint 13: Frontier Model Distillation Pipeline
- Build EnrichIndex: Claude/GPT generate summaries for FTS5 chunks
- Build query2doc: expand sparse queries with LLM-generated context
- Build distillation scheduler: prioritize high-value chunks for enrichment
- Gate: Enriched chunks produce measurably better retrieval than raw chunks


## PHASE C: Self-Evolution (Weeks 6-9)

### Sprint 14: Weekly Research Scraper
- Automated scrapers for: arXiv cs.AI/CL/SE, Papers With Code, HF trending, GitHub trending
- Grok integration for real-time X/Twitter AI trend detection
- Deduplication against existing knowledge
- Weekly digest generation and auto-ingestion into agent_memory
- Recency-tagged storage (freshness scoring in retrieval)
- Gate: First automated weekly digest produced and ingested

### Sprint 15: Multi-Agent Mesh Coordination
- Subagent spawning from agent/core.py
- Shared knowledge bus (FTS5 agent_memory as coordination layer)
- Task dispatch with HybridRAG3's dispatch board pattern
- Agent-to-agent artifact handoff
- Gate: 3 subagents complete a coordinated task (research + implement + verify)

### Sprint 16: LimitlessApp V2 Memory Integration
- Wire LimitlessApp as permanent cross-session memory layer
- Every JCoder interaction persisted and searchable
- Model-agnostic storage (survives model swaps)
- Cross-session pattern detection
- Gate: JCoder recalls context from 5 sessions ago accurately

### Sprint 21: Weekly Software Evolution Engine
- evolution_runner.py: Archive baseline, propose mutations, validate, deploy
- Evolution ledger (SQLite): track every attempt, accepted or rejected
- Safety invariants: immutable eval harness, immutable evolution system, immutable rollback
- Continual learner regression gate
- Git worktree isolation for safe testing
- Gate: First successful evolution cycle (mutation proposed, tested, accepted or rejected with full audit trail)

### Sprint 22: VM Tournament Mode (10x Parallel Evolution)
- Docker/worktree isolation for 10 parallel clones
- Identical eval suite across all clones
- Tournament selection: champion vs baseline
- Archive every baseline forever
- Gate: 10 clones run in parallel, champion selected, decision documented

### Sprint 23: Concentration Rotation and Meta-QA
- Attention decay modeling: predict when agent output quality degrades
- Role rotation at predicted degradation points
- Meta-QA layer: QA the QA process, catch rubber-stamping
- Random deep inspection sweep scheduling
- Gate: Demonstrate that rotated agents catch bugs that non-rotated agents miss


## PHASE D: Discovery Mode (Weeks 10+)

### Sprint 24: Recursive Meta-Learning
- Evolution targets the evolution system itself (Level 2 meta-learning)
- Research pipeline self-optimization (Level 3 meta-research)
- Track which sources/techniques led to accepted evolutions
- Auto-deprioritize low-value sources, amplify high-value ones
- Gate: Demonstrated improvement in evolution acceptance rate over 4 weeks

### Sprint 25: Supercomputer Burst Pipeline
- Scripts for spinning up Lambda Labs / Vast.ai instances
- Automated deployment of JCoder + all indexes to cloud
- 100x tournament mode for weekend burst sprints
- Full OpenCodeInstruct (5M entries) indexing
- Massive distillation runs across all 11.6M chunks
- Gate: Weekend burst produces more evolution progress than 4 normal weeks

### Sprint 26: Autonomous Research Lab
- JCoder identifies research gaps autonomously from eval failures
- Proposes hypotheses from cross-domain synthesis
- Implements prototypes during evolution cycles
- Human role: set direction, review weekly evolution proposals via Telegram
- Gate: JCoder independently discovers and implements a novel technique that improves benchmark scores


## STANDING DATA SPRINTS (parallel, ongoing)

### Data Sprint D1: Remaining Downloads
- github_codereview (2.58 GB)
- commit_chronicle (10.7M commits)
- opencodeinstruct (5M -- BEAST or cloud burst)
- github_code_2025 (1.5M repos)

### Data Sprint D2: Weekly Freshness Ingestion
- arXiv weekly digest
- GitHub trending weekly
- HuggingFace new datasets weekly
- Grok X/Twitter trend reports weekly

### Data Sprint D3: Documentation Scraping
- Python stdlib docs (docs.python.org)
- FastAPI, Pydantic, pytest, httpx docs
- Framework docs (LangChain, CrewAI, AutoGen, OpenHands)


## METRICS AND MILESTONES

| Milestone | Target | Metric |
|-----------|--------|--------|
| Baseline established | Sprint 8 | HumanEval+, MBPP+, Golden Set scores |
| Self-learning live | Sprint 9 | 5% improvement on hard queries |
| Retrieval unified | Sprint 10 | ISREL improvement across 68 indexes |
| Prompt evolved | Sprint 12 | Evolved > hand-written on golden set |
| First evolution cycle | Sprint 21 | Full audit trail, accepted or rejected |
| 10x tournament | Sprint 22 | 10 parallel clones, champion selected |
| Novel discovery | Sprint 26 | JCoder-originated technique improves scores |


## HARDWARE REQUIREMENTS BY PHASE

| Phase | Hardware | Why |
|-------|----------|-----|
| A (Hardening) | Toaster (16 GB) | Code fixes, no heavy compute |
| B (Intelligence) | Toaster + Ollama | Self-learning needs local LLM |
| C (Evolution) | BEAST (128 GB / 48 GB VRAM) | 10x parallel clones, distillation |
| D (Discovery) | BEAST + weekend cloud bursts | 100x tournaments, massive indexing |
