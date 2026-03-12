# JCoder Learning Curricula -- 2026-03-11
# Intelligent Lesson Plans by Category (Basics to Advanced)

## Priority Order
1. Agent Reasoning & Trajectory Learning (HIGHEST -- core capability)
2. Self-Improvement & Evolution (HIGHEST -- enables recursive growth)
3. RAG Systems Mastery (HIGHEST -- JCoder's retrieval backbone)
4. Tool Use & Function Calling (HIGH -- practical agent capability)
5. Code Generation & Reasoning (HIGH -- benchmark performance)
6. Code Review & Quality (MEDIUM -- engineering maturity)
7. Security & Vulnerability (MEDIUM -- reliability layer)
8. Research Methodology (ONGOING -- discovery engine)


---

## Category 1: Agent Reasoning & Trajectory Learning

### Why First
Agent reasoning is JCoder's PRIMARY differentiator. Without understanding how
agents solve problems step-by-step, JCoder cannot self-improve effectively.
Trajectory data is the richest learning signal available.

### Sprint C1-1: Foundations of Agent Problem Solving
**Goal**: Understand basic agent-environment interaction patterns
**Data Sources** (existing indexes):
- code_act.fts5.db (7,140 code agent + 71K general conversations)
- menvdata_traj.fts5.db (3,800 multi-language agent traces)
**Key Concepts**:
- Agent-environment loop: observe -> think -> act -> observe
- Tool calling as action (shell commands, file ops, API calls)
- Reward signals: test pass/fail, syntax validity, runtime success
- State tracking: what the agent knows vs what it needs to find
**Assessment**: Can JCoder identify the 3 phases of an agent trajectory
  (setup, exploration, solution) from raw trace data?

### Sprint C1-2: Multi-Step Reasoning Traces
**Goal**: Learn from successful multi-step problem solving
**Data Sources** (existing indexes):
- openhands_traj.fts5.db (192K chunks, 67K multi-turn solving traces)
- sweagent_traj.fts5.db (163K chunks, 80K trajectories)
- swesmith_traj.fts5.db (59K chunks, 5K curated traces)
**Key Concepts**:
- Backtracking: recognizing dead ends and recovering
- Tool composition: chaining grep -> read -> edit -> test
- Error-driven refinement: using failures to narrow search space
- Trajectory length vs success correlation
**Assessment**: Given a failing trajectory, can JCoder identify the
  decision point where the agent went wrong?

### Sprint C1-3: Advanced Agent Strategies
**Goal**: Learn sophisticated agent patterns from curated data
**Data Sources** (existing + new):
- swe_gym_traj.fts5.db (6K curated solving traces, existing)
- coderforge.fts5.db (589K chunks, 51K agentic trajectories, existing)
- swe_rebench_v2.fts5.db (32K multilingual, NEW download)
- multi_swe_bench.fts5.db (multilingual SWE, NEW download)
- trail_agent.fts5.db (TRAIL agentic eval, NEW download)
**Key Concepts**:
- Repository-level understanding (not just file-level)
- Cross-file dependency tracing
- Hypothesis-driven debugging (form theory, test it, refine)
- When to explore broadly vs exploit known patterns
- Multilingual agent strategies (Python vs JS vs Java patterns)
**Assessment**: Can JCoder propose a 5-step plan for resolving a
  GitHub issue it hasn't seen before?

### Sprint C1-4: Meta-Agent Reasoning (ADVANCED)
**Goal**: Learn to reason ABOUT reasoning
**Data Sources** (new downloads + expert knowledge):
- openthoughts_agent.fts5.db (OpenThoughts-Agent-v1, NEW)
- nemotron_agentic.fts5.db (Nemotron-Agentic-v1, NEW)
- mixture_of_thoughts.fts5.db (350K reasoning traces, NEW)
- Expert doc: recursive_meta_learning_architecture.md (ingested)
**Key Concepts**:
- Reasoning trace analysis: what makes a good chain of thought?
- Self-monitoring: detecting when reasoning is going off-track
- Strategy selection: when to use step-by-step vs holistic reasoning
- Meta-cognitive control: Thompson sampling over strategies
**Assessment**: Given 3 reasoning traces for the same problem, can JCoder
  identify which is best and explain why?


---

## Category 2: Self-Improvement & Evolution

### Why Second
Self-improvement is the MULTIPLIER. Every other capability grows faster when
JCoder can optimize its own algorithms and prompts automatically.

### Sprint C2-1: Foundations of Self-Learning
**Goal**: Understand basic self-improvement mechanisms
**Data Sources** (existing modules):
- self_learning.fts5.db (609 chunks, self-learning papers)
- research_papers.fts5.db (635 chunks, ML research)
- Expert doc: weekly_software_evolution_protocol.md (ingested)
**Key Concepts**:
- Reward modeling: defining what "better" means (eval scores, latency, safety)
- Thompson sampling: explore vs exploit across strategies
- Experience replay: learning from past successes and failures
- Stigmergy: leaving pheromone trails for future decisions
**Assessment**: Can JCoder explain when Thompson sampling beats greedy selection?

### Sprint C2-2: Reinforcement Learning for Code (GRPO/RLVR)
**Goal**: Understand the RL paradigm that trains coding models
**Data Sources** (new expert knowledge doc + papers):
- Expert doc: rl_for_code_grpo_rlvr.md (TO CREATE AND INGEST)
- verifiable_coding.fts5.db (verifiable problems with test suites, NEW)
**Key Concepts**:
- GRPO (Group Relative Policy Optimization): no critic needed, group-based baselines
- RLVR (RL with Verifiable Rewards): unit tests as automatic reward signals
- DeepCoder-14B: 14B model rivaling o3-mini via pure RL ($26K training)
- SWE-RL: RL from real-world code execution feedback
- DeepSWE: 59% SWE-bench via pure RL (no supervised fine-tuning)
**Assessment**: Can JCoder design a RLVR training loop for a specific task?

### Sprint C2-3: Evolutionary Self-Improvement
**Goal**: Learn evolutionary approaches to code and prompt optimization
**Data Sources** (expert knowledge):
- Expert doc: evolutionary_code_optimization.md (TO CREATE AND INGEST)
- Expert doc: weekly_software_evolution_protocol.md (already ingested)
**Key Concepts**:
- Darwin Godel Machine: self-improving coding agents via evolutionary search
- Huxley-Godel Machine (ICLR 2026 oral): improved variant with habitat selection
- AlphaEvolve/OpenEvolve: Google's evolutionary code optimization
- Prompt evolution: tournament selection, crossover, mutation of system prompts
- Safety invariants: immutable eval, immutable evolution system, immutable rollback
**Assessment**: Can JCoder design a safe 3-generation evolution experiment?

### Sprint C2-4: Recursive Meta-Learning (ADVANCED)
**Goal**: Learn to learn to learn -- multiple levels of self-improvement
**Data Sources** (expert knowledge):
- Expert doc: recursive_meta_learning_architecture.md (already ingested)
- Expert doc: frontier_distillation_strategy.md (already ingested)
**Key Concepts**:
- Level 0: Direct learning (query -> retrieve -> generate)
- Level 1: Learning to learn (optimize retrieval and generation strategies)
- Level 2: Learning to learn to learn (evolve the optimization itself)
- Level 3: Learning to research (optimize what to learn next)
- ALMA: meta-learning for LLM agent training (adjusts learning rates per task)
- CoCoS: self-correction via confidence estimation
**Assessment**: Can JCoder propose a concrete Level 2 experiment?


---

## Category 3: Tool Use & Function Calling

### Why Third
Practical tool use is what separates a chatbot from an agent. JCoder must
master function calling to interact with real-world APIs and tools.

### Sprint C3-1: Function Calling Basics
**Goal**: Understand structured function call generation
**Data Sources** (existing + new):
- glaive_code_assistant.fts5.db (existing, general code assistant)
- hermes_function_calling.fts5.db (NousResearch/hermes-function-calling-v1, NEW)
- glaive_function_calling_v2.fts5.db (glaiveai/glaive-function-calling-v2, NEW)
**Key Concepts**:
- JSON function schema definition
- Parameter extraction from natural language
- Single vs parallel function calls
- Error handling and retry patterns
**Assessment**: Given a tool schema, can JCoder generate correct function calls?

### Sprint C3-2: Multi-Tool Orchestration
**Goal**: Chain multiple tools to solve complex tasks
**Data Sources** (existing + new):
- function_calling (existing, 60K specs -- gated, needs auth)
- jupyter_agent.fts5.db (51K notebook agent interactions, NEW)
**Key Concepts**:
- Tool dependency graphs (output of A feeds input of B)
- Parallel tool execution (independent calls run simultaneously)
- Fallback chains (if tool A fails, try tool B)
- Context accumulation across tool calls
**Assessment**: Can JCoder plan a 4-tool chain to answer a complex question?

### Sprint C3-3: Advanced Tool Patterns (ADVANCED)
**Goal**: Master sophisticated tool interaction patterns
**Data Sources** (agent trajectory indexes + expert knowledge):
- All trajectory indexes (code_act, openhands_traj, sweagent_traj, etc.)
- Expert doc: autonomous_agent_blueprint.md (already ingested)
**Key Concepts**:
- Dynamic tool discovery (finding the right tool at runtime)
- Tool creation (writing new tools to solve novel problems)
- Multi-agent tool sharing (coordination layer)
- Sandboxed execution (safe tool running in isolated environments)
**Assessment**: Can JCoder dynamically write and execute a new tool?


---

## Category 3: RAG Systems Mastery

### Why Third (Moved to HIGHEST Priority)
RAG is JCoder's BACKBONE. Every query JCoder answers flows through retrieval.
Understanding how to design, optimize, inspect, and improve RAG systems is
essential for JCoder to improve itself AND help users build better RAG systems.
HybridRAG3 is the reference implementation -- JCoder should know it cold.

### Sprint CR-1: RAG Foundations & Architecture
**Goal**: Understand RAG pipeline stages and architecture patterns
**Data Sources** (expert knowledge):
- rag_architecture_design_patterns.md (ingested, 4 chunks)
- rag_optimization_retrieval_generation.md (ingested, 6 chunks)
**Key Concepts**:
- Naive RAG vs Advanced RAG vs Modular RAG vs Agentic RAG
- 6-step pipeline: retrieve -> build context -> build prompt -> LLM -> cost -> log
- GraphRAG (Microsoft): Entity knowledge graphs, community detection, 80% vs 50% accuracy
- CRAG: Retrieval evaluator (correct/incorrect/ambiguous) with web fallback
- Self-RAG: Reflection tokens (RETRIEVE, ISREL, ISSUP, ISUSE) for adaptive retrieval
- Speculative RAG (ICLR 2025): Drafter-verifier architecture, beats Self-RAG and CRAG
- CoRAG (NeurIPS 2025): Chain-of-retrieval with iterative query reformulation
- The 80% Rule: 80% of RAG failures trace to chunking, not retrieval or generation
**Assessment**: Can JCoder identify which RAG architecture pattern fits a given use case?

### Sprint CR-2: Retrieval Optimization & Hybrid Search
**Goal**: Master state-of-the-art retrieval techniques
**Data Sources** (expert knowledge + HybridRAG3 code):
- rag_optimization_retrieval_generation.md (ingested, 6 chunks)
- HybridRAG3 src/core/query_engine.py (reference implementation)
**Key Concepts**:
- Hybrid search: BM25/FTS5 + dense vector, fused via Reciprocal Rank Fusion (RRF)
- SPLADE learned sparse retrieval: 30K-dim sparse vectors, outperforms BM25
- ColBERT late interaction: Token-level MaxSim, SIGIR 2025 Best Paper
- HyDE: Generate hypothetical documents, embed them, use for retrieval
- Query2Doc, query decomposition, step-back prompting, RAG-Fusion
- Reranking: Cross-encoder (33-40% accuracy gain, +120ms), but destroys unanswerable detection
- BM25 tuning: k1=1.5, b=0.75 defaults
- FTS5 sweet spot: 10K-100K docs, <3ms, zero infrastructure
**Assessment**: Given a retrieval quality issue, can JCoder diagnose whether it's
  a chunking, embedding, query, ranking, or filtering problem?

### Sprint CR-3: Data Pipeline -- Downloading, Parsing, Indexing
**Goal**: Master end-to-end data ingestion for RAG
**Data Sources** (expert knowledge):
- rag_data_pipeline_parsing_indexing.md (ingested, 6 chunks)
**Key Concepts**:
- Parse-Transform-Index (PTI) pipeline: RAG equivalent of ETL
- Metadata preservation: Every chunk must carry source, title, author, date, version, hash
- Provenance tracking: OpenLineage, graph databases, append-only ledgers
- Incremental indexing: Delta indexing, version tracking, separate new-content index
- Parser comparison: Docling (97.9% table accuracy, air-gap safe), LlamaParse (fast cloud),
  Unstructured.io (automation), Apache Tika (1000+ legacy formats)
- Chunking hierarchy: Fixed (baseline) < Recursive (69%, benchmark leader) < Semantic (variable)
  < Contextual retrieval (Anthropic, 49% fewer failures) < Late chunking (efficient)
- Index types: FTS5 (portable, <3ms), HNSW (production, sub-ms), IVF (cost-sensitive)
- Download best practices: Staging area, SHA256 verify, SQLite ledger, resume support
**Assessment**: Can JCoder design a complete ingestion pipeline for a new document type?

### Sprint CR-4: Bulk Transfer & Corporate Network Techniques
**Goal**: Handle data acquisition in restricted environments
**Data Sources** (expert knowledge):
- rag_data_pipeline_parsing_indexing.md (corporate network section)
**Key Concepts**:
- Proxy-aware downloading: HTTPS_PROXY, NTLM/Kerberos auth, system proxy detection
- SSL/certificate handling: Custom CA bundles, SSL_CERT_FILE, certificate export
- Rate limiting: Token bucket per connection, Retry-After headers, off-hours scheduling
- Air-gapped transfer: Download externally, SHA256 manifest, approved media, verify on target
- HuggingFace corporate: HF_ENDPOINT mirror, hf_transfer (Rust, 3-5x faster), snapshot_download
- Firewall workarounds: Range requests for split downloads, pre-download to approved storage
- Resumable downloads: HTTP Range headers, byte offset tracking in SQLite ledger
- Parallel downloads: Thread pool (4-8 workers), per-file progress, backoff on 429/503
**Assessment**: Can JCoder download and index a dataset through a corporate proxy?

### Sprint CR-5: Online, Offline & Hybrid AI for RAG
**Goal**: Master dual-mode RAG with model routing
**Data Sources** (expert knowledge + HybridRAG3 code):
- rag_online_offline_hybrid_ai.md (ingested, 4 chunks)
- HybridRAG3 src/core/llm_router.py (reference implementation)
**Key Concepts**:
- Online API strategies: Per-model prompt engineering (GPT/Claude/Gemini), cost optimization
- Caching: Prompt caching + semantic caching + RAGCache for KV tensors (60% cost reduction)
- Prompt compression: LLMLingua (20x compression)
- Offline runtimes: Ollama (easy) vs vLLM (throughput) vs llama.cpp (portable)
- VRAM management: Model weights + KV cache + overhead formula, Q4_K_M quantization
- KV cache optimization: OLLAMA_KV_CACHE_TYPE=q8_0 halves cache, no quality loss
- Cascade routing (ICML 2025): Route simple->cheap, complex->expensive (40% fewer API calls)
- HybridRAG3 pattern: OllamaRouter + VLLMRouter + APIRouter, mode-specific config branches
- Fallback chains: Claude -> GPT -> Gemini -> local, with circuit breakers
**Assessment**: Can JCoder design a cost-optimal routing strategy for a given workload?

### Sprint CR-6: RAG for Secure & Auditable Environments
**Goal**: Design RAG for corporate/regulated/air-gapped deployments
**Data Sources** (expert knowledge):
- rag_secure_auditable_environments.md (ingested, 6 chunks)
**Key Concepts**:
- Air-gapped architecture: Embedded stores, pre-pulled containers, physical media transfer
- RBAC: Metadata filtering (primary), pre-filter/post-filter patterns, policy engines (Cerbos/OpenFGA)
- Audit logging: Every query fully logged (user, query, retrieved docs, response, model, cost)
- Zero-trust RAG: Don't trust input, retrieval, generation, network, or cache
- Injection defense: Prompt scanning, source-bounded generation, injection traps
- Data provenance chain: Query -> chunks -> sources -> documents -> ingestion record
- Software approval: Permissive licenses only (Apache 2.0, MIT, BSD), SBOM, model vetting
- Compliance docs: Theory of operation, data flow diagrams, threat model, privacy assessment
- Network isolation: Localhost-only binding, network gate, no outbound in offline mode
**Assessment**: Can JCoder design a RAG deployment that passes a security audit?

### Sprint CR-7: RAG Code Inspection & Improvement (ADVANCED)
**Goal**: Inspect, diagnose, and improve existing RAG systems (including HybridRAG3)
**Data Sources** (expert knowledge + HybridRAG3 code):
- rag_code_inspection_refactoring.md (ingested, 6 chunks)
- HybridRAG3 full codebase (reference implementation)
**Key Concepts**:
- Component-level auditing: Test retrieval and generation independently
- Retrieval metrics: Precision@K, Recall@K, MRR, NDCG@10
- Generation metrics: Faithfulness, hallucination rate, citation coverage
- Failure modes: Missing content, poor chunking, scattered evidence, ranking errors, distractors
- Observability tools: LangSmith, Arize Phoenix, Langfuse, TruLens, RAGAS
- Anti-patterns: Monolithic pipeline, arbitrary chunking, no eval loops, no uncertainty
- Refactoring priority: Chunking > hybrid search > reranking > query optimization > prompt > caching
- HybridRAG3 improvement opportunities: HyDE for low-confidence, CRAG evaluation,
  CoRAG iterative retrieval, adaptive chunking, DSPy auto-tuning, MetaRAG consistency,
  semantic caching
- Automatic tuning: DSPy (prompt optimization), RAGAS (eval), ARES (synthetic eval datasets)
**Assessment**: Given a RAG system with degraded accuracy, can JCoder identify the root cause
  and propose a specific fix using the refactoring priority order?


---

## Category 5: Code Generation & Reasoning

### Sprint C4-1: Basic Code Generation
**Goal**: Generate correct, tested code from specifications
**Data Sources** (existing):
- humanevalplus.fts5.db (164 problems)
- mbppplus.fts5.db (378 problems)
- code_alpaca.fts5.db, code_exercises.fts5.db
- python_instructions.fts5.db, python_code_18k.fts5.db
**Key Concepts**:
- Spec-to-code translation
- Test case generation alongside code
- Edge case identification
- Docstring/comment quality

### Sprint C4-2: Complex Reasoning Problems
**Goal**: Solve problems requiring multi-step reasoning
**Data Sources** (existing + new):
- code_contests.fts5.db (competitive programming)
- ds1000.fts5.db (data science problems)
- math_instruct.fts5.db
- mixture_of_thoughts.fts5.db (350K reasoning traces, NEW)
- verifiable_coding.fts5.db (problems with test suites, NEW)
**Key Concepts**:
- Chain-of-thought for code (STaR reasoning)
- Best-of-N generation with verification
- Decomposition: breaking complex problems into sub-problems
- Reasoning trace quality (verbose vs concise vs structured)

### Sprint C4-3: Competition-Level & Benchmark Mastery (ADVANCED)
**Goal**: Achieve top scores on coding benchmarks
**Data Sources** (existing + new):
- livecodebench.fts5.db (recent competition problems, NEW)
- code_contests.fts5.db (existing, competitive)
- BigCodeBench, CRUXEval, DebugBench, Codeforces (all existing)
**Key Concepts**:
- Time-constrained generation strategies
- Correctness verification before submission
- LiveCodeBench: contamination-free eval with recent problems
- Leaderboard analysis: what top models do differently


---

## Category 6: Code Review & Quality

### Sprint C5-1: Review Fundamentals
**Data Sources** (existing):
- codereview_python.fts5.db (11,890 chunks)
- codereview_general.fts5.db
**Key Concepts**: Bug detection, style enforcement, refactoring suggestions

### Sprint C5-2: Large-Scale Review Patterns
**Data Sources** (existing):
- github_codereview.fts5.db (2.58 GB, real-world reviews)
- code_refine.fts5.db (123K before/after pairs)
**Key Concepts**: Multi-file review, architecture-level feedback, PR workflow

### Sprint C5-3: Security-Aware Review (ADVANCED)
**Data Sources** (existing security indexes):
- vuln_cwe_patch.fts5.db, cve_cwe_all.fts5.db, vuln_security_dpo.fts5.db
**Key Concepts**: OWASP patterns, vulnerability-specific review, supply chain


---

## Category 7: Security & Vulnerability

### Sprint C6-1: Vulnerability Basics
**Data Sources**: securecode_web.fts5.db, cve_cwe_all.fts5.db
**Key Concepts**: CWE taxonomy, common vulnerability patterns, OWASP Top 10

### Sprint C6-2: Patch Analysis
**Data Sources**: vuln_cwe_patch.fts5.db (39K with fixes), cve_training.fts5.db
**Key Concepts**: Root cause analysis, patch correctness, regression testing

### Sprint C6-3: Adversarial Robustness (ADVANCED)
**Data Sources**: vuln_security_dpo.fts5.db, adversarial_self_play module
**Key Concepts**: Prompt injection defense, hallucination resistance, safety alignment


---

## Category 8: Research Methodology (ONGOING)

### Sprint C7-1: Research Foundations
**Data Sources**: Expert doc: research_methodology_expert.md (already ingested)
**Key Concepts**: 4-tier source hierarchy, literature review, novelty detection

### Sprint C7-2: Frontier Intelligence
**Data Sources**: Expert doc: frontier_distillation_strategy.md (already ingested)
**Key Concepts**: Multi-model distillation, information propagation timeline

### Sprint C7-3: Novel Discovery (ADVANCED)
**Data Sources**: Expert doc: knowledge_freshness_and_discovery.md (already ingested)
**Key Concepts**: Cross-domain synthesis, hypothesis generation, recursive meta-research


---

## Download Queue (Priority Order)

### Immediate (Toaster-safe, <500 MB each):
1. hermes_function_calling - NousResearch/hermes-function-calling-v1
2. glaive_function_calling_v2 - glaiveai/glaive-function-calling-v2
3. verifiable_coding - open-r1/verifiable-coding-problems-python
4. livecodebench - livecodebench/code_generation_lite
5. trail_agent - PatronusAI/TRAIL

### Medium (may be large, monitor disk):
6. openthoughts_agent - open-thoughts/OpenThoughts-Agent-v1-SFT
7. jupyter_agent - jupyter-agent/jupyter-agent-dataset
8. nemotron_agentic - nvidia/Nemotron-Agentic-v1
9. swe_rebench_v2 - nebius/SWE-rebench-V2

### Large (needs BEAST or careful disk management):
10. mixture_of_thoughts - open-r1/Mixture-of-Thoughts (350K rows)
11. multi_swe_bench - ByteDance-Seed/Multi-SWE-bench

### Previously queued (not yet downloaded):
12. github_codereview (2.58 GB)
13. commit_chronicle (10.7M)
14. opencodeinstruct (5M, BEAST only)
15. github_code_2025 (1.5M repos, BEAST only)


---

## Ingestion Tracker

### Already Ingested to agent_memory (396 chunks):
- 27 curriculum docs (foundations)
- 13 expert knowledge docs (research, evolution, distillation, etc.)
- 7 new docs from this session (research methodology, freshness, evolution,
  meta-learning, ecosystem, distillation, canary)

### Ready to Ingest (this session):
- rl_for_code_grpo_rlvr.md (Sprint C2-2 lesson plan)
- evolutionary_code_optimization.md (Sprint C2-3 lesson plan)

### Ingest After Download Completes:
- Each new FTS5 index becomes available for federated search
- High-value chunks get distilled into agent_memory summaries
