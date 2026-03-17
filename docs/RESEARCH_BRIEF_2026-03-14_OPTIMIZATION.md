# JCoder Optimization Research Brief
**Date:** 2026-03-14
**Sources:** Web research across 40+ papers and tools (2024-2026)

## Executive Summary

Research identifies 3 categories of high-ROI improvements for JCoder:
1. **Retrieval quality** -- AST-aware chunking, RRF fusion, structural metadata
2. **Evolution intelligence** -- MAP-Elites archives, island model, surrogate scoring
3. **Self-improvement loop** -- evolutionary memory, self-referential mutation, meta-prompting

---

## Part 1: Code RAG Optimization

### Tier 1 (Highest Impact)

**AST-Aware Chunking** (cAST, EMNLP 2025)
- Current: line/size-based splitting breaks functions across chunk boundaries
- Fix: Use tree-sitter to chunk at function/class/method boundaries
- Impact: +2.7 to +5.5 points on code benchmarks (RepoEval, SWE-bench)
- Tool: tree-sitter (multi-language), ASTChunk Python library
- This is the single largest retrieval gain available

**RRF Fusion of FTS5 + FAISS**
- Formula: score = 1/(rank + 60) for each system, then sum
- Impact: 15-30% recall improvement over either method alone
- No model changes needed -- just rank fusion at query time
- BM25/FTS5 irreplaceable for exact identifier matching (calculateTaxRate)
- Dense vectors catch semantic queries ("how do I sort a list")

**Parent-Child Chunk Indexing**
- Index small chunks (individual functions, ~200 tokens) as children
- Store parent_id pointing to containing file/class
- Search against children for precision, return parent for context
- Directly implementable in SQLite with one column addition

### Tier 2 (High Impact)

**Structural Metadata as FTS5 Fields**
- Index alongside code body: file path, language, imports, class name, function signature
- Makes retrieval structure-aware without full GraphRAG
- File-path boosting: chunks from same file/directory as previous hits get priority

**Corrective RAG Loop (CRAG pattern)**
- Retrieve -> evaluate relevance -> reformulate query -> retry (2-3 rounds max)
- Catches first-pass misses on complex multi-file queries
- JCoder already has CorrectiveRetriever (Sprint 10) -- wire it into the main pipeline

**Code-Specific Embeddings**
- nomic-embed-code (7B, Apache 2.0) outperforms Voyage Code 3 and OpenAI
- Needs BEAST for reasonable throughput (7B model)
- Trained on CoRNStack with curriculum-based hard negative mining

### Tier 3 (Worth Investigating)

**Import/Call Graph as Secondary Index**
- Simple SQLite adjacency table: function_id -> calls -> function_id
- When a chunk is retrieved, auto-pull imported/called function chunks
- InlineCoder pattern: trace actual call chains, not just similarity

**Query Decomposition for Complex Queries**
- Split "how does auth flow work?" into sub-queries
- Retrieve separately, merge results
- A-RAG (Feb 2026): agentic retriever chooses keyword vs semantic vs chunk-read per query

**Persistent Retrieval Memory**
- Cache successful query-retrieval-answer triples
- RAG-EVO showed this significantly improves accuracy over time
- JCoder already has PersistentMemory (Sprint 16) -- extend to retrieval caching

---

## Part 2: Evolution & Tournament Optimization

### Tier 1 (Highest Impact)

**MAP-Elites Archive (replaces flat bracket elimination)**
- Instead of keeping only winners, maintain a 2D grid
- Axes: e.g., latency vs accuracy, or context_window vs top_k
- Each cell keeps the best config for that niche
- Prevents convergence collapse (all configs drifting to same local optimum)
- Used by AlphaEvolve (DeepMind, May 2025) at production scale
- AlphaEvolve found first improvement to matrix multiplication in 56 years

**Island Model with Ring-Topology Migration**
- Run 3-4 independent tournament populations (islands)
- Each island evolves configs independently
- Every N generations, migrate top config to neighbor island
- Prevents premature convergence while allowing good solutions to spread
- Used by AlphaEvolve, OpenEvolve, CodeEvolve

**Evolutionary Memory (LoongFlow, Dec 2025)**
- Track which mutations improved fitness and which didn't
- Before generating new mutations, provide history as context:
  "Increasing top_k above 12 consistently hurt accuracy"
- The LLM uses this memory for informed mutation decisions
- JCoder already has RecursiveMetaLearner (Sprint 24) -- extend it

### Tier 2 (High Impact)

**Surrogate-Assisted Pre-Screening**
- Train lightweight predictor on past tournament results
- Filter obviously bad mutations before expensive full evaluation
- Even a decision tree on historical data helps
- Analogous to zero-cost proxies in Neural Architecture Search

**Bi-Level Parameter Separation**
- Structural params (model, retrieval strategy): evolve slowly
- Knob params (temperature, top_k): evolve fast
- Different mutation rates for different parameter classes
- Prevents wasting compute on slow-changing structural decisions

**Dual-Model Mutation (AlphaEvolve pattern)**
- Fast/cheap model (phi4-mini) for bulk config generation
- Powerful model (phi4:14b) for targeted refinement of top candidates
- Matches AlphaEvolve's Gemini Flash/Pro split
- Perfect for BEAST: run phi4-mini in parallel while phi4:14b refines winners

### Tier 3 (Worth Investigating)

**Self-Referential Mutation Rules (PromptBreeder, ICML 2024)**
- Evolve the mutation strategy itself alongside configs
- Start with simple rules, let the system discover what works
- Two-level evolution: task configs + meta-mutation prompts

**EvoPrompt for System Prompts (ICLR 2024)**
- Evolve system prompts alongside numeric config knobs
- LLM generates prompt variants via mutation/crossover
- Up to 25% improvement over human-engineered prompts on BIG-Bench Hard

**Adaptive Mutation Frequency (IPBT, Nov 2025)**
- Start with frequent mutations (explore broadly)
- Detect convergence, reduce mutation intensity
- If top-3 configs within 2% of each other, slow down

**Meta-Prompting for Mutation (CodeEvolve, Oct 2025)**
- Instead of random perturbation, ask LLM:
  "Given these tournament results [top 3 configs with scores],
   what config change is most likely to improve code correctness?"
- Three operators: inspiration crossover, meta-prompting, depth-based refinement

---

## Part 3: Key Papers & Tools Reference

### Code RAG
| Paper/Tool | Year | Key Contribution |
|------------|------|------------------|
| cAST (EMNLP 2025) | 2025 | AST-based chunking, +5.5 on RepoEval |
| nomic-embed-code | 2025 | 7B code embedder, Apache 2.0, SOTA |
| A-RAG | 2026 | Agentic retriever, +5-13% over flat |
| InlineCoder | 2025 | Call-chain context inlining |
| CodeRAG | 2025 | Requirement graph for multi-file codegen |
| Self-RAG (ICLR 2024) | 2024 | Reflection tokens for self-critique |
| CRAG | 2024 | Corrective retrieval with evaluator |

### Evolution & Optimization
| Paper/Tool | Year | Key Contribution |
|------------|------|------------------|
| AlphaEvolve (DeepMind) | 2025 | Full codebase evolution, MAP-Elites + island model |
| OpenEvolve | 2025 | Open-source AlphaEvolve implementation |
| CodeEvolve | 2025 | Island GA, LLM ensemble, 3 modular operators |
| GigaEvo | 2025 | Modular MAP-Elites, async DAG evaluation |
| LoongFlow | 2025 | Hybrid evolutionary memory + LLM reasoning |
| EvoPrompt (ICLR 2024) | 2024 | Evolutionary prompt optimization, +25% |
| PromptBreeder (ICML 2024) | 2024 | Self-referential mutation of meta-prompts |
| VQ-Elites | 2025 | Auto-discover behavioral niches via VQ-VAE |
| IPBT | 2025 | Adaptive mutation scheduling via Bayesian opt |

---

## Part 3: Self-Evolving AI & Autonomous Agents

### Tier 1 (Highest Impact)

**Darwin Godel Machine (DGM) -- Sakana AI, 2025**
- Self-improving agent that modifies its own source code
- Uses LLM to propose improvements, validates in sandboxed environments
- SWE-bench: improved own performance from 20% to 50%
- Evolutionary lineage tracking (expanding population of variants)
- **Actionable**: JCoder's self-learning pipeline could propose modifications to its own
  retrieval/generation code, validate against benchmarks in sandbox, promote only improvements

**Planner-Coder-Tester-Reviewer Architecture (AgentMesh/HyperAgent)**
- Four specialized agent roles with orchestrated handoffs
- Constrained command vocabulary (search_code, edit_file, run_tests)
- SWE-Agent: constraining to structured commands paradoxically improves performance
- **Actionable**: Wire JCoder's multi_agent.py (Sprint 15) with explicit roles and command sets

**Self-Healing Execution Loop**
- Generate code -> execute tests -> diagnose failures via RAG -> repair -> re-test
- 95% reduction in manual maintenance in test automation contexts
- **Actionable**: Feed error traces through JCoder's RAG indexes to find similar past errors
  and their solutions, then use retrieved context to guide repair attempts

### Tier 2 (High Impact)

**GRPO/RLVR for Retrieval Optimization**
- Use verifiable rewards (test pass/fail, lint scores) as automatic reward signals
- Generate multiple retrieval strategies per query, score by outcome, update weights
- No human labeling needed -- pure execution-based feedback
- **Warning**: RLVR does NOT elicit fundamentally new reasoning (OpenReview 2025).
  It amplifies existing patterns. Prolonged training can cause reward hacking.
- **Actionable**: Lightweight version -- use code execution outcomes to tune RAG
  retrieval weights without full RL training

**Devstral Small 2 (24B) as Primary Model**
- 68% SWE-bench Verified, Apache 2.0, 256K context
- Fits single RTX 4090 or BEAST's 48GB VRAM with headroom
- Outperforms current Devstral Small by large margin
- **Actionable**: Evaluate as JCoder primary once available on Ollama

**Persistent Agentic Memory (AgeMem pattern)**
- Expose memory as explicit tool actions: store/retrieve/update/summarize/forget
- Let the agent decide when to use each operation
- Memory is #1 developer complaint about coding agents in 2025-2026
- **Actionable**: JCoder already has PersistentMemory (Sprint 16) -- extend with
  explicit memory tool actions in the agent command vocabulary

### Tier 3 (Worth Investigating)

**Model Distillation on BEAST**
- Run large teacher (Devstral 2 123B or cloud APIs) to generate training data
- Distill into 24B student model for daily use
- Projects: DistilMistral, Hermes-DPO-Lite demonstrate practicality

**Inspiration-Based Crossover (CodeEvolve)**
- Feed two successful configs into LLM context, ask for intelligent hybrid
- More effective than random parameter crossover

---

## Part 4: Key Papers & Tools Reference (Self-Evolving AI)

| Paper/Tool | Year | Key Contribution |
|------------|------|------------------|
| Darwin Godel Machine (Sakana AI) | 2025 | Self-modifying agent, 20%->50% SWE-bench |
| CodeEvolve (Inter & Co) | 2025 | Open-source evolutionary coding agent |
| AdaEvolve | 2026 | Hierarchical adaptive optimization |
| AgentMesh | 2025 | Planner-Coder-Debugger-Reviewer |
| HyperAgent | 2025 | 4 specialized agents (Planner/Navigator/Editor/Executor) |
| SWE-Agent (Princeton NLP) | 2025 | Constrained Agent-Computer Interface |
| GRPO (DeepSeek) | 2025 | Group Relative Policy Optimization |
| DAPO (ByteDance) | 2025 | Fixes GRPO instabilities for long CoT |
| Devstral Small 2 (Mistral) | 2025 | 24B, 68% SWE-bench, Apache 2.0 |
| Devstral 2 (Mistral) | 2025 | 123B, 72.2% SWE-bench, open-weight SOTA |

---

## Implementation Priority (for BEAST deployment)

### Phase 1: Quick Wins (first week on BEAST)
1. AST-aware re-chunking with tree-sitter (JCoder)
2. Evolutionary memory in tournament runner
3. FTS5 source_path indexing (HybridRAG3)
4. Reranker revival with Ollama backend (HybridRAG3)

### Phase 2: Architecture Upgrades (weeks 2-3)
5. MAP-Elites archive replacing flat bracket
6. Island model with 4 populations
7. Self-healing execution loop (generate->test->RAG diagnose->repair)
8. Planner-Coder-Tester-Reviewer agent roles
9. Corrective retrieval loop (HybridRAG3)

### Phase 3: Model & Advanced (weeks 4+)
10. Devstral Small 2 evaluation as primary model
11. nomic-embed-code deployment (7B)
12. Surrogate scoring for mutation pre-screening
13. GRPO-style verifiable reward signals for retrieval tuning
14. Model distillation pipeline (teacher->student)
15. Self-referential mutation rules (PromptBreeder pattern)
