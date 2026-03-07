# Agentic Self-Learning: Deep-Dive Subagent Findings

Date: 2026-03-07
Companion to: 11_RESEARCH_AGENTIC_SELF_LEARNING.md, 11a_CITATIONS_RANKED.md

These are the detailed findings from 7 parallel research subagents deployed
on 2026-03-07. Each section contains full technical details beyond what the
summary document covers.

---

## 1. GEPA: Genetic-Pareto Reflective Prompt Evolution (ICLR 2026 Oral)

**Full name**: Genetic Evolution with Prompt-level Adaptation
**Paper**: "GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning"
**Authors**: Lakshya Agrawal, Shangyin Tan, Omar Khattab (DSPy creator), Matei Zaharia
(Spark/Databricks founder), Dan Klein, Ion Stoica -- UC Berkeley, Stanford, UT Austin,
Notre Dame, Databricks
**arXiv**: https://arxiv.org/abs/2507.19457
**Code**: https://github.com/gepa-ai/gepa
**DSPy integration**: `dspy.GEPA` (production-ready)
**Already deployed at**: Shopify, Databricks, Dropbox, OpenAI, Pydantic, MLflow

### The Five-Step Self-Evolution Loop

1. **Select**: Pick a candidate from the Pareto frontier (not just global best, but
   candidates excelling on DIFFERENT subsets of examples)
2. **Execute**: Run on a minibatch, capturing FULL execution traces (reasoning steps,
   tool calls, tool outputs, error messages, profiling data)
3. **Reflect**: LLM reads traces and produces Actionable Side Information (ASI) --
   natural language diagnosis of failures. This is the text-optimization analogue of
   a gradient.
4. **Mutate**: Generate improved candidates informed by accumulated lessons from ALL
   ancestors in the search tree. Supports system-aware merge of two Pareto-optimal
   candidates.
5. **Accept**: If mutated candidate improves on ANY dimension, add to pool and update
   Pareto front.

### Key Innovation: ASI Over Scalar Rewards

Instead of collapsing execution into a single number (reward), GEPA lets an LLM read
the full trace and diagnose WHY something failed. This produces surgical, targeted
mutations rather than random exploration.

### Results vs Baselines

| Comparison | Result |
|-----------|--------|
| vs GRPO (RL) | +6% average, up to +19pp on specific tasks |
| vs MIPROv2 (SOTA prompt optimizer) | +10pp aggregate, +12pp on AIME-2025 |
| Rollout efficiency | 35x fewer rollouts (100-500 vs 5,000-25,000+) |
| Prompt length | Up to 9.2x shorter than MIPROv2 |
| ARC-AGI | 32% -> 89% via architecture discovery |
| MATH | 93% accuracy (DSPy full program) |
| Cost savings | 90x vs using a larger model |

### JCoder Implementation Notes

GEPA is immediately usable via `pip install dspy` and calling `dspy.GEPA`. The core
loop (select -> execute -> reflect -> mutate -> accept) maps directly to Claude CLI
tool loops. The Pareto frontier approach prevents premature convergence.

---

## 2. Self-Evolving Curriculum (SEC)

**Paper**: "Self-Evolving Curriculum for LLM Reasoning" (arXiv:2505.14970)
**Authors**: Xiaoyin Chen, Jiarui Lu, Yoshua Bengio et al.

### Core Mechanism: Non-Stationary Multi-Armed Bandit

SEC frames curriculum selection as a MAB problem:
- Training problems partitioned into N categories (by difficulty)
- Each category is an "arm"
- Q-values updated via TD(0): `Q_{t+1}(c) = alpha * r_t(c) + (1 - alpha) * Q_t(c)`
- Sampling via Boltzmann distribution over Q-values

### The p=0.5 Insight

Under binary rewards (pass/fail), expected absolute advantage = `2*sqrt(p*(1-p))`,
which peaks at p=0.5. SEC automatically gravitates toward problems at the agent's
zone of proximal development without any human heuristics.

### Results (Qwen2.5-3B/7B)

| Benchmark | Random | SEC | Gain |
|-----------|--------|-----|------|
| Countdown (OOD, 3B) | 0.479 | 0.542 | +13% |
| Zebra Puzzles (OOD, 3B) | 0.285 | 0.345 | +21% |
| ARC-1D (OOD, 3B) | 0.313 | 0.381 | +22% |
| AIME24 (7B) | 0.138 | 0.175 | +27% |

### Related Curriculum Papers

- **Actor-Curator** (Feb 2026): Neural network curator, no manual bucketing, +28.6%
  on AIME2024, 80% training speedup
- **VCRL**: Variance of group rewards as difficulty proxy, same p=0.5 optimum
- **AdaRFT** (Apr 2025): Adjustable target difficulty, 2x training speedup
- **TACLer** (Jan 2026): Model-tailored curriculum, 50%+ compute savings
- **Negative finding**: One paper found NO curriculum benefit on synthetic math tasks,
  suggesting benefits may be task-dependent

---

## 3. ReasoningBank + Remember Me, Refine Me (ReMe)

### ReasoningBank (arXiv:2509.25140, Sep 2025, Google)

**Memory format**: {title, description, content} triples
**Retrieval**: Cosine similarity, top-k=1 (diminishing returns beyond k=2)
**No fine-tuning needed** -- entirely prompt-driven

**Key mechanisms**:
- Binary classifier judges Success/Failure (no ground truth needed)
- Separate extraction prompts for success (transferable strategies) and failure
  (preventive strategies)
- Critical finding: Success+Failure memory = 49.7% vs Success-only = 46.5%
  (failures produce constructive contrastive signals)

**Memory-Aware Test-Time Scaling (MaTTS)**:
- Parallel: Generate N trajectories, self-contrast across all N
- Sequential: Iterative refinement within single trajectory

**Emergent evolution stages**: Procedural -> Reflective -> Adaptive -> Compositional

### Remember Me, Refine Me (arXiv:2512.10696, Dec 2025, Alibaba/AgentScope)

**Memory format**: 5-tuple (scenario, knowledge, keywords, confidence, tools)
**Code**: https://github.com/agentscope-ai/ReMe

**Three core mechanisms**:

1. **Multi-faceted distillation**: Success patterns + failure analysis + comparative
   insight (best vs worst trajectory for same task). N=8 trajectories sampled per
   task at temp=0.9.

2. **Context-adaptive reuse**: Retrieve (top-5) -> Rerank (LLM) -> Rewrite
   (reorganize for current context). Rewriting is the key differentiator vs raw paste.

3. **Utility-based refinement**:
   - Validation gate: LLM scores actionability, accuracy, relevance, clarity, uniqueness
   - Rejection threshold: score < 0.3
   - Deletion trigger: retrieved 5+ times AND helped < 50% of the time
   - Deduplication: similarity-based before storage

**Key result**: Qwen3-8B with ReMe (55.03% Pass@4) OUTPERFORMS memoryless Qwen3-14B
(54.65%). Memory compensates for model size.

### Comparative Analysis

| Aspect | ReasoningBank | ReMe |
|--------|--------------|------|
| Memory format | {title, desc, content} | 5-tuple with confidence |
| Retrieval | Top-1, raw injection | Top-5 + rerank + rewrite |
| Pruning | None (append-only) | Utility-based (success rate tracking) |
| Failure learning | Separate extraction prompt | Contrastive best/worst analysis |
| Evolution | Emergent | Engineered lifecycle |
| Fine-tuning | No | No |

---

## 4. Self-Play SWE-RL (SSR) + PSV

### SSR: Self-Play SWE-RL (arXiv:2512.18552, Dec 2025, Meta FAIR)

A single 32B LLM plays two roles via different prompting:

**Bug Injection Agent**:
- Explores repository, discovers test infrastructure
- Produces 5-part bug artifact: bug patch, test script, test files, test parser, test-weakening patch
- Minimum thresholds: 10+ passing tests, 2+ changed files, 3+ failing tests
- "Removal + history" strategy prevents trivial one-line bugs

**Bug Solving Agent**:
- Attempts to find and fix the injected bug
- Uses only the test patch as specification (no natural language)

**Automatic difficulty escalation**: Reward function incentivizes bugs at ideal
difficulty (challenging but solvable with low success rate). As solver improves,
injector learns to create harder bugs.

**Results**:
- SWE-bench Verified: +10.4 absolute points over human-data-trained baseline
- SWE-bench Pro: +7.8 absolute points
- Generalizes to natural-language issues (never seen during training)

### PSV: Propose, Solve, Verify (arXiv:2512.18160, Dec 2025)

Uses FORMAL VERIFICATION (Verus for Rust) instead of unit tests.

**Three-phase loop**:
1. Propose: Difficulty-aware proposer generates formal specs (pre/postconditions)
2. Solve: Multiple candidates via expert iteration
3. Verify: Verus verifier provides binary pass/fail (mathematically guaranteed)

**Results**:

| Benchmark | PSV | AlphaVerus | Improvement |
|-----------|-----|------------|-------------|
| Dafny2Verus (274) | 65.63% | 24.06% | 2.73x |
| MBPP-Verified (78) | 36.78% | 6.48% | 5.68x |
| HumanEval-Verified (85) | 16.18% | 7.24% | up to 9.6x |

**Critical ablation**: Removing formal verification = 51.5% COLLAPSE in pass@1.
Verification is the essential ingredient, not optional.

---

## 5. RWML: Reinforcement World Model Learning (arXiv:2602.05842, Feb 2026, Microsoft)

### Two-Stage Pipeline

**Stage 1 -- World Model Learning (self-supervised)**:
1. Collect (state, action, next-state) triplets from rollouts
2. Model predicts next state given current state + action
3. Predicted vs actual state compared via embedding cosine similarity
4. Binary reward: 1.0 if distance < threshold, 0.0 otherwise
5. Training via GRPO with curriculum (subsample easy predictions)

**Stage 2 -- Policy Learning**:
- Start from world-model-trained weights
- Apply GRPO with task-success rewards
- World model pre-training provides better initialization

### Key Innovation: Semantic Embedding Rewards

- Operates in embedding space, not token space
- Many wordings can describe same state transition
- Less catastrophic forgetting than token-level SFT (2.38 MMLU drop vs 10.1)
- LLM-as-a-judge alternative COLLAPSED performance (32.6% -> 3.6%)

### Results

| Method | ALFWorld | tau-squared |
|--------|----------|-------------|
| Base ReACT | 37.7% | 31.9% |
| Policy RL alone | 81.0% | 38.0% |
| **RWML + Policy RL** | **87.9%** | **43.7%** |
| Expert imitation | 82.5% | 43.7% |

Matches expert-data performance WITHOUT expert demonstrations.

### Related World Model Work

- **DynaWeb** (Jan 2026): Sutton's Dyna for web agents, 40% real + 60% dreamed data
- **AWM** (Feb 2026, Snowflake): 1,000 synthetic code environments, Arctic-AWM-14B
- **WALL-E 2.0**: Neurosymbolic rules, 98% ALFWorld after 4 iterations
- **R-WoM**: Retrieval-augmented world models, +25.3% on procedural tasks

---

## 6. Lifelong Agents ICLR 2026 Workshop

**Date**: April 26, 2026, Rio de Janeiro
**Key frameworks**:

### EvolveR (arXiv:2510.16079)

Three-stage closed loop:
1. Offline self-distillation extracts guiding/cautionary principles from trajectories
2. Online interaction retrieves principles via search to guide decisions
3. Policy evolution via GRPO

Key finding: Self-distilled principles outperform external teacher models at larger scales.

### StuLife Benchmark (arXiv:2508.19005)

1,284 tasks simulating a college journey. Even GPT-5 scored only 17.9/100, revealing
fundamental gaps in long-term retention and self-motivated initiative.

### Anti-Forgetting Techniques

- Neural ODE + Memory-Augmented Transformers: 24% forgetting reduction
- Nested Learning (Google, NeurIPS 2025): Stack of learning at different time-scales
- Sparse Memory Fine-Tuning (Meta FAIR): Nearly eliminates catastrophic forgetting

### Core Tension: Stability-Plasticity Dilemma

Too much plasticity = forget prior knowledge. Too much stability = can't adapt.
Every technique threads this needle differently.

---

## 7. Claude CLI Self-Learning Patterns: Feasibility Assessment

### What WORKS Today

| Pattern | Mechanism | Reliability |
|---------|-----------|-------------|
| Persistent memory | CLAUDE.md + auto-memory files | HIGH (200-line limit) |
| Tool-use loops | think -> act -> observe -> correct -> repeat | HIGH |
| Self-evaluation | Lint/test results fed back for iteration | HIGH |
| Recursive decomposition | Parallel/sequential/background subagents | HIGH |
| Task DAGs | Dependency graphs across subagents | MEDIUM-HIGH |

### What EXISTS but Has Limits

| Pattern | Mechanism | Limitation |
|---------|-----------|------------|
| Cross-session memory | memsearch (Milvus vector store) | Requires MCP plugin |
| Knowledge accumulation | metaswarm JSONL knowledge base | Manual reflection triggers |
| Experience replay | Session summaries re-injected | Lossy compression at boundaries |

### Hard Limits (Cannot Be Overcome)

| Constraint | Value |
|-----------|-------|
| Context window | 200K tokens (~160K effective) |
| Performance degradation | Starts at ~150K tokens |
| CLAUDE.md limit | ~200 lines / 2K tokens |
| Auto-compaction trigger | 64-75% capacity (lossy) |
| Session boundary | Full context loss |
| Model weights | FROZEN -- no actual learning |

### Notable Community Projects

- **metaswarm**: 18 agents, 13 skills, TDD enforcement, self-reflecting JSONL KB
  (github.com/dsifry/metaswarm)
- **Ruflo v3**: 60+ agents, swarm coordination, fault-tolerant consensus
  (github.com/ruvnet/ruflo)
- **memsearch**: Vector-indexed persistent memory via hooks, daily session summaries,
  forked subagent for memory search
  (milvus.io/blog/adding-persistent-memory-to-claude-code-with-the-lightweight-memsearch-plugin.md)
- **recursive-decomposition-skill**: RLM strategies for 100x context extension
  (github.com/massimodeluisa/recursive-decomposition-skill)

### Bottom Line

All "self-learning" in Claude CLI is EXTRINSIC -- writing structured data to files
that get re-injected into future context windows. The model itself never changes.
The ceiling is determined by: context window size, compaction quality, and how well
the external memory system indexes and retrieves past experience. The most successful
patterns combine deterministic hooks (reliability) + structured knowledge bases
(experience accumulation) + explicit verification loops (self-correction).
