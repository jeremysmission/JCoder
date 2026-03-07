# Agentic Self-Learning Research: Cutting-Edge Techniques (2025-2026)

Date: 2026-03-07
Status: Research Summary -- Comprehensive Survey of Breaking Techniques
Companion to: 09_RESEARCH_SELF_EVOLUTION.md (covers SICA, DGM, STOP, EvoAgentX from Feb 2026)
Deep dives: 11a_CITATIONS_RANKED.md (20 ranked papers with links)
             11b_DEEP_DIVE_SUBAGENT_FINDINGS.md (full technical details from 7 research agents)

---

## 1. Executive Summary

The period from September 2025 to March 2026 has produced an explosion of research in
agentic self-learning. Two landmark ICLR 2026 workshops -- "AI with Recursive Self-
Improvement" and "Lifelong Agents: Learning, Aligning, Evolving" -- signal that the field
has crossed from theoretical speculation to deployed systems. The core paradigm shift:
agents that improve themselves autonomously through experience, self-play, procedural
memory, and evolutionary prompt optimization -- without requiring gradient updates or
human supervision at each step.

This document catalogs the 15 most promising techniques, assesses which are implementable
with Claude CLI (no fine-tuning required), ranks the top 10 online sources for staying
current, and identifies the techniques closest to practical breakthrough.

---

## 2. Theoretical Foundations

### 2.1 The MASE Framework and Three Laws (arXiv:2508.07407)

The most comprehensive taxonomy of self-evolving agents comes from the MASE (Multi-Agent
Self-Evolving) framework, which establishes three foundational laws inspired by Asimov:

| Law | Name | Constraint |
|-----|------|------------|
| 1st | **Endure** | Modifications must maintain system stability and safety |
| 2nd | **Excel** | Evolution must not degrade existing task performance |
| 3rd | **Evolve** | Subject to Laws 1-2, the agent must autonomously improve |

The framework categorizes all self-evolution into three optimization axes:
- **Single-agent optimization**: LLM behavior, prompts, memory, tools
- **Multi-agent optimization**: architecture search, workflow generation, swarm evolution
- **Domain-specific optimization**: coding, science, reasoning, multimodal

### 2.2 Generator-Verifier-Updater (GVU) Operator (arXiv:2512.02731)

The most rigorous mathematical foundation for self-improvement comes from the GVU
framework, which decomposes any self-improvement process into three canonical functions:

1. **Generator (G)**: Samples candidate solutions from current policy
2. **Verifier (V)**: Scores candidates using an internal potential function
3. **Updater (U)**: Adjusts parameters based on weighted high-scoring samples

**Key mathematical results:**

- **Self-Improvement Coefficient (kappa)**: The Lie derivative of the capability functional
  along the resource flow. Positive kappa = sustained improvement; kappa ~ 0 = stagnation.

- **Variance Inequality**: The sufficient condition for positive expected capability gain:
  `rho > (eta*L/2)(rho^2 + 1/SNR(G) + 1/SNR(V))`
  where rho = verifier-external alignment, eta = step size, L = curvature,
  SNR = signal-to-noise ratios for generation and verification.

- **Hallucination Barrier**: When generator and verifier are identical (diagonal GVU),
  verification noise mirrors generation noise, typically preventing sustained improvement.
  This explains why external verification (unit tests, formal proofs, human review) is
  critical for real self-improvement.

- **Verification Dominance**: For any fixed generator noise, sufficiently strong verification
  enables positive expected gain. This is why code agents (with test execution as verifier)
  improve faster than prose agents.

- **Universality**: ANY first-order statistical update admits an implicit GVU decomposition.
  This unifies AlphaZero, RLHF, Self-Instruct, Constitutional AI, and code agents as
  topological realizations of the same operator.

**JCoder Relevance**: This proves mathematically that our existing GVU-like loop (generate
code -> run tests -> update strategy) is the canonical mechanism. The key lever is
verification quality, not generation quality. Invest in better test suites, not bigger
models.

### 2.3 Intrinsic Metacognitive Learning (OpenReview, ICLR 2026)

Position paper arguing that truly self-improving agents require three intrinsic
metacognitive components:

1. **Metacognitive Knowledge**: Self-assessment of capabilities, tasks, and strategies
2. **Metacognitive Planning**: Deciding what and how to learn based on self-understanding
3. **Metacognitive Evaluation**: Reflecting on learning experiences to improve future learning

**Key finding**: Current self-improving agents rely on rigid, externally-designed loops that
fail to generalize across task domains and struggle to scale with increasing capabilities.
Many foundational elements for intrinsic metacognition already exist in current AI systems
but remain underdeveloped.

**JCoder Relevance**: Our meta-cognition module already implements a lightweight version of
this. The paper validates our approach and suggests extending it with dynamic strategy
selection based on self-assessed competence per task type.

---

## 3. Breaking Techniques (Last 6 Months)

### 3.1 PRAXIS: Procedural Recall for Agents (arXiv:2511.22074, Nov 2025)

**What**: A lightweight post-training learning mechanism that stores action consequences and
retrieves them by jointly matching environmental + internal states.

**How it works**:
1. Agent takes actions in environment (e.g., web browsing)
2. State-action-result triples are stored in procedural memory in real time
3. On new tasks, current state is matched against stored episodes
4. Retrieved exemplars augment action selection (prompt injection)

**Results**: On REAL web browsing benchmark:
- Improved task completion accuracy across all model backbones
- Reduced steps-to-completion from 25.2 to 20.2 (average across models)
- Shows preliminary generalization to unseen tasks

**Why it matters**: No fine-tuning required. Pure inference-time learning from experience.
Works with any LLM backbone including frozen API models.

**Claude CLI feasibility**: HIGH. Can be implemented as a JSON experience store with
semantic state matching. JCoder already has FTS5 indexes for retrieval.

### 3.2 RISE: Recursive IntroSpEction (NeurIPS 2024, deployed 2025-2026)

**What**: Transforms single-turn prompts into multi-turn MDPs where the agent iteratively
detects and corrects its own mistakes.

**How it works**:
1. Agent generates initial response
2. Agent reviews its own response for errors
3. Agent generates a corrected version
4. Process repeats for N turns (typically 3-5)
5. Training: bootstrap on-policy rollouts with best-of-N at each turn

**Results**:
- LLaMA3-8B: +8.2% improvement over 5-turn introspection
- Mistral-7B: +6.6% improvement
- LLaMA2-7B: +17.7% improvement (largest gain on weakest model)

**Why it matters**: The improvement is largest on weaker models, suggesting this is a
powerful technique for getting more out of smaller, locally-runnable models.

**Claude CLI feasibility**: HIGH. Multi-turn self-correction is trivially implementable
with Claude CLI tool loops. Already partially implemented in JCoder's bridge pattern.

### 3.3 DEEVO: Debate-Driven Evolutionary Prompt Optimization (Aug 2025)

**What**: Evolves LLM prompts through structured debates with Elo-based selection.

**How it works**:
1. Maintain a population of prompt variants
2. Pairwise debates: two prompts compete on the same task
3. Elo ratings track prompt fitness over time
4. Intelligent crossover: combine high-Elo prompts
5. Mutation: LLM-driven prompt modification
6. Age-quota and newcomer-veteran balancing maintain diversity

**Key advantage**: Requires NO ground truth feedback. Works on open-ended tasks where
correctness is subjective. Significantly outperforms manual prompt engineering and
alternative optimization approaches.

**Claude CLI feasibility**: MEDIUM-HIGH. Requires running multiple prompt variants in
parallel and maintaining a leaderboard. Could be implemented with subagent tournament
brackets. Compute-intensive but architecturally simple.

### 3.4 GEPA: Genetic-Pareto Reflective Prompt Evolution (ICLR 2026 Oral)

**What**: GEPA = Genetic-Pareto. Full title: "GEPA: Reflective Prompt Evolution Can
Outperform Reinforcement Learning." Accepted as ORAL at ICLR 2026 (top ~1%).
Authors: Lakshya Agrawal, Omar Khattab (DSPy creator), Matei Zaharia (Spark founder),
Dan Klein, Ion Stoica -- UC Berkeley, Stanford, UT Austin, Databricks.
**Already deployed at**: Shopify, Databricks, Dropbox, OpenAI, Pydantic, MLflow.
**Code**: github.com/gepa-ai/gepa | Integrated as `dspy.GEPA`.

**How it works** (5-step loop):
1. **Select**: Pick candidate from Pareto frontier (not just global best)
2. **Execute**: Run on minibatch, capture FULL execution traces
3. **Reflect**: LLM reads traces, produces Actionable Side Information (ASI) --
   natural language diagnosis of WHY something failed. This is the text-optimization
   analogue of a gradient.
4. **Mutate**: Generate improved candidates informed by ALL ancestor lessons.
   Supports system-aware merge of two Pareto-optimal candidates.
5. **Accept**: If mutated candidate improves on ANY dimension, add to pool.

**Results**:
- vs GRPO (RL): +6% average, up to +19pp, with **35x fewer rollouts**
- vs MIPROv2 (SOTA prompt optimizer): +10pp aggregate, +12pp on AIME-2025
- ARC-AGI: 32% -> 89% via architecture discovery
- MATH: 93% accuracy | Prompts up to 9.2x shorter than MIPROv2
- 90x cost savings vs using a larger model

**Why it matters**: Proves that LLM self-reflection on execution traces learns FASTER
than RL reward signals. This is a paradigm shift -- natural language feedback > scalar
rewards for prompt optimization.

**Claude CLI feasibility**: HIGH. The entire loop maps to Claude CLI tool cycles.
Available immediately via `pip install dspy` + `dspy.GEPA`.

### 3.5 SeRL: Self-Play RL with Limited Data (arXiv:2505.20347, May 2025)

**What**: Bootstraps LLM training from limited initial data via two self-play modules:

1. **Self-Instruction**: Generates additional training instructions from available data
   with robust filtering strategies
2. **Self-Rewarding**: Majority-voting mechanism estimates response quality without
   external annotations

**Results**: Matches performance of models trained on high-quality data with verifiable
rewards, using only limited seed data.

**Why it matters**: Eliminates the data bottleneck. An agent can bootstrap from a handful
of examples into a capable specialist.

**Claude CLI feasibility**: MEDIUM. The self-instruction component is directly implementable
(generate problems, filter, solve). Self-rewarding via majority vote requires multiple
inference passes but is feasible.

### 3.6 SWE-RL (SSR): Self-Play for Software Engineering (arXiv:2512.18552, Dec 2025)

**What**: A single 32B LLM (Meta FAIR) plays two roles via different prompting:
bug-injection agent AND bug-solving agent. No human-curated issues needed.

**How bug injection works**:
1. Agent explores repository, discovers test infrastructure
2. Produces 5-part bug artifact: bug patch, test script, test files, test parser,
   test-weakening patch
3. Minimum thresholds enforced: 10+ passing tests, 2+ changed files, 3+ failing tests
4. "Removal + history" strategy prevents collapse to trivial one-line bugs

**Automatic difficulty escalation**: Reward function incentivizes bugs at ideal
difficulty (challenging but solvable with low success rate). As solver improves,
injector learns to create harder bugs -- open-ended curriculum with no human input.

**Results**:
- SWE-bench Verified: **+10.4 absolute points** over human-data-trained baseline
- SWE-bench Pro: **+7.8 absolute points**
- Generalizes to natural-language issues (NEVER seen during training)

**Why it matters**: Proves a coding agent can self-improve without ANY human-labeled
data. The dual-role self-play is analogous to AlphaGo but for software engineering.

**Claude CLI feasibility**: HIGH. Claude CLI can inject bugs, run tests, attempt repairs.
The full inject-repair loop is implementable as a tool-use cycle.

### 3.7 PSV: Propose, Solve, Verify (arXiv:2512.18160, Dec 2025)

**What**: Self-play framework using FORMAL VERIFICATION (Verus for Rust) instead of
unit tests. The verification signal is mathematically guaranteed correct for all inputs.

**How it works**:
1. **Propose**: Difficulty-aware proposer generates formal specs (pre/postconditions).
   Categorizes problems as EASY/MEDIUM/HARD/IMPOSSIBLE based on solver pass rate.
2. **Solve**: Multiple candidates via expert iteration
3. **Verify**: Verus formal verifier provides binary pass/fail

**Results**:

| Benchmark | PSV | AlphaVerus (baseline) | Improvement |
|-----------|-----|-----------------------|-------------|
| Dafny2Verus (274) | 65.63% | 24.06% | 2.73x |
| MBPP-Verified (78) | 36.78% | 6.48% | 5.68x |
| HumanEval-Verified (85) | 16.18% | 7.24% | up to 9.6x |

**Critical ablation**: Removing formal verification = **51.5% COLLAPSE** in pass@1.
Verification is not optional -- it IS the essential ingredient.

**Why it matters**: Proves the GVU Verification Dominance theorem in practice. Formal
verification creates the strongest possible self-improvement loop.

**Claude CLI feasibility**: MEDIUM. Requires formal verification backend. Could
approximate with property-based testing (Hypothesis) or comprehensive test suites.
**Code**: github.com/abwilf/psv

### 3.8 Self-Evolving Curriculum (SEC) (arXiv:2505.14970, May 2025)

**What**: Automatic curriculum learning that co-evolves the training difficulty alongside
the RL fine-tuning process.

**How it works**:
1. Start with a pool of training problems at various difficulty levels
2. A curriculum policy selects which problems to train on next
3. The curriculum policy is itself trained via RL alongside the main model
4. As the agent improves, the curriculum automatically increases difficulty

**Why it matters**: Solves the "too easy / too hard" problem in agent training. Static
curricula waste compute on problems that are already solved or hopelessly beyond reach.

**Claude CLI feasibility**: MEDIUM-HIGH. Can be approximated by tracking success rates per
difficulty tier and dynamically adjusting which problems are attempted next.

### 3.9 Remember Me, Refine Me: Dynamic Procedural Memory (arXiv:2512.10696, Dec 2025)

**What**: A dynamic procedural memory framework for experience-driven agent evolution.

**How it works**:
- Agents store successful action sequences as procedural templates
- High-frequency successful paths are solidified into reusable workflows
- Templates are continuously refined based on new experiences
- Failed approaches are explicitly recorded to prevent repetition

**Two paradigms**:
1. **Strategic guidance**: Retrieve relevant past experiences to inform planning
2. **Procedural solidification**: Codify frequent successes into code/templates

**Claude CLI feasibility**: HIGH. This is essentially what JCoder's experience store and
meta-cognition loop already do. The "solidification" step (converting patterns to code)
is the next frontier.

### 3.10 RWML: Reinforcement World Model Learning (arXiv:2602.05842, Feb 2026)

**What**: Self-supervised method that learns action-conditioned world models for LLM agents
on textual states using sim-to-real gap rewards.

**How it works**:
1. Agent takes action in environment, observes resulting state
2. A world model predicts what the next state SHOULD be
3. The gap between predicted and actual state provides a reward signal
4. The world model improves to better predict environment dynamics

**Why it matters**: Agents can "imagine" action consequences before executing them,
reducing expensive real-world interactions. The sim-to-real reward is self-generated.

**Claude CLI feasibility**: MEDIUM. Requires maintaining an internal model of expected
outcomes. Could be approximated with a prediction-verification loop in tool use.

### 3.11 Absolute Zero / R-Zero: Zero-Data Self-Evolution (2025)

**What**: Self-evolving agents that improve from literally zero training data. The agent
generates its own problems, solves them, verifies solutions, and learns from the cycle.

**Why it matters**: Eliminates ALL external data dependencies. The agent is entirely
self-bootstrapping.

**Claude CLI feasibility**: HIGH for code domains (generate problem -> solve -> run tests).
This is the purest form of the self-play loop.

### 3.12 OpenAI Self-Evolving Agents Cookbook (Nov 2025)

**What**: Production-ready pipeline for autonomous agent retraining.

**Architecture**:
1. **Baseline Agent** generates outputs
2. **Feedback Collection**: Human review OR LLM-as-judge
3. **Multi-Grader Evaluation**: 4 complementary graders
   - Rule-based (entity preservation)
   - Metric-based (length deviation)
   - Embedding-based (cosine similarity)
   - LLM-as-judge (holistic quality)
4. **Metaprompt Agent**: Analyzes failures and proposes prompt updates
5. **VersionedPrompt**: Tracks prompt history with rollback capability
6. Retry up to 3 times per section with evolved prompts
7. Deploy highest-scoring version

**Claude CLI feasibility**: HIGH. This entire pipeline is provider-agnostic and maps
directly to Claude CLI tool loops. The grader abstraction, versioned prompts, and
metaprompt agent pattern are immediately implementable.

---

## 4. The ICLR 2026 RSI Workshop: State of the Field

The ICLR 2026 Workshop on "AI with Recursive Self-Improvement" (April 2026) represents
the field's most authoritative snapshot. Key organizing principles:

### 4.1 Five Lenses for Analyzing Self-Improvement

| Lens | Question | Examples |
|------|----------|----------|
| Change Targets | What changes? | Parameters, world models, memory, tools, architecture |
| Temporal Regime | When? | Within episodes, test-time, post-deployment |
| Mechanisms | How? | Reward learning, imitation, evolutionary search |
| Operating Context | Where? | Web/UI, games, robotics, science, enterprise |
| Evidence | Does it work? | Benchmarks, ablations, deployment metrics |

### 4.2 Known Risks

The workshop explicitly acknowledges five failure modes:
1. **Reward/specification hacking** (DGM's objective hacking incident)
2. **Memory drift and stale context** (procedural memory goes stale)
3. **Brittle self-edits** (small prompt changes cause cascading failures)
4. **Unbounded exploration** (agent explores unproductive branches)
5. **Regression** (new capabilities destroy old ones)

### 4.3 Invited Speakers (Top Researchers in the Field)

- **Jeff Clune** (UBC/DeepMind) -- open-ended evolution pioneer
- **Chelsea Finn** (Stanford) -- meta-learning / few-shot adaptation
- **Graham Neubig** (CMU/OpenHands) -- coding agents
- **Matej Balog** (DeepMind) -- program synthesis
- **Bang Liu** (Mila) -- multi-agent systems
- **Yu Su** (Ohio State) -- tool-augmented agents
- **Arman Cohan** (Yale) -- NLP / scientific agents

---

## 5. Techniques Crackable with Claude CLI

The following techniques require NO fine-tuning, NO gradient access, and NO custom
training infrastructure. They are implementable purely through Claude CLI tool loops,
persistent memory, and subagent orchestration.

### Tier 1: Immediately Implementable (Days)

| Technique | Core Mechanism | JCoder Integration Point |
|-----------|---------------|--------------------------|
| **PRAXIS** | Store state-action-result triples, retrieve by state similarity | FTS5 experience store + semantic matching |
| **RISE** | Multi-turn self-correction loops | Bridge pattern (already partially implemented) |
| **OpenAI Cookbook Pipeline** | Multi-grader eval + metaprompt evolution | Eval framework + versioned prompts |
| **Remember Me, Refine Me** | Solidify frequent successes into templates | Meta-cognition -> procedural code generation |
| **Absolute Zero Loop** | Generate problem -> solve -> verify -> learn | Self-play with pytest as verifier |

### Tier 2: Implementable with Moderate Effort (Weeks)

| Technique | Core Mechanism | What's Needed |
|-----------|---------------|---------------|
| **GEPA** | Reflective prompt mutation + genetic selection | Prompt population manager + fitness tracker |
| **DEEVO** | Debate-driven tournament brackets | Subagent tournament orchestrator + Elo system |
| **SEC** | Dynamic difficulty adjustment | Success rate tracker per difficulty tier |
| **SWE-RL** | Bug injection -> repair self-play | Code mutation engine + test harness |
| **SeRL** | Self-instruction + majority-vote rewards | Problem generator + N-way voting |

### Tier 3: Requires Additional Infrastructure

| Technique | What's Needed |
|-----------|---------------|
| **PSV** | Formal verification backend (Lean, Coq, or heavy property testing) |
| **RWML** | World model architecture + sim-to-real gap measurement |
| **GVU (full)** | Requires weight updates for the Updater -- fine-tuning needed |

---

## 6. On the Brink: Techniques About to Break Through

These are the techniques at the tipping point -- published but not yet widely adopted,
with the highest ratio of potential impact to implementation difficulty:

### 6.1 Procedural Memory Consolidation (PRAXIS + Remember Me, Refine Me)

**Status**: Papers published Nov-Dec 2025, no mainstream adoption yet.
**Why it's about to break**: Every agentic framework (LangChain, CrewAI, AutoGen) is
adding "memory" features, but none have implemented true procedural learning from
experience. The first framework to ship this wins the next wave.
**JCoder opportunity**: We already have the memory infrastructure. Adding PRAXIS-style
state matching to our experience store is a weekend project.

### 6.2 Prompt Evolution via Genetic Algorithms (GEPA, DEEVO, EvoPrompt)

**Status**: GEPA accepted as ICLR 2026 oral (highest honor). DEEVO published Aug 2025.
**Why it's about to break**: Prompt engineering is still manual everywhere. Automated
prompt evolution that outperforms human experts (proven) will become table stakes.
**JCoder opportunity**: Implement a prompt tournament system where prompt variants compete
on our 200q eval set. Best prompts survive, worst get mutated.

### 6.3 Self-Play for Code (SWE-RL, Absolute Zero, PSV)

**Status**: Multiple papers converging on the same idea from different angles.
**Why it's about to break**: Code has a unique advantage -- automated verification via
test execution. This makes code the ideal domain for self-play (the GVU Verification
Dominance theorem explains why mathematically).
**JCoder opportunity**: Generate coding challenges, solve them, verify with tests. The
harder the challenges we can generate, the better we get. Self-bootstrapping.

### 6.4 Metacognitive Strategy Selection

**Status**: Position paper at ICLR 2026, foundational but not yet implemented.
**Why it's about to break**: Current agents use fixed strategies regardless of task.
An agent that dynamically selects its approach based on self-assessed competence will
dramatically outperform static agents on diverse workloads.
**JCoder opportunity**: Our meta-cognition module can be extended to maintain competence
scores per task type and select strategies accordingly.

---

## 7. Top 10 Online Sources for Breaking Agentic Research

### Primary Research Sources

| Rank | Source | URL | Why |
|------|--------|-----|-----|
| 1 | **arXiv cs.AI + cs.CL** | arxiv.org/list/cs.AI/recent | All cutting-edge papers appear here first, 1-2 weeks before conferences |
| 2 | **Awesome Self-Evolving Agents (GitHub)** | github.com/EvoAgentX/Awesome-Self-Evolving-Agents | Most comprehensive curated list, updated weekly, full taxonomy |
| 3 | **ICLR/NeurIPS/ICML OpenReview** | openreview.net | Peer-reviewed papers with reviewer comments, workshop proceedings |
| 4 | **HuggingFace Daily Papers** | huggingface.co/papers | Community-curated, fast surfacing of important papers |
| 5 | **Emergent Mind** | emergentmind.com | Topic-clustered AI paper tracking with trend analysis |

### Community and Industry Sources

| Rank | Source | URL | Why |
|------|--------|-----|-----|
| 6 | **Sebastian Raschka's Newsletter** | magazine.sebastianraschka.com | Deep technical analysis of LLM trends, annual State of LLMs |
| 7 | **VoltAgent Awesome AI Agent Papers (GitHub)** | github.com/VoltAgent/awesome-ai-agent-papers | 2026-focused curation covering agent engineering, memory, eval |
| 8 | **Autonomous Agents Papers (GitHub)** | github.com/tmgthb/Autonomous-Agents | Updated daily, comprehensive agent research paper tracker |
| 9 | **OpenAI Cookbook** | developers.openai.com/cookbook/ | Production-ready patterns (Self-Evolving Agents, Coding Agents) |
| 10 | **LessWrong / AI Alignment Forum** | lesswrong.com | Early discussion of RSI risks, safety considerations, novel ideas |

### Honorable Mentions

- **Google DeepMind Blog** (deepmind.google/research) -- AlphaCode, Gemini agent research
- **Apple ML Research** (machinelearning.apple.com/research) -- Self-play negotiations, RL for agents
- **BAIR Blog** (bair.berkeley.edu/blog) -- UC Berkeley AI research, robotics + agents
- **arXiv Daily Substack** (rosinality.substack.com) -- Daily paper digests with commentary

---

## 8. Taxonomy of All Self-Evolving Agent Techniques (2025-2026)

Compiled from the Awesome-Self-Evolving-Agents repository and survey papers:

### 8.1 Single-Agent Optimization

**Training-Based LLM Behavior**:
- Supervised Fine-Tuning: ToRA, STaR, NExT, MuMath-Code
- Reinforcement Learning: Self-Rewarding LMs, Agent Q, Absolute Zero, R-Zero, SPIRAL,
  SeRL, SSRL, Parallel-R1, Vision-Zero

**Test-Time Behavior Optimization**:
- Feedback-Based: CodeT, LEVER, Math-Shepherd, Rewarding Progress
- Search-Based: Self-Consistency, Tree of Thoughts, Buffer of Thoughts, Graph of Thoughts,
  Forest-of-Thought, Deductive Beam Search
- Reasoning-Based: START, CoRT

**Prompt Optimization**:
- Edit-Based: GPS, GrIPS, TEMPERA, Plum
- Evolutionary: EvoPrompt, Promptbreeder, GEPA, DEEVO
- Generative: APE, PromptAgent, DSPy, APOHF, Self-Supervised Prompt Optimization
- Text Gradient-Based: TextGrad, Semantic Backpropagation, GPO, REVOLVE

**Memory Optimization**:
- A-MEM, Mem0, Memento, Memory-R1, M3-Agent, PRAXIS, LEGOMem, ReasoningBank

**Tool Optimization**:
- Training: ToolEVO, BUTTON, Tool-Star, SPORT-Agents, AutoTIR, ReTool, ToolRL
- Inference: EASYTOOL, DRAFT, MCP-Zero
- Creation: CREATOR, CLOVA, Alita

### 8.2 Multi-Agent Optimization

- Architecture Search: MetaAgent (FSM-based), MaAS (supernet search)
- Workflow Generation: AFlow, WorkflowLLM, FlowReasoner, FlowAgent, MermaidFlow
- Swarm Systems: GPTSwarm, AgentNet, MAS-ZERO, ScoreFlow

### 8.3 Domain-Specific (Code)

- SICA, SWE-RL, PSV, Absolute Zero, CodeRL, STOP, DGM

---

## 9. Implementation Roadmap for JCoder

Based on this research, the recommended implementation priority for JCoder:

### Phase 1: Quick Wins (This Sprint)

1. **PRAXIS-style experience store**: Add state-matching retrieval to existing FTS5
   experience database. When the agent encounters a familiar state, retrieve and inject
   relevant past action-result pairs into the prompt.

2. **RISE self-correction loop**: Wrap the existing bridge pattern in a multi-turn
   correction cycle. After initial code generation, have the agent review its own output,
   identify issues, and produce a corrected version. Cap at 3 iterations.

3. **Versioned prompts**: Track prompt variants with timestamps and eval scores. Enable
   rollback to previous best-performing prompts.

### Phase 2: Competitive Advantage (Next Sprint)

4. **Prompt tournament (GEPA/DEEVO)**: Maintain a population of system prompt variants.
   Run them against the 200q eval set. Use Claude to analyze why winners won and losers
   lost. Mutate accordingly. Track Elo ratings.

5. **SWE-RL self-play loop**: Generate coding challenges of increasing difficulty. Agent
   attempts to solve them. Verified by test execution. Failed challenges become the next
   training set.

6. **Dynamic difficulty curriculum (SEC)**: Track success rates by problem difficulty.
   Automatically focus training time on the frontier -- problems that are challenging but
   not impossible.

### Phase 3: Full Self-Evolution (Future)

7. **Metacognitive strategy selection**: Agent maintains competence scores per task type
   and dynamically selects its approach (which tools to use, how much context to retrieve,
   whether to use self-correction loops).

8. **Procedural solidification**: Automatically convert frequently-successful action
   sequences into reusable code templates. The agent literally writes its own tools.

9. **Multi-agent evolutionary archive**: Maintain multiple agent variants with different
   strengths. Route tasks to the best-suited variant. Cross-pollinate successful strategies.

---

## 10. Key Takeaways

1. **Verification > Generation**: The GVU theorem proves that improving verification
   quality (better tests, formal proofs) matters MORE than improving generation quality.
   Invest in test infrastructure first.

2. **Procedural memory is the next frontier**: All major frameworks are racing to add
   experience-based learning. PRAXIS and Remember Me, Refine Me are the leading approaches.
   JCoder's existing experience store gives us a head start.

3. **Prompt evolution works**: GEPA (ICLR 2026 oral) and DEEVO prove that automated
   prompt optimization outperforms human experts. This should be standard practice.

4. **Code is the ideal self-play domain**: Automated test execution provides oracle-quality
   verification, making code the strongest possible environment for self-improvement
   (Verification Dominance theorem).

5. **3 iterations is the sweet spot**: Across all techniques, the iteration cliff appears
   at iteration 3-5. Budget for 3-5 meaningful improvement cycles, not 10+.

6. **The Three Laws matter**: Endure (safety), Excel (no regression), Evolve (improve).
   Every self-improvement loop needs regression detection and rollback capability.

7. **No fine-tuning required for biggest wins**: The most impactful techniques (PRAXIS,
   RISE, GEPA, SWE-RL self-play) work with frozen API models. Fine-tuning is a
   second-order optimization, not a prerequisite.

---

## 11. Sources

### Survey Papers
- A Comprehensive Survey of Self-Evolving AI Agents (arXiv:2508.07407)
- A Survey of Self-Evolving Agents (arXiv:2507.21046)
- Memory in the Age of AI Agents (arXiv:2512.13564)
- Mem^p: Exploring Agent Procedural Memory (arXiv:2508.06433)

### Technique Papers
- Self-Improving AI Agents through Self-Play / GVU Framework (arXiv:2512.02731)
- PRAXIS: Procedural Recall for Agents (arXiv:2511.22074)
- RISE: Recursive Introspection (arXiv:2407.18219, NeurIPS 2024)
- DEEVO: Tournament of Prompts (arXiv:2506.00178)
- GEPA: Reflective Prompt Evolution (arXiv:2507.19457, ICLR 2026 Oral)
- SeRL: Self-Play RL with Limited Data (arXiv:2505.20347)
- Self-Evolving Curriculum for LLM Reasoning (arXiv:2505.14970)
- Remember Me, Refine Me: Dynamic Procedural Memory (arXiv:2512.10696)
- RWML: Reinforcement World Model Learning (arXiv:2602.05842)
- Intrinsic Metacognitive Learning (OpenReview, ICLR 2026)
- Language Self-Play for Data-Free Training (arXiv:2509.07414)

### Position Papers and Workshops
- ICLR 2026 Workshop: AI with Recursive Self-Improvement (recursive-workshop.github.io)
- ICLR 2026 Workshop: Lifelong Agents (lifelongagent.github.io)
- ICLR 2026 Workshop: MemAgents (OpenReview:U51WxL382H)

### Industry Resources
- OpenAI Self-Evolving Agents Cookbook (developers.openai.com/cookbook)
- Awesome Self-Evolving Agents (github.com/EvoAgentX/Awesome-Self-Evolving-Agents)
- VoltAgent AI Agent Papers (github.com/VoltAgent/awesome-ai-agent-papers)
- Autonomous Agents Papers (github.com/tmgthb/Autonomous-Agents)

### News and Analysis
- The AI Research Landscape in 2026 (labs.adaline.ai)
- 6 AI Breakthroughs That Will Define 2026 (infoworld.com)
- 7 Agentic AI Trends to Watch in 2026 (machinelearningmastery.com)
- State of LLMs 2025 (magazine.sebastianraschka.com)
