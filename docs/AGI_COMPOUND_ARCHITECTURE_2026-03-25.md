# JCoder AGI Compound Architecture

## The Three Compounding Loops

### Loop 1: GVU Spiral (Self-Play × Experience Replay × Prompt Evolution)
- Generator = Adversarial Self-Play (creates challenges)
- Verifier = Experience Replay (records what worked, P2Value scoring)
- Updater = Prompt Evolution (modifies system prompt based on outcomes)
- **Compound: each revolution makes the NEXT harder AND more informative**
- Missing: FeedbackRouter that routes Self-Play outcomes to BOTH systems simultaneously

### Loop 2: Structural Accelerator (AST Graph × Strategy Evolver)
- AST Graph knows what connects to what (1533 nodes, 8715 edges)
- Strategy Evolver picks optimal retrieval per query type
- **Compound: structural context makes strategy selection structure-aware**
- Missing: AST blast_radius as feature input to query classification

### Loop 3: Quality-Diversity Flywheel (QD Archive × All 5 Systems)
- MAP-Elites archive maintains N specialized configurations
- Each system contributes to discovering which config works in which niche
- **Compound: multi-dimensional archive stores entire strategy bundles**

## The Two Meta-Accelerators

### Accelerator 1: Recursive Meta-Learning
- Level 1: Evolution optimizes agent configs
- Level 2: Meta-learning optimizes the evolution system
- Level 3: Meta-research optimizes the research pipeline
- **Makes the RATE of improvement increase**

### Accelerator 2: Darwin Godel Machine Pattern
- Evolve not just prompts/strategies, but THE CODE OF THE EVOLVER ITSELF
- Self-referential: mutation operators that improve their own mutation operators

## The Escape Velocity Metric
Track: `revolution_delta / revolution_time`
If positive across 3+ revolutions → escape velocity achieved.

## Key Research Sources
- SPIRAL (arXiv 2506.24119) — self-play on zero-sum games
- Absolute Zero (arXiv 2505.03335) — NeurIPS 2025 spotlight
- Darwin Godel Machine (Sakana AI) — 20% → 50% SWE-bench
- AlphaEvolve (DeepMind) — 23% speedup on Gemini training
- Meta Self-Play SWE-RL (arXiv 2512.18552) — +10.4 points, zero human data
- ICML 2025: "Truly Self-Improving Agents Require Intrinsic Metacognitive Learning"
- ICLR 2026 Workshop on Recursive Self-Improvement

Signed: Claude Opus 4.6 | 2026-03-25
