# The Orchestrator — Top 1% Design

## What Makes This Different

Most agent evolution systems are "run N copies, pick best." That's 50th percentile.

Top 1% means the orchestrator is itself intelligent:
- It knows WHEN to kill underperformers (not just at the end)
- It knows HOW to cross-pollinate ideas between copies
- It tracks WHY winners win (not just that they scored higher)
- It improves ITS OWN orchestration strategy over time

## Architecture

```
                    ┌─────────────────────────┐
                    │   META-ORCHESTRATOR      │
                    │   (optimizes itself)      │
                    │                           │
                    │  - Population size        │
                    │  - Mutation rate          │
                    │  - Selection pressure     │
                    │  - Cross-pollination      │
                    │  - Early termination      │
                    └─────────┬───────────────┘
                              │
              ┌───────────────┼───────────────┐
              │               │               │
        ┌─────▼─────┐  ┌─────▼─────┐  ┌─────▼─────┐
        │  COPY 1   │  │  COPY 2   │  │  COPY N   │
        │ (worktree)│  │ (worktree)│  │ (worktree)│
        │           │  │           │  │           │
        │ Mutate    │  │ Mutate    │  │ Mutate    │
        │ Challenge │  │ Challenge │  │ Challenge │
        │ Evaluate  │  │ Evaluate  │  │ Evaluate  │
        │ Learn     │  │ Learn     │  │ Learn     │
        └─────┬─────┘  └─────┬─────┘  └─────┬─────┘
              │               │               │
              └───────────────┼───────────────┘
                              │
                    ┌─────────▼───────────────┐
                    │   TOURNAMENT             │
                    │                           │
                    │  - Procedural challenges  │
                    │  - Cross-domain transfer  │
                    │  - Contamination firewall │
                    │  - Execution-validated    │
                    └─────────┬───────────────┘
                              │
                    ┌─────────▼───────────────┐
                    │   CHAMPION SELECTION     │
                    │                           │
                    │  - Best → candidate       │
                    │  - Baseline preserved     │
                    │  - 1 week validation      │
                    │  - Promote or rollback    │
                    └─────────────────────────┘
```

## The Weekly Cycle

### Monday: Scrape + Ingest
- Weekly scraper runs: arXiv, GitHub trending, HuggingFace, X.com
- New papers/code/discussions ingested into FTS5 indexes
- FAISS rebuilt with new content

### Tuesday-Thursday: Evolution Tournament
- 10 copies spawned in isolated git worktrees
- Each copy gets the latest ingested knowledge
- 10 rounds of procedural challenges per copy
- After each round: mutate config, prompts, retrieval strategies
- Successive Halving: kill bottom 50% after round 5
- Cross-pollinate: best mutation from top copy injected into remaining copies

### Friday: Champion Selection
- Surviving copies evaluated on held-out test set
- Best copy becomes candidate champion
- Candidate runs alongside baseline for comparison

### Weekend: Validation
- Candidate processes real user queries alongside baseline
- Compare: answer quality, speed, safety, hallucination rate
- If candidate > baseline on all metrics → promote

### Next Monday: Promote or Rollback
- If promoted: candidate becomes new baseline
- If rolled back: baseline continues, evolution adjusts parameters
- Meta-orchestrator records what worked/didn't for next week

## Isolation Strategy

Each copy runs in a **git worktree** (not full VM — lighter, faster):
```bash
git worktree add ../jcoder_copy_1 -b evolution/copy-1
git worktree add ../jcoder_copy_2 -b evolution/copy-2
...
```

Each worktree gets its own:
- Config (agent.yaml, models.yaml mutations)
- Prompt variants (system prompts)
- Strategy population (retrieval configs)
- Experience replay database
- But SHARES the FTS5/FAISS indexes (read-only)

## What Each Copy Can Mutate

Per round, each copy randomly mutates 1-3 of:
1. System prompt wording (prompt evolution)
2. Retrieval strategy parameters (top_k, fusion method, rrf_k)
3. Scoring weights (keyword vs LLM judge ratio)
4. FTS5 query construction (stopwords, boosting terms)
5. Reranker aggressiveness (pool multiplier)
6. Context budget allocation (how many chars per chunk)
7. Temperature / sampling parameters

What copies CANNOT mutate (safety rails):
- Core retrieval algorithms (FAISS, FTS5)
- Network security (NetworkGate)
- File system access (allowed_dirs)
- Model selection (stays on approved stack)

## Successive Halving (Hyperband)

Instead of running all 10 copies for all 10 rounds:
- Round 1-3: All 10 copies run (explore widely)
- Round 4: Kill bottom 5 (keep top 5)
- Round 5-7: Top 5 run (focus resources)
- Round 8: Kill bottom 2 (keep top 3)
- Round 9-10: Top 3 run (final tournament)

This saves 40% of compute vs running all 10 for all 10 rounds.

## Cross-Pollination Protocol

After round 5 (when we've seen enough data):
- Identify the SINGLE BEST MUTATION from the top copy
- Inject it into all surviving copies
- This prevents local optima while preserving diversity
- Only inject ONE mutation (not the entire config) to maintain evolutionary pressure

## The Meta-Orchestrator

The orchestrator tracks across weeks:
- Which mutation types produced the most improvement
- Which population sizes found better champions
- Which selection pressures (aggressive vs conservative) worked
- Whether cross-pollination helped or hurt

Every 4 weeks, the meta-orchestrator adjusts:
- Population size (maybe 8 is better than 10, or 15)
- Mutation rate (maybe more aggressive mutations work better)
- Selection timing (maybe kill after round 3, not round 5)
- Cross-pollination frequency

This makes the orchestrator ITSELF evolve — it gets better
at finding better agents. This is the recursive improvement
that leads to escape velocity.

## Success Metric

The orchestrator tracks one number across weeks:
**Champion improvement per week**

If this number is INCREASING, the system is compounding.
If it plateaus, the meta-orchestrator adjusts parameters.
If it regresses, rollback to previous orchestration strategy.

## Implementation Priority

1. Git worktree isolation (already supported by Claude Code)
2. Procedural challenge generator (built: procedural_challenges.py)
3. Config mutation operators (built: strategy_evolver.py has the pattern)
4. Tournament runner with Successive Halving
5. Champion validation pipeline
6. Meta-orchestrator feedback loop

Signed: Claude Opus 4.6 | 2026-03-25
