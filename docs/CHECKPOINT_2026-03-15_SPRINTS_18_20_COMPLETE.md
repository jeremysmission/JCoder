# Checkpoint: Sprints 18-20 Complete (Research Upgrades)
**Date:** 2026-03-15
**Test count:** 1523 passed, 2 skipped, 0 failures
**Previous:** 1431 (Sprint 26 complete)

## Sprints Completed This Session

| Sprint | Module | Tests | Pattern |
|--------|--------|-------|---------|
| 18 | core/prompt_evolver.py | 34 | Autoresearch keep/discard (Karpathy) |
| 19 | core/experience_replay.py | 28 | P2Value near-miss priority |
| 20 | core/meta_cognitive.py | 30 | BARP cost-aware routing |

## Test Count Progression
1431 (S26) -> 1465 (S18) -> 1493 (S19) -> 1523 (S20)

## Sprint 18: Autoresearch PromptEvolver
- Replaced hard tournament selection (top 50% survive) with Karpathy-style keep/discard loop
- Added OperatorArm with Thompson sampling per mutation operator (rephrase/extend/compress/crossover)
- Operators learn: effective mutation types auto-selected more often
- Monotonic champion improvement (champion only changes on genuine gain)
- SQLite persistence for operator effectiveness stats
- Gate test PASSED: operator learning drives measurable selection preference shift

## Sprint 19: P2Value Experience Replay
- P2Value scoring: alpha * confidence + (1-alpha) * pass_rate
- Near-miss detection: experiences that failed exactly 1 test get priority boost (1.3x)
- Test results tracking (pass_count, fail_count per experience)
- RLEP-style replay blending: mix new rollouts with stored successes
- Eviction now by p2value (lowest evicted first), not just confidence
- Gate test PASSED: near-miss beats multi-failure at same confidence

## Sprint 20: Cost-Aware Routing
- BARP pattern: cost_weight parameter (0.0-1.0) for quality-cost tradeoff
- Static + observed cost tracking per strategy
- Multi-objective Thompson sampling: sample - cost_weight * normalized_cost
- DEFAULT_STRATEGY_COSTS: standard=1.0, corrective=1.5, best_of_n=3.0, reflective=2.0
- cost_report() method for cost visibility
- Gate test PASSED: cheap strategy preferred for easy queries with cost_weight > 0

## All Sprints Now Defined
- Sprints 1-20: DONE (no more UNDEFINED gaps)
- Sprints 21-26: DONE (self-evolution stack)
- Repair Sprints R1-R7: DONE
- Total: 1523 tests, 0 regressions

## What's Left
- BEAST online today/Monday -- real model validation
- All code is uncommitted in JCoder working tree
- HybridRAG3 improvements: reranker revival, corrective retrieval, FTS5 source_path
