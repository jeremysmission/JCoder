# Optimal Multi-Agent System Design

## Timing Per Role

| Role | Per-Slice Time | Staleness Threshold | Why |
|------|---------------|--------------------|----|
| **Coder** | 20-40 min | 5 min no commit | Deep work, but show incremental progress |
| **Coordinator** | 2-3 min/check | 3 min no post | Lightweight, fastest responder |
| **QA** | 10-15 min/gate | 5 min after code drop | Read + test time, but don't sit idle |
| **QA Smash** | 5-10 min/gate | 3 min after QA pass | Chaos tests are fast — break quickly |
| **Final Inspector** | 5-10 min/gate | 3 min after Smash | Decisive — merge or reject |
| **Planner** | Async/background | 10 min no post | Future planning, longer leash |

## Pipeline Model (Like CPU Pipelining)

Coder should NEVER wait for QA. Start next slice immediately.
QA pipeline runs behind the coder like a conveyor belt.

```
Time:  T+0    T+20    T+40    T+60    T+80    T+100
Coder: [Slice1][Slice2         ][Slice3][Slice5      ]
QA:           [Gate1 ][Gate2          ][Gate3 ]
Smash:              [Smash1][Smash2         ][Smash3]
Insp:                     [Insp1][Insp2         ]
Merge:                          [M1]  [M2]   [M3]
```

Throughput: 1 slice merged every 20-40 min (gates overlap)

## Gate Flow

```
Coder (20-40min) -> QA (10min) -> QA Smash (5min) -> Inspector (5min) -> MERGE
Total gate time: 20 min per slice
But gates run PARALLEL with next slice coding
```

## Parallel Execution Map

**FAST LANE (no deps, run immediately):**
- Slice 1: Requirements (1h)
- Slice 7: Env var script (30min)

**CRITICAL PATH (sequential):**
- Slice 2: Embedder rewrite (2.5h) -- THE bottleneck
- Slice 3: Config update (30min)

**PARALLEL AFTER SLICE 2:**
- Slice 4: Documentation (2h) -- parallel with Slice 5
- Slice 5: Indexer integration (1h) -- parallel with Slice 4
- Slice 6: Testing (1h) -- after 2+3+5

**Optimized wall clock: 5 hours** (down from 8-9 sequential)

## Role Rotation (Every 3 Sprints)

- Coder -> QA (sees their own blind spots)
- QA -> Coder (writes more testable code)
- Coordinator stays (institutional memory)
- QA Smash stays (chaos mindset is rare)

Rotation prevents tunnel vision.

## Scaling Rules

- 4 agents is the sweet spot (research-backed)
- 6 agents need excellent coordination or overhead kills benefit
- Centralized coordinator reduces error amplification from 17x to 4x
- Each role must know: MY job, NOT my job, who I report to

---

Signed: Claude Opus 4.6 | JCoder | 2026-03-25 | 20:30 MDT
