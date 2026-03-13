# JCoder Sprint Completion Plan -- 2026-03-13

Last updated: 2026-03-13 America/Denver

This document is the canonical execution spine from the current JCoder state
to project completion. It resolves the undefined Sprint `17-20` gap and breaks
the remaining work into concrete slices with explicit exit gates.

## Current State

| Track | Status |
|-------|--------|
| Product sprints `1-14` | DONE |
| GUI parity sprint `13.1` | DONE |
| Repair sprints `R1-R2` | DONE |
| Repair sprints `R3-R7` | OPEN |
| Data sprint `D1` | ACTIVE |
| Data sprint `D2` | STANDING |
| Data sprint `D3` | PARTIAL |

## Completed Product Sprints

| Sprint | Focus | Status |
|--------|-------|--------|
| 1 | Foundation and planning | DONE |
| 2 | Data acquisition | DONE |
| 3 | Ingestion pipeline | DONE |
| 4 | RAG infrastructure | DONE |
| 4.5 | Agent framework | DONE |
| 5 | Integration | DONE |
| 6 | Production basics | DONE |
| 7 | API eval baseline and ingestion closeout | DONE WITH DEFERRED FOLLOW-UPS |
| 8 | Self-learning pipeline activation | DONE |
| 9 | Reasoning and reflection integration | DONE |
| 10 | Corrective retrieval and smart orchestrator | DONE |
| 11 | Prompt evolution and adversarial hardening | DONE |
| 12 | Cascade router and model orchestration | DONE |
| 13 | End-to-end self-learning validation | DONE |
| 13.1 | GUI command center and CLI parity shell | DONE |
| 14 | Weekly knowledge scraper | DONE |

## Near-Term Completion Spine

| Sprint | Focus | Status | Exit Gate |
|--------|-------|--------|-----------|
| 17 | Live backend readiness and full-stack RAG validation | ACTIVE | Configured vLLM-style endpoints on `8000/8001/8002` are up, `doctor` is green for the live stack, GUI `ask` passes, CLI `ask` passes, and live `agent run` stays green |
| 18 | Retrieval quality, distillation, and knowledge freshness | PLANNED | High-signal `D1` and `D3` corpora are ingested, retrieval eval is re-baselined, weights are tuned, and a measured distillation pilot lands |
| 19 | Repair closure and portability hardening | PLANNED | `R3-R7` are complete, portability is validated on non-dev paths, and the focused/full regression bundles are green |
| 20 | Production packaging and daily-driver cutover | PLANNED | Install/package flow works, `doctor` is operator-ready, runbooks are complete, and the 8-hour soak plus resume/cutover drills pass |

## Long-Range Product Spine

| Sprint | Focus | Status | Exit Gate |
|--------|-------|--------|-----------|
| 15 | Multi-agent mesh coordination | PLANNED | Three coordinated subagents can research, implement, and verify a task with clean handoff artifacts |
| 16 | LimitlessApp V2 memory integration | PLANNED | JCoder recalls and uses context from prior sessions through the external memory layer |
| 21 | Weekly software evolution engine | PLANNED | First audited evolution cycle completes with accept/reject evidence and rollback safety |
| 22 | VM tournament mode | PLANNED | Ten isolated clones run the same eval suite and a champion is selected deterministically |
| 23 | Concentration rotation and meta-QA | PLANNED | Rotated agents demonstrably catch misses that a static single-agent flow misses |
| 24 | Recursive meta-learning | PLANNED | Evolution acceptance rate improves over a sustained four-week window |
| 25 | Supercomputer burst pipeline | PLANNED | A weekend cloud burst produces more net evolution progress than four normal weeks |
| 26 | Autonomous research lab | PLANNED | JCoder proposes, implements, and validates a novel benchmark-improving technique |

## Sprint Slices

### Sprint 17: Live Backend Readiness and Full-Stack RAG Validation

1. `17.1` Bring up the configured vLLM-style services on `8000/8001/8002` and get `measure` plus `doctor` to reflect the live stack accurately.
2. `17.2` Run GUI-driven live `ask` against the configured stack and capture a clean startup, streamed output, and successful answer path.
3. `17.3` Run matching CLI live `ask` and long-running `agent run` smokes against the same stack so GUI and CLI share one validation story.
4. `17.4` Freeze the live-stack benchmark and latency snapshot that becomes the post-GUI parity baseline.

### Sprint 18: Retrieval Quality, Distillation, and Knowledge Freshness

1. `18.1` Clear the highest-value `D1` backlog from `config/download_queue.json`.
2. `18.2` Finish the highest-value `D3` documentation scraping gaps for framework/operator docs.
3. `18.3` Re-run retrieval evaluation, tune federated weights/RRF, and document measurable lift.
4. `18.4` Run a targeted online distillation pilot on the highest-gap categories and record improvement vs spend.

### Sprint 19: Repair Closure and Portability Hardening

1. `19.1` Finish `R3` silent-failure elimination with log assertions.
2. `19.2` Finish `R4` path/config portability and validate non-dev-box roots.
3. `19.3` Finish `R5` SQLite connection consolidation beyond the already-fixed federated/search path.
4. `19.4` Finish `R6` pipeline hardening for download, ingest, and session safety limits.
5. `19.5` Finish `R7` test infrastructure, cleanup discipline, and suite ergonomics.

### Sprint 20: Production Packaging and Daily-Driver Cutover

1. `20.1` Finalize install/package surfaces, entry points, and first-run setup.
2. `20.2` Harden `doctor`, runbooks, and operator guidance for real workstation use.
3. `20.3` Run the long-duration soak, crash-resume drill, and cutover checklist.
4. `20.4` Publish the release-ready documentation set and declare daily-driver readiness.

### Sprint 15: Multi-Agent Mesh Coordination

1. `15.1` Define subagent roles and artifact contracts.
2. `15.2` Implement shared knowledge-bus coordination.
3. `15.3` Validate three-agent coordinated execution on a real task.

### Sprint 16: LimitlessApp V2 Memory Integration

1. `16.1` Define the persistent memory contract and sync points.
2. `16.2` Persist cross-session artifacts with search/retrieval hooks.
3. `16.3` Validate recall quality across multiple prior sessions.

### Sprint 21-26: Self-Evolution to Research Lab

1. `21` Build the audited evolution runner and immutable rollback path.
2. `22` Scale the evolution runner to tournament mode across isolated clones.
3. `23` Add concentration rotation, deep inspection sweeps, and QA-of-QA.
4. `24` Let the evolution system optimize its own selection and sourcing strategy.
5. `25` Add weekend cloud burst capability for massive parallel experimentation.
6. `26` Transition to autonomous gap discovery, prototype generation, and benchmark-proven novel research output.

## Required Parallel Tracks

| Track | Requirement before final completion |
|-------|------------------------------------|
| Repair `R3-R7` | Must be fully closed before Sprint `20` can claim daily-driver readiness |
| Data `D1` | High-signal backlog must be cleared before Sprint `18` can claim retrieval-quality closure |
| Data `D2` | Must be running as a standing freshness loop before Sprint `21` |
| Data `D3` | Framework/operator docs must be searchable before Sprint `20` operator cutover |

## Execution Order

1. Sprint `17` with `R3-R4` in parallel.
2. Sprint `18` with `D1-D3` and `R5-R6` in parallel.
3. Sprint `19`, finishing the remaining repair and test infrastructure debt.
4. Sprint `20`, the operator/daily-driver cutover.
5. Sprint `15-16`, which can overlap the back half of Sprint `20` once the live stack is stable.
6. Sprint `21-23`, the first self-evolution operating loop.
7. Sprint `24-26`, the meta-learning and autonomous research endgame.

## Definition of Project Completion

- Product sprints `1-26` are complete.
- Repair sprints `R1-R7` are complete.
- Data sprint `D1` backlog is cleared and `D2-D3` are operating loops instead of one-off chores.
- GUI and CLI both pass live-stack validation on the intended backend.
- The full regression suite is green on the supported environments.
- Operator docs, runbooks, and launch/cutover evidence are published.
