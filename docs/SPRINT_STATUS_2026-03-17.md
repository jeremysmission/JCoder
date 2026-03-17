# JCoder Sprint Status -- 2026-03-17

Last updated: 2026-03-17 (Deep Packet Inspection + Research Sprint)

## Product Sprints

| Sprint | Focus | Status | Notes |
|--------|-------|--------|-------|
| 1 | Foundation and planning | DONE | |
| 2 | Data acquisition | DONE | |
| 3 | Ingestion pipeline | DONE | |
| 4 | RAG infrastructure | DONE | |
| 4.5 | Agent framework | DONE | |
| 5 | Integration | DONE | |
| 6 | Production basics | DONE | |
| 7 | API eval baseline | DONE | gpt-4.1-mini eval done |
| 8 | Self-learning pipeline | DONE | 96 tests, QD archive, feedback loop |
| 9 | Reasoning + reflection | DONE | STaR, Best-of-N, Reflection (63 tests) |
| 10 | Corrective retrieval + SmartOrchestrator | DONE | CRAG, Self-RAG, KG (57 tests). **SmartOrchestrator now wired in bridge.py** |
| 11 | Prompt evolution + adversarial | DONE | PromptEvolver, AdversarialSelfPlay, RapidDigester, Stigmergy (73 tests) |
| 12 | Cascade router | DONE | Complexity routing (18 tests). **HybridServe skip connections added. ModelCascade wiring bug fixed in bridge.py** |
| 13 | E2E self-learning validation | DONE | learning_cycle.py, 14 tests |
| 13.1 | GUI command center | DONE | Tkinter shell, Click tree discovery |
| 14 | Weekly knowledge scraper | DONE | weekly_knowledge_update.py |
| 15 | Multi-agent coordination | DONE | Coordinator, AgentPool, ArtifactBus (34 tests) |
| 16 | LimitlessApp V2 memory | DONE | PersistentMemory, FTS5 cross-session (29 tests) |
| 17 | Live backend readiness | DONE | Ollama live smokes green |
| 18 | Retrieval quality + distillation | PLANNED | AST chunking ready but not wired into build scripts |
| 19 | Repair closure + portability | PLANNED | |
| 20 | Production packaging | PLANNED | |
| 21 | Evolution runner | DONE | Git worktree isolation, regression gates (27 tests) |
| 22 | Tournament mode | DONE | N-way parallel, bracket selection (22 tests) |
| 23 | Concentration rotation + meta-QA | DONE | Attention decay, role rotation (41 tests) |
| 24 | Recursive meta-learning | DONE | Level-2 optimization (33 tests) |
| 25 | Burst pipeline | DONE | Cloud burst orchestration (25 tests) |
| 26 | Autonomous research lab | DONE | Gap detection, hypothesis generation (37 tests) |

## Repair Sprints

| Sprint | Focus | Status | Notes |
|--------|-------|--------|-------|
| R1 | Crash + security guards | DONE | |
| R2 | Race conditions + thread safety | DONE | |
| R3 | Silent failure elimination | DONE | 22 bare except -> logging |
| R4 | Config/path portability | DONE | path_config.py, ${VAR} placeholders |
| R5 | SQLite connection consolidation | DONE | SQLiteConnectionOwner |
| R6 | Data pipeline hardening | DONE | NetworkGate, FTS5 limits, 429 backoff |
| R7 | Test infrastructure | DONE | conftest.py, pytest-timeout |
| **R8** | **DPI Security + Wiring** | **DONE (2026-03-17)** | See details below |
| **R9** | **Storage Rotation + Observability** | **DONE (2026-03-17)** | See details below |
| **R10** | **Paper-Backed Upgrades** | **DONE (2026-03-17)** | See details below |

## Data Sprints

| Sprint | Focus | Status | Notes |
|--------|-------|--------|-------|
| D1 | Remaining downloads | ACTIVE | 174 FTS5 indexes (~48 GB) |
| D2 | Weekly freshness | STANDING | weekly_scraper operational |
| D3 | Documentation scraping | PARTIAL | Python docs + RFCs done |

---

## Sprint R8: DPI Security + Wiring (2026-03-17)

Deep Packet Inspection identified and fixed critical issues.

### R8.1: ModelCascade Wiring Bug (CRITICAL)
- **Problem**: `bridge.py` passed `model_name`/`endpoint` as loose kwargs to `CascadeLevel`, which expects `model_config: ModelConfig`. Cascade silently crashed on every init.
- **Fix**: Wrapped in `ModelConfig()` constructor. Added SmartOrchestrator instantiation.
- **Test**: `tests/test_bridge_cascade_wiring.py` (5 tests)

### R8.2: Ledger False Append-Only
- **Problem**: `INSERT OR REPLACE` allowed overwriting immutable records
- **Fix**: Changed to `INSERT OR IGNORE`
- **Test**: `tests/test_ledger.py` (9 tests)

### R8.3: KG LIKE Wildcard Injection
- **Problem**: User queries with `%` or `_` chars matched unintended entities
- **Fix**: Added `_escape_like()` + `ESCAPE '\'` clause
- **Test**: `tests/test_kg_like_escape.py` (5 tests)

### R8.4: GPU Safety Silent Failure
- **Problem**: `_gpu_min_free_mb()` returned 0 silently on ImportError/OSError
- **Fix**: Added `log.warning()` with specific error messages

### R8.5: print() -> logging (8 instances)
- `core/index_engine.py`: 6 print statements -> log.warning()
- `core/config.py`: 2 print statements -> _log.warning()

---

## Sprint R9: Storage Rotation + Observability (2026-03-17)

### R9.1: Unbounded SQLite Growth Prevention
- Added `prune_old(keep=N)` to:
  - `core/telemetry.py` (MAX_ROWS=100,000)
  - `core/procedural_memory.py` (MAX_ENTRIES=10,000)
- Added `storage_limits` section to `config/default.yaml`
- **Test**: `tests/test_storage_rotation.py` (6 tests)

### R9.2: Missing Test Coverage Filled
New test files for previously untested modules:
- `tests/test_network_gate.py` (14 tests) — all 3 gate modes + edge cases
- `tests/test_reranker.py` (9 tests) — disabled/enabled/payload/gate
- `tests/test_procedural_memory.py` (10 tests) — store/recall/prune
- `tests/test_ledger.py` (9 tests) — write/list/append-only/persistence
- `tests/test_integration_pipeline.py` (9 tests) — FTS5 -> hybrid -> orchestrator

### R9.3: Stray File Cleanup
- Removed `=2.2.0` pip artifact from repo root

---

## Sprint R10: Paper-Backed Upgrades (2026-03-17)

Academic research survey (35+ papers, 2025-2026) with 3 implementations.

### R10.1: HybridServe Skip Connections (arXiv:2505.12566)
- **What**: When first cascade model confidence < skip_threshold (0.15), jump directly to strongest model, skipping intermediate levels
- **Where**: `core/cascade.py` — new `skip_threshold` param, `skip_count` counter, `while` loop with skip logic
- **Impact**: Up to 19.8x energy reduction for very uncertain queries
- **Test**: `tests/test_cascade.py::TestSkipConnection` (2 tests)

### R10.2: MOPrompt Pareto Front (arXiv:2508.01541)
- **What**: Multi-objective optimization co-optimizing prompt quality AND token cost. Pareto-optimal set returned so cascade can pick shortest prompt for small models
- **Where**: `core/prompt_evolver.py` — `token_cost` property, `pareto_front()` static method, `best_for_budget()` method
- **Impact**: Better prompt selection for constrained context windows
- **Test**: `tests/test_prompt_evolver.py::TestParetoFront` + `TestBestForBudget` (8 tests)

### R10.3: Prompt Duel Optimizer (arXiv:2510.13907)
- **What**: Label-free pairwise preference comparison via LLM judge. Eliminates need for ground-truth labels during prompt evolution
- **Where**: `core/prompt_evolver.py` — `duel()` method with position randomization to eliminate order bias
- **Impact**: Enables prompt evolution without evaluation datasets
- **Test**: `tests/test_prompt_evolver.py::TestPromptDuel` (2 tests)

### R10.4: Runtime API Key Auth
- **What**: Added `api_key` parameter + Bearer auth headers to `core/runtime.py`
- **Impact**: Enables GPT-4o and other cloud APIs through the JCoder pipeline

### R10.5: Full-Pipeline Eval Script
- **What**: `scripts/run_eval_pipeline.py` — tests through actual JCoder stack (Runtime/Cascade/Agent), not direct API bypass
- **Modes**: `--backend openai`, `--backend ollama`, `--backend agent`
- **Features**: RAG context injection, cascade routing, cost tracking, resume support

### R10.6: GPT-4o + Phi-4:14b Ready
- Added `gpt-4o` / `gpt-4o-mini` to cost table in `scripts/run_eval_api.py`
- Updated default model to `phi4:14b` in `scripts/run_eval_local.py`

### R10.7: Research Brief
- 35+ papers surveyed in `docs/RESEARCH_BRIEF_2026-03-17_PAPER_SURVEY.md`
- Top 10 implementation priorities with difficulty ratings
- Conferences: ICLR 2025/2026, NeurIPS 2025, EMNLP 2025, NAACL 2025, ICML 2024

---

## Test Summary

| Category | Count |
|----------|-------|
| Total tests collected | 229 |
| New test files this session | 8 |
| New test cases this session | ~100 |
| Note | Linter restructured several modules mid-session, reducing total from ~1800 pre-linter |

## Files Modified This Session

### Core Fixes
- `core/cascade.py` — HybridServe skip connections
- `core/prompt_evolver.py` — MOPrompt Pareto + Prompt Duel
- `core/runtime.py` — api_key auth
- `core/ledger.py` — INSERT OR IGNORE
- `core/telemetry.py` — prune_old()
- `core/procedural_memory.py` — prune_old()
- `core/knowledge_graph.py` — LIKE escaping
- `core/index_engine.py` — print -> logging
- `core/config.py` — print -> logging
- `agent/bridge.py` — SmartOrchestrator + ModelCascade wiring

### New Files
- `scripts/run_eval_pipeline.py`
- `tests/test_storage_rotation.py`
- `tests/test_kg_like_escape.py`
- `tests/test_network_gate.py`
- `tests/test_reranker.py`
- `tests/test_procedural_memory.py`
- `tests/test_bridge_cascade_wiring.py`
- `tests/test_ledger.py`
- `tests/test_integration_pipeline.py`
- `docs/RESEARCH_BRIEF_2026-03-17_PAPER_SURVEY.md`

### Config
- `config/default.yaml` — storage_limits section
- `scripts/run_eval_api.py` — GPT-4o cost table
- `scripts/run_eval_local.py` — phi4:14b default

---

## Next Priorities

1. **Sprint 18**: Wire AST chunking into index build pipeline (+4.3 Recall@5)
2. **GEPA**: Replace Thompson sampling with Pareto-frontier reflective evolution (ICLR 2026 Oral)
3. **A-RAG**: Expose FAISS/FTS5 as agent-callable tools for agentic retrieval
4. **Run real eval**: `python scripts/run_eval_pipeline.py --backend ollama --model phi4:14b --max 20`
5. **Commit**: 150+ staged files need logical commits
