# Checkpoint: DPI + Research Sprint (2026-03-17)

## Session Summary
Deep Packet Inspection of entire JCoder codebase followed by targeted fixes,
paper-backed upgrades, and comprehensive test coverage expansion.

## What Was Done

### Phase 1: Deep Packet Inspection
- Full security audit (SQL injection, path traversal, command injection)
- Reliability analysis (missing imports, silent failures, thread safety)
- Scalability review (unbounded storage, full-table scans)
- Architecture assessment (dead code, hardcoded constants, design gaps)
- Test suite quality review (mocking patterns, coverage gaps, anti-patterns)

### Phase 2: Critical Bug Fixes (Repair Sprint R8)
1. **CascadeLevel wiring crash** in bridge.py — ModelCascade silently failing
2. **Ledger false append-only** — INSERT OR REPLACE -> INSERT OR IGNORE
3. **KG LIKE wildcard injection** — user queries with % or _ matched wrong entities
4. **GPU safety silent failure** — added logging to _gpu_min_free_mb()
5. **print() -> logging** — 8 instances across index_engine.py and config.py

### Phase 3: Storage Rotation (Repair Sprint R9)
1. `prune_old()` added to TelemetryStore and ProceduralMemory
2. `storage_limits` config section in default.yaml
3. 5 new test files covering previously untested modules
4. Stray `=2.2.0` pip artifact removed

### Phase 4: Paper-Backed Upgrades (Repair Sprint R10)
1. **HybridServe skip connections** (arXiv:2505.12566) — cascade skips to strongest model on very low confidence
2. **MOPrompt Pareto front** (arXiv:2508.01541) — multi-objective prompt quality vs token cost
3. **Prompt Duel Optimizer** (arXiv:2510.13907) — label-free pairwise preference via LLM judge
4. **Runtime API key auth** — enables GPT-4o through JCoder pipeline
5. **Full-pipeline eval script** — 3 backend modes (openai/ollama/agent)

### Phase 5: Research Survey
- 35+ papers surveyed from ICLR, NeurIPS, EMNLP, NAACL, ICML (2025-2026)
- Top 10 implementation priorities identified
- Saved to docs/RESEARCH_BRIEF_2026-03-17_PAPER_SURVEY.md

## Test Results
- 105 tests passing in targeted run (all touched files)
- 30 prompt evolver tests passing
- 8 new test files, ~100 new test cases

## Files Modified (13 core files)
core/cascade.py, core/prompt_evolver.py, core/runtime.py, core/ledger.py,
core/telemetry.py, core/procedural_memory.py, core/knowledge_graph.py,
core/index_engine.py, core/config.py, agent/bridge.py,
scripts/run_eval_api.py, scripts/run_eval_local.py, config/default.yaml

## Files Created (12 new files)
scripts/run_eval_pipeline.py, tests/test_storage_rotation.py,
tests/test_kg_like_escape.py, tests/test_network_gate.py,
tests/test_reranker.py, tests/test_procedural_memory.py,
tests/test_bridge_cascade_wiring.py, tests/test_ledger.py,
tests/test_integration_pipeline.py,
docs/SPRINT_STATUS_2026-03-17.md,
docs/RESEARCH_BRIEF_2026-03-17_PAPER_SURVEY.md,
docs/CHECKPOINT_2026-03-17_DPI_AND_RESEARCH_SPRINT.md

## Next Steps
1. Wire AST chunking into index build pipeline (Sprint 18)
2. Implement GEPA reflective prompt evolution (ICLR 2026 Oral)
3. Run real eval: phi4:14b and gpt-4o through pipeline
4. Commit all changes
