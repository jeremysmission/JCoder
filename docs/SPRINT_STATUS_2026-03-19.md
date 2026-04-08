# JCoder Sprint Status -- 2026-03-19

Status note (2026-04-08):
- This file remains the historical status record for the 2026-03-19 session.
- The current forward sprint queue now lives in `docs/SPRINT_STATUS_2026-04-08.md`.
- The current demo-first plan now lives in `docs/SPRINT_PLAN_R28_R32_2026-04-08.md`.

Last updated: 2026-03-19 (QA + Sprint R12-R17 execution session)

## Session Summary

Started with 19 test failures (2627 passing), ended with 0 failures (2731 passing).
Completed 6 repair sprints: R12 (retrieval quality), R13-R17 (new sprints created and executed).

## 2026-03-21 Jeremy Directive: Beast 70B Track

- JCoder should use Beast as the biggest-model lab for 70B-class import and test work.
- Keep work-project GPU standards and the HybridRAG3 Python stack out of this lane.
- Why this belongs on Beast: the workstation has two hardware-validated 24 GB RTX 3090s and was bought specifically for larger local AI experiments.
- Current constraint: NVLink is inactive, so the first 70B validation must assume software sharding across two separate GPUs rather than one fused 48 GB device.
- Coordination follow-up: Sprint R19 and the QA queue now own the first Beast-only 70B import, dual-GPU evidence capture, and rollback note.

## Test Suite Growth

| Metric | Before | After |
|--------|--------|-------|
| Passing | 2627 | 2731 |
| Failing | 19 | 0 |
| Skipped | 6 | 6 |
| New test files | -- | 7 |
| Net new tests | -- | +104 |

## Sprints Completed This Session

### Pre-existing Bug Fixes (19 failures -> 0)
- `core/experience_replay.py`: Implemented P2Value priority scoring system (16 test fixes)
- `core/telemetry.py`: Added thread-safe close() with per-thread connections (2 test fixes)
- `agent/session.py`: Fixed cleanup race condition with 10ms grace period (1 test fix)

### R12: Retrieval Quality Revolution [COMPLETE]
- R12.1: AST chunking (already done)
- R12.2: Adaptive-k retrieval (already done)
- R12.3: Cross-index MinHash dedup wired into build_fts5_indexes.py (14 tests)
- R12.4: Embedder config upgraded to nomic-embed-code
- R12.5: Quality scoring (quality_score column in FTS5, _estimate_quality heuristic)
- R12.6: Confidence gating (already done)
- Quality filtering wired into index_engine.py search_fts5_direct()

### R13: Open Knowledge Fallback [COMPLETE]
- SmartOrchestrator: open_knowledge_fallback mode
- Falls back to LLM parametric knowledge when retrieval returns 0 results or all scores < 0.15
- Config toggle: agent.yaml -> retrieval.open_knowledge_fallback
- 14 tests (test_open_knowledge_fallback.py)

### R14: Retrieval Eval Baseline [COMPLETE]
- scripts/run_retrieval_eval.py: measures Recall@5, MRR@10, symbol hit rate
- Uses 40 golden questions from evaluation/golden_questions_v1.json
- No LLM needed -- pure retrieval quality measurement
- 13 tests (test_retrieval_eval.py)

### R15: Doctor + Session Robustness [COMPLETE]
- Doctor: added embedder model check (nomic-embed-code verification)
- Session: corrupted JSON handled gracefully with warning
- Session: stale session warning (> 24 hours)
- 14 tests (test_doctor_and_session.py)

### R16: Distillation Pipeline [COMPLETE]
- Fixed stdout replacement in distill_weak_topics.py (broke pytest)
- Verified existing pipeline: FTS5 storage, knowledge file output, RAG context, budget cap
- 12 tests (test_distillation_pipeline.py)

### R17: Config Consolidation [COMPLETE]
- Fixed memory.yaml: hardcoded D:/JCoder_Data -> ${JCODER_DATA}/indexes
- Fixed sanitizer.py: unified JCODER_DATA_DIR -> JCODER_DATA env var
- Verified embedder config consistency (models.yaml + memory.yaml dimensions match)
- Verified CLI entry points (ask, doctor commands wired)
- 18 tests (test_config_consolidation.py)

## Files Modified

### Core Fixes
- `core/experience_replay.py` -- P2Value system (compute_p2value, pass_count/fail_count fields, replay_blend fix)
- `core/telemetry.py` -- Thread-safe close(), per-thread connections
- `core/smart_orchestrator.py` -- Open knowledge fallback
- `core/index_engine.py` -- Quality score filtering in search_fts5_direct
- `agent/session.py` -- Cleanup grace period, corrupted session handling, stale warning

### Config
- `config/models.yaml` -- nomic-embed-code + code_model/text_model
- `config/memory.yaml` -- ${JCODER_DATA} env var
- `config/agent.yaml` -- retrieval.open_knowledge_fallback toggle

### Scripts
- `scripts/build_fts5_indexes.py` -- Dedup integration, quality_score column
- `scripts/run_retrieval_eval.py` -- NEW: retrieval quality eval harness
- `scripts/distill_weak_topics.py` -- Fixed stdout replacement guard

### CLI
- `cli/doctor_cmd.py` -- Embedder model check

### Ingestion
- `ingestion/sanitizer.py` -- Unified JCODER_DATA env var

### New Test Files
- `tests/test_build_fts5_dedup.py` (14 tests)
- `tests/test_open_knowledge_fallback.py` (14 tests)
- `tests/test_retrieval_eval.py` (13 tests)
- `tests/test_doctor_and_session.py` (14 tests)
- `tests/test_distillation_pipeline.py` (12 tests)
- `tests/test_config_consolidation.py` (18 tests)

### New Docs
- `docs/SPRINT_PLAN_R13_R17_2026-03-19.md`
- `docs/SPRINT_R12_RETRIEVAL_QUALITY.md` (updated to COMPLETE)
- `docs/SPRINT_STATUS_2026-03-19.md` (this file)

## Next Priorities

1. **Beast 70B bring-up**: import one Beast-only 70B Ollama model outside repo storage and capture `nvidia-smi` plus `ollama ps` evidence for both GPUs
2. **Boundary check**: verify the work stack and HybridRAG3 Python environment remain untouched by Beast-only model experiments
3. **Run retrieval eval**: `python scripts/run_retrieval_eval.py` to establish baseline
4. **Pull nomic-embed-code**: `ollama pull nomic-embed-code` on BEAST
5. **Rebuild FAISS indexes** with new code embedder
6. **Run distillation pilot**: `python scripts/distill_weak_topics.py` with GPT-5
7. **Sprint 20**: Production packaging and daily-driver cutover
8. **Data sprints**: Continue D1 backlog (OpenCodeReasoning, SWE-smith)
