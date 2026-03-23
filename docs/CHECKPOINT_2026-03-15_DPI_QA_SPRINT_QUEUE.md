# DPI QA Sprint Queue -- 2026-03-15
**Assessor:** Claude Opus 4.6 (Master QA)
**Verdict:** B+ (STRONG) -- Production-viable with targeted fixes
**1699 tests, 0 failures, 174 FTS5 indexes (~40+ GB)**

## ALREADY FIXED THIS SESSION
- [x] agent/tools.py ToolRegistry: 657L -> 428L (extracted tool_schemas.py, tool_defs.py)
- [x] ingestion/sanitizer.py SanitizationPipeline: 620L -> 535L (extracted archive/comment helpers)
- [x] Weekly scraper: 97 fresh chunks ingested
- [x] Download script fixes: hermes multi-config, mixture_of_thoughts config routing

---

## SPRINT R8: Pre-BEAST Security Fixes (CRITICAL)
Priority: Must fix before BEAST deployment

### R8.1 SQL Injection -- scripts/weekly_knowledge_update.py:189 -- FIXED
- Added _SAFE_ID regex whitelist + double-quote identifier quoting
- Changed to `with sqlite3.connect()` context manager (fixes resource leak too)

### R8.2 Bare except: -- scripts/download_best_practices.py:413 -- FALSE POSITIVE
- The `except:` is inside a documentation string (teaching example), not executable code

### R8.3 Silent Error in bridge.py:647 -- FIXED
- Changed `except Exception: return ""` to log at DEBUG level

### R8.4 Test Coverage: Critical Modules -- PARTIAL
- [x] core/orchestrator.py -- 13 tests (test_orchestrator.py)
- [x] core/runtime.py -- 18 tests (test_runtime.py)
- [x] psutil added as optional dep (pyproject.toml [ops])
- [ ] agent/core.py (~400 LOC) -- Agent main loop, still needs tests
- [ ] core/retrieval_engine.py -- still needs tests

---

## SPRINT R8.5: Critical Feature Wiring (CRITICAL -- from architecture review)

### R8.5.1 Wire ModelCascade in bridge.py -- NOT INSTANTIATED
- Imported at line 42-43 but never created in _try_init_pipeline()
- Config flag `cascade_enabled` is read (line 701-702) but unused
- Sprint 12 feature completely non-functional
- Fix: ~20 lines in _try_init_pipeline()

### R8.5.2 Wire SmartOrchestrator in bridge.py -- NOT INSTANTIATED
- Imported at line 28-29 but never created
- Config flag `smart_orchestrator_enabled` read but unused
- Sprint 10 Self-RAG feature non-functional
- Fix: ~20 lines in _try_init_pipeline()

---

## SPRINT R9: Error Handling & Test Coverage (HIGH)
Priority: Next sprint after R8

### R9.1 Silent Error Swallowing -- 74+ instances
Top offenders:
- agent/bridge.py:647 -- inline `except Exception: return ""` (no logging)
- ingestion/sanitizer.py -- 11 instances
- core/adaptive_research.py -- 5 instances
- core/rapid_digest.py -- 4 instances
- cli/agent_cmd.py -- 4 instances
Action: Audit & narrow top-20 worst blocks, add logging

### R9.2 Test Coverage: Core Engine Modules
- core/retrieval_engine.py (~300 LOC) -- CRITICAL
- core/runtime.py (~250 LOC) -- HIGH
- core/index_engine.py (~400 LOC) -- HIGH
- agent/llm_backend.py (~300 LOC) -- HIGH
- core/config.py (~300 LOC) -- HIGH

### R9.3 Resource Cleanup
- scripts/weekly_knowledge_update.py:182-192: sqlite3.connect() without `with`
- core/sqlite_owner.py:55: `except sqlite3.Error: pass` on cleanup
- Add atexit/signal handlers for graceful shutdown

### R9.4 psutil Dependency
- Used in scripts/bootstrap_beast.py:104, not in pyproject.toml
- Add as optional dep

---

## SPRINT R10: Code Quality (MEDIUM)
Priority: Address during normal development

### R10.1 Print -> Logging (131+ instances)
- ingestion/repo_loader.py, evaluation/agent_eval_runner.py, scripts/
- Replace print() with logging.getLogger(__name__)

### R10.2 Global State Mutation (18+ instances)
- _pass, _fail, _DOWNLOADERS, _EVAL_SET_CACHE
- Race-condition risk under ThreadPoolExecutor
- Lock mutable globals behind threading.Lock in bridge.py

### R10.3 Classes at Tolerance Edge
- core/rapid_digest.py RapidDigest ~523L
- gui/tk_app.py Main class ~515L
- Monitor, refactor if they grow

### R10.4 TODO Cleanup
- scripts/build_format_smoke_pack.py:94 -- URL in TODO

### R10.5 vLLM Port Hardcoding
- scripts/run_vllm.ps1 hardcodes ports vs config/ports.yaml
- Single source of truth needed

---

## SPRINT R11: Polish (LOW)
- L1: Version pinning (>= -> ~=) in pyproject.toml
- L2: Magic numbers in core/best_of_n.py -> named constants
- L3: DRY violation in 8 download scripts (identical _DOWNLOADER pattern)
- L4: Structured JSON logging
- L5: SQLite migration framework
- L6: Missing __all__ exports in core/, cli/, ingestion/
- L7: wheel unpinned in build deps
- L8: .tmp_* dirs cleanup

---

## SPRINT R12: Beast 70B Qualification Gate (HIGH)
Priority: Must pass before JCoder claims Beast as the 70B test lane

### R12.1 Beast-only storage and boundary check
- Verify the 70B pull path resolves to Beast global model storage, never under JCoder or other repo directories
- Verify work GPU standards and the HybridRAG3 Python stack remain unchanged

### R12.2 Dual-GPU evidence capture
- Capture `nvidia-smi` telemetry during model load and one prompt
- Confirm both 24 GB RTX 3090s engage through memory residency and or sustained utilization
- Record the current sharding assumption explicitly: NVLink is inactive, so Beast is operating as two independent GPUs

### R12.3 Operator rollback proof
- Document how to remove the imported 70B model and revert Beast-only launch or environment changes
- Keep JCoder, Hustle, and Beast coordination notes aligned on the same storage boundary and hardware assumptions

### Exit Criteria
- There is a saved proof pack showing Beast-only storage, dual-GPU engagement, and rollback steps for the first 70B import and prompt run

---

## GRADE CARD
| Dimension | Grade |
|-----------|-------|
| Architecture | A |
| Security | A |
| Configuration | A |
| Test Quality | A- |
| Dependency Management | A |
| Tool Safety | A |
| Test Coverage | C+ (33 modules at zero) |
