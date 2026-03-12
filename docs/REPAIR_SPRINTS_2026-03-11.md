# JCoder Repair Sprint Plan

- Created: 2026-03-11 America/Denver
- Author: Claude_Deep_Packet_Inspector
- Source: Full codebase deep inspection (188 source files, 56 test files, 9 config files)
- Findings: 3 CRITICAL, 9 HIGH, 26 MEDIUM, 16 LOW, 17 INFO
- Sprints: 7 repair sprints, dependency-ordered

## Execution Order Rationale

1. **R1 first**: crash and security -- nothing else matters if the runtime crashes on every unusual LLM response or an LLM can exec arbitrary code
2. **R2 second**: race conditions -- these corrupt state silently and are hardest to debug after the fact
3. **R3 third**: silent failures -- bare `except: pass` blocks hide the bugs R1-R2 are fixing; must be observable before further work
4. **R4 fourth**: config portability -- hardcoded paths prevent the project from running on any machine other than the current dev box
5. **R5 fifth**: SQLite connection consolidation -- performance and resource hygiene across 14 files; depends on R3 (failures now logged, so connection issues are visible)
6. **R6 sixth**: data pipeline hardening -- depends on R4 (paths correct) and R5 (connections healthy)
7. **R7 last**: test infrastructure -- depends on all production fixes being in place so tests cover the right behavior

---

## Repair Sprint 1: Critical Crash and Security Guards -- COMPLETED

**Goal**: Eliminate the three findings that can crash the runtime or execute arbitrary code.

**Scope**: 3 CRITICAL findings, ~8 files

**Status**: COMPLETED 2026-03-11. All 3 CRITICAL fixes applied and verified. 804 passed, 2 skipped.

### R1-1: Guard LLM response parsing (CRITICAL)
- **File**: `core/runtime.py:115`
- **Problem**: `response.json()["choices"][0]["message"]["content"]` -- any unexpected LLM response structure raises unhandled `KeyError`/`IndexError`, crashing the calling pipeline
- **Fix**: Wrap in `try/except (KeyError, IndexError, TypeError)` with descriptive error log and return a structured error response
- **Acceptance**: Unit test with malformed response dicts (missing `choices`, empty `choices`, missing `content` key, `None` response body)

### R1-2: Sandbox LLM code execution (CRITICAL)
- **File**: `tools/hard_challenges.py:386-387`
- **Problem**: `exec(code, ns)` runs raw LLM-generated code in the current interpreter with full filesystem/network access
- **Fix**: Run generated code in a subprocess with restricted permissions, or at minimum: (a) use `ast.parse()` to reject imports of `os`, `sys`, `subprocess`, `shutil`, `socket`, `pathlib`; (b) set a `signal.alarm` or `threading.Timer` timeout; (c) restrict the namespace dict to safe builtins only
- **Acceptance**: Test that import-os code is rejected; test that infinite loop times out; test that legitimate code still runs

### R1-3: Guard SQLite connections in scripts (CRITICAL)
- **Files**: `scripts/build_fts5_indexes.py:154`, `scripts/build_se_indexes.py:218`, `scripts/overnight_download.py:371,384`, `tools/ingest_datasets.py:78`, `tools/build_faiss_indexes.py:56,212`
- **Problem**: `conn = sqlite3.connect(...)` without `try/finally` or context manager. Exception during build leaves `.fts5.db` file-locked on Windows
- **Fix**: Wrap all script-level `sqlite3.connect()` calls in `with contextlib.closing(sqlite3.connect(...)) as conn:` or equivalent `try/finally`
- **Acceptance**: Verify each script's connection is closed on both success and exception paths; Windows file-lock test (attempt to open DB after simulated build failure)

---

## Repair Sprint 2: Race Conditions and Thread Safety -- COMPLETED

**Goal**: Eliminate all identified race conditions that corrupt state under concurrent access.

**Scope**: 5 findings across 5 files

**Status**: COMPLETED 2026-03-11. All 5 thread-safety fixes applied and verified. 804 passed, 2 skipped.

**Changes applied**:
- R2-1: `core/federated_search.py` -- `_pool_lock` guards ThreadPoolExecutor lazy init and `close()`
- R2-2: `core/self_learning_pipeline.py` -- `_query_lock` guards `_query_count` increment
- R2-3: `agent/core.py` -- `_run_lock`/`_running` flag prevents concurrent `run()`/`resume()`; `resume()` no longer resets token counters; `_save_session()` clears `_running` on terminal states
- R2-4: `core/cascade.py` -- `_runtime_lock` guards `_get_runtime()` lazy init and `close()`
- R2-5: `core/meta_cognitive.py` -- `_arms_lock` guards `_load_arms()`, `_get_arm()`, `strategy_report()`, `best_strategy_per_type()`

### R2-1: Lock ThreadPoolExecutor lazy init (HIGH)
- **File**: `core/federated_search.py:253-256`
- **Problem**: Two concurrent `search()` calls both observe `self._pool is None`, each create a `ThreadPoolExecutor`, orphaning one and leaking its threads
- **Fix**: Add `threading.Lock` around the `self._pool is None` check and pool creation
- **Acceptance**: Test: 10 concurrent `search()` calls on a fresh instance; assert only 1 pool exists after all complete

### R2-2: Atomic query counter (HIGH)
- **File**: `core/self_learning_pipeline.py:184`
- **Problem**: `self._query_count += 1` is non-atomic read-modify-write; concurrent `answer()` calls corrupt the counter
- **Fix**: Use `threading.Lock` around the increment, or replace with `itertools.count()` (thread-safe `__next__`)
- **Acceptance**: Test: 100 concurrent `answer()` calls; assert final count == 100

### R2-3: Guard Agent concurrent state (HIGH)
- **File**: `agent/core.py:148,409-410`
- **Problem**: `run()` and `resume()` on the same Agent instance race on `_history`, `_steps`, `_total_input_tokens`, `_total_output_tokens`; `resume()` also resets token counters to 0, discarding prior-session totals
- **Fix**: (a) Add a `_running` flag with lock that prevents concurrent `run()`/`resume()`; (b) `resume()` should accumulate token counters, not reset them
- **Acceptance**: Test: calling `run()` while another `run()` is active raises `RuntimeError`; test: `resume()` preserves prior token totals

### R2-4: Lock cascade runtime dict (MEDIUM)
- **File**: `core/cascade.py:146`
- **Problem**: `_runtimes` dict lazily populated inside `route()` without a lock; concurrent routing orphans HTTP connection pools
- **Fix**: `threading.Lock` around the lazy init check
- **Acceptance**: Test: 5 concurrent `route()` calls for the same level; assert only 1 Runtime instance exists

### R2-5: Lock meta-cognitive arm state (MEDIUM)
- **File**: `core/meta_cognitive.py:165`
- **Problem**: `_arms` dict is in-memory mutable state shared across threads with no lock; concurrent `select_strategy`/`report_outcome` calls race on arm updates
- **Fix**: Add `threading.Lock` around all `_arms` reads and writes
- **Acceptance**: Test: concurrent select + report cycles; assert arm stats are consistent

---

## Repair Sprint 3: Silent Failure Elimination

**Goal**: Replace all bare `except Exception: pass` blocks with logged errors so failures are observable.

**Scope**: 8 files, ~20 catch blocks

### R3-1: knowledge_graph.py (HIGH)
- **Lines**: 349, 374
- **Problem**: `_add_entity` and `_add_relation` silently discard all SQLite write failures
- **Fix**: Replace `pass` with `logger.warning("Failed to add %s: %s", type, exc)` at minimum
- **Acceptance**: Test: simulate SQLite write error; assert warning is logged

### R3-2: adversarial_self_play.py (HIGH)
- **Lines**: 234, 472
- **Problem**: Game-round failures and `_persist` failures both silently swallowed; training appears successful even when all rounds fail
- **Fix**: Log at WARNING level; count failed rounds; include failure count in training summary
- **Acceptance**: Test: all rounds fail; assert training summary includes failure count > 0

### R3-3: rapid_digest.py (MEDIUM)
- **Multiple locations**: `_triage`, `_extract`, `_synthesize`, `generate_prototype`, `_persist`
- **Problem**: Bare `except Exception: return {}` and `except Exception: pass` make all processing failures silent
- **Fix**: Log at WARNING level in each catch block
- **Acceptance**: Test: simulate LLM failure; assert warning logged and caller receives empty dict (not crash)

### R3-4: prompt_evolver.py (MEDIUM)
- **Lines**: 353, 388
- **Problem**: `_mutate` and `_crossover` return `None` silently on any error; evolver appears to stall
- **Fix**: Log at WARNING level; return `None` is acceptable but must be logged
- **Acceptance**: Test: simulate mutation error; assert warning logged

### R3-5: synthesis_matrix.py (MEDIUM)
- **Lines**: 252, 314
- **Problem**: `_extract_themes` and `_classify_positions` fall through to heuristic scoring silently
- **Fix**: Log at INFO level (heuristic fallback is by-design, but should be observable)
- **Acceptance**: Test: simulate LLM failure; assert info log entry about fallback

### R3-6: smart_orchestrator.py (MEDIUM)
- **Lines**: 125, 152
- **Problem**: Reflection and telemetry failures use bare `pass` with no log entry
- **Fix**: Log at WARNING level
- **Acceptance**: Test: simulate telemetry write error; assert warning logged

### R3-7: layered_triage.py (MEDIUM)
- **Line**: 243
- **Problem**: `_drone_pass` falls back to satellite scores on LLM failure with no log
- **Fix**: Log at WARNING level
- **Acceptance**: Test: simulate drone-pass LLM error; assert warning logged and satellite fallback used

### R3-8: cascade.py (LOW)
- **Line**: 207
- **Problem**: `except Exception: continue` in routing loop; escalation errors invisible
- **Fix**: Log at WARNING with the exception and the current cascade level
- **Acceptance**: Test: simulate level-1 failure; assert warning logged and escalation to level-2 occurs

---

## Repair Sprint 4: Config and Path Portability

**Goal**: Remove all hardcoded paths and harden config loading so JCoder runs on any machine.

**Scope**: 7 findings across ~8 files

### R4-1: Guard config dataclass unpacking (HIGH)
- **File**: `core/config.py:260-267`
- **Problem**: `RetrievalConfig(**retrieval_raw)` crashes with unhelpful `TypeError` on any unknown YAML key
- **Fix**: Filter each raw dict against `dataclass_fields` before unpacking; log a warning for any unknown keys dropped; catch `TypeError` with a descriptive startup error
- **Acceptance**: Test: YAML with an unknown key loads successfully with a warning; test: YAML with a typo'd key logs which key was dropped

### R4-2: Remove hardcoded path from memory.yaml (HIGH)
- **File**: `config/memory.yaml:20`
- **Problem**: `data_dir: "D:/JCoder_Data/indexes"` is a hardcoded absolute path in a tracked config file
- **Fix**: Change to a relative path or `${JCODER_DATA}/indexes` with env var expansion in the loader
- **Acceptance**: Test: config loads correctly with `JCODER_DATA` set to a temp directory; test: config loads correctly with the env var unset (falls back to relative path)

### R4-3: Remove hardcoded D:\ from scripts (HIGH)
- **Files**: `scripts/overnight_download.py:454,486`, `scripts/data_status.py:73`, `scripts/build_se_indexes.py:43-45`, `cli/bench_cmd.py:72`
- **Problem**: `shutil.disk_usage('D:\\')` and hardcoded archive paths bypass `JCODER_DATA` env var
- **Fix**: Derive drive letter from configured data directory; replace hardcoded archive paths with config-driven or env-var-driven paths
- **Acceptance**: Test: each script resolves paths from config/env var, not hardcoded drive letter

### R4-4: Fix module-import-time env read (LOW)
- **File**: `ingestion/sanitizer.py:121-126`
- **Problem**: `SanitizationConfig.clean_archive_dir` default evaluates `os.environ.get("JCODER_DATA_DIR", ...)` at class definition time, not at instantiation
- **Fix**: Use `dataclasses.field(default_factory=...)` to defer evaluation
- **Acceptance**: Test: set env var after import; assert new SanitizationConfig() picks up the new value

### R4-5: Fix dead batching code in prisma_tracker (HIGH)
- **File**: `core/prisma_tracker.py:56-59`
- **Problem**: `_flush_every = 20000` is dead code; `_log()` flushes synchronously on every call regardless
- **Fix**: Either wire `_flush_every` into a counter-based flush (add `_pending_count`, flush when `>= _flush_every`), or remove the dead field and document the always-flush design
- **Acceptance**: If batching: test that flush only happens every N calls. If always-flush: field removed and docstring updated.

### R4-6: Fix prisma_tracker unbounded title cache (MEDIUM)
- **File**: `core/prisma_tracker.py:60`
- **Problem**: `_title_cache` dict grows unbounded; long-running processes accumulate memory
- **Fix**: Add LRU eviction (e.g., `functools.lru_cache` on the lookup method, or cap dict at 10K entries with oldest-evict)
- **Acceptance**: Test: insert 15K titles; assert cache size stays at or below cap

---

## Repair Sprint 5: SQLite Connection Consolidation

**Goal**: Replace the new-connection-per-call anti-pattern in 14 modules with a shared connection cache.

**Scope**: 14 files, 1 pattern

### R5-1: Implement threading.local() connection pattern

**Affected files** (all in `core/`):
1. `telemetry.py`
2. `experience_replay.py`
3. `meta_cognitive.py`
4. `stigmergy.py`
5. `star_reasoner.py`
6. `active_learner.py`
7. `continual_learner.py`
8. `prompt_evolver.py`
9. `quality_diversity.py`
10. `adversarial_self_play.py`
11. `procedural_memory.py`
12. `knowledge_graph.py`
13. `rapid_digest.py`
14. `prisma_tracker.py`

**Problem**: Each method call creates a new `sqlite3.connect(...)`, uses it, and closes it. High-frequency paths (telemetry, experience replay) cycle hundreds of connections per session.

**Fix**: Add a `_get_conn(self) -> sqlite3.Connection` method to each class that:
1. Uses `threading.local()` to store one connection per thread
2. Creates the connection on first access with WAL mode enabled
3. Provides an explicit `close()` method that closes all thread-local connections
4. Uses `atexit.register` as a safety net for cleanup

**Implementation pattern** (reusable base or mixin):
```python
import threading
import sqlite3
import atexit

class _SQLiteOwner:
    def __init__(self, db_path: str):
        self._db_path = db_path
        self._local = threading.local()
        self._all_conns: list[sqlite3.Connection] = []
        self._lock = threading.Lock()
        atexit.register(self.close)

    def _get_conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, 'conn', None)
        if conn is None:
            conn = sqlite3.connect(self._db_path, check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn = conn
            with self._lock:
                self._all_conns.append(conn)
        return conn

    def close(self):
        with self._lock:
            for c in self._all_conns:
                try:
                    c.close()
                except Exception:
                    pass
            self._all_conns.clear()
```

**Acceptance**:
- Test: 3 threads calling the same module concurrently; assert no `sqlite3.OperationalError` (database locked)
- Test: `close()` called; assert subsequent `_get_conn()` creates a fresh connection
- Benchmark: measure connection creation count before/after; assert reduction from N-per-call to 1-per-thread

### R5-2: Fix stigmergy O(n) query pattern
- **File**: `core/stigmergy.py`
- **Problem**: `_get_pheromones()` issues one `SELECT` per `chunk_id` instead of batched `WHERE chunk_id IN (...)`
- **Fix**: Batch into a single parameterized query
- **Acceptance**: Test: 100 chunk_ids; assert only 1 SQL query executed

### R5-3: Fix star_reasoner blanket UPDATE
- **File**: `core/star_reasoner.py`
- **Problem**: `_increment_staleness()` has no `WHERE` clause -- updates every row in the table on every query
- **Fix**: Add `WHERE trace_id NOT IN (?)` to exclude the current query's traces, or scope to traces older than a threshold
- **Acceptance**: Test: insert 10 traces, run query using 2; assert only 8 traces have incremented staleness

### R5-4: Fix knowledge_graph O(n) patterns
- **File**: `core/knowledge_graph.py`
- **Lines**: 386-392 (`_find_seeds` per-word SELECT), 407-411 (`_personalized_pagerank` loads entire relations table), 463 (BFS `list.pop(0)`)
- **Fix**: (a) Batch seed lookup into single query; (b) Add LIMIT or streaming to PageRank query; (c) Use `collections.deque.popleft()`
- **Acceptance**: Test: graph with 10K relations; assert PageRank does not load all into memory (mock cursor to verify LIMIT)

---

## Repair Sprint 6: Data Pipeline Hardening

**Goal**: Close silent data gaps and resource issues in the download/indexing pipeline.

**Scope**: 7 findings across ~10 files

### R6-1: Log HuggingFace API schema changes (HIGH)
- **File**: `core/download_manager.py:43-63`
- **Problem**: `fetch_huggingface_parquet_urls` silently returns `[]` when API response shape changes; corpus gaps are invisible
- **Fix**: Log at WARNING when the response structure doesn't match expected shape; include the actual response keys in the log
- **Acceptance**: Test: mock HuggingFace response with changed schema; assert warning logged with diagnostic info

### R6-2: Add FTS5 query token limit (MEDIUM)
- **File**: `core/index_engine.py:196-206`
- **Problem**: FTS5 sanitizer converts every token to an OR term; long queries explode into expensive full-index scans
- **Fix**: Cap FTS5 query at 10 tokens (configurable); truncate with a log entry if exceeded
- **Acceptance**: Test: 25-token query is truncated to 10; test: 5-token query is untouched

### R6-3: Enforce NetworkGate in download pipeline (MEDIUM)
- **Files**: All download scripts, `core/download_manager.py`
- **Problem**: `network.mode = "offline"` in config is not checked by any download path
- **Fix**: Pass `NetworkGate` to `DownloadManager.__init__`; check gate before any HTTP call; raise `NetworkGateError` if blocked
- **Acceptance**: Test: set network mode to offline; assert download attempt raises `NetworkGateError`

### R6-4: Cap session file load size (MEDIUM)
- **File**: `agent/session.py:102-119`
- **Problem**: `json.loads(path.read_text())` loads entire session file with no size limit
- **Fix**: Check `path.stat().st_size` before reading; reject files above a configurable cap (default 10 MB) with a descriptive error
- **Acceptance**: Test: session file > 10 MB; assert `ValueError` raised with size info

### R6-5: Guard agent write_file errors (HIGH)
- **File**: `agent/tools.py:657-665`
- **Problem**: `open()` and `f.write()` not wrapped; OSError propagates as generic error with no file context
- **Fix**: Wrap in `try/except OSError as exc:` returning `ToolResult(False, f"Write failed for {resolved}: {exc}")`
- **Acceptance**: Test: write to read-only path; assert ToolResult.success is False with path in error message

### R6-6: Add fetch_json size limit (LOW)
- **File**: `core/download_manager.py:138-140`
- **Problem**: `fetch_text()` reads entire response body with no size limit
- **Fix**: Add `max_bytes` parameter (default 50 MB); read in chunks and abort if exceeded
- **Acceptance**: Test: mock response > 50 MB; assert download aborted with size error

### R6-7: Limit build_fts5_indexes error reporting (MEDIUM)
- **File**: `scripts/build_fts5_indexes.py:186-189`
- **Problem**: After 5 errors, subsequent errors are silently swallowed; no per-file diagnostics available
- **Fix**: Log all errors to a sidecar `.errors.log` file; keep the console cap at 5 for readability but always write full errors to the log
- **Acceptance**: Test: simulate 20 parse errors; assert all 20 appear in the error log file

---

## Repair Sprint 7: Test Infrastructure and Coverage

**Goal**: Harden the test suite so it catches regressions reliably across machines and environments.

**Scope**: 56 test files, infrastructure additions

### R7-1: Create conftest.py with shared fixtures
- **File**: `tests/conftest.py` (new)
- **Provide**: `tmp_db_path` fixture (temp SQLite path with cleanup), `mock_llm` fixture (configurable response factory), `mock_gate` fixture (NetworkGate in offline mode), `agent_memory` fixture (temp AgentMemory with cleanup), project root `sys.path` setup
- **Acceptance**: At least 5 existing test files refactored to use shared fixtures; all 817+ tests still pass

### R7-2: Add suite-level timeout
- **File**: `pyproject.toml` or `conftest.py`
- **Problem**: No timeout; a blocking test hangs the entire run
- **Fix**: Add `timeout = 120` (2 minutes) as default via `pytest-timeout` in `pyproject.toml` `[tool.pytest.ini_options]`; tag genuinely long tests with `@pytest.mark.timeout(600)`
- **Acceptance**: A test with `time.sleep(300)` is killed after 120 seconds

### R7-3: Fix resource leaks in test cleanup
- **Files**: `test_federated_integration.py`, `test_agent_memory.py`, `test_hard01_extreme_stress.py`
- **Problem**: Cleanup calls not in `try/finally`; Windows file locks on assertion failure
- **Fix**: Move all `.close()` calls into `finally` blocks or use the new `conftest.py` fixtures with `yield`
- **Acceptance**: Deliberately fail an assertion mid-test; assert no file locks remain (temp dir is deletable)

### R7-4: Replace wall-clock assertions with marks
- **Files**: `test_hard01_extreme_stress.py`, `test_hard04_invariants.py`, `test_hard05_full_chaos.py`, `test_federated_integration.py`
- **Problem**: `assert elapsed < 2.0` fails on slower machines
- **Fix**: Mark timing-sensitive tests with `@pytest.mark.slow`; add `addopts = "-m 'not slow'"` as default; CI can opt in with `-m slow`
- **Acceptance**: Default `pytest` run skips slow tests; `pytest -m slow` runs them

### R7-5: Add download manager coverage
- **File**: `tests/test_download_manager.py` (expand)
- **Missing**: SHA-256 verify (success + mismatch), retry logic, 4xx/5xx HTTP responses, concurrent downloads, ledger persistence across restarts
- **Acceptance**: At least 8 new tests covering the listed gaps; all pass

### R7-6: Add write/edit tool path restriction tests
- **File**: `tests/test_agent.py` (expand)
- **Problem**: Only `read_file` is tested through the `allowed_dirs` guard; write and edit could bypass it
- **Fix**: Add tests that `write_file` and `edit_file` outside `allowed_dirs` return `ToolResult(success=False)`
- **Acceptance**: 2 new tests; both pass

### R7-7: Fix weak chaos test assertions
- **Files**: `test_hard03_adversarial_llm.py:194`, `test_hard02_chaos_pipeline.py:121`
- **Problem**: `or`-based assertions that always pass; NaN bypass in score check
- **Fix**: Split into separate assertions (weights sum to 1.0 AND all non-negative); replace `x == x` with explicit `not math.isnan(x)`
- **Acceptance**: Test with `float('nan')` score injected; assert test fails (currently passes)

### R7-8: Fix test isolation issues
- **Files**: `test_hard04_invariants.py:27`, `test_hard05_full_chaos.py:29` (module-level `random.seed`), `test_build_se_indexes.py:23-26` (module-level `sys.platform` mutation)
- **Fix**: Move `random.seed()` into fixtures or test function bodies; move `sys.platform` override into a fixture with proper teardown
- **Acceptance**: Run tests in randomized order (`pytest-randomly`); all pass

---

## Summary

| Sprint | Focus | Findings Addressed | Estimated Files |
|--------|-------|--------------------|-----------------|
| R1 | Crash and Security | 3 CRITICAL | ~8 |
| R2 | Race Conditions | 2 HIGH + 2 MEDIUM | ~5 |
| R3 | Silent Failures | 2 HIGH + 6 MEDIUM | ~8 |
| R4 | Config Portability | 3 HIGH + 2 MEDIUM + 1 LOW | ~8 |
| R5 | SQLite Connections | 14 MEDIUM + 3 MEDIUM | ~17 |
| R6 | Pipeline Hardening | 2 HIGH + 4 MEDIUM + 1 LOW | ~10 |
| R7 | Test Infrastructure | 4 CRITICAL (test) + 8 coverage gaps | ~12 |
| **Total** | | **3C + 9H + 26M + 4L** | **~68 files** |

## Dependencies

```
R1 (crash/security) -- no deps, start immediately
  |
  v
R2 (race conditions) -- depends on R1 (runtime stable)
  |
  v
R3 (silent failures) -- depends on R2 (races fixed, errors now meaningful)
  |
  v
R4 (config portability) -- can run parallel with R3
  |
  v
R5 (SQLite connections) -- depends on R3 (failures logged) + R4 (paths correct)
  |
  v
R6 (pipeline hardening) -- depends on R4 + R5
  |
  v
R7 (test infrastructure) -- depends on all production fixes
```

R3 and R4 can run in parallel. All others are sequential.
