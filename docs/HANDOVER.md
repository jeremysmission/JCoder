# JCoder Handover -- Session End at 2d1437b

## What JCoder Is
Fully local, offline CLI AI coding assistant. Ingests a codebase, builds a hybrid vector+keyword index, answers questions about the code using a local LLM (vLLM with dual RTX 3090 tensor parallelism).

## Current State
- **7 commits**, 2,518 lines across 25 Python files
- **35/35 pytest** passing
- **RetrievalScore: 36/40 (90%)** on 40-question golden benchmark (mock mode, sparse-only)
- **AnswerScore: 22/40 (55%)** (mock LLM returns dummy text, score is noise until real LLM)
- Frozen benchmark snapshot at `D:\JCoder_bench_snapshot` (git worktree pinned at e524139)

## 2026-03-13 Addendum

- The canonical sprint status board is `docs/SPRINT_STATUS_2026-03-11.md`.
- The canonical completion spine is `docs/SPRINT_COMPLETION_PLAN_2026-03-13.md`.
- Sprint `17-20` are now explicitly assigned and sliced; the old undefined gap is closed.

## Architecture

```
main.py -> cli/commands.py -> core/orchestrator.py
                                  |
                          core/retrieval_engine.py
                           /          \
              core/embedding_engine.py  core/index_engine.py
              (vLLM /v1/embeddings)     (FAISS + FTS5 + RRF)
                                          \
                                   core/reranker.py
                                   (vLLM /score)
                                          \
                                   core/runtime.py
                                   (vLLM /v1/chat/completions)
```

### Key modules
| Module | Purpose |
|--------|---------|
| `core/index_engine.py` | FAISS GPU/CPU + SQLite FTS5 + RRF fusion. Per-index FTS5 DB. search_content normalization. path_prior_boost. |
| `core/embedding_engine.py` | Calls vLLM embedding endpoint, L2-normalizes vectors |
| `core/runtime.py` | Calls vLLM chat/completions with system prompt + context |
| `core/reranker.py` | Cross-encoder rescoring via vLLM /score |
| `core/network_gate.py` | offline/localhost/allowlist HTTP policy enforcement |
| `core/orchestrator.py` | Wires retriever + runtime, returns AnswerResult with chunks |
| `core/mock_backend.py` | Hash-based deterministic mock embedder, reranker, LLM |
| `core/eval_guard.py` | SHA-256 benchmark integrity verification |
| `ingestion/chunker.py` | tree-sitter AST chunking with char-based fallback. Supports .yaml/.yml. |
| `ingestion/repo_loader.py` | File discovery + FileValidator (skip binary/oversized/empty) |
| `cli/commands.py` | Click CLI: ingest, ask, eval, measure, doctor, seal-benchmarks |
| `cli/doctor.py` | Environment readiness checks |

## Commit History
```
2d1437b path_prior_boost, NetworkGate, FileValidator, measure, diagnose-retrieval
bbdc947 Per-index FTS5 DB + search_content normalization + OR-primary
ed1a99e Sparse-only mock mode, FTS5 rebuild on load, frozen benchmark snapshot
e524139 Identifier-heavy sparse heuristic, YAML indexing, skip evaluation/
4f6f4c9 Split eval into RetrievalScore (A) and AnswerScore (B)
4ad8430 Track 1+2: Golden questions eval (40q) + pytest suite (35 tests)
124f4e7 Phase 0.7: Mock-validated RAG skeleton
```

## What Works
- `python main.py --mock ingest <path> --index-name <name>` -- ingest a repo
- `python main.py --mock eval --benchmark evaluation/golden_questions_v1.json --index-name bench_snapshot` -- run golden eval
- `python main.py --mock eval ... --diagnose-retrieval` -- show source rankings for failures
- `python main.py measure` -- GPU/torch/CUDA measurement
- `python main.py doctor` -- environment checks
- `python main.py --mock ask "question" --index-name <name>` -- ask questions
- `python main.py gui` / `jcoder-gui` -- launch the generated command-center GUI
- `python -m pytest tests/ -q` -- 35 tests

## Session 3: GUI Command Center (2026-03-13)

### Changes
- Added a new `gui/` package:
  - `gui/theme.py` ports the HybridRAG3 dark palette and button hierarchy.
  - `gui/command_catalog.py` builds GUI metadata from the live Click tree.
  - `gui/runner.py` streams subprocess output without freezing tkinter.
  - `gui/app.py` adds the command-center shell with search, generated forms, command preview, output console, and stop support.
- Added `cli/gui_cmd.py` so the GUI launches as `jcoder gui`.
- Added the `jcoder-gui` entry point in `pyproject.toml`.
- Updated `README.md`, sprint tracking, and checkpoint docs for the GUI lane.

### Behavior
- Every executable leaf command in the CLI is available in the GUI through generated forms.
- Root CLI options `--config-dir` and `--mock` are surfaced globally.
- Standard commands run in a background subprocess and stream logs into the GUI.
- The `interactive` REPL launches in a separate console window because it needs live terminal input.

### Test Coverage
- `python -m py_compile gui\\theme.py gui\\command_catalog.py gui\\runner.py gui\\app.py cli\\gui_cmd.py`
  - Result: passed
- `python -m pytest tests\\test_gui_command_catalog.py tests\\test_eval_and_cli.py -q --basetemp .tmp_pytest_gui`
  - Result: `23 passed`
- `python -m py_compile agent\\config_loader.py tests\\test_config_loader.py`
  - Result: passed
- `python -m pytest tests\\test_config_loader.py -q --basetemp .tmp_pytest_cfg_loader`
  - Result: `63 passed`
- Windowed startup smoke:
  - Constructed `JCoderGuiApp`, confirmed widget/catalog initialization, ran `doctor.check` through the GUI runner, and observed streamed output plus a clean command completion path.
  - Launched `python main.py gui`, confirmed the GUI process stayed alive until intentionally terminated by the smoke harness.
- GUI-driven backend smoke:
  - Instantiated the real `JCoderGuiApp` against a temporary config/data root with `--mock` enabled.
  - Ran `ingest` against a one-file sample repo, then ran `ask` against the generated temporary index through the same GUI runner path.
  - Both commands completed with streamed output and exit code `0`.
- Live GUI backend smoke:
  - Ran `agent.complete` through the real `JCoderGuiApp` against local Ollama with `phi4-mini:latest`.
  - The command completed successfully and produced completion output in the GUI.
  - The earlier `SQLite objects created in a thread` federated-search failure did not appear in the GUI output.

### Live Validation Follow-Up
- A direct live `agent run` smoke against local Ollama first exposed a federated-search defect where discovered FTS5 indexes reused SQLite connections across worker threads.
- Fixed `agent/config_loader.py` so federated FTS5 indexes stay lazy and let `IndexEngine` open worker-safe connections on demand.
- A subsequent live `agent run` smoke exposed a second federated-search issue: direct FTS5 queries assumed every index had `chunk_id`, but `research_papers.fts5.db` uses the legacy schema `content/source/category`.
- Fixed `core/index_engine.py` so direct federated FTS5 search adapts to both modern and legacy schemas.
- Added regression coverage in `tests/test_config_loader.py` for both threaded federation and legacy FTS5 schema compatibility.
- Re-ran the long-running live `agent run` smoke against local Ollama. It completed successfully with `Summary: READY`, and the earlier SQLite/thread and `chunk_id` warnings no longer appeared.

### Open Items
- Bring up the configured vLLM-style endpoints on ports `8000/8001/8002` and run a live RAG-backed `ask` smoke from the GUI.
- Run the matching CLI live-stack `ask` validation once the vLLM-style stack is online.
- The `interactive` REPL is intentionally external-console today, not an embedded terminal tab.

### Git State
- Authoritative worktree: `D:\HybridRAG3\_jcoder_worktree`
- Current commit: `5d94699` (`Safekeep GUI rollout, federated fixes, and sprint completion plan`)
- Checkout state: `safekeep/jcoder-2026-03-13-115114`
- Safekeep remote branch: `origin/safekeep/jcoder-2026-03-13-115114`
- Local `master` remains at `a90a4d3`
- Remote: `origin https://github.com/jeremysmission/JCoder.git`
- `master` vs `origin/master`: `2 ahead, 0 behind`
- Related worktrees:
  - `D:\JCoder` on `master`
  - `D:\HybridRAG3\_jcoder_worktree` on `safekeep/jcoder-2026-03-13-115114`
  - `D:\JCoder_bench_snapshot` pinned at `352cc58`
- Current worktree is clean after the safekeep commit and push.
- Full regression status at safekeep time:
  - `1136 passed, 2 failed, 2 skipped`
  - the two failures are wall-clock thresholds in `tests/test_hard01_extreme_stress.py`

## Known Issues / Remaining 4 Retrieval Failures
| ID | Question | Expected File | Problem |
|----|----------|---------------|---------|
| G028 | What Protocol interfaces are defined? | core/interfaces.py | 54-line file, BM25 loses to longer files |
| G030 | How does the mock reranker score results? | core/mock_backend.py | Not in sparse top results at all |
| G034 | What does the doctor command check? | cli/doctor.py | Not in sparse top results at all |
| G037 | What GPU safety margin is configured? | config/policies.yaml | 31-line file, too short for BM25 |

**Root cause**: Mock mode is sparse-only (FTS5 BM25). Short files with few tokens lose to longer files that happen to contain the same keywords. Real embeddings (dense FAISS) will fix this -- semantic similarity doesn't care about file length.

## What's Next (Priority Order)
1. **Add git remote** -- no remote configured yet
2. **GPU bring-up** -- install NVIDIA drivers, run `jcoder measure`, fill in vLLM perf numbers
3. **Real embeddings** -- switch from mock to vLLM embedding server, re-ingest, expect 4 retrieval failures to flip
4. **Answer quality tuning** -- AnswerScore is meaningless in mock (dummy answers). Tune with real LLM.
5. **Reranker** -- cross-encoder rescoring for final precision boost
6. **tree-sitter expansion** -- add grammars for more languages

## Config
- `config/default.yaml` -- model endpoints, dimensions, storage paths
- `config/policies.yaml` -- timeouts, batch sizes, caps, GPU safety margin
- `config/ports.yaml` -- vLLM server port assignments

## Session 2: Measure Hardening (6108f7c)

### Changes
- `jcoder measure` now outputs structured JSON at `JCoder_Data/metrics/measurements.json` + pretty terminal table
- JSON schema: `system` / `endpoints` / `performance` (exact keys locked by tests)
- Endpoint probing tries `/v1/models` then `/models` fallback (vLLM + OpenAI compatible)
- `--run-bench` flag: runs perf probes ONLY if all 3 endpoints (llm/embed/rerank) are reachable
- `--allow-wait` flag: gates the 10-minute drift measurement sleep
- All failures graceful: no CUDA = system fields empty; no endpoints = nulls; no crash

### Bug Fixes
- ZeroDivisionError in bench probes: mock HTTP calls return in 0.0s on Windows; fixed via `_safe_elapsed()` (floor 1e-6s) and p95 latency floor of 0.01ms
- Mock routing order: `/chat/completions` checked before `/models` to prevent substring false matches

### Test Coverage
- 48/48 tests passing (35 original + 5 new bench tests + 8 schema tests)
- Tests cover: schema keys/types, graceful degradation (no CUDA, no endpoints), bench with unreachable endpoints, bench with reachable mocked endpoints

### Commits
```
6108f7c Measure: optional --run-bench perf probes (graceful + tested)
88c8932 Measure: robust /v1/models probing + schema tests
801fe24 Harden measure command: structured JSON output, endpoint probes, graceful degradation
```

## No Remote
JCoder has no git remote configured. To add one:
```bash
cd /d/JCoder
git remote add origin <your-repo-url>
git push -u origin master
```
