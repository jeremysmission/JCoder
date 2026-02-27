# 04 -- Reusable Code Inventory

JCoder is a fully local offline CLI AI coding assistant. It reuses proven,
production-tested code from two existing projects to avoid reinventing the
wheel. This document catalogs every reusable file, what it does, what ports
directly, and what needs adaptation.


## HybridRAG3 Code (D:\HybridRAG3 -- 15 files, 8,700 lines)

### 1. src/core/llm_router.py -- 1,812 lines

**Full path:** `D:\HybridRAG3\src\core\llm_router.py`

**What it does:**
Switchboard that routes AI queries to the correct backend based on mode.
Four backend router classes (OllamaRouter, VLLMRouter, APIRouter,
TransformersRouter) behind a single `LLMRouter.query()` orchestrator.
Handles credential resolution, model discovery, httpx client factory with
proxy/CA awareness, error classification, and streaming generation.

**Reuse as-is:**
- OllamaRouter class (localhost Ollama HTTP client with streaming)
- VLLMRouter class (localhost vLLM with OpenAI-compatible endpoint)
- LLMResponse dataclass
- `_build_httpx_client()` factory (proxy/CA bundle handling)
- Error classification and retry logic

**Needs adaptation:**
- Replace APIRouter with direct vLLM OpenAI-compatible client for local
  inference; add tensor parallel config for multi-GPU
- Strip TransformersRouter (HuggingFace retired, not needed)
- Simplify LLMRouter orchestrator -- JCoder only needs offline backends
- Add model-specific context window enforcement

**Estimated effort:** Medium


### 2. src/core/vector_store.py -- 653 lines

**Full path:** `D:\HybridRAG3\src\core\vector_store.py`

**What it does:**
SQLite + numpy memmap hybrid vector store. Stores chunk metadata and text in
SQLite with FTS5 full-text search, stores embedding vectors in memory-mapped
float16 numpy files for low-RAM operation. Provides hybrid dense+sparse
retrieval with Reciprocal Rank Fusion (RRF) scoring. Deterministic chunk IDs
via INSERT OR IGNORE for crash-safe re-indexing.

**Reuse as-is:**
- Entire hybrid search pipeline (dense cosine + BM25 sparse + RRF fusion)
- SQLite schema and migration logic
- EmbeddingMemmapStore (float16 memmap for memory efficiency)
- ChunkMetadata dataclass
- FTS5 keyword search integration

**Needs adaptation:**
- Upgrade embedding dimension from 768 to match Nomic Embed Code 7B output
- Add GPU FAISS index option for workstation-class hardware
- Add language/file-type metadata columns for code-aware filtering

**Estimated effort:** Low


### 3. src/core/embedder.py -- 210 lines

**Full path:** `D:\HybridRAG3\src\core\embedder.py`

**What it does:**
HTTP client wrapper for the Ollama `/api/embed` endpoint. Converts text to
embedding vectors via the local nomic-embed-text model. Handles batch
embedding with L2 normalization, model availability checking, and lazy httpx
import for fast startup.

**Reuse as-is:**
- Batch embedding pipeline with progress tracking
- L2 normalization logic
- Lazy httpx import pattern
- Model availability check on init

**Needs adaptation:**
- Switch from Ollama `/api/embed` to vLLM `/v1/embeddings` endpoint for
  Nomic Embed Code 7B
- Add code-specific preprocessing (strip comments vs. keep them, language
  detection for embedding prompt prefixes)

**Estimated effort:** Low


### 4. src/core/config.py -- 853 lines

**Full path:** `D:\HybridRAG3\src\core\config.py`

**What it does:**
Single source of truth for all settings. YAML-based configuration with a
dataclass hierarchy providing typed access, IDE autocomplete, and typo
protection. Resolution priority: env vars > YAML file > hardcoded defaults.
Includes profile system (laptop_safe, desktop_power, server_max) and a
safety net that warns on YAML key mismatches instead of silently dropping
them.

**Reuse as-is:**
- `load_config()` function and YAML loading pattern
- Dataclass hierarchy pattern with `_dict_to_dataclass` converter
- Profile system architecture (adapt profile names)
- Environment variable override mechanism
- Key mismatch warning safety net

**Needs adaptation:**
- New config schema for JCoder (CLI settings, scraper schedule, evolution
  parameters, vLLM endpoints, tree-sitter language list)
- Remove HybridRAG-specific sections (hallucination guard, reranker, GUI)
- Add code-specific sections (supported languages, AST chunking params,
  evolution thresholds)

**Estimated effort:** Medium


### 5. src/core/query_engine.py -- 507 lines

**Full path:** `D:\HybridRAG3\src\core\query_engine.py`

**What it does:**
The main RAG pipeline orchestrator. Six-step pipeline: search (via Retriever)
-> build context from chunks -> construct prompt with source-bounding rules ->
call LLM (via LLMRouter) -> calculate cost -> log query for audit trail.
Every failure path returns a safe result (never throws to caller). QueryResult
dataclass carries answer, sources, chunk count, latency, and cost.

**Reuse as-is:**
- Pipeline architecture (retrieve -> context -> prompt -> generate -> log)
- QueryResult dataclass
- Safe failure pattern (every error path returns a result, never raises)
- Cost calculation integration point

**Needs adaptation:**
- Code-specific prompting (ask about functions, classes, patterns rather than
  document facts)
- Add reranker step between retrieval and context building
- Code-aware citation format (file:line instead of document:page)
- Streaming response support for CLI display

**Estimated effort:** Medium


### 6. src/core/grounded_query_engine.py -- 462 lines

**Full path:** `D:\HybridRAG3\src\core\grounded_query_engine.py`

**What it does:**
Extends QueryEngine with hallucination detection and blocking. Adds three
stages to the pipeline: retrieval gate (refuse if evidence too weak), prompt
hardening (inject grounding rules), and NLI verification (check claims
against source text). GroundedQueryResult carries grounding score, safety
flag, and claim-level verification breakdown.

**Reuse as-is:**
- Retrieval gate pattern (refuse to answer on weak evidence)
- GroundedQueryResult dataclass extension pattern
- Pipeline extension via subclass (zero blast radius to base engine)
- Source-bounded generation with 9-rule prompt

**Needs adaptation:**
- Replace NLI verification with code-specific verification (does the
  suggested code actually match patterns in the indexed codebase?)
- Adapt prompt rules for code generation context
- Gate thresholds tuned for code retrieval confidence levels

**Estimated effort:** Medium


### 7. src/core/chunker.py -- 250 lines

**Full path:** `D:\HybridRAG3\src\core\chunker.py`

**What it does:**
Text chunking with smart boundary detection. Splits text at natural break
points (paragraph > sentence > newline > hard cut) with configurable overlap.
Prepends section headings by looking backward up to 2000 chars to find the
nearest heading (ALL CAPS, numbered sections, colon-terminated lines).
ChunkerConfig dataclass controls size, overlap, and heading behavior.

**Reuse as-is:**
- ChunkerConfig dataclass
- Overlap logic (ensures facts spanning chunk boundaries appear in at least
  one chunk)
- Smart boundary detection heuristics

**Needs adaptation:**
- Add tree-sitter AST-based chunking for code (split at function/class
  boundaries instead of character counts)
- Language detection to select the right tree-sitter grammar
- Code-specific heading detection (function signatures, class declarations,
  module docstrings)
- Preserve indentation integrity within code chunks

**Estimated effort:** High (major new feature -- tree-sitter integration)


### 8. src/core/indexer.py -- 583 lines

**Full path:** `D:\HybridRAG3\src\core\indexer.py`

**What it does:**
Document indexing pipeline orchestrator. Scans folder, runs preflight
integrity checks (catches corrupt files before parsing), then processes each
file through parse -> chunk -> embed -> store. Block-based processing keeps
RAM stable on large files. Deterministic chunk IDs enable crash-safe
re-indexing via INSERT OR IGNORE. Hash-based change detection (size + mtime)
skips unchanged files on re-index.

**Reuse as-is:**
- Pipeline architecture (scan -> preflight -> parse -> chunk -> embed -> store)
- Block-based processing for large files
- Hash-based change detection (skip unchanged files)
- Crash-safe re-indexing via deterministic chunk IDs
- Progress tracking and error isolation (one bad file does not crash the run)

**Needs adaptation:**
- Add language detection (Python, Rust, Go, JS, etc.) as metadata
- Code-specific metadata extraction (imports, exports, class hierarchy)
- Deduplication across repos (same utility file in multiple projects)
- File-type-specific parsers (source code, markdown docs, config files)

**Estimated effort:** Medium


### 9. src/core/ibit.py -- 407 lines

**Full path:** `D:\HybridRAG3\src\core\ibit.py`

**What it does:**
Built-In Test engine borrowed from aviation/military hardware testing. Two
modes: IBIT (initial, 6 checks at startup, <500ms) and CBIT (continuous,
3 lightweight checks every 60s). Checks config validity, path existence,
database integrity, embedder health, router connectivity, and full pipeline
end-to-end. Each check is independent and catches its own exceptions.
IBITCheck dataclass carries name, pass/fail, detail, and elapsed time.

**Reuse as-is:**
- Entire IBIT/CBIT pattern and IBITCheck dataclass
- Independent check execution (one failure does not block others)
- Startup health verification sequence

**Needs adaptation:**
- Add JCoder-specific checks: vLLM health, FAISS GPU index status, scraper
  process alive, tree-sitter grammars installed, disk space for index
- Adjust CBIT interval and checks for CLI (no GUI status bar, log to file)

**Estimated effort:** Low


### 10. src/core/boot.py -- 365 lines

**Full path:** `D:\HybridRAG3\src\core\boot.py`

**What it does:**
Single entry point for system startup. Runs every validation step in
dependency order: load config -> resolve credentials -> validate config +
credentials together -> construct services (API client, Ollama client, etc.)
-> return ready-to-use instance. BootResult dataclass reports success,
available modes, warnings, and errors. Does not crash on missing API
credentials -- marks online mode unavailable and continues with offline.

**Reuse as-is:**
- Boot pipeline pattern (sequential validation with clear pass/fail)
- BootResult dataclass
- Graceful degradation (missing optional components do not block startup)
- Structured logging of every boot step

**Needs adaptation:**
- Simpler boot sequence (no cloud credentials, no network gate for
  offline-only operation)
- Add vLLM server health check and GPU detection
- Add tree-sitter grammar validation

**Estimated effort:** Low


### 11. src/security/credentials.py -- 784 lines

**Full path:** `D:\HybridRAG3\src\security\credentials.py`

**What it does:**
Canonical credentials module -- the one and only place that reads API
credentials. Three-tier resolution: Windows Credential Manager (keyring) ->
environment variables -> config file. Supports multiple env var aliases for
each credential type (key, endpoint, deployment, API version). ApiCredentials
dataclass records where each credential came from. Never logs actual key
values -- only masked previews.

**Reuse as-is:**
- Pattern only (resolution order, masking, source tracking)
- ApiCredentials dataclass structure

**Needs adaptation:**
- Simplify to only store a Claude API key (for distillation sessions)
- Remove Azure/OpenAI alias complexity
- Keep keyring + env var fallback chain

**Estimated effort:** Low


### 12. src/core/network_gate.py -- 567 lines

**Full path:** `D:\HybridRAG3\src\core\network_gate.py`

**What it does:**
Centralized network access control with three modes: offline (localhost
only), online (localhost + configured API endpoint), admin (unrestricted,
maintenance only). Every outbound connection must check the gate. Full audit
trail of allowed and denied connections with timestamp, URL, purpose, mode,
and result. Thread-safe singleton with NetworkMode enum.

**Reuse as-is:**
- Gate pattern and NetworkMode enum
- Audit trail logging (every connection attempt recorded)
- Thread-safe singleton pattern

**Needs adaptation:**
- Invert logic for scraper use case: whitelist for scraper target URLs
  instead of blocking unknown destinations
- Add scraper-specific modes (scrape session active vs. idle)
- Domain-level allow/block lists for web scraping

**Estimated effort:** Medium


### 13. src/core/exceptions.py -- 540 lines

**Full path:** `D:\HybridRAG3\src\core\exceptions.py`

**What it does:**
Typed exception hierarchy where every error has a clear name, message, and
fix suggestion. Base HybridRAGError carries `fix_suggestion` and `error_code`
fields plus a `to_dict()` method for JSON logging. Specific types include
AuthRejectedError, EndpointNotConfiguredError, ModelNotFoundError, and
others. Enables precise error handling: callers catch specific types and show
targeted messages.

**Reuse as-is:**
- HybridRAGError base class with fix_suggestion and error_code
- `to_dict()` serialization for structured logging
- Exception hierarchy pattern

**Needs adaptation:**
- Rename base class to JCoderError
- Replace HybridRAG-specific exception types with JCoder equivalents
  (ScraperError, IndexCorruptError, EvolutionFailedError, etc.)

**Estimated effort:** Low


### 14. src/monitoring/logger.py -- 144 lines

**Full path:** `D:\HybridRAG3\src\monitoring\logger.py`

**What it does:**
Structured JSON logging with zero external dependencies. StructuredLogger
wraps stdlib logging and accepts keyword arguments that get serialized to
JSON with timestamps. LoggerSetup manages daily log files. Includes
pre-built log entry builders for audit events (AuditLogEntry), cost events
(CostLogEntry), and query events (QueryLogEntry).

**Reuse as-is:**
- StructuredLogger class (JSON logging with kwargs)
- LoggerSetup with daily file rotation
- `get_app_logger()` factory function
- Log entry builder pattern

**Needs adaptation:**
- Add JCoder-specific log entry builders (ScrapeLogEntry, EvolutionLogEntry,
  IndexLogEntry)
- Minor: rename internal references

**Estimated effort:** Low


### 15. src/core/cost_tracker.py -- 563 lines

**Full path:** `D:\HybridRAG3\src\core\cost_tracker.py`

**What it does:**
Token and cost tracking with two-tier storage: in-memory list for instant
GUI access plus SQLite database for cross-session history. Singleton pattern
enforced by `get_cost_tracker()`. Listener pattern for live GUI updates.
CostEvent dataclass tracks session, profile, model, mode, tokens in/out,
cost in USD, data bytes, and latency. Auto-flush to SQLite every 30 seconds
and on shutdown. Rate storage per 1M tokens (industry standard).

**Reuse as-is:**
- CostEvent dataclass and recording pipeline
- SQLite storage schema and auto-flush mechanism
- Singleton and listener patterns
- Rate management (per 1M token pricing)

**Needs adaptation:**
- Track Claude API costs during distillation sessions
- Track local vLLM inference costs (compute time, GPU utilization)
- Remove GUI listener pattern (CLI-only)
- Add session-level cost roll-up for evolution cycle budgeting

**Estimated effort:** Low


---


## LimitlessApp Code (D:\Projects\KnowledgeBase\LimitlessApp -- 5 files, 2,718 lines)

### 16. claude_memory_bridge.py -- 1,024 lines

**Full path:** `D:\Projects\KnowledgeBase\LimitlessApp\claude_memory_bridge.py`

**What it does:**
Closes the human-out-of-the-loop cycle between Claude Code sessions and
persistent AI memory. Reads Claude Code JSONL session logs, extracts
decisions, actions, file changes, and problems using rule-based pattern
matching (zero LLM cost, deterministic output). SessionParser handles JSONL
format (user/assistant/tool messages). Bridge manifest tracks processed
sessions to avoid reprocessing. Output is compressed structured data written
to Claude Code memory directory.

**Reuse as-is:**
- SessionParser (JSONL parsing for Claude Code session format)
- Pattern matching extractors for decisions and file changes
- Bridge manifest for idempotent processing
- Deduplication logic

**Needs adaptation:**
- Target code-specific patterns: bug fixes, refactors, new features,
  dependency changes, test results
- Extract code blocks and diffs from session content
- Adapt output format for JCoder's knowledge base instead of Claude Code
  memory directory

**Estimated effort:** Medium


### 17. compaction_engine.py -- 627 lines

**Full path:** `D:\Projects\KnowledgeBase\LimitlessApp\compaction_engine.py`

**What it does:**
Cross-session intelligence synthesis. Takes raw session-by-session
extractions and compacts them into coherent institutional memory. Six
operations: deduplication, consolidation (merge related decisions into
themes), pattern detection (file hotspot analysis), temporal weighting
(recent sessions weighted higher), priority ranking, and briefing generation
under 200 lines. FileHotspotAnalyzer tracks which files get edited most
frequently. Achieved 17,000:1 compression ratio (250MB JSONL to 86 lines).

**Reuse as-is:**
- FileHotspotAnalyzer (high-churn file detection)
- Deduplication and consolidation pipeline
- Temporal weighting with configurable recency factor
- Briefing generation under line budget

**Needs adaptation:**
- Code-specific compaction: preserve code blocks, strip prose boilerplate
- Add function-level hotspot tracking (not just file-level)
- Adapt output format for scraper content summarization

**Estimated effort:** Low


### 18. evolution_loop.py -- 443 lines

**Full path:** `D:\Projects\KnowledgeBase\LimitlessApp\evolution_loop.py`

**What it does:**
Self-scoring feedback loop that makes the system evolve. Five-step cycle:
run pipeline -> score output quality -> decide what worked -> adjust
extraction patterns -> repeat. QualityScorer measures five dimensions:
specificity (names files, versions, tools), actionability (tells you what
to do), signal density (information per line), noise ratio (garbage that got
through), and coverage (fraction of sessions producing useful data). Score
history tracked in JSON for trend analysis.

**Reuse as-is:**
- QualityScorer class and five scoring dimensions
- Score history tracking with JSON persistence
- Feedback loop pattern (run -> score -> adjust -> repeat)
- Self-adjustment of extraction patterns based on scores

**Needs adaptation:**
- Add code-specific quality metrics: test pass rate, coverage delta, lint
  score improvement, complexity reduction
- Add evolution-specific scoring for the 10x Code Evolver (does the evolved
  code actually perform better?)
- Threshold tuning for code quality vs. knowledge quality

**Estimated effort:** Medium


### 19. deep_extractor.py -- 429 lines

**Full path:** `D:\Projects\KnowledgeBase\LimitlessApp\deep_extractor.py`

**What it does:**
LLM-powered deep extraction (Tier 2) that sits on top of rule-based
extraction. Adds six capabilities: intent extraction, forward action
generation, contradiction detection, theme synthesis, risk identification,
and knowledge gap detection. Minimal OllamaClient handles connection to
local Ollama with graceful degradation (falls back to Tier 1 if Ollama
unavailable). Auto-detects hardware tier (laptop vs. workstation).

**Reuse as-is:**
- OllamaClient with graceful degradation pattern
- Two-tier architecture (rule-based always runs, LLM enhances when available)
- Contradiction detection logic
- Hardware auto-detection

**Needs adaptation:**
- Code-specific deep analysis: complexity scoring, duplication detection,
  style consistency, architectural pattern recognition
- Switch from Ollama to vLLM for inference (or keep Ollama as fallback)
- Add code-aware prompts for each extraction dimension

**Estimated effort:** Medium


### 20. limitless_run.py -- 195 lines

**Full path:** `D:\Projects\KnowledgeBase\LimitlessApp\limitless_run.py`

**What it does:**
Master orchestrator that runs the entire pipeline end-to-end with a single
command. Five phases: bridge (JSONL extraction) -> compaction (cross-session
synthesis) -> deep extraction (LLM-powered, if available) -> evolution
scoring (quality measurement) -> memory write. Auto-detects hardware tier.
Error handling isolates phase failures (one broken phase does not block
others). Score history persisted to JSON. Designed for Windows Task
Scheduler, manual runs, or cron jobs.

**Reuse as-is:**
- Pipeline orchestrator pattern with phase isolation
- Hardware auto-detection
- Score history persistence
- Error collection and summary reporting
- Exit code conventions (0 = success, 1 = no input, 2 = partial failure)

**Needs adaptation:**
- New pipeline stages for JCoder: scrape -> summarize -> ingest -> evolve
- Replace Claude Code JSONL input with scraper output
- Add checkpoint/resume for long-running scraper sessions
- CLI argument parsing for manual overrides

**Estimated effort:** Low


---


## Summary Table

| #  | File                      | Source       | Lines | Reuse | Adapt | Effort |
|----|---------------------------|--------------|------:|------:|------:|--------|
| 1  | llm_router.py             | HybridRAG3   | 1,812 |   900 |   900 | Medium |
| 2  | vector_store.py           | HybridRAG3   |   653 |   500 |   150 | Low    |
| 3  | embedder.py               | HybridRAG3   |   210 |   150 |    60 | Low    |
| 4  | config.py                 | HybridRAG3   |   853 |   400 |   450 | Medium |
| 5  | query_engine.py           | HybridRAG3   |   507 |   300 |   200 | Medium |
| 6  | grounded_query_engine.py  | HybridRAG3   |   462 |   250 |   210 | Medium |
| 7  | chunker.py                | HybridRAG3   |   250 |   100 |   150 | High   |
| 8  | indexer.py                | HybridRAG3   |   583 |   350 |   230 | Medium |
| 9  | ibit.py                   | HybridRAG3   |   407 |   300 |   100 | Low    |
| 10 | boot.py                   | HybridRAG3   |   365 |   250 |   115 | Low    |
| 11 | credentials.py            | HybridRAG3   |   784 |   200 |   100 | Low    |
| 12 | network_gate.py           | HybridRAG3   |   567 |   250 |   300 | Medium |
| 13 | exceptions.py             | HybridRAG3   |   540 |   200 |   100 | Low    |
| 14 | logger.py                 | HybridRAG3   |   144 |   130 |    14 | Low    |
| 15 | cost_tracker.py           | HybridRAG3   |   563 |   350 |   200 | Low    |
| 16 | claude_memory_bridge.py   | LimitlessApp | 1,024 |   400 |   600 | Medium |
| 17 | compaction_engine.py      | LimitlessApp |   627 |   400 |   225 | Low    |
| 18 | evolution_loop.py         | LimitlessApp |   443 |   200 |   240 | Medium |
| 19 | deep_extractor.py         | LimitlessApp |   429 |   200 |   225 | Medium |
| 20 | limitless_run.py          | LimitlessApp |   195 |   150 |    45 | Low    |

| Source       | Files | Lines  | Reuse As-Is | Needs Adapt | New Code |
|--------------|------:|-------:|------------:|------------:|---------:|
| HybridRAG3   |    15 |  8,700 |       4,130 |       3,279 |   ~1,300 |
| LimitlessApp |     5 |  2,718 |       1,350 |       1,335 |     ~400 |
| **Total**    |**20** |**11,418** | **5,480** |   **4,614** |**~1,700**|

Estimated time saved: 60-70% vs building from scratch. The RAG pipeline
(files 1-10) and evolution loop (files 16-20) represent months of iteration,
debugging, and production hardening.


---


## Adaptation Priority

Files ranked by port order based on dependency chains. Nothing in a later
tier can work without the earlier tiers being in place.

### Tier 1 -- Foundation (port first, everything depends on these)

| Priority | File              | Why first                                      |
|----------|-------------------|-------------------------------------------------|
| 1        | config.py         | Every module imports config                      |
| 2        | logger.py         | Every module logs                                |
| 3        | exceptions.py     | Every module raises typed errors                 |

### Tier 2 -- Storage Layer (needed before any indexing or querying)

| Priority | File              | Why this order                                   |
|----------|-------------------|-------------------------------------------------|
| 4        | embedder.py       | Needed by both indexer and search                |
| 5        | vector_store.py   | Storage backend for chunks and embeddings        |
| 6        | chunker.py        | Produces chunks that vector_store stores         |

### Tier 3 -- Pipeline (depends on Tier 2)

| Priority | File              | Why this order                                   |
|----------|-------------------|-------------------------------------------------|
| 7        | indexer.py        | Depends on embedder, vector_store, chunker       |
| 8        | llm_router.py     | LLM backend needed before query engine           |
| 9        | query_engine.py   | Depends on vector_store, embedder, llm_router    |
| 10       | grounded_query_engine.py | Extends query_engine                      |

### Tier 4 -- System Health (can be added after core pipeline works)

| Priority | File              | Why this order                                   |
|----------|-------------------|-------------------------------------------------|
| 11       | boot.py           | Wires everything together at startup             |
| 12       | ibit.py           | Health checks need all components to exist       |
| 13       | credentials.py    | Only needed if Claude distillation is configured |
| 14       | network_gate.py   | Only needed when scraper is added                |
| 15       | cost_tracker.py   | Nice-to-have, not blocking for core function     |

### Tier 5 -- Evolution Cycle (weekly pipeline, port last)

| Priority | File                    | Why this order                             |
|----------|-------------------------|--------------------------------------------|
| 16       | compaction_engine.py    | Standalone, no dependencies on other LA files |
| 17       | evolution_loop.py       | Scoring needs compacted data               |
| 18       | claude_memory_bridge.py | Extraction patterns adapt last             |
| 19       | deep_extractor.py       | Enhancement layer, needs base extraction   |
| 20       | limitless_run.py        | Orchestrator, needs all other phases       |


---


## Key Dependencies Between Files

```
config.py
  |
  +-- logger.py
  +-- exceptions.py
  |
  +-- embedder.py ------+
  |                      |
  +-- vector_store.py ---+-- indexer.py
  |                      |
  +-- chunker.py --------+
  |
  +-- llm_router.py ----+-- query_engine.py -- grounded_query_engine.py
  |
  +-- boot.py (wires all of the above)
  +-- ibit.py (tests all of the above)
  |
  +-- credentials.py (optional, for Claude API key)
  +-- network_gate.py (optional, for scraper URL control)
  +-- cost_tracker.py (optional, for cost visibility)

compaction_engine.py (standalone)
  |
  +-- evolution_loop.py (depends on compaction output)
  |
  +-- claude_memory_bridge.py (feeds compaction)
  +-- deep_extractor.py (enhances extraction)
  |
  +-- limitless_run.py (orchestrates all above)
```


---


## What Is NOT Reused (must be built new for JCoder)

These capabilities do not exist in either source project and must be written
from scratch:

1. **Tree-sitter AST chunking** -- code-aware splitting at function/class
   boundaries. Integrates with chunker.py but the AST logic is new.

2. **Web scraper** -- fetches documentation from configured URLs on a
   schedule. No scraper exists in either project.

3. **CLI interface** -- JCoder is CLI-only. HybridRAG3 has a tkinter GUI,
   LimitlessApp has a tkinter GUI. Neither has a CLI.

4. **Code evolution engine** -- the 10x Code Evolver that runs the
   generate -> test -> score -> improve loop on actual code. evolution_loop.py
   provides the scoring framework but the code generation and testing
   integration is new.

5. **Multi-language support** -- language detection, per-language embedding
   prefixes, language-specific chunking rules. Neither project handles
   multiple programming languages.

6. **vLLM tensor parallel config** -- multi-GPU inference configuration for
   the home workstation (dual RTX 3090). llm_router.py has a VLLMRouter
   stub but no tensor parallel support.

Estimated new code: ~1,700 lines across these six areas.
