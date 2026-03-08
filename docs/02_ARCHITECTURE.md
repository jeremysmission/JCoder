# JCoder Architecture

Document: 02_ARCHITECTURE.md
Version: RevB (2026-03-07)
Status: Mixed -- sections marked IMPLEMENTED or ROADMAP

Legend:
- [IMPLEMENTED] -- code exists and tests pass
- [PARTIAL] -- core exists, details differ from original design
- [ROADMAP] -- design only, not yet built

---

## Table of Contents

1. [System Overview](#system-overview) [PARTIAL]
2. [System Diagram](#system-diagram) [PARTIAL]
3. [Subsystem 1 -- CLI Agent](#subsystem-1----cli-agent) [IMPLEMENTED]
4. [Subsystem 2 -- RAG Engine](#subsystem-2----rag-engine) [PARTIAL]
5. [Subsystem 3 -- Weekly Scraper](#subsystem-3----weekly-scraper) [ROADMAP]
6. [Subsystem 4 -- 10x Code Evolver](#subsystem-4----10x-code-evolver) [ROADMAP]
7. [Data Flow -- User Query](#data-flow----user-query) [IMPLEMENTED]
8. [Data Flow -- Weekly Scraper Cycle](#data-flow----weekly-scraper-cycle) [ROADMAP]
9. [Data Flow -- 10x Evolution Cycle](#data-flow----10x-evolution-cycle) [ROADMAP]
10. [Process Architecture](#process-architecture) [PARTIAL]
11. [Storage Layout](#storage-layout) [PARTIAL]
12. [Model Inventory](#model-inventory) [PARTIAL]
13. [Reusable Code Mapping](#reusable-code-mapping) [PARTIAL]
14. [Component Interaction Matrix](#component-interaction-matrix) [PARTIAL]

---

## Implementation Status (2026-03-07)

What exists today (~8,000+ lines, 788 tests, 0 failures):

| Component | Status | Location |
|-----------|--------|----------|
| CLI Agent (Click + Rich) | IMPLEMENTED | `cli/`, `agent/core.py` |
| Agent Tools (read/write/edit/bash/glob/grep/git) | IMPLEMENTED | `agent/tools.py` |
| Agent Bridge (self-learning pipeline) | IMPLEMENTED | `agent/bridge.py` |
| Telemetry + Feedback Loop | IMPLEMENTED | `core/telemetry.py`, `cli/agent_cmd.py` |
| Experience Replay | IMPLEMENTED | `core/experience_replay.py` |
| Meta-Cognitive Controller (Thompson Sampling) | IMPLEMENTED | `core/meta_cognitive.py` |
| Active Learner (uncertainty + committee) | IMPLEMENTED | `core/active_learner.py` |
| Quality-Diversity Archive (MAP-Elites) | IMPLEMENTED | `core/quality_diversity.py`, `agent/bridge.py` |
| Agent Memory | IMPLEMENTED | `agent/memory.py` |
| FTS5 Index + Federated Search | IMPLEMENTED | `core/index_engine.py`, `core/federated_search.py` |
| Index Discovery (auto-scan FTS5 DBs) | IMPLEMENTED | `core/index_discovery.py` |
| Dual LLM Backend (Ollama + Anthropic) | IMPLEMENTED | `agent/llm_backend.py` |
| 8 Query Profiles + Prompt Modes | IMPLEMENTED | `agent/prompts.py`, `config/` |
| 200-Question Eval Set + Scorer | IMPLEMENTED | `evaluation/` |
| Session Persistence | IMPLEMENTED | `agent/session.py` |
| FIM Code Completion | IMPLEMENTED | `agent/prompts.py`, `cli/agent_cmd.py` |
| Data Ingestion Pipeline | IMPLEMENTED | `ingestion/` |
| RAG Engine (retrieval + generation) | PARTIAL | `core/retrieval_engine.py` |
| FAISS GPU Vector Store | NOT YET | design only |
| AST Chunker (tree-sitter) | NOT YET | design only |
| Reranker Integration | NOT YET | design only |
| Weekly Scraper | NOT YET | design only |
| 10x Code Evolver | NOT YET | design only |
| QLoRA Fine-Tuning | NOT YET | design only |
| vLLM Serving | NOT YET | design only (current: Ollama) |

Current model stack (Ollama-served, not vLLM):
- Primary: Devstral Small 2 24B (Apache 2.0, 256K ctx) -- BEAST only
- Code embedder: nomic-embed-code (768-dim) -- BEAST only
- Toaster fallback: phi4-mini (3.8B), FTS5-only search

---

## System Overview

JCoder is a fully local, offline, free CLI AI coding assistant. It runs entirely
on a single workstation with no cloud dependencies, no API keys, no telemetry,
and no internet requirement after initial model downloads.

Four subsystems cooperate:

| # | Subsystem        | Purpose                              | Runtime      |
|---|------------------|--------------------------------------|--------------|
| 1 | CLI Agent        | User-facing REPL, tool execution     | Interactive  |
| 2 | RAG Engine       | Code-aware retrieval + generation    | Library      |
| 3 | Weekly Scraper   | Autonomous knowledge ingestion       | Scheduled    |
| 4 | 10x Code Evolver | Self-improvement loop + fine-tuning  | Scheduled    |

Hardware target: 128 GB RAM, 48 GB GPU (dual RTX 3090 NVLink), 2 TB NVMe,
Windows 11, 2 Gbps internet.

---

## System Diagram

```
+=====================================================================+
|                         JCoder System                                |
+=====================================================================+
|                                                                      |
|  +---------------------+        +------------------------------+     |
|  |   [1] CLI Agent     |        |   [2] RAG Engine             |     |
|  |                     |        |                              |     |
|  |  Click + Rich +     |  query |  AST Chunker (tree-sitter)  |     |
|  |  prompt_toolkit     |------->|  Embedder (Nomic 7B/vLLM)   |     |
|  |                     |        |  FAISS GPU + SQLite FTS5     |     |
|  |  Tools:             |  cited |  Hybrid Search (RRF)         |     |
|  |   Read, Write,      |<-------|  Reranker (Qwen3-4B)        |     |
|  |   Edit, Bash,       |  resp  |  Query Engine                |     |
|  |   Glob, Grep, Git   |        |                              |     |
|  |                     |        +-----+------------------------+     |
|  |  Sliding context    |              |                              |
|  |  window (multi-turn)|              | index                        |
|  +---------------------+              |                              |
|           |                           v                              |
|           |  exec              +------------------------------+      |
|           +--- bash --->       |   Local Filesystem           |      |
|           |                    |   (code, tests, projects)    |      |
|           |  read/write        +------------------------------+      |
|           +--- files --->             ^                              |
|                                       |                              |
|  +---------------------+             | ingest                        |
|  | [3] Weekly Scraper  |             |                              |
|  |                     |-------------+                              |
|  |  GitHub Trending    |  summaries + embeddings                    |
|  |  Hacker News        |                                            |
|  |  Reddit (r/prog)    |                                            |
|  |  arXiv CS           |                                            |
|  |  Release Notes      |                                            |
|  |                     |                                            |
|  |  Dedup Guard        |                                            |
|  |  LLM Summarizer     |                                            |
|  +---------------------+                                            |
|                                                                      |
|  +---------------------+        +------------------------------+     |
|  | [4] 10x Evolver    |        |   vLLM Server (TP=2)         |     |
|  |                     |  gen   |                              |     |
|  |  Test Runner        |------->|  Qwen3-Coder-Next 80B MoE   |     |
|  |  Weakness Analyzer  |        |  ~37 GB Q3, ~71 tok/s       |     |
|  |  Improvement Gen    |<-------|                              |     |
|  |  Regression Guard   |  resp  |  Nomic Embed Code 7B        |     |
|  |  10x Repeat Loop    |        |  ~4.4 GB Q4, 768 dims       |     |
|  |                     |        |                              |     |
|  |  QLoRA Fine-Tuning  |        |  Qwen3-Reranker-4B          |     |
|  |  Claude Distillation|        |  ~3 GB                      |     |
|  +---------------------+        +------------------------------+     |
|                                                                      |
+======================================================================+
```

Data flow summary:

```
User --[natural language]--> CLI Agent --[query]--> RAG Engine
                                                       |
                                          retrieve + rerank + generate
                                                       |
                                                       v
User <--[cited response]--- CLI Agent <--[answer]-- LLM (vLLM)

Weekly Scraper --[summaries]--> RAG Engine index (background, scheduled)
10x Evolver --[test+improve]--> codebase (background, scheduled)
```

---

## Subsystem 1 -- CLI Agent [IMPLEMENTED]

> **Status**: Core agent loop, tools, bridge, sessions, profiles all implemented.
> Actual implementation uses Click + Rich (no prompt_toolkit REPL yet).
> Tools match the design. Context window management implemented via token budgets.

### Purpose

The CLI Agent is the user-facing interface. The user types natural language
instructions. The agent decomposes them into tool calls, executes code changes,
runs tests, and reports results. It maintains a multi-turn conversation with
a sliding context window to stay within the LLM's token budget.

### Components

| Component         | Library          | Role                                    |
|-------------------|------------------|-----------------------------------------|
| REPL Loop         | prompt_toolkit   | Input handling, history, keybindings     |
| Output Renderer   | Rich             | Syntax highlighting, tables, panels      |
| CLI Framework     | Click            | Argument parsing, subcommands            |
| Conversation Mgr  | custom           | Multi-turn history, sliding window       |
| Tool Registry     | custom           | Tool definitions, argument schemas       |
| Planner           | LLM-driven       | Decides which tools to call and in what order |

### Tools

| Tool   | Description                                        |
|--------|----------------------------------------------------|
| Read   | Read file contents by path                         |
| Write  | Write or overwrite a file                          |
| Edit   | Apply targeted string replacement to a file        |
| Bash   | Execute a shell command and capture output          |
| Glob   | Fast file pattern matching across the codebase      |
| Grep   | Regex content search across files                   |
| Git    | Git operations (status, diff, commit, branch, log) |

### Context Window Management

The agent maintains a sliding window over conversation history. When the total
token count approaches the LLM context limit, the oldest turns are summarized
and compressed. The current strategy:

1. Keep the system prompt (always present, never evicted)
2. Keep the last N turns verbatim (configurable, default 10)
3. Summarize evicted turns into a running "session summary" block
4. Tool outputs are truncated to a configurable max length before storage

This keeps the LLM focused on the current task while retaining awareness of
earlier decisions made in the session.

### Agent Loop Pseudocode

```
while True:
    user_input = prompt()
    conversation.append(role="user", content=user_input)

    while not done:
        response = llm.generate(conversation, tools=tool_registry)

        if response.has_tool_calls:
            for call in response.tool_calls:
                result = execute_tool(call)
                conversation.append(role="tool", content=result)
        else:
            display(response.text)
            conversation.append(role="assistant", content=response.text)
            done = True

    conversation.trim_to_window()
```

---

## Subsystem 2 -- RAG Engine [PARTIAL]

> **Status**: FTS5 search, federated search with RRF, retrieval engine, and index
> discovery all implemented. FAISS GPU, AST chunker (tree-sitter), reranker, and
> vLLM embedding server are NOT yet built. Current search is FTS5-only with keyword
> fallback. Embedding requires Ollama nomic-embed-code (available on BEAST only).

### Purpose

Code-aware retrieval-augmented generation. When the CLI Agent needs to answer
a question about the codebase, understand a library, or reference scraped
knowledge, it queries the RAG Engine. The RAG Engine retrieves relevant chunks,
reranks them, and generates a cited answer.

### Pipeline Architecture

```
Query
  |
  v
+------------------+
| Embed query      |  Nomic Embed Code 7B via vLLM
+------------------+
  |
  v
+------------------+    +------------------+
| Dense search     |    | Sparse search    |
| (FAISS GPU)      |    | (SQLite FTS5)    |
+------------------+    +------------------+
  |                       |
  v                       v
+-------------------------------------+
| Reciprocal Rank Fusion (RRF)        |
| Merge dense + sparse ranked lists   |
+-------------------------------------+
  |
  v
+------------------+
| Reranker         |  Qwen3-Reranker-4B (second-pass scoring)
| (top-K -> top-N) |
+------------------+
  |
  v
+------------------+
| Build prompt     |  Context chunks + system rules + user query
+------------------+
  |
  v
+------------------+
| LLM Generate     |  Qwen3-Coder-Next 80B via vLLM
+------------------+
  |
  v
+------------------+
| Citation attach  |  Map claims back to source chunks
+------------------+
  |
  v
Answer with citations
```

### Components

**AST Chunker (tree-sitter)**

Unlike the HybridRAG3 text chunker that splits at paragraph/sentence boundaries,
the JCoder AST chunker parses source code into an Abstract Syntax Tree and
splits at function and class boundaries. This means each chunk is a semantically
complete unit of code -- a whole function, a whole class, a whole module-level
block -- never a fragment cut mid-statement.

Tree-sitter supports 100+ languages. Priority targets: Python, JavaScript,
TypeScript, Rust, Go, Java, C, C++.

Fallback: for files tree-sitter cannot parse (config files, markdown, plain
text), the system falls back to the HybridRAG3 text chunker with heading
detection and overlap.

**Embedder: Nomic Embed Code 7B**

- Served by vLLM on localhost (not Ollama -- vLLM handles batched inference
  more efficiently for large indexing jobs)
- 768-dimensional embeddings optimized for code semantics
- Q4 quantization, approximately 4.4 GB VRAM
- Batched embedding for indexing throughput

**Vector Store: FAISS GPU + SQLite FTS5 Hybrid**

Two storage backends working in parallel:

| Backend      | What it stores              | Good at                     |
|--------------|-----------------------------|-----------------------------|
| FAISS (GPU)  | Embedding vectors (float16) | Semantic similarity search  |
| SQLite FTS5  | Chunk text + metadata       | Keyword/exact-match search  |

FAISS runs on GPU for sub-millisecond nearest-neighbor search across hundreds
of thousands of vectors. SQLite FTS5 provides BM25 keyword matching for exact
identifiers, function names, error messages, and other literal strings that
semantic search misses.

**Hybrid Search: Reciprocal Rank Fusion (RRF)**

Both FAISS and FTS5 return ranked lists. RRF merges them:

```
RRF_score(doc) = sum( 1 / (k + rank_i(doc)) ) for each ranker i
```

Where k is a constant (default 60). This gives a unified ranking that captures
both semantic relevance and keyword relevance without requiring score
normalization between heterogeneous backends.

**Reranker: Qwen3-Reranker-4B**

After hybrid search returns the top-K candidates (default K=50), the reranker
rescores each candidate against the original query using cross-attention. This
is more accurate than embedding similarity alone because it sees the query and
document together, not as independent vectors.

- 81.20 MTEB-Code benchmark score
- Approximately 3 GB VRAM
- Reduces top-50 to top-N (default N=10) for context injection

**Query Engine**

Orchestrates the full pipeline: embed query, search, fuse, rerank, build
prompt, generate, attach citations. Ported from HybridRAG3's
grounded_query_engine.py with modifications for code-specific prompting.

---

## Subsystem 3 -- Weekly Scraper [ROADMAP]

> **Status**: Design only. No code exists. Depends on BEAST hardware and vLLM
> serving. Deferred until core agent + RAG pipeline is validated on BEAST.

### Purpose

Autonomous agent that runs once per week (Windows Task Scheduler) to ingest
fresh knowledge from the software development ecosystem. This keeps JCoder's
RAG index current with new libraries, breaking changes, security advisories,
and best practices without manual curation.

### Sources

| Source            | What it captures                          | Method        |
|-------------------|-------------------------------------------|---------------|
| GitHub Trending   | Popular new repos, rising projects        | GitHub API    |
| Hacker News       | Top stories, show HN, ask HN             | HN API        |
| Reddit            | r/programming, r/python, r/rust, etc.    | Reddit API    |
| arXiv             | CS papers (cs.CL, cs.SE, cs.AI)          | arXiv API     |
| Release Notes     | Major framework releases (Python, Node,  | RSS/Atom      |
|                   | Rust, Go, etc.)                          |               |

### Pipeline

```
Scrape sources (API calls)
  |
  v
Dedup guard (hash-based, prevents re-ingestion of seen URLs/DOIs)
  |
  v
LLM summarization (Qwen3-Coder-Next 80B)
  |  - Condense each item to 200-500 word technical summary
  |  - Extract: title, topic tags, language tags, key takeaways
  |
  v
AST chunk (for code snippets) + text chunk (for prose)
  |
  v
Embed (Nomic Embed Code 7B)
  |
  v
Index into FAISS + SQLite FTS5
  |
  v
Log summary report (what was added, what was deduped, errors)
```

### Dedup Guard

Every scraped item gets a deterministic content hash (SHA-256 of URL + title +
date). Before processing, the hash is checked against a persistent SQLite table.
If found, the item is skipped. This prevents the same trending repo from being
re-ingested every week it stays on the trending page.

### Schedule

- Trigger: Windows Task Scheduler, weekly (configurable day/time)
- Runtime: approximately 15-30 minutes depending on volume
- Network: requires internet access (the only subsystem that does)
- Failure mode: log errors, skip failed sources, index whatever succeeded

---

## Subsystem 4 -- 10x Code Evolver [ROADMAP]

> **Status**: Design only. No code exists. Depends on BEAST hardware, vLLM,
> and QLoRA training. Deferred to Sprint 7+.

### Purpose

Self-improving code loop that autonomously finds weaknesses in JCoder's own
codebase and generates targeted improvements. Ported from LimitlessApp's
evolution architecture with the addition of QLoRA fine-tuning for model
specialization.

### Evolution Cycle (10 iterations per run)

```
+---> Run full test suite
|       |
|       v
|     Analyze results
|       |  - Identify failing tests
|       |  - Identify slow tests
|       |  - Identify untested code paths
|       |  - Identify code quality issues
|       v
|     Generate improvements (LLM)
|       |  - Target weakest areas first
|       |  - Produce concrete code patches
|       v
|     Apply improvements
|       |
|       v
|     Verify (run tests again)
|       |  - If regression: revert and try different approach
|       |  - If pass: keep improvement, record delta
|       v
+---- Repeat (10x per cycle)
```

### Distillation Pipeline

Weekly distillation from Claude Max ($200/month subscription) generates
high-quality training pairs for local model fine-tuning:

```
Claude Max (cloud, $200/mo)
  |
  |  Send complex coding tasks
  |  Receive expert-quality responses
  |
  v
Training pair extraction
  |  - (prompt, response) pairs
  |  - Quality filtering (only keep high-scoring pairs)
  |  - Dedup against existing training set
  |
  v
QLoRA fine-tuning (local, dual RTX 3090)
  |  - Base: Qwen3-Coder-Next 80B
  |  - Adapter: 4-bit quantized LoRA layers
  |  - Training data: accumulated distillation pairs
  |
  v
Merge adapter into serving model
  |
  v
Regression test (must not degrade existing benchmarks)
  |
  v
Deploy updated model to vLLM
```

### Components

| Component          | Origin       | Role                                     |
|--------------------|--------------|------------------------------------------|
| Test Runner        | new          | Execute pytest, capture structured results|
| Weakness Analyzer  | LimitlessApp | Parse test results, rank improvement targets |
| Improvement Gen    | LimitlessApp | LLM-driven code generation for fixes     |
| Regression Guard   | new          | Automated revert on test failure          |
| Distillation Agent | new          | Claude Max API interaction, pair extraction|
| QLoRA Trainer      | new          | Fine-tuning pipeline using transformers + PEFT |

---

## Data Flow -- User Query [IMPLEMENTED]

> **Status**: Implemented. Actual flow uses Ollama (not vLLM) and FTS5 (not FAISS).
> No reranker step yet. Otherwise matches the design.

Step-by-step trace of a user asking a question in the CLI.

```
Step 1: User types "How does the authentication middleware work?"
           |
Step 2: CLI Agent appends to conversation history
           |
Step 3: Agent sends conversation to LLM (Qwen3-Coder-Next 80B via vLLM)
           |
Step 4: LLM decides to use RAG tool (queries the codebase)
           |
Step 5: RAG Engine receives query text
           |
Step 6: Embedder encodes query -> 768-dim vector (Nomic Embed Code 7B)
           |
Step 7: FAISS GPU returns top-50 dense matches (cosine similarity)
           |
Step 8: SQLite FTS5 returns top-50 sparse matches (BM25 keyword)
           |
Step 9: RRF merges both lists into unified top-50 ranking
           |
Step 10: Reranker (Qwen3-Reranker-4B) rescores top-50 -> top-10
           |
Step 11: Query Engine builds prompt:
           [system rules] + [top-10 chunks with source paths] + [user query]
           |
Step 12: LLM generates answer with inline citations [1], [2], ...
           |
Step 13: CLI Agent receives response, appends to conversation
           |
Step 14: Rich renders formatted answer with syntax-highlighted code blocks
           |
Step 15: User sees answer. Conversation continues (multi-turn).
```

Latency budget (estimated):

| Step          | Target      |
|---------------|-------------|
| Embed query   | ~20 ms      |
| FAISS search  | ~5 ms       |
| FTS5 search   | ~10 ms      |
| RRF merge     | ~1 ms       |
| Rerank (4B)   | ~200 ms     |
| LLM generate  | ~2-5 s      |
| **Total**     | **~2.5-5.5 s** |

---

## Data Flow -- Weekly Scraper Cycle [ROADMAP]

> **Status**: Design only. No code exists.

```
Step 1:  Windows Task Scheduler triggers scraper (e.g., Sunday 02:00)
            |
Step 2:  Scraper fetches GitHub Trending API
            |  - Top 25 repos per language (Python, Rust, Go, TS, JS)
            |  - Extracts: name, description, stars, language, README snippet
            |
Step 3:  Scraper fetches Hacker News top stories (past 7 days)
            |  - Top 50 stories with comments
            |  - Extracts: title, URL, top comments, discussion summary
            |
Step 4:  Scraper fetches Reddit posts (past 7 days)
            |  - Subreddits: programming, python, rust, golang, webdev
            |  - Top 20 posts per subreddit
            |
Step 5:  Scraper fetches arXiv papers (past 7 days)
            |  - Categories: cs.CL, cs.SE, cs.AI, cs.PL
            |  - Extracts: title, abstract, authors, arxiv ID
            |
Step 6:  Scraper fetches release notes via RSS
            |  - Python, Node.js, Rust, Go, major frameworks
            |  - Extracts: version, changelog highlights
            |
Step 7:  Dedup guard checks all items against seen-items SQLite table
            |  - Hash: SHA-256(URL + title + date)
            |  - Skip already-seen items
            |
Step 8:  LLM summarizes each new item (Qwen3-Coder-Next 80B)
            |  - 200-500 word technical summary per item
            |  - Structured output: title, tags, languages, takeaways
            |
Step 9:  Chunker processes summaries
            |  - Code snippets: AST chunker (tree-sitter)
            |  - Prose: text chunker with heading detection
            |
Step 10: Embedder encodes all chunks (Nomic Embed Code 7B, batched)
            |
Step 11: Chunks + embeddings indexed into FAISS + SQLite FTS5
            |
Step 12: Dedup guard records all processed item hashes
            |
Step 13: Summary report logged:
            |  - Items processed: N
            |  - Items deduped: M
            |  - Errors: K
            |  - Index size delta: +X chunks
            |
Step 14: Process exits. Next run in 7 days.
```

---

## Data Flow -- 10x Evolution Cycle [ROADMAP]

> **Status**: Design only. No code exists.

```
Step 1:  Trigger (manual or scheduled)
            |
Step 2:  Snapshot current codebase state (git stash or branch)
            |
Step 3:  === BEGIN ITERATION 1 of 10 ===
            |
Step 4:  Run full test suite (pytest)
            |  - Capture: pass/fail per test, duration, coverage
            |
Step 5:  Weakness Analyzer parses results
            |  - Failing tests -> highest priority
            |  - Slow tests (>1s) -> optimization targets
            |  - Uncovered code paths -> test generation targets
            |  - Code smells (long functions, deep nesting) -> refactor targets
            |
Step 6:  Improvement Generator (LLM) produces patches
            |  - Reads weakness report + relevant source files via RAG
            |  - Generates concrete code changes (not suggestions)
            |  - One patch per weakness, ranked by expected impact
            |
Step 7:  Apply top patch to codebase
            |
Step 8:  Run full test suite again
            |  - If all tests pass and no regression: KEEP
            |  - If any test regresses: REVERT, try next patch
            |
Step 9:  Record delta (what changed, what improved, what was tried)
            |
Step 10: === END ITERATION 1, BEGIN ITERATION 2 ===
            |  (repeat Steps 4-9 ten times)
            |
Step 11: After 10 iterations:
            |  - Commit all kept improvements
            |  - Generate evolution report (what improved, what failed)
            |
Step 12: === WEEKLY DISTILLATION (separate schedule) ===
            |
Step 13: Send batch of complex coding tasks to Claude Max API
            |  - Tasks drawn from: recent user queries, known weak areas,
            |    coding challenges, code review scenarios
            |
Step 14: Receive expert responses, extract (prompt, response) pairs
            |
Step 15: Quality filter: keep only pairs scoring above threshold
            |
Step 16: Add to QLoRA training dataset
            |  - Dedup against existing training pairs
            |
Step 17: Run QLoRA fine-tuning on dual RTX 3090
            |  - Base model: Qwen3-Coder-Next 80B
            |  - 4-bit QLoRA adapters
            |  - Training: ~2-4 hours depending on dataset size
            |
Step 18: Regression test fine-tuned model against benchmark suite
            |  - Must match or exceed base model scores
            |
Step 19: If pass: merge adapter, deploy to vLLM
            |  If fail: discard adapter, log failure analysis
            |
Step 20: Process complete.
```

---

## Process Architecture [PARTIAL]

> **Status**: Current reality: single-process CLI agent with Ollama as model server.
> vLLM tensor-parallel setup is ROADMAP (requires BEAST hardware). No separate
> embedder or reranker server processes yet.

JCoder runs as multiple processes that communicate via localhost HTTP and
filesystem:

```
+--------------------------+
| vLLM Server (TP=2)       |  Long-running daemon
|  Port: 8000              |  Serves: Qwen3-Coder-Next 80B
|  GPU: both RTX 3090      |  API: OpenAI-compatible /v1/chat/completions
+--------------------------+

+--------------------------+
| vLLM Server (embedder)   |  Long-running daemon
|  Port: 8001              |  Serves: Nomic Embed Code 7B
|  GPU: shared (4.4 GB)    |  API: OpenAI-compatible /v1/embeddings
+--------------------------+

+--------------------------+
| vLLM Server (reranker)   |  Long-running daemon (or loaded on-demand)
|  Port: 8002              |  Serves: Qwen3-Reranker-4B
|  GPU: shared (3 GB)      |  API: /v1/rerank or custom endpoint
+--------------------------+

+--------------------------+
| JCoder CLI Agent         |  Interactive process (foreground)
|  PID: user-started       |  Talks to: vLLM (8000), RAG Engine (in-process)
|  stdin/stdout: terminal  |  Embedder client -> port 8001
+--------------------------+

+--------------------------+
| Weekly Scraper           |  Scheduled process (Task Scheduler)
|  Runs: once/week         |  Talks to: vLLM (8000), port 8001
|  Duration: 15-30 min     |  Writes to: FAISS index, SQLite
+--------------------------+

+--------------------------+
| 10x Evolver              |  Scheduled or manual process
|  Runs: on-demand/weekly  |  Talks to: vLLM (8000), port 8001
|  Duration: 1-4 hours     |  Reads/writes: codebase, test results
+--------------------------+
```

### Inter-Process Communication

| From             | To            | Protocol              | Purpose              |
|------------------|---------------|-----------------------|----------------------|
| CLI Agent        | vLLM (8000)   | HTTP (OpenAI API)     | LLM generation       |
| CLI Agent        | vLLM (8001)   | HTTP (OpenAI API)     | Query embedding      |
| CLI Agent        | vLLM (8002)   | HTTP                  | Reranking            |
| CLI Agent        | FAISS index   | In-process (Python)   | Vector search        |
| CLI Agent        | SQLite        | In-process (Python)   | FTS5 + metadata      |
| Weekly Scraper   | vLLM (8000)   | HTTP (OpenAI API)     | Summarization        |
| Weekly Scraper   | vLLM (8001)   | HTTP (OpenAI API)     | Chunk embedding      |
| Weekly Scraper   | FAISS index   | In-process (Python)   | Index writes         |
| Weekly Scraper   | SQLite        | In-process (Python)   | Index writes         |
| 10x Evolver      | vLLM (8000)   | HTTP (OpenAI API)     | Code generation      |
| 10x Evolver      | filesystem    | Bash (subprocess)     | Test execution       |
| 10x Evolver      | Git           | Bash (subprocess)     | Version control      |

### Startup Order

1. Start vLLM server (LLM) -- loads Qwen3-Coder-Next 80B with TP=2
2. Start vLLM server (embedder) -- loads Nomic Embed Code 7B
3. Start vLLM server (reranker) -- loads Qwen3-Reranker-4B (optional, on-demand)
4. Health check all three endpoints (IBIT)
5. Start JCoder CLI Agent (interactive)

Scraper and Evolver are independent scheduled tasks that assume vLLM servers
are already running. If servers are down, they fail fast with a clear error.

---

## Storage Layout [PARTIAL]

> **Status**: Actual layout differs from design. Real code lives in `agent/`,
> `core/`, `cli/`, `ingestion/`, `evaluation/`, `config/`, `scripts/` (flat, no
> `src/` prefix). No `src/rag/`, `src/scraper/`, `src/evolver/` dirs yet.

```
D:\JCoder\                          Project root
|
+-- src/                            Source code
|   +-- cli/                        CLI Agent (Click + Rich + prompt_toolkit)
|   |   +-- agent.py                Agent loop, tool dispatch
|   |   +-- repl.py                 REPL interface
|   |   +-- tools/                  Tool implementations
|   |       +-- read.py
|   |       +-- write.py
|   |       +-- edit.py
|   |       +-- bash.py
|   |       +-- glob_tool.py
|   |       +-- grep_tool.py
|   |       +-- git.py
|   |
|   +-- rag/                        RAG Engine
|   |   +-- chunker_ast.py          AST chunker (tree-sitter)
|   |   +-- chunker_text.py         Fallback text chunker (from HybridRAG3)
|   |   +-- embedder.py             vLLM embedding client
|   |   +-- vector_store.py         FAISS GPU + SQLite FTS5 hybrid
|   |   +-- search.py               Hybrid search + RRF
|   |   +-- reranker.py             Qwen3-Reranker-4B client
|   |   +-- query_engine.py         Full RAG pipeline orchestrator
|   |   +-- indexer.py              Document indexing pipeline
|   |
|   +-- scraper/                    Weekly Scraper
|   |   +-- github_trending.py      GitHub Trending API client
|   |   +-- hackernews.py           Hacker News API client
|   |   +-- reddit.py               Reddit API client
|   |   +-- arxiv.py                arXiv API client
|   |   +-- releases.py             RSS/Atom release note fetcher
|   |   +-- dedup.py                Dedup guard (SHA-256 + SQLite)
|   |   +-- summarizer.py           LLM summarization driver
|   |   +-- scraper_run.py          Pipeline orchestrator
|   |
|   +-- evolver/                    10x Code Evolver
|   |   +-- test_runner.py          pytest execution + result parsing
|   |   +-- analyzer.py             Weakness analysis
|   |   +-- generator.py            LLM-driven improvement generation
|   |   +-- guard.py                Regression guard (revert on failure)
|   |   +-- distillation.py         Claude Max training pair extraction
|   |   +-- trainer.py              QLoRA fine-tuning pipeline
|   |   +-- evolver_run.py          10x loop orchestrator
|   |
|   +-- core/                       Shared infrastructure
|   |   +-- config.py               YAML config system (from HybridRAG3)
|   |   +-- llm_router.py           vLLM client router (from HybridRAG3)
|   |   +-- boot.py                 System bootstrap (from HybridRAG3)
|   |   +-- ibit.py                 Health monitoring (from HybridRAG3)
|   |   +-- network_gate.py         Network access control (from HybridRAG3)
|   |   +-- credentials.py          Credential management (from HybridRAG3)
|   |
|   +-- monitoring/                 Logging, metrics
|       +-- logger.py
|
+-- config/                         Configuration files
|   +-- default_config.yaml         Main config (paths, models, thresholds)
|   +-- scraper_sources.yaml        Scraper source definitions
|   +-- evolver_config.yaml         Evolver thresholds and schedule
|
+-- tests/                          Test suite
|   +-- test_cli/
|   +-- test_rag/
|   +-- test_scraper/
|   +-- test_evolver/
|   +-- test_core/
|
+-- docs/                           Documentation
|   +-- 00_ROADMAP.md
|   +-- 01_STACK.md
|   +-- 02_ARCHITECTURE.md          (this file)
|
+-- logs/                           Runtime logs
|   +-- jcoder.log
|   +-- scraper.log
|   +-- evolver.log
|
+-- tools/                          Utility scripts
|
+-- scripts/                        Operational scripts

D:\JCoder_Data\                     Data directory (outside project root)
|
+-- models/                         Downloaded model weights
|   +-- qwen3-coder-next-80b-q3/    LLM (~37 GB)
|   +-- nomic-embed-code-7b-q4/     Embedder (~4.4 GB)
|   +-- qwen3-reranker-4b/          Reranker (~3 GB)
|   +-- qlora-adapters/             Fine-tuned adapter checkpoints
|
+-- index/                          RAG index files
|   +-- faiss.index                 FAISS GPU index (float16 vectors)
|   +-- chunks.db                   SQLite database (metadata + FTS5)
|   +-- dedup.db                    Scraper dedup hash table
|
+-- training/                       Fine-tuning data
|   +-- distillation_pairs.jsonl    (prompt, response) training pairs
|   +-- training_log.jsonl          Training run history
|
+-- cache/                          Transient caches
    +-- conversation_history/       Session conversation backups
    +-- scraper_raw/                Raw scraper downloads (pre-summary)
```

### VRAM Budget (48 GB total across dual RTX 3090 NVLink)

| Model                      | VRAM     | GPU Assignment     |
|----------------------------|----------|--------------------|
| Qwen3-Coder-Next 80B (Q3) | ~37 GB   | TP=2 (split across both GPUs) |
| Nomic Embed Code 7B (Q4)  | ~4.4 GB  | GPU 0 (remaining)  |
| Qwen3-Reranker-4B         | ~3.0 GB  | GPU 1 (remaining)  |
| FAISS GPU index            | ~0.5 GB  | GPU 0 (estimated for 500K vectors) |
| **Total**                  | **~44.9 GB** | **6.1 GB headroom** |

---

## Model Inventory [PARTIAL]

> **Status**: Design specified Qwen-family models. Actual approved stack is
> Devstral Small 2 24B (primary), phi4-mini (toaster), nomic-embed-code (embedder).
> No vLLM -- all served via Ollama. Reranker not implemented.
> JCoder is personal project -- no NDAA restrictions on model selection.

| Model                       | Role      | Params | Quant | Size    | Context | Throughput   |
|-----------------------------|-----------|--------|-------|---------|---------|--------------|
| Qwen3-Coder-Next 80B MoE   | LLM       | 80B    | Q3    | ~37 GB  | 128K    | ~71 tok/s    |
| Nomic Embed Code 7B         | Embedder  | 7B     | Q4    | ~4.4 GB | 8192    | batched      |
| Qwen3-Reranker-4B           | Reranker  | 4B     | FP16  | ~3 GB   | 8192    | ~200 ms/50   |

All models served by vLLM with OpenAI-compatible API endpoints. No Ollama
dependency (vLLM provides better batched throughput for the scale of operations
in JCoder).

Serving configuration:

```
# LLM (tensor parallel across both GPUs)
vllm serve qwen3-coder-next-80b-q3 \
    --tensor-parallel-size 2 \
    --port 8000 \
    --max-model-len 131072 \
    --gpu-memory-utilization 0.85

# Embedder (single GPU, small footprint)
vllm serve nomic-embed-code-7b-q4 \
    --port 8001 \
    --task embedding

# Reranker (single GPU, small footprint)
vllm serve qwen3-reranker-4b \
    --port 8002 \
    --task rerank
```

---

## Reusable Code Mapping [PARTIAL]

> **Status**: Some code was borrowed from HybridRAG3 (config patterns, FTS5,
> federated search, telemetry). LimitlessApp code has NOT been ported yet.
> Actual reuse ratio differs from original estimates.

### From HybridRAG3 (D:\HybridRAG3) -- 9,894 lines total

| Source File (HybridRAG3)                    | Lines | Target File (JCoder)       | Adaptation Notes                                        |
|---------------------------------------------|-------|----------------------------|---------------------------------------------------------|
| `src/core/llm_router.py`                    | 1,811 | `src/core/llm_router.py`  | Strip OllamaRouter + APIRouter, keep VLLMRouter, add tensor-parallel config. Remove Azure/OpenAI cloud paths entirely -- JCoder is local-only. Retain streaming support and error classification. |
| `src/core/vector_store.py`                  | 990   | `src/rag/vector_store.py`  | Replace memmap backend with FAISS GPU. Keep SQLite FTS5 layer intact. Change float32/float16 disk format to FAISS IVF index format. Keep deterministic chunk ID system (INSERT OR IGNORE). Dimension change: configurable (768 default for Nomic Embed Code). |
| `src/core/embedder.py`                      | 211   | `src/rag/embedder.py`      | Replace Ollama HTTP client with vLLM OpenAI-compatible embedding endpoint. Keep batch processing logic. Keep L2 normalization. Change default model to nomic-embed-code-7b. |
| `src/core/config.py`                        | 621   | `src/core/config.py`       | Keep dataclass architecture and YAML loading. Add new config sections for: scraper sources, evolver thresholds, vLLM endpoints, FAISS parameters, reranker settings. Remove Azure/cloud config sections. |
| `src/core/grounded_query_engine.py`         | 507   | `src/rag/query_engine.py`  | Keep 8-step pipeline structure. Replace hallucination guard (NLI) with reranker-based confidence scoring. Add code-specific prompt engineering (language-aware context formatting). Keep citation mapping. |
| `src/core/chunker.py`                       | 402   | `src/rag/chunker_text.py`  | Keep as fallback for non-code files. Heading detection and overlap logic unchanged. Used for markdown, config files, plain text. New AST chunker (tree-sitter) handles source code files. |
| `src/core/indexer.py`                       | 485   | `src/rag/indexer.py`       | Keep scan-parse-chunk-embed-store pipeline. Add tree-sitter AST chunking path for code files. Keep hash-based change detection and crash recovery. Keep pre-flight integrity checks. |
| `src/core/ibit.py`                          | 284   | `src/core/ibit.py`         | Keep IBIT/CBIT architecture. Add health checks for: vLLM servers (3 endpoints), FAISS index integrity, GPU memory utilization. Remove Ollama-specific checks. |
| `src/core/boot.py`                          | 383   | `src/core/boot.py`         | Keep boot pipeline structure (config -> validate -> construct -> ready). Replace Ollama/API client construction with vLLM client construction. Add FAISS index loading to boot sequence. Remove credential resolution for cloud APIs. |
| `src/security/credentials.py`               | 785   | `src/core/credentials.py`  | Heavily simplified. Remove all Azure/OpenAI/cloud credential handling. Keep keyring integration for Claude Max API key (used by distillation agent only). Keep env var resolution pattern. |
| `src/security/network_gate.py`              | 506   | `src/core/network_gate.py` | Keep three-mode architecture (offline/online/admin). Default mode: offline (localhost only). Online mode: only for scraper and distillation agent. Admin mode: for model downloads and pip operations. |

Reuse summary: 6,985 lines ported with adaptation, approximately 2,909 lines of new code needed for AST chunking, FAISS integration, reranker client, and CLI agent.

### From LimitlessApp (D:\Projects\KnowledgeBase\LimitlessApp) -- 2,718 lines total

| Source File (LimitlessApp)                  | Lines | Target File (JCoder)           | Adaptation Notes                                        |
|---------------------------------------------|-------|--------------------------------|---------------------------------------------------------|
| `claude_memory_bridge.py`                   | 1,024 | `src/evolver/distillation.py`  | Repurpose session extraction logic for Claude Max API interaction. Replace memory persistence with training pair extraction. Keep structured output parsing and quality scoring. |
| `compaction_engine.py`                      | 487   | `src/evolver/analyzer.py`      | Repurpose knowledge compression for test result analysis. Keep ranking and prioritization logic. Replace "memory compaction" framing with "weakness ranking" framing. |
| `evolution_loop.py`                         | 412   | `src/evolver/evolver_run.py`   | Core of the 10x loop. Keep iteration structure and delta tracking. Add regression guard (revert on failure). Add git integration for version control of improvements. |
| `deep_extractor.py`                         | 398   | `src/evolver/generator.py`     | Repurpose deep analysis for code improvement generation. Replace "insight extraction" with "patch generation." Keep LLM interaction patterns and structured output parsing. |
| `limitless_run.py`                          | 397   | `src/evolver/evolver_run.py`   | Merge pipeline orchestration into evolver_run.py. Keep scheduling logic, error handling, and summary reporting. |

Reuse summary: 2,718 lines ported with adaptation. The LimitlessApp code provides the self-improvement loop skeleton; JCoder adds code-specific test integration, git-based version control of improvements, and QLoRA fine-tuning (new code).

### Total Reuse

| Source         | Lines Reused | New Lines Needed | Reuse Ratio |
|----------------|-------------|------------------|-------------|
| HybridRAG3     | 6,985       | ~2,909           | 71%         |
| LimitlessApp   | 2,718       | ~1,200           | 69%         |
| CLI Agent      | 0           | ~3,000           | 0% (new)    |
| **Total**      | **9,703**   | **~7,109**       | **58%**     |

Estimated total codebase at v1.0: approximately 16,800 lines of Python.

---

## Component Interaction Matrix [PARTIAL]

> **Status**: Only CLI Agent and RAG Engine columns are meaningful today.
> Scraper and Evolver columns are aspirational.

Which subsystems depend on which shared components:

| Component          | CLI Agent | RAG Engine | Scraper | Evolver |
|--------------------|-----------|------------|---------|---------|
| config.py          | yes       | yes        | yes     | yes     |
| llm_router.py      | yes       | yes        | yes     | yes     |
| boot.py            | yes       | no         | no      | no      |
| ibit.py            | yes       | no         | no      | no      |
| network_gate.py    | yes       | no         | yes     | yes     |
| credentials.py     | no        | no         | no      | yes     |
| vector_store.py    | no        | yes        | yes     | no      |
| embedder.py        | no        | yes        | yes     | no      |
| query_engine.py    | no        | yes        | no      | no      |
| indexer.py         | no        | yes        | yes     | no      |
| FAISS index        | no        | yes        | yes     | no      |
| SQLite FTS5        | no        | yes        | yes     | no      |
| tree-sitter        | no        | yes        | yes     | no      |
| vLLM (port 8000)   | yes       | yes        | yes     | yes     |
| vLLM (port 8001)   | no        | yes        | yes     | no      |
| vLLM (port 8002)   | no        | yes        | no      | no      |
| filesystem (code)  | yes       | no         | no      | yes     |
| internet           | no        | no         | yes     | yes (distillation only) |

---

## Design Principles

1. **Offline by default.** The network gate blocks all outbound traffic unless
   explicitly permitted. Only the scraper and distillation agent need internet.
   The CLI Agent and RAG Engine never touch the network beyond localhost.

2. **No cloud dependencies.** Every model runs locally via vLLM. No API keys
   required for core operation. Claude Max is optional (distillation only).

3. **Crash recovery everywhere.** Deterministic chunk IDs, INSERT OR IGNORE,
   hash-based change detection, and git-based versioning mean any process can
   be killed and restarted without data loss or corruption.

4. **Subsystem independence.** Each subsystem can be developed, tested, and
   run independently. The CLI Agent works without the scraper. The scraper
   works without the evolver. The evolver works without the CLI.

5. **Localhost HTTP as IPC.** vLLM's OpenAI-compatible API provides a clean,
   well-documented interface between processes. No custom RPC, no shared
   memory, no message queues. Any HTTP client can call any model server.

6. **GPU-first where it matters.** FAISS on GPU for sub-millisecond search.
   vLLM with tensor parallelism for maximum generation throughput. The 48 GB
   VRAM budget is carefully allocated to keep all three models resident
   simultaneously.
