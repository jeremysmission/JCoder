# Session Retrospective — 2026-03-24

## Greatest Achievements

### 1. The One-Word Fix That Unblocked Everything
The entire self-learning pipeline had been dead for days. Every eval returned 0 chunks for all 200 questions. The root cause? A single column name: `content` vs `search_content`. The FTS5 databases used `search_content` but three scripts queried `SELECT content FROM chunks`. One word, zero results.

Finding this took tracing the full pipeline from CLI entry point through embedding engine, index engine, retrieval, and into the learning cycle. The fix was trivial — the diagnosis was not.

**Lesson:** When a system returns zero results, the bug is almost never in the algorithm. It's in the wiring. Check column names, table names, endpoint URLs before debugging logic.

### 2. The Extension Allowlist — Finishing What Was Started
The night sprint team claimed the P1 extension drift was "fixed" with a single-source-of-truth pattern. QA found it was only half done — `sanitizer.py` and `prep_stage_for_index.py` still had hardcoded lists. We also found a C# taxonomy split (`c_sharp` vs `csharp`) that would have silently stranded all C# output in a dead bucket.

**Lesson:** "Fixed" means verified in every consumer, not just the source. A registry is only as good as the number of places that actually import from it.

### 3. 37% Code Reduction Without Breaking a Single Test
Split 5 oversized modules (3,138 LOC total) down to 1,989 LOC across 10 files. The key insight was that `tools.py` contained 410 lines of JSON schemas *identical* to `tool_schemas.py` — pure duplication that had drifted. The `core.py` refactor eliminated 150 lines of duplicated loop logic by unifying `run()` and `resume()` into a shared `_iterate()` method.

**Lesson:** Before splitting a large file, check if it already has a sibling that contains the same content. Duplication hides in plain sight.

### 4. Research-Driven Architecture Decisions
Deep web research across GitHub, Reddit, HuggingFace, and X.com surfaced findings that changed our direction:
- **phi4's 16K context limit** — confirmed it's adequate for our RAG chunk sizes, saving us from an unnecessary model switch
- **nomic-embed-text-v2-moe** — 62% better separation than v1, validated with a 3-text benchmark before committing to the upgrade
- **FlashRank reranker** — 3MB TinyBERT model that correctly scores 0.9997 for relevant and 0.0000 for irrelevant passages
- **Ollama bug #6262** — batch embedding quality degrades silently. Setting `OLLAMA_NUM_PARALLEL=1` prevents it. Nobody told us; we found it by reading GitHub issues.

**Lesson:** Never assume the AI's reasoning about architecture is correct. Research first, verify independently, then build. The 20 minutes spent reading GitHub issues about Ollama batch embedding saved us from silently corrupted FAISS indexes.

### 5. The D: Drive Purge
Found and eliminated hardcoded `D:/` paths across both repos — 14 references in HybridRAG3 guides, 4 files in JCoder source code. The retired D: drive was an external SSD that no longer exists on this machine. Every one of those paths would have caused a silent failure or confusing error on a fresh clone.

**Lesson:** When infrastructure changes (drives, paths, hostnames), grep the entire codebase. Config files and documentation rot faster than code.

---

## Greatest Lessons Learned

### Trust But Verify
The night sprint handoff claimed 2,862 tests passing, all FAISS indexes built, extension fix complete. Independent verification found: extension fix was half-done, FAISS indexes were 0.2% coverage on large databases, and the learning cycle was completely broken. The handoff was honest about what *scripts were written* but optimistic about what *actually worked end-to-end*.

### The Scoring Function IS the Bottleneck
We wired hybrid FAISS+FTS5+RRF retrieval into the learning cycle and the eval score barely moved (4.17% to 5.22%). The retrieval was finding better results, but the keyword-overlap scoring function couldn't measure the improvement. Adding the LLM-as-judge showed the real quality — phi4 rated relevant context as 9/10 while the keyword scorer said 0%. The eval method determines what you can see.

### Small Changes, Large Blast Radius
Adding `.cc` and `.cxx` to `LANGUAGE_MAP` in `chunker.py` automatically propagated to `sanitizer.py`, `prep_stage_for_index.py`, `repo_loader.py`, `corpus_pipeline.py`, and `build_fts5_indexes.py` — because we built the single-source-of-truth pattern first. One line change, 6 files updated. That's the payoff of good architecture.

### Background Work Needs Retry Logic
The FAISS v2-moe rebuild ran for 25 minutes, embedded 19,232 chunks, then crashed on a single Ollama timeout (KeyError on missing `data` key in response). Inline scripts without retry logic waste hours of GPU time. Always use the battle-tested script with exponential backoff for long-running jobs.

---

## By The Numbers

| Metric | Value |
|--------|-------|
| Commits pushed (JCoder) | 24 |
| Commits pushed (HybridRAG3) | 3 |
| Tests passing | 2,862 (JCoder) + 1,297 (HybridRAG3) |
| Modules split | 5 (3,138 -> 1,989 LOC) |
| Docstrings added | 30 functions (HybridRAG3) |
| Broken doc links fixed | 11 |
| D: drive references purged | 18 |
| FAISS vectors indexed | 187,368 (jcoder_self) |
| New models pulled | 2 (qwen3.5:9b, nomic-embed-text-v2-moe) |
| Doctor checks | 16/16 GREEN |
| Research sources consulted | 40+ (GitHub, Reddit, HuggingFace, arXiv, X.com) |

---

## What I Would Do Differently

1. **Run the FAISS rebuild with the proper script from the start** — not an inline Python one-liner. Lost 25 minutes of GPU time to a missing try/except.

2. **Check column names FIRST when retrieval returns zero** — this should have been caught in the night sprint's QA. A simple `PRAGMA table_info(chunks)` would have revealed the mismatch in 2 seconds.

3. **Benchmark the eval scoring function before optimizing retrieval** — we spent time improving retrieval quality that the keyword scorer couldn't measure. Should have added LLM-as-judge first, then optimized retrieval.

---

Signed: **Claude Opus 4.6 (1M context)**
Repos: JCoder, HybridRAG3_Educational
Date: 2026-03-24, approximately 03:00 - 10:30 MDT (~7.5 hours)
Events span: Night sprint handoff from 2026-03-23 02:15 MDT through this session ending 2026-03-24 ~10:30 MDT
