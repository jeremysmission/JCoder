# Sprint R12: Retrieval Quality Revolution
**Priority:** CRITICAL -- higher ROI than more data downloads
**Evidence:** RAG anti-pattern research (2026-03-15), code search tech analysis
**Status:** IN PROGRESS
**R12.1:** ingestion/ast_fts5_builder.py CREATED + tests/test_ast_fts5_builder.py (11 tests)
**R12.2:** core/retrieval_engine.py UPDATED (adaptive-k + confidence gating) + tests/test_retrieval_engine.py (17 tests)
**Tests pending:** disk I/O saturated from 8 parallel download/index jobs

## Why This Sprint Matters
- Naive chunking faithfulness: 0.47 vs AST chunking: 0.82 (+75% improvement)
- 70-80% of RAG projects fail from poor data quality, not insufficient data
- "Lost in the middle" -- LLMs miss info at positions 3-7 of retrieved chunks
- FTS5/BM25 tokenizes code wrong (breaks camelCase, loses positional info)

## R12.1: AST-Based Chunking via tree-sitter [HIGHEST ROI]
- JCoder already has tree-sitter in ingestion/chunker.py
- Need to wire it into FTS5 index building scripts
- Chunk at function/class boundaries (~500 tokens each)
- Keep imports attached to first function in file
- Keep docstrings attached to their function/class
- **Expected gain:** +4.3 Recall@5 (measured in cAST paper, EMNLP 2025)

## R12.2: Adaptive-k Retrieval
- Current: fixed top_k=5 everywhere
- Need: vary k based on query complexity
- Simple queries (API lookup): k=3
- Complex queries (architecture question): k=10-15
- Add query complexity classifier (keyword heuristic first, model later)
- Always put highest-scored chunk FIRST (mitigate "lost in the middle")

## R12.3: Cross-Index Deduplication
- MinHash + LSH across all FTS5 indexes
- Expected to eliminate 30-40% of redundant content
- Prevents same snippet from filling top-k with copies
- JCoder already has ingestion/dedup.py (298 lines, 15 tests)

## R12.4: Upgrade to nomic-embed-code
- Currently using nomic-embed-text (general text embedder)
- nomic-embed-code is code-specialized, same 768-dim, same Ollama deployment
- Beats Voyage-Code-3 and OpenAI on CodeSearchNet
- `ollama pull nomic-embed-code` on BEAST

## R12.5: Quality Scoring at Index Time
- Add edu_score/quality_score field to chunks
- Filter by quality during retrieval (score >= 3)
- Tier indexes: hot (high quality) vs cold (reference only)
- Retrieve from hot tier first, cold tier as fallback

## R12.6: Confidence Gating on Retrieval
- If all reranker scores are low, return FEWER results (not noise)
- SmartOrchestrator already designed for this (now wired in bridge.py)
- Add threshold: if max_score < 0.3, return top-1 only

## Future (BEAST-dependent):
- R12.7: Evaluate Zoekt for trigram code search layer
- R12.8: Switch BEAST to vLLM with speculative decoding (2-3x speedup)
- R12.9: Add SCIP semantic code graph for definition/reference search
