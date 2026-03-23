# Sprint R12: Retrieval Quality Revolution
**Priority:** CRITICAL -- higher ROI than more data downloads
**Evidence:** RAG anti-pattern research (2026-03-15), code search tech analysis
**Status:** COMPLETE (2026-03-19)

## Summary of Changes
- R12.1: AST chunking wired (ast_fts5_builder.py + chunker.py, 36 tests)
- R12.2: Adaptive-k retrieval + confidence gating (retrieval_engine.py, 17 tests)
- R12.3: Cross-index MinHash dedup wired into build_fts5_indexes.py (14 tests)
- R12.4: Embedder config upgraded to nomic-embed-code (config/models.yaml)
- R12.5: Quality scoring (quality_score column in FTS5 schema, _estimate_quality heuristic)
- R12.6: Confidence gating via SmartOrchestrator in bridge.py

## Why This Sprint Matters
- Naive chunking faithfulness: 0.47 vs AST chunking: 0.82 (+75% improvement)
- 70-80% of RAG projects fail from poor data quality, not insufficient data
- "Lost in the middle" -- LLMs miss info at positions 3-7 of retrieved chunks
- FTS5/BM25 tokenizes code wrong (breaks camelCase, loses positional info)

## R12.1: AST-Based Chunking via tree-sitter [DONE]
- tree-sitter wired into ingestion/chunker.py and ast_fts5_builder.py
- Chunk at function/class boundaries (~500 tokens each)
- Keep imports attached to first function, docstrings to their function/class
- Graceful fallback: AST -> heuristic regex -> character splitting
- **Tests:** 36 passing (test_ast_fts5_builder.py + test_chunker.py)

## R12.2: Adaptive-k Retrieval [DONE]
- Query complexity classifier (keyword heuristic)
- Simple queries (API lookup): k=3, Complex queries: k=10-15
- Highest-scored chunk always FIRST (mitigate "lost in the middle")
- **Tests:** 17 passing (test_retrieval_engine.py)

## R12.3: Cross-Index Deduplication [DONE]
- MinHash + LSH (ingestion/dedup.py) wired into scripts/build_fts5_indexes.py
- Persistent state (D:/JCoder_Data/dedup_state/) for incremental builds
- --no-dedup flag to skip when needed
- **Tests:** 14 passing (test_build_fts5_dedup.py)

## R12.4: Upgrade to nomic-embed-code [DONE]
- config/models.yaml: name -> "nomic-embed-code", added code_model/text_model
- config/memory.yaml: already had code_model: "nomic-embed-code"
- DualEmbeddingEngine auto-routes code to nomic-embed-code, text to nomic-embed-text
- **Note:** Existing FAISS indexes must be rebuilt with new embedder (FTS5 unaffected)
- Run `ollama pull nomic-embed-code` before enabling embedder

## R12.5: Quality Scoring at Index Time [DONE]
- quality_score column added to FTS5 schema (0-5 scale)
- _estimate_quality() heuristic: docstrings, definitions, length, curated source, imports/types
- **Tests:** included in test_build_fts5_dedup.py

## R12.6: Confidence Gating on Retrieval [DONE]
- SmartOrchestrator wired in bridge.py
- If max_score < 0.3, returns top-1 only (not noise)

## Future (BEAST-dependent):
- R12.7: Evaluate Zoekt for trigram code search layer
- R12.8: Switch BEAST to vLLM with speculative decoding (2-3x speedup)
- R12.9: Add SCIP semantic code graph for definition/reference search
