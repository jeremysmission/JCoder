# RAG Anti-Patterns Research -- 2026-03-15
## CHALLENGES TO OUR CURRENT APPROACH

### WARNING: More Data Can HURT
- **70-80% of enterprise RAG projects fail in production**
- **42% cite poor data cleaning as primary cause**
- ANN recall drops from 0.95 to 0.71 at 50x corpus size
- Retrieval accuracy alone explains only 60% of end-to-end quality
- "Context rot" -- LLMs perform WORSE with more context to juggle

### The "Lost in the Middle" Problem
- LLMs use info at START and END of context but MISS info in the MIDDLE
- If best chunk is at position 5 of 10, the LLM likely ignores it
- Practical fix: put most relevant chunk FIRST, not just in top-k

### Chunking Is Root Cause of 80% of Failures
- Naive fixed-size chunking: faithfulness 0.47-0.51
- Semantic/AST chunking: faithfulness 0.79-0.82
- **We MUST prioritize AST-based chunking over downloading more data**

### Deduplication Distorts Retrieval
- Near-duplicates across 174 indexes waste top-k slots on redundant info
- Same fact in 15 indexes fills retrieval with repetitions, crowding out unique content
- **We should dedup ACROSS indexes, not just within**

### Optimal Retrieval is ADAPTIVE, Not Fixed
- Static top-k=5 is wrong for both simple and complex queries
- Performance peaks at chunk size 512 tokens, declines at 1024+
- **Adaptive-k per query** beats fixed-k significantly

### Fine-Tuning + RAG Together is Best
- Fine-tuning adds +6% accuracy, RAG adds +5% ON TOP of that
- Neither alone is optimal -- the gains are cumulative

## WHAT THIS MEANS FOR JCODER

### STOP doing:
- Blindly downloading every dataset without quality filtering
- Fixed top-k retrieval
- Naive text chunking for code

### START doing:
- Quality scoring on all indexed data (edu_score >= 3 filter)
- Cross-index deduplication (MinHash across all FTS5 DBs)
- AST-based chunking via tree-sitter
- Adaptive retrieval (vary k based on query complexity)
- "Lost in the middle" mitigation (best chunk always first)
- Monitoring retrieval QUALITY not just latency

### PRIORITY SHIFT:
Instead of "500 GB of everything", aim for "100-200 GB of HIGH QUALITY curated data"
with proper deduplication, quality filtering, and AST chunking.
More data with poor chunking/dedup is WORSE than less data with good infrastructure.
