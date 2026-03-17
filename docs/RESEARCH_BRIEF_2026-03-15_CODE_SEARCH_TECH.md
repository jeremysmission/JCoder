# Code Search Tech: What The Industry Actually Uses -- 2026-03-15

## THE THREE-LAYER ARCHITECTURE (Industry Consensus)
1. **Trigram/ngram inverted index** -- exact text + regex (Zoekt, Blackbird)
2. **Semantic code graph** -- go-to-def, find-refs (SCIP, Kythe)
3. **Vector embeddings** -- NL-to-code retrieval (FAISS, Turbopuffer)

**Nobody serious uses just one of these.** We currently have FTS5 (BM25 word index) + some FAISS. We're missing layers 1 and 2.

## WHY FTS5/BM25 IS WRONG FOR CODE SEARCH
- BM25 tokenizes on WORD boundaries -- breaks camelCase, snake_case
- `getUserId` becomes `get`, `User`, `Id` -- loses positional info
- Trigram index keeps EXACT character sequences: `get`, `etU`, `tUs`, `Use`, etc.
- Zoekt/Blackbird-style ngram indexes have ZERO false positives for substring search
- FTS5 is designed for natural language, NOT code

## WHAT LEADERS USE

### Sourcegraph: Zoekt (trigram, Go)
- Positional trigram inverted index
- Memory-mapped shard files (~3x corpus size)
- Sub-50ms on 48 TB of code (GitLab scale)
- `github.com/sourcegraph/zoekt`

### GitHub: Blackbird (dynamic ngram, Rust)
- Variable-length ngrams (not just trigrams)
- Sharded by git blob SHA (natural dedup)
- 640 queries/sec, 120K docs/sec indexing
- Custom-built from scratch in Rust

### Google: Kythe (ngram + semantic graph)
- 86 TB corpus, 200 queries/sec, 50ms median
- Language-agnostic semantic graph

### Cursor: tree-sitter + Turbopuffer
- AST chunking at function/class boundaries (~500 tokens)
- Merkle tree for incremental re-indexing
- Vector DB with namespace-per-codebase
- Importance ranking + smart truncation

## BEST EMBEDDING MODELS FOR CODE (2026)
1. **CodeXEmbed-7B** (Salesforce) -- #1 on CoIR benchmark, 20%+ above Voyage-Code
2. **Nomic Embed Code** (Nomic AI) -- Apache 2.0, beats Voyage and OpenAI, runs on Ollama
3. **Voyage-Code-3** -- strong but proprietary
4. We're using nomic-embed-text -- should upgrade to nomic-embed-code

## VECTOR DB OPTIONS FOR OFFLINE
- **LanceDB** (Rust, embedded, disk-native) -- best for our use case
- **FAISS** (our current) -- fast GPU, but library not database
- **Qdrant** (Rust, filterable HNSW) -- good if we need proper DB features

## RECOMMENDED ARCHITECTURE EVOLUTION

### Phase 1: Fix chunking (highest ROI)
- Switch to tree-sitter AST chunking for all code indexes
- ~500 token chunks at function/class boundaries
- Already have tree-sitter -- just not using it for FTS5 building

### Phase 2: Add trigram layer
- Evaluate Zoekt for exact code search alongside FTS5
- Zoekt handles substring/regex, FTS5 handles NL queries
- OR: build simple trigram index in Python/Rust

### Phase 3: Upgrade embeddings
- Switch from nomic-embed-text to nomic-embed-code
- Same 768-dim, same Ollama deployment, better code retrieval

### Phase 4: Add semantic graph
- SCIP or tree-sitter based definition/reference graph
- Enables "find all callers of this function" type queries
