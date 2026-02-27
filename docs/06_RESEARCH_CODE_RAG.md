# 06 -- Code-Aware RAG: Research Findings

Document: 06_RESEARCH_CODE_RAG.md
Created: 2026-02-26
Status: Research compilation -- not yet implemented

---

## 1. Code Parsing and Chunking

### 1.1 tree-sitter (Foundation Layer)

tree-sitter is the industry-standard incremental parsing library used by nearly every
serious code intelligence tool (Aider, Cursor, Sourcegraph Cody, TabbyML, Neovim,
Zed, GitHub Linguist).

- **Current version**: v0.26.4 (February 2026)
- **tree-sitter-language-pack**: Single pip install covers 165+ language grammars
- **Output**: Concrete syntax trees (CSTs) with byte-range spans for every node
- **Performance**: Incremental re-parse on edit in microseconds
- **License**: MIT

tree-sitter gives us node types (function_definition, class_declaration,
import_statement, comment, string_literal, etc.) with exact byte offsets. This is
the primitive that all intelligent chunking strategies build on.

### 1.2 cAST Algorithm (EMNLP 2025, Carnegie Mellon)

The Context-Aware Syntax Tree (cAST) algorithm is the current state of the art for
code chunking in retrieval-augmented generation pipelines.

**Algorithm**:
1. Parse source file into AST via tree-sitter
2. Recursive split-then-merge: walk the AST top-down, splitting nodes that exceed
   the target chunk size, then merging adjacent small siblings that fit together
3. Size measurement: count non-whitespace characters (not raw token count), which
   correlates better with semantic density in code
4. Boundary preservation: chunks always align to AST node boundaries -- never split
   mid-function or mid-class unless the node itself exceeds the size limit

**Benchmark results vs. naive line-based or fixed-token chunking**:
- +5.5 points on RepoEval (repository-level code completion)
- +4.3 points on CrossCodeEval (cross-file code understanding)
- +2.7 points on SWE-bench (real-world bug fixing)

**Optimal chunk size**: 200-800 tokens. Below 200 tokens, chunks lose surrounding
context. Above 800 tokens, retrieval precision degrades because unrelated code
gets bundled together.

**Reference**: arxiv.org/abs/2506.15655

### 1.3 code-chunk Library (supermemoryai)

The code-chunk library (github.com/supermemoryai/code-chunk) adds a
**contextualizedText** field to each chunk. This field prepends:

- **Scope chain**: The full nesting path (e.g., `module.ClassName.method_name`)
- **Relevant imports**: Only the imports actually referenced by code in the chunk
- **Entity signatures**: Function signatures, class declarations, type annotations
  for entities defined or referenced in the chunk

This matters because a chunk containing `return self._cache[key]` is nearly useless
without knowing that `self` is an instance of `CacheManager` and `_cache` is a
`Dict[str, EmbeddingVector]`. The contextualizedText field makes each chunk
self-contained for embedding and retrieval.

### 1.4 Multi-File Context via Dependency Graphs

tree-sitter can resolve import statements across files, producing a dependency graph
that answers questions like:

- "Which files import this module?"
- "What is the full call chain from endpoint to database?"
- "If I change this function signature, what breaks?"

This graph is essential for multi-hop retrieval: a query about "how does the /query
endpoint handle authentication" might need chunks from the route handler, the auth
middleware, and the token validation utility -- three separate files linked by imports.

---

## 2. Vector Database: LanceDB

### 2.1 Why LanceDB Over FAISS

LanceDB is recommended over FAISS for code RAG workloads. The comparison:

| Feature                    | FAISS              | LanceDB                     |
|----------------------------|--------------------|-----------------------------|
| Architecture               | In-memory library  | Embedded DB (like SQLite)    |
| Server required            | No                 | No                           |
| Hybrid search (BM25+vec)   | Manual assembly    | Native, single query         |
| SQL filtering              | No                 | Yes (Lance SQL)              |
| Incremental updates        | Rebuild index      | Add/delete without rebuild   |
| Disk-based queries         | No (must load all) | Memory-mapped, near-memory   |
| GPU indexing               | Yes                | Yes (5-10x on RTX 3090)     |
| Persistence                | Manual serialize   | Automatic (Lance format)     |
| Metadata storage           | External           | Built-in columnar storage    |

### 2.2 Key Capabilities

- **Embedded**: No server process. Opens a directory, reads/writes Lance files.
  Deployment is `pip install lancedb` and nothing else.
- **Rust core + Apache Arrow**: Columnar storage with zero-copy reads. The Python
  API is a thin wrapper over Rust.
- **Native hybrid search**: A single query can combine BM25 full-text search,
  vector similarity, and SQL WHERE clauses. This is critical for code search where
  you need exact identifier matching (BM25) alongside semantic understanding (vector).
- **GPU-accelerated indexing**: Building HNSW indexes on an RTX 3090 is 5-10x faster
  than CPU. For a corpus of 40K chunks, index build drops from minutes to seconds.
- **Incremental HNSW updates**: Add new vectors or delete old ones without rebuilding
  the entire index. This supports the "re-index changed files only" workflow.
- **Memory-mapped queries**: The full index does not need to fit in RAM. Queries
  run from disk at near-memory speed via mmap. This means a 1-billion-vector index
  can be queried in under 100ms on commodity hardware.
- **Built-in re-ranking**: Supports cross-encoder re-rankers or custom user-defined
  functions (UDFs) as a post-retrieval step, eliminating the need for a separate
  re-ranking service.

### 2.3 Storage Format

LanceDB uses the Lance columnar format (an evolution of Parquet optimized for ML
workloads). Data is stored as versioned, append-only files. This gives:

- Automatic versioning (roll back to any previous state)
- Concurrent readers with a single writer
- Efficient scans over metadata columns without loading vectors

---

## 3. Code Search Strategies

### 3.1 Hybrid Search Is Mandatory

Pure vector search fails on code because:

- Identifiers are arbitrary strings. `calculateTaxRate` and `compute_tax_amount` are
  semantically similar but lexically different. Vector search handles this.
- But `calculateTaxRate` and `calculateTaxRate` must be an exact match. A developer
  searching for a specific function name needs BM25-style exact matching.
- API names, error codes, enum values, and configuration keys are all exact-match
  queries that vector search handles poorly.

The consensus across all surveyed tools (Sourcegraph, Aider, Cursor) is that code
search must combine:

1. **BM25 full-text search**: For exact identifiers, error messages, enum values
2. **Dense vector search**: For semantic queries ("function that validates JWT tokens")
3. **Structured filters**: Language, file path, recency, symbol type

### 3.2 Code-Aware Tokenization

Standard NLP tokenizers (BPE, WordPiece) mangle code identifiers:

- `calculateTaxRate` becomes `["calc", "ulate", "Tax", "Rate"]` -- losing the
  compound word structure
- `HTTP_STATUS_CODE` becomes `["HTTP", "_", "STATUS", "_", "CODE"]` -- the
  underscores waste token budget

Code-aware tokenization should:

1. Split camelCase into `["calculate", "Tax", "Rate"]`
2. Split snake_case into `["HTTP", "STATUS", "CODE"]`
3. Preserve the original form alongside the splits (index both `calculateTaxRate`
   and its components)
4. Normalize common abbreviations (`cfg` -> `config`, `ctx` -> `context`) as
   additional index terms

### 3.3 Zoekt (Sourcegraph)

Zoekt is the trigram-based code search engine built by Google and maintained by
Sourcegraph. It is the gold standard for regex and substring search over code
repositories.

- Indexes source code into trigram posting lists
- Supports full regular expression queries with sub-second response
- Handles repositories with millions of files
- Not a vector database -- complements semantic search as the "exact match" layer

For a JCoder-scale project (thousands of files, not millions), the BM25 capability
built into LanceDB likely suffices. Zoekt becomes relevant at Sourcegraph scale.

---

## 4. GraphRAG for Code

### 4.1 Code-Graph-RAG (vitali87)

Code-Graph-RAG (github.com/vitali87/code-graph-rag) is a purpose-built GraphRAG
system for source code.

**Architecture (4-pass)**:
1. **Parse**: tree-sitter extracts AST nodes (functions, classes, imports, calls)
2. **Graph construction**: Nodes become graph vertices; edges represent calls,
   imports, inheritance, containment
3. **Community detection**: Leiden algorithm groups tightly-connected code into
   communities (similar to modules or subsystems)
4. **Query**: Graph traversal + LLM summarization of relevant communities

**Implementation**:
- Graph database: Memgraph (Bolt protocol, Cypher queries)
- Language support: 11 languages via tree-sitter
- Interface: MCP server (Model Context Protocol), so any MCP-compatible LLM client
  can query the code graph directly

**Use case**: "Explain how authentication works in this codebase" -- the graph
traversal finds the auth module community, pulls in related middleware and token
validation, and summarizes the full flow.

### 4.2 Microsoft GraphRAG v1.0

Microsoft's general-purpose GraphRAG (applied to code at SAP) uses an LLM to
extract entities and relationships from text, builds a knowledge graph, and
uses community summaries for retrieval.

- Applied to code documentation and source at SAP with reported 94% of
  LLM-only-based performance at significantly lower per-query cost
- Best for "global" queries that require understanding the entire codebase
  structure rather than finding a specific function
- Heavyweight: requires LLM calls during indexing to extract entities

### 4.3 LightRAG

LightRAG is a lighter alternative to Microsoft GraphRAG:

- Supports Ollama as the LLM backend (relevant for offline/air-gapped use)
- Simpler graph construction (keyword co-occurrence + LLM extraction)
- **Known limitation**: Struggles with models under 8B parameters for reliable
  JSON extraction during graph construction. phi4-mini (3.8B) may not produce
  clean enough structured output.

---

## 5. Advanced RAG Techniques

### 5.1 HyDE (Hypothetical Document Embeddings)

**How it works**:
1. User asks: "function that retries HTTP requests with exponential backoff"
2. LLM generates a hypothetical code snippet that would answer this query
3. Embed the hypothetical snippet (not the question)
4. Search the vector index for real code similar to the hypothetical

**Why it helps for code**: Natural language questions and code live in different
embedding spaces. HyDE bridges this gap by converting the question into the same
modality (code) as the indexed documents.

**Cost**: One extra LLM call per query (can use a small model like phi4-mini).

### 5.2 Self-RAG (ICLR 2024)

**How it works**:
1. Model receives a query
2. Model decides whether retrieval is needed (some questions are answerable from
   parametric knowledge alone)
3. If retrieval is triggered, model retrieves and reads passages
4. Model self-critiques: "Is this passage relevant? Is my answer supported by it?"
5. Model generates final answer, citing which passages support which claims

**Reference**: selfrag.github.io

**Application to code**: Useful for mixed queries where some parts need retrieval
("what does function X do?") and others do not ("what is a binary search?").

### 5.3 RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval)

**How it works**:
1. Cluster leaf chunks (individual functions) by semantic similarity
2. Summarize each cluster into a higher-level description
3. Repeat: cluster the summaries, summarize again
4. Result: a tree where leaves are raw code and internal nodes are progressively
   more abstract summaries (function -> module -> subsystem -> architecture)

**Benchmark**: +20% accuracy on multi-hop questions compared to flat retrieval.

**Application to code**: A query about "how does the data pipeline work" retrieves
a subsystem-level summary node rather than individual function chunks, giving the
LLM a coherent architectural overview.

**Reference**: arxiv.org/abs/2401.18059

### 5.4 Agentic RAG

**How it works**:
1. Agent receives a complex query
2. Agent decomposes it into sub-queries
3. For each sub-query, agent selects the appropriate tool (vector search, BM25,
   graph traversal, file read, grep)
4. Agent evaluates whether retrieved context is sufficient
5. If not, agent reformulates and retries
6. Agent synthesizes final answer from all gathered context

**Application to code**: "Why does the /query endpoint sometimes return 500?" might
require: (a) reading the route handler, (b) searching for exception handlers,
(c) checking the error logs schema, (d) tracing the call chain to find unhandled
edge cases. No single retrieval call answers this.

### 5.5 Parent-Child Chunking

**How it works**:
1. Index fine-grained chunks (individual functions, 200-400 tokens)
2. Also store the parent context (the full class or module containing each function)
3. At query time, match on the fine-grained chunk (high precision)
4. Return the parent context to the LLM (broader understanding)

**Why it matters**: A matched function is often meaningless without its class context
(instance variables, other methods it calls, class-level docstring). Parent-child
chunking retrieves the function for precision but gives the LLM the class for
comprehension.

---

## 6. How Existing Tools Work

### 6.1 Aider

- **Indexing**: tree-sitter parses every file in the repo, extracts "tags" (function
  definitions, class declarations, import statements)
- **Repo map**: Builds a directed graph of definitions and references, then uses
  PageRank to rank symbols by importance
- **Context selection**: Given a user message, selects the most relevant files and
  symbols to fit within a 1K token budget for the repo map
- **Key insight**: The repo map is a compressed representation of the entire
  repository structure, not a full dump. It tells the LLM "these are the most
  important symbols and where they live."

**Reference**: aider.chat/docs/repomap.html

### 6.2 Cursor

- **Incremental sync**: Uses a Merkle tree to detect which files changed since the
  last index update, re-indexing only those files
- **Chunking**: Semantic boundary splitting (respects function/class boundaries)
- **Privacy**: File path obfuscation in the cloud index -- paths are hashed so the
  server cannot reconstruct the directory structure
- **Context window**: Fills the LLM context with retrieved chunks, prioritized by
  relevance score and recency

### 6.3 Sourcegraph Cody

- **3-layer context architecture**:
  1. Local file: The file currently being edited (highest priority)
  2. Local repo: Other files in the same repository (tree-sitter indexed)
  3. Remote repos: Other repositories in the organization (Zoekt + SCIP indexed)
- **Context budget**: Up to 1M tokens of context (using long-context models)
- **Precision tools**: SCIP (Sourcegraph Code Intelligence Protocol) provides
  go-to-definition and find-references at the index level, not just text search

### 6.4 TabbyML

- **Completion-focused**: Primarily used for code completion rather than chat
- **tree-sitter tags**: Extracts the same kind of tags as Aider (definitions,
  references) for RAG-assisted completion
- **Adaptive caching**: Caches recent completions and retrieval results, serving
  them instantly if the cursor context has not changed significantly

---

## 7. Benchmarks

### 7.1 CodeRAG-Bench (NAACL 2025)

- **Scale**: 9,000 programming tasks with 25 million retrieval documents
- **Coverage**: Code generation, code understanding, bug fixing, documentation Q&A
- **Purpose**: Standardized evaluation of retrieval quality for code-oriented RAG
- **Key finding**: Retrieval consistently helps, but the quality of chunking and
  the retrieval model matter more than the generator model size

**Reference**: code-rag-bench.github.io

### 7.2 CoIR (ACL 2025)

- **Scale**: 10 datasets, 8 task types, 7 domains, 2 million documents
- **Coverage**: Code-to-code search, text-to-code search, code-to-text search,
  cross-language code search
- **Purpose**: Comprehensive information retrieval benchmark specifically for code
- **Key finding**: General-purpose embedding models (e.g., text-embedding-3-small)
  underperform code-specialized models by 10-15 points on code retrieval tasks

**Reference**: github.com/CoIR-team/coir

---

## 8. Sources

- tree-sitter releases: github.com/tree-sitter/tree-sitter/releases
- cAST paper: arxiv.org/abs/2506.15655
- code-chunk: github.com/supermemoryai/code-chunk
- LanceDB: lancedb.com
- Code-Graph-RAG: github.com/vitali87/code-graph-rag
- Self-RAG: selfrag.github.io
- RAPTOR: arxiv.org/abs/2401.18059
- Aider repo map: aider.chat/docs/repomap.html
- CodeRAG-Bench: code-rag-bench.github.io
- CoIR: github.com/CoIR-team/coir
