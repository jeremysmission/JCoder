# Data Dumps and Indexing Methods Research

Date: 2026-02-26
Status: Research Summary -- Comprehensive Findings

---

## 1. Executive Summary

This document covers dataset selection, ranking, and licensing; AST-based code chunking;
multi-representation and hierarchical indexing strategies; FAISS incremental update and
compression tradeoffs; embedding model benchmarks; recommended shard structure; TTL
(time-to-live) policies; and a phased download plan. The total data footprint across
all phases is approximately 327 GB, with the most valuable data concentrated in Phase 1
(~22 GB of instruction data yielding the highest quality-per-gigabyte).

---

## 2. Dataset Rankings

### 2.1 Top Dataset by Value per Gigabyte

**CoRNStack (Nomic)** -- the single highest-value dataset for code retrieval training:

- **Size:** 15 GB
- **Content:** 21 million code retrieval triplets (query, positive, negative)
- **License:** Apache 2.0
- **Why it leads:** Pre-built triplets eliminate the need for mining hard negatives,
  which is the most expensive and error-prone step in training retrieval models. At
  21M triplets in 15 GB, the information density is exceptionally high.

### 2.2 Instruction Datasets Ranked

Ranked by training value per gigabyte, accounting for data quality, diversity, and
licensing:

| Rank | Dataset       | Size   | Pairs/Examples | License       | Notes                          |
|------|---------------|--------|---------------:|---------------|--------------------------------|
| 1    | Glaive v3     | 2 GB   | 1,000,000      | Apache 2.0    | Highest volume, multi-language |
| 2    | CodeSearchNet | 3.5 GB | 2,000,000      | MIT           | NL-to-code search pairs        |
| 3    | Magicoder     | 500 MB | 185,000        | Apache 2.0    | OSS-Instruct, high quality     |
| 4    | CodeFeedback  | 2 GB   | 287,000 (filtered) | Apache 2.0 | Multi-turn debugging pairs     |

**Notes on ranking:**

- Glaive v3 leads on raw volume and language coverage.
- CodeSearchNet is essential for retrieval training (NL queries paired with code).
- Magicoder punches above its weight due to OSS-Instruct quality -- 185K examples
  from Magicoder may be more valuable than 500K from lower-quality sources.
- CodeFeedback's 287K is the post-filtering count; the raw dataset is larger but
  contains noise that should be removed.

### 2.3 Large-Scale Code Sources

| Dataset                      | Size         | Content                 | License     | Access                        |
|-----------------------------|-------------|-------------------------|-------------|-------------------------------|
| Stack Overflow dump          | 98 GB (compressed) | All Q&A through 2024  | CC-BY-SA 4.0 | BitTorrent community mirrors only |
| The Stack v2 (Python subset) | 191 GB      | 47 million Python files | Permissive mix | HuggingFace                 |

**Stack Overflow access warning:** The Stack Overflow data dump is no longer available
on archive.org. The only reliable source is BitTorrent community mirrors. Verify
checksums before using, as mirrors may be incomplete or outdated.

---

## 3. AST-Based Code Chunking

### 3.1 cAST: Code-Aware Semantic Tree Chunking (EMNLP 2025)

Traditional code chunking splits on line count or token count, which frequently breaks
functions mid-body, splits imports from the code that uses them, and separates class
definitions from their methods. cAST uses the Abstract Syntax Tree to identify natural
boundaries in code.

**Benchmark improvements over naive chunking:**

- **+5.5 points on RepoEval** (repository-level code completion)
- **+2.7 points on SWE-bench** (issue resolution)

These gains come purely from better chunking -- the same embedding model and retrieval
pipeline produce significantly better results when chunks respect code structure.

### 3.2 Optimal Chunk Size

**200-800 tokens** is the optimal range, with the sweet spot depending on the code's
granularity:

- **200-400 tokens:** Best for function-level retrieval (individual functions,
  standalone utilities).
- **400-600 tokens:** Best for class-level retrieval (methods with their class context).
- **600-800 tokens:** Best for module-level retrieval (files with imports and
  multiple related functions).

**Measurement:** Use **non-whitespace character count** rather than token count for
chunk size boundaries. Token counts vary significantly across tokenizers, but
non-whitespace character count is stable and correlates well with semantic content
density. A rough conversion: 1 token is approximately 3-4 non-whitespace characters
for code.

### 3.3 AST Chunk Boundaries

The following AST node types should be treated as hard boundaries (never split across):

- Function/method definitions
- Class definitions
- Import blocks
- Decorator chains (decorator + decorated function are one unit)
- Try/except blocks (the try and its handlers are one unit)

Soft boundaries (prefer to split here but may combine for size):

- Top-level assignments
- Comment blocks
- Blank line separators (2+ consecutive blank lines suggest a logical section break)

---

## 4. Multi-Representation Indexing

### 4.1 Dual Embedding Strategy

For each code chunk, generate and store **two embeddings:**

1. **Code embedding:** The raw code text embedded directly.
2. **Natural language embedding:** A generated description of what the code does,
   embedded as natural language.

This enables both **code-to-code search** (finding similar code) and **NL-to-code
search** (finding code from a natural language description). Without the dual
representation, NL queries perform poorly against code embeddings because the semantic
spaces are different.

**Implementation:**

- Generate NL descriptions using a code summarization model or the same LLM used for
  other tasks.
- Store both embeddings in the same FAISS index with metadata indicating which type
  each vector is.
- At query time, determine whether the query is code or natural language and search
  against the appropriate embedding type.

### 4.2 Hierarchical Indexing

A three-level hierarchy provides the best balance of speed, precision, and coverage:

| Level | Granularity          | Index Type | Use Case                           |
|-------|---------------------|-----------|------------------------------------|
| L1    | Repo/module summaries | FAISS IVF  | Coarse routing: "which module?"    |
| L2    | Function/class chunks | FAISS IVF  | Main retrieval: "which function?"  |
| L3    | Line-level           | BM25       | Fine-grained: "which line?"        |

**Query flow:**

1. L1 narrows to relevant modules (fast, broad filter).
2. L2 retrieves specific functions/classes within those modules (main index).
3. L3 (optional) pinpoints exact lines when the query is very specific.

L1 and L2 use dense vector search (FAISS). L3 uses BM25 (sparse keyword search)
because line-level matching is better served by exact token matching than by semantic
similarity.

---

## 5. FAISS Incremental Updates

### 5.1 Adding and Removing Vectors

FAISS supports incremental updates through:

- **add_with_ids():** Add new vectors with explicit IDs. Use this for new code files
  or updated chunks. The IDs must be unique int64 values -- use a hash of the file
  path and chunk offset.

- **remove_ids():** Remove vectors by their IDs. Use this when files are deleted or
  chunks are re-generated (remove old, add new).

### 5.2 IVF Retraining

For IVF (Inverted File) indexes, the cluster centroids are computed at index build
time and do not update automatically. As the data distribution shifts (new code added,
old code removed), the centroids become stale and retrieval quality degrades.

**Recommendation:** Retrain IVF centroids monthly, or when more than 20% of vectors
have been added/removed since the last training. Retraining requires a full pass over
the data but does not require re-embedding -- only the index structure is rebuilt.

### 5.3 Compression Tradeoffs

| Method | Size Reduction | Recall Impact             | Recommendation         |
|--------|---------------|---------------------------|------------------------|
| SQ8    | 4x smaller    | ~1-2% recall loss         | Recommended            |
| PQ     | 16-64x smaller| Drops to 56% correlation  | Avoid for code         |

**SQ8 (Scalar Quantization, 8-bit):** Reduces each float32 dimension to uint8,
achieving 4x compression with minimal quality loss. The 1-2% recall loss is acceptable
for code retrieval where top-10 results are re-ranked.

**PQ (Product Quantization):** Aggressive compression that clusters dimension groups.
While effective for natural language and image embeddings, PQ drops to 56% correlation
on code embeddings. Code embeddings have more uniformly important dimensions than NL
embeddings, so the lossy compression of PQ disproportionately damages code retrieval.
**Do not use PQ for code indexes.**

---

## 6. Embedding Model Benchmarks

### 6.1 Code-Specific Embedding Models

| Model                  | Parameters | Avg NDCG@10 (CodeSearchNet) | CoIR NDCG@10 | License    | Deployment   |
|-----------------------|-----------:|----------------------------:|-------------:|------------|-------------|
| Nomic Embed Code 7B   | 7B         | 81.2                        | --           | Apache 2.0 | Local/Ollama |
| Voyage Code 3         | Unknown    | 81.7                        | --           | API only   | API only     |
| CodeRankEmbed 137M    | 137M       | --                          | 60.1         | Apache 2.0 | Local        |

### 6.2 Analysis

- **Nomic Embed Code 7B** is the recommended model for local deployment. At 81.2
  NDCG@10 on CodeSearchNet, it is within 0.5 points of the best API-only model (Voyage
  Code 3) while being fully local and Apache 2.0 licensed. It runs on Ollama, which
  aligns with JCoder's existing embedding infrastructure.

- **Voyage Code 3** achieves the highest score at 81.7, but it is API-only with no
  local deployment option. This makes it unsuitable for offline or air-gapped
  environments and adds per-query cost. It could be used for evaluation/benchmarking
  but should not be a production dependency.

- **CodeRankEmbed 137M** is the lightweight option. At 137M parameters it runs on CPU
  without issues, but its 60.1 CoIR NDCG@10 reflects significantly lower retrieval
  quality. Use it only for rapid prototyping or resource-constrained environments where
  7B is too large.

---

## 7. Recommended Shard Structure

Organize the index into six shards, each independently searchable and updatable:

| Shard              | Content                                  | Estimated Size | Update Frequency |
|--------------------|------------------------------------------|---------------|------------------|
| core-instructions  | Glaive, Magicoder, CodeFeedback          | ~5 GB          | Monthly          |
| stackoverflow      | Stack Overflow Q&A dump                  | ~98 GB         | Quarterly        |
| docs               | Official documentation, API references   | ~5 GB          | Monthly          |
| code-python        | The Stack v2 Python subset, local repos  | ~200 GB        | Monthly          |
| code-multi         | Non-Python code (JS, TS, Rust, Go, etc.) | ~50 GB         | Quarterly        |
| bugs               | Bug reports, CVE data, error patterns    | ~2 GB          | Monthly          |

**Shard independence:** Each shard has its own FAISS index and metadata store. Queries
can target specific shards (e.g., "search only docs") or all shards simultaneously.
This allows incremental updates to one shard without rebuilding others.

---

## 8. TTL (Time-to-Live) Policy

### 8.1 No Expiry

The following data types do not degrade with age and should have no TTL:

- **Stack Overflow answers:** Programming knowledge in accepted answers remains
  accurate indefinitely (language syntax, algorithm implementations, etc.).
- **Official documentation:** API references, language specifications, and framework
  docs for specific versions are permanently valid for those versions.
- **Instruction datasets:** Curated training pairs (Glaive, Magicoder, etc.) are
  static resources that do not go stale.

### 8.2 One-Year TTL

The following data types can become outdated and should be refreshed or removed after
one year:

- **Blog posts and tutorials:** May reference deprecated APIs, outdated best practices,
  or superseded library versions.
- **Commit data and changelogs:** Relevant for understanding recent project evolution
  but becomes noise for older changes.
- **Package version metadata:** Dependency recommendations change frequently.

### 8.3 TTL Implementation

TTL is implemented as a metadata field on each chunk, not as automatic deletion. A
monthly maintenance job:

1. Identifies chunks past their TTL.
2. Flags them for review (not automatic deletion).
3. Optionally de-prioritizes expired chunks in retrieval scoring (multiply relevance
   score by a decay factor).
4. Permanently removes chunks only when confirmed stale by the refresh pipeline.

---

## 9. Recommended Download Order

The phased download plan prioritizes highest-value-per-gigabyte data first. Each phase
is independently useful -- you can start training and retrieval after Phase 1 without
waiting for later phases.

### Phase 1: Instruction Data (~22 GB)

**Download first. Highest training value per gigabyte.**

| Dataset       | Size   | Priority | Rationale                              |
|---------------|--------|----------|----------------------------------------|
| CoRNStack     | 15 GB  | P1       | 21M triplets, retrieval training       |
| Glaive v3     | 2 GB   | P1       | 1M instruction pairs                   |
| CodeSearchNet | 3.5 GB | P1       | NL-to-code search pairs                |
| Magicoder     | 500 MB | P1       | OSS-Instruct, high quality             |
| CodeFeedback  | 2 GB   | P1       | Multi-turn debugging                   |

**Time estimate:** 1-2 hours on a typical broadband connection.

### Phase 2: Documentation (~5 GB)

**Download second. Enables documentation-grounded generation.**

| Dataset              | Size   | Priority | Rationale                        |
|---------------------|--------|----------|----------------------------------|
| Python stdlib docs   | 500 MB | P2       | Core language reference           |
| Top 100 PyPI docs    | 2 GB   | P2       | Most-used library documentation   |
| MDN Web Docs         | 1 GB   | P2       | Web standards reference           |
| Rust stdlib docs     | 500 MB | P2       | Rust language reference           |
| Go stdlib docs       | 500 MB | P2       | Go language reference             |
| Misc framework docs  | 500 MB | P2       | Django, Flask, FastAPI, etc.      |

**Time estimate:** 30-60 minutes.

### Phase 3: Stack Overflow (~100 GB)

**Download third. Massive volume of community knowledge.**

| Dataset                 | Size   | Priority | Rationale                       |
|------------------------|--------|----------|---------------------------------|
| Stack Overflow dump     | 98 GB  | P3       | All Q&A through latest dump     |

**Access:** BitTorrent community mirrors only (no longer on archive.org). Verify
SHA256 checksums after download. The compressed dump is 98 GB; uncompressed XML is
approximately 180 GB.

**Time estimate:** 8-24 hours depending on seed availability.

### Phase 4: Bulk Code (~200 GB)

**Download last. Largest footprint, lowest per-GB value.**

| Dataset                        | Size   | Priority | Rationale                    |
|-------------------------------|--------|----------|------------------------------|
| The Stack v2 (Python subset)   | 191 GB | P4       | 47M Python files             |
| Additional language subsets     | ~10 GB | P4       | JS, TS, Rust, Go subsets     |

**Time estimate:** 24-72 hours.

### Total Footprint

| Phase   | Size     | Cumulative |
|---------|----------|-----------|
| Phase 1 | ~22 GB   | ~22 GB    |
| Phase 2 | ~5 GB    | ~27 GB    |
| Phase 3 | ~100 GB  | ~127 GB   |
| Phase 4 | ~200 GB  | ~327 GB   |

**Storage recommendation:** Allocate 500 GB for raw downloads plus processed indexes.
An NVMe SSD is strongly recommended for the FAISS indexes (Phase 1 and 2 indexes);
bulk storage (Phase 3 and 4 raw data) can be on HDD.

---

## 10. Sources

- cAST: Code-Aware Semantic Tree Chunking -- arxiv.org/abs/2506.15655 (EMNLP 2025)
- Nomic Embed Code -- HuggingFace model card (Apache 2.0)
- CoIR: Code Information Retrieval Benchmark -- ACL 2025
- FAISS Wiki -- github.com/facebookresearch/faiss/wiki (incremental updates, compression)
- CodeSearchNet -- GitHub (Microsoft, 2M code-NL pairs)
- CoRNStack -- Nomic AI (21M retrieval triplets, Apache 2.0)
- Glaive Code Assistant v3 -- HuggingFace (1M instruction pairs)
- Magicoder: Empowering Code Generation with OSS-Instruct -- ICML 2024
- CodeFeedback: Multi-Turn Code Refinement -- HuggingFace
- The Stack v2 -- HuggingFace (BigCode, permissive subset)
