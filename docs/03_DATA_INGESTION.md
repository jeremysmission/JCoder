# JCoder: Data Ingestion Plan

## Overview

This document specifies the full data ingestion pipeline for JCoder: what to
download, how to process it, how to embed it, how to store it, and how to keep
it current. All operations run locally on the build machine (128 GB RAM, 48 GB
dual-3090, 2 TB NVMe, 2 Gbps internet). No cloud dependencies.

---

## Tiered Download Plan

Downloads are organized into three tiers by priority and size. Tier 1 is the
minimum viable knowledge base. Tier 2 adds depth. Tier 3 adds breadth.

### Tier 1 -- Immediate (~10 GB, <1 min at 2 Gbps)

High-value, small datasets that give JCoder immediate utility. Download these
first before anything else.

| Source | Size | Format | Content | Download Command |
|--------|------|--------|---------|------------------|
| tldr-pages | 5 MB | Markdown | Concise CLI command examples for 1,000+ commands | `git clone https://github.com/tldr-pages/tldr data/raw/tldr` |
| Python 3.12 docs | 30 MB | HTML (tar.bz2) | Complete stdlib reference, tutorial, howtos, C API | `wget -P data/raw/python-docs https://docs.python.org/3/archives/python-3.12.9-docs-html.tar.bz2` |
| MDN Web Docs | 200 MB | HTML/Markdown | JS/CSS/HTML reference, Web API docs, HTTP reference | `git clone --depth 1 https://github.com/mdn/content data/raw/mdn-content` |
| CodeSearchNet | 2 GB | JSONL | 6M function-docstring pairs across 6 languages (Python, Java, JS, Go, Ruby, PHP) | `huggingface-cli download code_search_net/code_search_net --local-dir data/raw/codesearchnet` |
| DevDocs.io dump | 1 GB | HTML/JSON | 400+ API docs (Rails, Node, React, Django, etc.) pre-formatted for offline use | `git clone --depth 1 https://github.com/freeCodeCamp/devdocs data/raw/devdocs && cd data/raw/devdocs && bundle exec thor docs:download --all` |
| Magicoder-OSS-Instruct | 1 GB | JSONL | 75K high-quality coding instruction-response pairs generated from OSS seeds | `huggingface-cli download ise-uiuc/Magicoder-OSS-Instruct-75K --local-dir data/raw/magicoder` |
| Glaive Code Assistant v3 | 2 GB | JSONL | 950K multi-turn code Q&A pairs covering debugging, generation, explanation | `huggingface-cli download glaiveai/glaive-code-assistant-v3 --local-dir data/raw/glaive-code` |
| OctoPack/CommitPackFT | 3 GB | JSONL | High-quality commit message + code diff pairs from filtered GitHub history | `huggingface-cli download bigcode/commitpackft --local-dir data/raw/commitpackft` |

**Tier 1 total**: ~10 GB raw, ~8 GB after processing.

### Tier 2 -- Overnight (~27 GB, ~2 min download)

Deeper reference material. Schedule to download while Tier 1 is being processed.

| Source | Size | Format | Content | Download Command |
|--------|------|--------|---------|------------------|
| Stack Overflow Posts.7z | 20 GB | XML (7z-compressed) | All 24M questions + 35M answers, votes, tags, timestamps | `wget -P data/raw/stackoverflow https://archive.org/download/stackexchange/stackoverflow.com-Posts.7z` |
| LeetCode solutions | 2 GB | JSON | Algorithm implementations with problem descriptions, test cases, complexity analysis | `huggingface-cli download greengerong/leetcode --local-dir data/raw/leetcode` |
| Awesome Lists dump | 500 MB | Markdown | Curated project lists from 500+ awesome-* repos, category-tagged | `python tools/scrapers/awesome_scraper.py --output data/raw/awesome-lists` |
| arXiv CS papers (2020-2026) | 5 GB | PDF/LaTeX/text | CS.SE, CS.AI, CS.PL, CS.CL papers -- latest research on code generation, LLMs, program analysis | `python tools/scrapers/arxiv_scraper.py --categories cs.SE cs.AI cs.PL cs.CL --start 2020-01 --output data/raw/arxiv` |

**Tier 2 total**: ~27 GB raw, ~22 GB after processing (SO XML compresses well).

### Tier 3 -- Weekend (~500 GB, ~30 min download)

Massive-scale code corpora. These give JCoder exposure to real-world code
patterns at GitHub scale.

| Source | Size | Format | Content | Download Command |
|--------|------|--------|---------|------------------|
| StarCoderData (permissive) | 250 GB | Parquet | GitHub code filtered to permissive licenses (MIT, Apache 2.0, BSD). 80+ languages. Deduplicated. | `huggingface-cli download bigcode/starcoderdata --include "*.parquet" --local-dir data/raw/starcoderdata` |
| The Stack v2 (filtered) | 200 GB | Parquet | Deduplicated source code from Software Heritage, 600+ languages, near-dedup applied | `huggingface-cli download bigcode/the-stack-v2 --include "*.parquet" --local-dir data/raw/the-stack-v2` |
| SlimPajama | 50 GB | JSONL | Clean web text (RedPajama filtered + deduplicated) for general knowledge grounding | `huggingface-cli download cerebras/SlimPajama-627B --include "*.jsonl.zst" --local-dir data/raw/slimpajama` |

**Tier 3 total**: ~500 GB raw, ~400 GB after dedup and quality filtering.

---

## Processing Pipeline

Every data source passes through the same six-stage processing pipeline. Stages
run sequentially per source but multiple sources can be processed in parallel.

### Stage 1: Format Conversion

Each source format gets converted to a common internal JSONL format. One JSON
object per document/function/post.

| Source Format | Conversion | Tool | Output |
|---------------|-----------|------|--------|
| XML (Stack Overflow) | Parse `<row>` elements, extract Title, Body, Score, Tags, AcceptedAnswerId | `xml.etree.ElementTree` streaming parser | JSONL with fields: id, title, body_html, body_text, score, tags, is_answer, parent_id |
| HTML (Python docs, MDN, DevDocs) | Strip nav/chrome, extract article body, convert to markdown | `beautifulsoup4` + `markdownify` | JSONL with fields: id, title, content_md, url_path, section_hierarchy |
| Markdown (tldr-pages, Awesome Lists) | Parse frontmatter, split by heading structure | Custom parser | JSONL with fields: id, command_name, platform, content_md, examples |
| Parquet (StarCoderData, The Stack v2) | Read with `pyarrow`, extract code + metadata columns | `pyarrow.parquet` | JSONL with fields: id, content, language, license, repo_name, file_path, size |
| JSONL (CodeSearchNet, Magicoder, etc.) | Normalize field names to common schema | Direct field mapping | JSONL with fields: id, content, language, source, metadata |
| PDF/LaTeX (arXiv papers) | Extract text with `pymupdf`, parse LaTeX with regex fallback | `pymupdf` + custom LaTeX stripper | JSONL with fields: id, title, abstract, body_text, arxiv_id, categories, date |

**Common schema** (all sources normalize to this):

```json
{
  "id": "sha256-of-content",
  "source": "stackoverflow",
  "source_id": "12345678",
  "content": "the actual text or code",
  "content_type": "code|documentation|qa|paper|tutorial",
  "language": "python",
  "license": "CC-BY-SA-4.0",
  "metadata": {
    "title": "How to sort a dict by value",
    "score": 4521,
    "tags": ["python", "dictionary", "sorting"],
    "url": "https://stackoverflow.com/q/12345678",
    "date": "2024-03-15"
  }
}
```

### Stage 2: Chunking

Two chunking strategies based on content type.

**Code chunking (tree-sitter AST)**:

- Parse each code file with the appropriate tree-sitter grammar
- Extract top-level units: functions, methods, classes, module-level blocks
- Each chunk = one semantic unit (function, class, or coherent block)
- If a function exceeds 512 tokens, split at logical boundaries (nested functions, loop blocks)
- Minimum chunk size: 32 tokens (skip trivial getters/setters unless part of a class)
- Preserve leading docstrings/comments as chunk metadata
- Attach import context: collect file-level imports, prepend to each chunk as metadata (not embedded content)
- Language grammars to install: Python, JavaScript, TypeScript, Rust, Go, C, C++, Java, Ruby, PHP, C#, Bash, SQL

```
tree-sitter parse flow:
  source_file
    -> function_definition    -> chunk 1
    -> class_definition
       -> method_definition   -> chunk 2
       -> method_definition   -> chunk 3
    -> function_definition    -> chunk 4
```

**Text chunking (documentation, Q&A, papers)**:

- Split by heading structure (h1/h2/h3 boundaries)
- Target chunk size: 384 tokens (optimal for Nomic Embed Code 7B)
- Overlap: 64 tokens between adjacent chunks from the same section
- Never split mid-sentence; backtrack to sentence boundary
- Preserve heading hierarchy as metadata: `["Python", "Data Structures", "Dictionaries", "dict.update()"]`
- Code blocks inside documentation are extracted as separate code chunks and linked back to the parent text chunk

**Stack Overflow special handling**:

- Pair each accepted answer with its question as a single QA chunk
- Include question title as prefix: `Q: {title}\n\n{question_body}\n\nA: {accepted_answer_body}`
- High-vote non-accepted answers (score >= 10) become separate chunks
- Strip HTML formatting, preserve code blocks as fenced markdown
- Filter: minimum question score >= 2, discard closed/duplicate questions

### Stage 3: Deduplication

Two-pass deduplication to remove redundant content.

**Pass 1 -- Exact deduplication (SHA256)**:

- Compute SHA256 of normalized content (whitespace-collapsed, lowercased for text; whitespace-normalized for code)
- Store hashes in a SQLite lookup table: `dedup_hashes(sha256 TEXT PRIMARY KEY, source TEXT, first_seen TEXT)`
- Skip any chunk whose hash already exists
- Expected removal rate: 5-15% across Tier 1-2, 20-30% across Tier 3

**Pass 2 -- Near-duplicate detection (MinHash LSH)**:

- Compute MinHash signatures (128 permutations) for each chunk
- Use Locality-Sensitive Hashing with 64 bands, 2 rows per band
- Jaccard similarity threshold: 0.85 (chunks >85% similar are considered duplicates)
- Keep the chunk with the highest quality score (see Stage 4)
- Library: `datasketch` MinHash + MinHashLSH
- Expected additional removal: 10-20% for code corpora (many forks, copy-paste)

**Dedup database**: Persistent SQLite at `data/dedup.db`, survives across ingestion runs.
New content (from weekly scraper) checks against existing hashes before ingestion.

### Stage 4: Quality Filtering

Remove auto-generated, minified, low-quality, and irrelevant content.

**Code quality filters**:

| Filter | Criteria | Action |
|--------|----------|--------|
| Minified code | Average line length > 200 chars, or >80% of lines lack whitespace | Discard |
| Auto-generated | Contains `DO NOT EDIT`, `auto-generated`, `generated by` in first 5 lines | Discard |
| Boilerplate | Identical structure to known templates (license headers, empty main) | Discard |
| Too short | <32 tokens after stripping comments | Discard |
| Too long | >2048 tokens (single function) | Split at Stage 2, flag for review |
| Low entropy | Shannon entropy < 3.0 bits/char (repetitive patterns) | Discard |
| Binary/data | High proportion of non-printable or numeric-only content | Discard |
| Test-only | File path matches `test_*`, `*_test.*`, `spec/*` and contains only assertions | Keep but tag as `content_type: test` |

**Text quality filters**:

| Filter | Criteria | Action |
|--------|----------|--------|
| Too short | <50 characters of actual content | Discard |
| Non-English | `langdetect` confidence <0.8 for English | Discard (English-only for v1) |
| Low-score SO posts | Question score < 2 or answer score < 1 | Discard |
| Spam indicators | Excessive URLs, promotional language patterns | Discard |
| Stale docs | Last updated before 2018 (for framework-specific docs) | Tag as `stale: true`, lower priority in retrieval |

**Quality score** (0.0 to 1.0, stored in metadata):

```
quality_score = weighted_average(
    0.3 * popularity,      # SO score, GitHub stars, citation count
    0.3 * recency,         # Decay function: 1.0 for 2026, -0.1 per year
    0.2 * completeness,    # Has docstring, type hints, error handling
    0.2 * uniqueness       # Inverse of near-duplicate cluster size
)
```

### Stage 5: Language Detection and Tagging

Every chunk gets a language tag for filtered retrieval.

**Code language detection**:

- Primary: tree-sitter grammar used during parsing (definitive)
- Fallback for ambiguous files: file extension mapping
- Fallback for extension-less: `guesslang` ML classifier
- Store as ISO-style tag: `python`, `javascript`, `typescript`, `rust`, `go`, `c`, `cpp`, `java`, `ruby`, `php`, `csharp`, `bash`, `sql`, `html`, `css`

**Natural language detection**:

- Library: `langdetect` (port of Google's language-detection)
- Store ISO 639-1 code: `en`, `zh`, `ja`, etc.
- v1 indexes English only; other languages are preserved in raw storage for future expansion

**Content type tagging**:

Each chunk gets exactly one content type tag:

| Tag | Description | Example Sources |
|-----|-------------|-----------------|
| `code` | Standalone code (functions, classes, scripts) | StarCoderData, The Stack, CodeSearchNet |
| `documentation` | API docs, reference material, tutorials | Python docs, MDN, DevDocs |
| `qa` | Question-answer pairs | Stack Overflow, Glaive Code Assistant |
| `instruction` | Coding instruction-response pairs | Magicoder, CommitPackFT |
| `paper` | Research papers and technical articles | arXiv |
| `tutorial` | Step-by-step guides and howtos | tldr-pages, Awesome Lists |
| `test` | Test code, assertions, fixtures | Extracted test files |

### Stage 6: Metadata Extraction

Final metadata assembly before embedding. Every chunk stored in the processing
database includes:

```json
{
  "chunk_id": "sha256-first-16-chars",
  "source": "stackoverflow",
  "source_id": "12345678",
  "content_type": "qa",
  "language_code": "python",
  "language_natural": "en",
  "license": "CC-BY-SA-4.0",
  "quality_score": 0.87,
  "token_count": 312,
  "date_created": "2024-03-15",
  "date_ingested": "2026-03-01",
  "tags": ["sorting", "dictionary", "python"],
  "title": "How to sort a dict by value in Python 3",
  "url": "https://stackoverflow.com/q/12345678",
  "section_path": null,
  "file_path": null,
  "repo_name": null,
  "stale": false
}
```

This metadata is stored in SQLite (see Index Storage) and used for filtered
retrieval (e.g., "search only Python documentation" or "search only code with
quality > 0.7").

---

## Embedding Pipeline

### Model

Nomic Embed Code 7B served via vLLM at `http://127.0.0.1:8101/v1/embeddings`.

| Property | Value |
|----------|-------|
| Model | Nomic Embed Code 7B (Q4_K_M) |
| VRAM | ~4.4 GB on GPU 0 |
| Dimensions | 768 |
| Max tokens | 8192 per chunk |
| API | OpenAI-compatible /v1/embeddings |

### Batching Strategy

```
Batch size:          64 chunks per API call
Max tokens per call: 64 * 384 avg = ~24,576 tokens (well within limits)
Concurrent workers:  4 (httpx async, 4 in-flight requests)
Effective batch:     256 chunks in pipeline at any time
```

The embedding service runs alongside the LLM on GPU 0. During bulk ingestion,
the LLM is not loaded (vLLM serves only the embedder). During normal operation,
the embedder shares GPU 0 with the reranker.

### Throughput Estimates

Measured on a single RTX 3090 with Nomic Embed Code 7B Q4:

| Metric | Value |
|--------|-------|
| Chunks per second | ~17 |
| Chunks per minute | ~1,000 |
| Chunks per hour | ~60,000 |

### Embedding Time by Tier

| Tier | Estimated Chunks | Embedding Time | Notes |
|------|-----------------|----------------|-------|
| Tier 1 (~10 GB raw) | ~2M chunks | ~33 hours | Run over 2 days, can parallelize |
| Tier 2 (~27 GB raw) | ~5M chunks | ~83 hours | Run over 4 days |
| Tier 3 (~500 GB raw) | ~80M chunks | ~1,333 hours (~56 days) | Subset strategy below |

**Tier 3 subset strategy**: Embedding all 80M chunks from Tier 3 takes nearly
two months. Instead, apply aggressive quality filtering first:

1. Filter to top 20% by quality score: 80M -> 16M chunks
2. Prioritize the 10 most-used languages: 16M -> 12M chunks
3. Embedding time for filtered set: ~200 hours (~8 days)
4. Remaining chunks stored in raw JSONL for on-demand embedding later

### Embedding Checkpointing

- Progress tracked in `data/embed_progress.db` (SQLite)
- Table: `embed_status(chunk_id TEXT PRIMARY KEY, embedded BOOLEAN, embed_time TEXT, batch_id TEXT)`
- On crash/restart, resume from last incomplete batch
- Batch results written to disk every 100 batches (6,400 chunks)

### Embedding Request Format

```python
# Single batch request to vLLM embeddings endpoint
response = httpx.post(
    "http://127.0.0.1:8101/v1/embeddings",
    json={
        "model": "nomic-ai/nomic-embed-code-v1",
        "input": [chunk.content for chunk in batch],  # list of 64 strings
        "encoding_format": "float"
    },
    timeout=120.0
)
vectors = [item["embedding"] for item in response.json()["data"]]
```

---

## Index Storage

### Dense Vector Index (FAISS)

| Property | Value |
|----------|-------|
| Index type | IVFFlat (GPU-accelerated training, CPU serving) |
| Dimensions | 768 |
| nlist (clusters) | sqrt(N) rounded up, retrained when N doubles |
| nprobe (search) | 32 (tunable, balances speed vs recall) |
| Metric | Inner product (cosine similarity with normalized vectors) |
| Storage format | Single `.faiss` file + `.faiss.meta` ID mapping |

**Index size estimates**:

| Tier | Chunks | Vector Size | Index Overhead | Total FAISS Size |
|------|--------|-------------|----------------|------------------|
| Tier 1 | 2M | 2M * 768 * 4B = 5.7 GB | ~1.5 GB (IVF lists, centroids) | ~7 GB |
| Tier 1+2 | 7M | 7M * 768 * 4B = 20 GB | ~5 GB | ~25 GB |
| Tier 1+2+3 (filtered) | 19M | 19M * 768 * 4B = 55 GB | ~14 GB | ~69 GB |

**GPU training**: IVF centroids are trained on GPU (fast), then the index is
moved to CPU RAM for serving. At 128 GB system RAM, even the full 69 GB index
fits comfortably with room for the rest of the application.

**Incremental additions**: New vectors from the weekly scraper are added to
the existing IVF index via `index.add_with_ids()`. Full retrain of centroids
happens monthly or when index size doubles (whichever comes first).

### Keyword Index (SQLite FTS5)

| Property | Value |
|----------|-------|
| Engine | SQLite FTS5 (built-in, no external dependencies) |
| Tokenizer | `porter unicode61` (stemming + Unicode normalization) |
| Indexed fields | content, title, tags |
| Storage | `data/indexes/fts5.db` |

```sql
CREATE VIRTUAL TABLE chunks_fts USING fts5(
    chunk_id,
    content,
    title,
    tags,
    language_code,
    content_type,
    tokenize='porter unicode61'
);
```

**FTS5 size estimates**:

| Tier | Chunks | Estimated FTS5 Size |
|------|--------|---------------------|
| Tier 1 | 2M | ~3 GB |
| Tier 1+2 | 7M | ~10 GB |
| Tier 1+2+3 | 19M | ~27 GB |

### Metadata Store (SQLite)

All chunk metadata lives in a separate SQLite database for fast filtered
lookups without touching the vector index.

```sql
CREATE TABLE chunk_meta (
    chunk_id       TEXT PRIMARY KEY,
    source         TEXT NOT NULL,
    source_id      TEXT,
    content_type   TEXT NOT NULL,
    language_code  TEXT,
    language_nat   TEXT DEFAULT 'en',
    license        TEXT,
    quality_score  REAL,
    token_count    INTEGER,
    date_created   TEXT,
    date_ingested  TEXT NOT NULL,
    tags           TEXT,  -- JSON array stored as text
    title          TEXT,
    url            TEXT,
    file_path      TEXT,
    repo_name      TEXT,
    stale          INTEGER DEFAULT 0
);

CREATE INDEX idx_meta_source ON chunk_meta(source);
CREATE INDEX idx_meta_language ON chunk_meta(language_code);
CREATE INDEX idx_meta_type ON chunk_meta(content_type);
CREATE INDEX idx_meta_quality ON chunk_meta(quality_score);
CREATE INDEX idx_meta_date ON chunk_meta(date_created);
```

**Metadata DB size estimates**:

| Tier | Chunks | Estimated Size |
|------|--------|----------------|
| Tier 1 | 2M | ~500 MB |
| Tier 1+2 | 7M | ~1.7 GB |
| Tier 1+2+3 | 19M | ~4.5 GB |

### Index Storage Summary

| Component | Tier 1 | Tier 1+2 | Tier 1+2+3 |
|-----------|--------|----------|------------|
| FAISS IVF | 7 GB | 25 GB | 69 GB |
| FTS5 | 3 GB | 10 GB | 27 GB |
| Metadata SQLite | 0.5 GB | 1.7 GB | 4.5 GB |
| **Total indexes** | **10.5 GB** | **36.7 GB** | **100.5 GB** |

---

## Weekly Scraper Design

After initial ingestion, JCoder stays current through an automated weekly
scraper that discovers, downloads, summarizes, and indexes new content.

### Sources

| Source | URL Pattern | Content Type | Expected Volume |
|--------|-------------|--------------|-----------------|
| GitHub Trending | `https://github.com/trending?since=weekly` | Repositories, README files, star code snippets | ~100 repos/week |
| Hacker News Top | `https://hacker-news.firebaseio.com/v0/topstories.json` | Technical articles, discussions | ~200 stories/week |
| Reddit r/programming | `https://www.reddit.com/r/programming/top/.json?t=week` | Articles, discussions, code snippets | ~50 posts/week |
| Reddit r/python | `https://www.reddit.com/r/python/top/.json?t=week` | Python-specific news, libraries, tutorials | ~50 posts/week |
| Reddit r/rust | `https://www.reddit.com/r/rust/top/.json?t=week` | Rust ecosystem updates | ~30 posts/week |
| Reddit r/golang | `https://www.reddit.com/r/golang/top/.json?t=week` | Go ecosystem updates | ~30 posts/week |
| Reddit r/typescript | `https://www.reddit.com/r/typescript/top/.json?t=week` | TS ecosystem updates | ~20 posts/week |
| arXiv cs.SE + cs.AI | `https://arxiv.org/list/cs.SE/new` | New research papers | ~50 papers/week |
| Framework releases | Per-framework RSS/changelog URLs | Release notes, migration guides, new API docs | ~20 releases/week |

**Framework release tracking** (checked weekly):

- Python: `https://docs.python.org/3/whatsnew/`
- Node.js: `https://nodejs.org/en/blog`
- React: `https://react.dev/blog`
- Rust: `https://blog.rust-lang.org/`
- Go: `https://go.dev/blog/`
- Django: `https://www.djangoproject.com/weblog/`
- FastAPI: `https://github.com/tiangolo/fastapi/releases`
- TypeScript: `https://devblogs.microsoft.com/typescript/`

### Scraper Architecture

```
tools/scrapers/weekly_scraper.py
    |
    +-- scrapers/
    |     +-- github_trending.py    # GitHub trending repos
    |     +-- hackernews.py         # HN top stories + linked articles
    |     +-- reddit.py             # Multi-subreddit scraper
    |     +-- arxiv.py              # arXiv new papers
    |     +-- releases.py           # Framework release notes
    |
    +-- summarizer.py               # Qwen3-Coder summarization
    +-- ingestor.py                 # Embed + index new chunks
    +-- changelog.py                # Weekly digest generator
    +-- scheduler.py                # Task scheduler wrapper
```

### Scraper Implementation

**HTTP client**: `httpx` with async support for parallel fetching.

```python
# Core scraper pattern
async def scrape_source(source_config: SourceConfig) -> list[RawDocument]:
    async with httpx.AsyncClient(
        timeout=30.0,
        follow_redirects=True,
        headers={"User-Agent": "JCoder/1.0 (offline-indexer; +https://github.com/jcoder)"}
    ) as client:
        # Respect robots.txt
        if not await check_robots_txt(client, source_config.base_url):
            log.warn(f"[WARN] Skipping {source_config.name}: blocked by robots.txt")
            return []

        # Rate limiting: 1 request per second per domain
        async with rate_limiter(source_config.domain, requests_per_second=1):
            response = await client.get(source_config.url)
            response.raise_for_status()

        return parse_response(response, source_config)
```

**Rate limiting rules**:

| Domain | Max Requests/Second | Notes |
|--------|---------------------|-------|
| github.com | 1 | Unauthenticated API limit: 60/hour |
| reddit.com | 0.5 | Reddit API rules: 1 req/2 sec |
| arxiv.org | 0.33 | arXiv asks for 3-second delays |
| All others | 1 | Default polite crawling rate |

**robots.txt compliance**: Every domain's robots.txt is fetched once per
scraper run, cached in memory, and checked before each request.

### Summarizer

Each scraped article or repository is summarized by Qwen3-Coder-Next 80B
into structured chunks suitable for RAG indexing.

```python
SUMMARIZE_PROMPT = """Summarize this technical content for a code knowledge base.

Output format:
- TITLE: one-line title
- TAGS: comma-separated topic tags
- SUMMARY: 2-3 paragraph technical summary
- KEY_POINTS: bulleted list of actionable takeaways
- CODE_SNIPPETS: any important code examples, preserved exactly

Content:
{content}
"""
```

Each summarized article produces 1-3 chunks depending on length. Code snippets
are extracted as separate code-type chunks linked to the parent summary.

### Incremental Ingest

New chunks from the weekly scraper follow the same pipeline as bulk ingestion
but are added incrementally without full index rebuilds.

1. **Dedup check**: SHA256 of URL + content checked against `data/dedup.db`
2. **Quality filter**: Same filters as bulk ingestion
3. **Embed**: Batch embed new chunks via Nomic Embed Code 7B
4. **FAISS add**: `index.add_with_ids(new_vectors, new_ids)` -- no retrain needed
5. **FTS5 insert**: Standard SQL INSERT into the FTS5 table
6. **Metadata insert**: INSERT into chunk_meta table

Expected weekly volume: 300-500 new chunks, ~1-2 minutes to embed and index.

### Dedup Guard

Before any new content enters the pipeline:

```python
def is_duplicate(url: str, content: str) -> bool:
    content_hash = hashlib.sha256(content.encode()).hexdigest()
    url_hash = hashlib.sha256(url.encode()).hexdigest()
    combined_hash = hashlib.sha256(f"{url_hash}:{content_hash}".encode()).hexdigest()

    with sqlite3.connect("data/dedup.db") as conn:
        exists = conn.execute(
            "SELECT 1 FROM dedup_hashes WHERE sha256 = ?", (combined_hash,)
        ).fetchone()
        if exists:
            return True
        conn.execute(
            "INSERT INTO dedup_hashes (sha256, source, first_seen) VALUES (?, ?, ?)",
            (combined_hash, url, datetime.now().isoformat())
        )
        return False
```

### Schedule

| Setting | Value |
|---------|-------|
| Trigger | Windows Task Scheduler |
| Schedule | Every Sunday at 02:00 AM |
| Runtime | ~30-60 minutes (scrape + summarize + embed + index) |
| Lock file | `data/scraper.lock` (prevents overlapping runs) |
| Timeout | 2 hours (kill if stuck) |

**Task Scheduler XML** (imported via `schtasks /create /xml`):

```xml
<Task>
  <Triggers>
    <CalendarTrigger>
      <StartBoundary>2026-03-01T02:00:00</StartBoundary>
      <ScheduleByWeek>
        <DaysOfWeek><Sunday/></DaysOfWeek>
        <WeeksInterval>1</WeeksInterval>
      </ScheduleByWeek>
    </CalendarTrigger>
  </Triggers>
  <Actions>
    <Exec>
      <Command>D:\JCoder\.venv\Scripts\python.exe</Command>
      <Arguments>tools/scrapers/weekly_scraper.py --all</Arguments>
      <WorkingDirectory>D:\JCoder</WorkingDirectory>
    </Exec>
  </Actions>
</Task>
```

### Weekly Changelog

After each scraper run, a digest is generated at `data/changelogs/YYYY-WNN.md`:

```markdown
# Weekly Knowledge Digest -- 2026-W10

## Stats
- Sources scraped: 9
- Articles fetched: 487
- New chunks added: 423
- Duplicates skipped: 64
- Total index size: 7,012,423 chunks

## Highlights
### GitHub Trending
- **repo/name** (Python, 1.2K stars this week): Description...

### New Papers
- **"Title of Paper"** (arXiv 2603.12345): Summary...

### Framework Releases
- **React 20.1**: New feature X, deprecation of Y...
```

### Storage Budget for Weekly Scraper

| Metric | Value |
|--------|-------|
| New raw content per week | ~50-100 MB |
| New embeddings per week | ~1-2 MB (500 chunks * 768 dims * 4 bytes) |
| New metadata per week | ~100 KB |
| Changelog per week | ~10 KB |
| **Monthly total** | **~400 MB - 1 GB** |
| **Annual total** | **~5-12 GB** |

---

## Disk Budget

Running total of all disk usage. Must fit in 2 TB.

### Raw Data

| Component | Size |
|-----------|------|
| Tier 1 raw downloads | 10 GB |
| Tier 2 raw downloads | 27 GB |
| Tier 3 raw downloads | 500 GB |
| **Subtotal raw data** | **537 GB** |

### Processed Data

| Component | Size |
|-----------|------|
| Tier 1 processed JSONL | 8 GB |
| Tier 2 processed JSONL | 22 GB |
| Tier 3 processed JSONL (filtered 20%) | 80 GB |
| **Subtotal processed** | **110 GB** |

### Indexes

| Component | Size |
|-----------|------|
| FAISS IVF (all tiers, filtered) | 69 GB |
| SQLite FTS5 | 27 GB |
| Metadata SQLite | 4.5 GB |
| Dedup database | 2 GB |
| Embed progress DB | 500 MB |
| **Subtotal indexes** | **103 GB** |

### Models

| Component | Size |
|-----------|------|
| Qwen3-Coder-Next 80B (Q3_K_M) | 37 GB |
| Nomic Embed Code 7B (Q4_K_M) | 4.4 GB |
| Qwen3-Reranker-4B (Q4) | 3 GB |
| tree-sitter grammars (13 languages) | 200 MB |
| **Subtotal models** | **44.6 GB** |

### Working Space

| Component | Size |
|-----------|------|
| Python venv | 5 GB |
| vLLM KV cache (disk swap) | 20 GB |
| Temp processing space | 50 GB |
| Download resume cache | 10 GB |
| Weekly scraper annual growth | 12 GB |
| Logs and changelogs | 2 GB |
| **Subtotal working space** | **99 GB** |

### Grand Total

| Category | Size |
|----------|------|
| Raw data | 537 GB |
| Processed data | 110 GB |
| Indexes | 103 GB |
| Models | 44.6 GB |
| Working space | 99 GB |
| **GRAND TOTAL** | **893.6 GB** |
| **Remaining on 2 TB** | **~1,106 GB (55% free)** |

**Notes**:

- Raw Tier 3 data (500 GB) can be deleted after processing+embedding to reclaim
  space if needed. This would bring the total to ~394 GB with 1,606 GB free.
- The 55% free headroom accommodates: future Tier 3 expansion, additional model
  downloads (fine-tuned checkpoints), and multi-year scraper accumulation.
- Disk usage monitoring script alerts at 80% capacity (1.6 TB).

---

## Download Automation Script Design

A single Python script handles all downloads with resume support, verification,
and automatic post-processing.

### Script: `tools/download_data.py`

```
Usage:
  python tools/download_data.py --tier 1          # Download Tier 1 only
  python tools/download_data.py --tier 2          # Download Tier 2 only
  python tools/download_data.py --tier 3          # Download Tier 3 only
  python tools/download_data.py --tier 1 2        # Download Tier 1 and 2
  python tools/download_data.py --all             # Download all tiers
  python tools/download_data.py --source tldr     # Download single source
  python tools/download_data.py --status          # Show download status
  python tools/download_data.py --verify          # Verify checksums only
  python tools/download_data.py --resume          # Resume interrupted downloads
```

### Features

**Resume support**:

- All HTTP downloads use `Range` headers to resume partial transfers
- `huggingface-cli download` has built-in resume support
- `git clone` uses `--depth 1` for initial clone; `git pull` for updates
- Download state tracked in `data/download_log.json`

**Progress display (Rich)**:

```
Downloading Tier 1 (8 sources, ~10 GB)
+-----------------------+--------+----------+-------+--------+
| Source                | Size   | Progress | Speed | ETA    |
+-----------------------+--------+----------+-------+--------+
| tldr-pages            | 5 MB   | DONE     | --    | --     |
| Python 3.12 docs      | 30 MB  | DONE     | --    | --     |
| MDN Web Docs          | 200 MB | 67%      | 180MB | 0:01   |
| CodeSearchNet         | 2 GB   | pending  | --    | --     |
| ...                   |        |          |       |        |
+-----------------------+--------+----------+-------+--------+
Overall: 235 MB / 10 GB [===>                    ] 2.3%  ETA: 0:38
```

**Checksum verification**:

- Each source has a known checksum (SHA256) stored in `config/data_checksums.yaml`
- After download completes, compute SHA256 of the downloaded file/directory
- If mismatch: log warning, offer re-download
- For HuggingFace datasets: verify against the repo's `.sha256` files
- For git clones: verify HEAD commit hash against expected

**Automatic format conversion**:

After each source downloads successfully, the conversion pipeline runs
automatically:

1. Detect format (XML, HTML, Parquet, JSONL, Markdown, PDF)
2. Run the appropriate converter from `tools/converters/`
3. Output normalized JSONL to `data/processed/{source_name}/`
4. Log conversion stats to `data/download_log.json`

**Logging**:

All download activity is logged to `data/download_log.json`:

```json
{
  "downloads": [
    {
      "source": "tldr-pages",
      "tier": 1,
      "url": "https://github.com/tldr-pages/tldr",
      "method": "git_clone",
      "started": "2026-03-01T10:00:00",
      "completed": "2026-03-01T10:00:03",
      "status": "complete",
      "size_bytes": 5242880,
      "checksum_sha256": "abc123...",
      "checksum_verified": true,
      "output_path": "data/raw/tldr",
      "conversion_status": "complete",
      "conversion_output": "data/processed/tldr/",
      "chunks_produced": 4521,
      "errors": []
    }
  ],
  "summary": {
    "tier_1_complete": true,
    "tier_2_complete": false,
    "tier_3_complete": false,
    "total_downloaded_gb": 10.2,
    "total_chunks": 2041523,
    "last_run": "2026-03-01T10:45:00"
  }
}
```

### Error Handling

| Error | Recovery |
|-------|----------|
| Network timeout | Retry 3 times with exponential backoff (5s, 15s, 45s) |
| Partial download | Resume from last byte using Range header |
| Checksum mismatch | Delete and re-download from scratch |
| Disk full | Abort with clear error message, log partial progress |
| HuggingFace rate limit | Wait for `Retry-After` header, then continue |
| Git clone failure | Delete partial clone directory, retry |
| Conversion failure | Log error, skip source, continue with next |

### Script Dependencies

```
httpx>=0.28          # HTTP client with streaming and Range support
rich>=13.0           # Progress bars, tables, panels
pyyaml>=6.0          # Config and checksum files
py7zr>=0.22          # Extract Stack Overflow .7z archive
pyarrow>=18.0        # Read Parquet files (Tier 3)
beautifulsoup4>=4.12 # HTML parsing for format conversion
markdownify>=0.14    # HTML to Markdown conversion
pymupdf>=1.25        # PDF text extraction (arXiv papers)
langdetect>=1.0.9    # Language detection
datasketch>=1.6      # MinHash deduplication
```

---

## Ingestion Order of Operations

The complete ingestion process from zero to fully indexed:

```
1. Install dependencies
   pip install -r requirements.txt

2. Start embedding server (dedicated, no LLM loaded)
   python -m vllm.entrypoints.openai.api_server \
       --model nomic-ai/nomic-embed-code-v1 \
       --port 8101 \
       --gpu-memory-utilization 0.3

3. Download Tier 1
   python tools/download_data.py --tier 1

4. Process Tier 1 (convert + chunk + dedup + filter + tag)
   python tools/process_data.py --tier 1

5. Embed Tier 1
   python tools/embed_data.py --tier 1

6. Build initial indexes
   python tools/build_index.py --rebuild

7. Verify: run test queries against Tier 1 index
   python tools/verify_index.py --smoke-test

8. Download + process + embed Tier 2 (can run overnight)
   python tools/download_data.py --tier 2
   python tools/process_data.py --tier 2
   python tools/embed_data.py --tier 2
   python tools/build_index.py --incremental

9. Download + process + embed Tier 3 (weekend job)
   python tools/download_data.py --tier 3
   python tools/process_data.py --tier 3
   python tools/embed_data.py --tier 3
   python tools/build_index.py --incremental

10. Enable weekly scraper
    schtasks /create /xml tools/scrapers/weekly_task.xml /tn "JCoder Weekly Scraper"

11. Final verification
    python tools/verify_index.py --full
```

**Estimated total time** (wall clock, sequential):

| Step | Duration |
|------|----------|
| Tier 1 download | <1 min |
| Tier 1 process | ~2 hours |
| Tier 1 embed | ~33 hours |
| Tier 2 download | ~2 min |
| Tier 2 process | ~4 hours |
| Tier 2 embed | ~83 hours |
| Tier 3 download | ~30 min |
| Tier 3 process | ~24 hours |
| Tier 3 embed (filtered) | ~200 hours |
| Index builds | ~4 hours |
| **Total** | **~14.5 days** |

JCoder is fully usable after Tier 1 completes (~35 hours). Tier 2 and 3
run in the background to expand the knowledge base over the following two weeks.
