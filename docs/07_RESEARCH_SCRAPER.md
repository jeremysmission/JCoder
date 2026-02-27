# 07 -- Autonomous Scraping: Research Findings

Document: 07_RESEARCH_SCRAPER.md
Created: 2026-02-26
Status: Research compilation -- not yet implemented

---

## 1. Sources Ranked by Signal-to-Noise

### Tier 1 -- High Signal, Low Noise (Poll Weekly or More)

These sources have structured APIs, high relevance density, and minimal junk content.

| Source               | Access Method                  | Volume         | Auth Required |
|----------------------|--------------------------------|----------------|---------------|
| GitHub Releases      | REST API (repos/:owner/:repo)  | 50-100 KB/week | Token (free)  |
| GitHub Trending      | Scrape trending page or API    | 10-20 KB/week  | None          |
| Security Advisories  | NVD API 2.0, GitHub Advisory   | 20-50 KB/week  | NVD: API key  |
| arXiv (cs.*)         | arxiv API (OAI-PMH or REST)    | 200-500 KB/week| None          |
| PyPI Releases        | RSS feeds + JSON API           | 50-100 KB/week | None          |

**GitHub Releases (50 repos)**: Monitor release events for the 50 most critical
dependencies and tools. The REST API returns structured JSON with tag name, body
(release notes in markdown), assets, and timestamps. Rate limit: 5,000 req/hour
with token authentication.

**GitHub Trending**: The trending page (github.com/trending) surfaces new projects
gaining stars rapidly. Can be scraped directly or accessed via community wrappers
like Github-Trending-API. Useful for discovering new tools before they appear in
blog posts.

**Security Advisories**: Two complementary sources:
- NVD API 2.0 (nvdlib): NIST National Vulnerability Database. Query by CPE name,
  CVE ID, or keyword. Returns structured CVSS scores, affected versions, and
  references. Rate limit: 5 req/30s without API key, 50 req/30s with key.
- GitHub Advisory Database (ghsa-client): GitHub's own advisory database, often
  faster than NVD for open-source vulnerabilities. GraphQL API.

**arXiv (cs.SE, cs.PL, cs.AI, cs.CL)**: The arxiv API supports OAI-PMH harvesting
and a simpler REST query interface. Mandatory 3-second delay between requests.
Abstracts only (full PDFs are large and rarely needed for awareness). Typical
volume: 200-500 KB/week of abstracts across the four categories.

**PyPI Releases**: Every PyPI package has an RSS feed at
pypi.org/rss/project/{name}/releases.xml and a JSON API at
pypi.org/pypi/{name}/json. No rate limit documented. Monitor the 50 key
dependencies for version bumps, security patches, and deprecation notices.

### Tier 2 -- Good Signal, Moderate Noise (Poll Weekly)

| Source               | Access Method                  | Volume          | Auth Required |
|----------------------|--------------------------------|-----------------|---------------|
| Hacker News          | Algolia API                    | 100-300 KB/week | None          |
| Official Blogs       | RSS/Atom via feedparser        | 50-200 KB/week  | None          |
| Changelog.com        | RSS feed                       | 20-50 KB/week   | None          |
| TLDR Newsletter      | RSS feed                       | 10-30 KB/week   | None          |

**Hacker News**: The Algolia API (hn.algolia.com/api) provides full search over HN
stories and comments. No authentication required. Filter by points > 50 to cut
noise. Typical query: search for "LLM" or "RAG" or "code generation" stories from
the past week with more than 50 points. Returns title, URL, points, and comment
count.

**Official Blogs**: Maintain a list of RSS/Atom feed URLs for key projects (Python,
VS Code, Rust, Go, Node.js, FastAPI, etc.). Use feedparser to poll weekly. Most
official blogs have high signal-to-noise ratios because they only post for
significant releases or announcements.

**Changelog.com and TLDR Newsletter**: Curated developer news. Already filtered
for relevance by human editors. RSS feeds available.

### Tier 3 -- Mixed Signal, Higher Noise (Poll Bi-Weekly)

| Source               | Access Method                  | Volume          | Auth Required |
|----------------------|--------------------------------|-----------------|---------------|
| Reddit               | PRAW or .json URL suffix       | 200-500 KB/poll | OAuth (free)  |
| npm Trends           | npm API + bundlephobia         | 20-50 KB/poll   | None          |
| crates.io Trends     | crates.io API                  | 10-30 KB/poll   | None          |

**Reddit**: PRAW (Python Reddit API Wrapper) or simply appending .json to any
Reddit URL. Free tier: 100 requests/minute after registering an OAuth app.
Subreddits: r/programming, r/python, r/rust, r/machinelearning. Bi-weekly polling
is sufficient because Reddit content has a longer shelf life than HN.

**npm/crates.io**: Track download trends and new releases for key JavaScript and
Rust packages. Useful for ecosystem awareness but lower priority than Python
tooling.

---

## 2. Recommended Scraping Stack

### 2.1 Crawl4AI (Primary Web Scraper)

- **GitHub stars**: 58K+
- **License**: Apache 2.0
- **Core capability**: Fetches web pages, executes JavaScript via Playwright,
  outputs clean markdown
- **Key features**:
  - Built-in Playwright integration for JS-rendered pages (SPAs, React sites)
  - Markdown output mode (not raw HTML)
  - Configurable extraction strategies
  - Async operation for parallel fetching
  - Proxy support
  - Respectful crawling (configurable delays, robots.txt)

**Reference**: docs.crawl4ai.com

### 2.2 trafilatura (Lightweight HTML-to-Text)

- **Purpose**: Extract article text from HTML, stripping navigation, ads, footers
- **Performance**: Benchmarks show trafilatura outperforms newspaper3k, readability,
  and jusText on precision and recall for main content extraction
- **Use case**: Processing blog posts, documentation pages, and news articles where
  Crawl4AI's full browser rendering is unnecessary
- **License**: Apache 2.0

**Reference**: trafilatura.readthedocs.io

### 2.3 httpx (API Client)

- **Purpose**: Direct API calls to GitHub, arXiv, NVD, PyPI, HN Algolia
- **Features**: Async support, HTTP/2, connection pooling, timeout configuration
- **Already in use**: HybridRAG3 uses httpx for Ollama communication

### 2.4 feedparser (RSS/Atom)

- **Purpose**: Parse RSS and Atom feeds from official blogs, PyPI, Changelog.com
- **Maturity**: 20+ years old, handles malformed feeds gracefully
- **License**: BSD

### 2.5 APScheduler (Task Scheduling)

- **Purpose**: Python-native job scheduling (cron-like, interval-based, or
  date-based triggers)
- **Features**:
  - In-process scheduler (no external daemon)
  - Persistent job store (SQLite or memory)
  - Missed job handling (run immediately or skip)
  - Timezone-aware
- **Use case**: Schedule Tier 1 sources hourly/daily, Tier 2 weekly, Tier 3
  bi-weekly, all within the JCoder process

**Reference**: apscheduler.readthedocs.io

---

## 3. Content Processing Pipeline

### 3.1 Pipeline Stages

```
1. FETCH
   |  httpx (APIs) / Crawl4AI (web pages) / feedparser (RSS)
   v
2. EXTRACT
   |  HTML -> markdown (trafilatura or Crawl4AI markdown mode)
   |  Preserve code blocks as-is (do not summarize code)
   v
3. CODE BLOCK EXTRACTION
   |  Regex or tree-sitter to identify and tag fenced code blocks
   |  Store language annotation (```python, ```rust, etc.)
   v
4. LLM SUMMARIZATION
   |  Local model (phi4-mini) generates 2-3 sentence summary
   |  Extract: topic tags, mentioned libraries, language, category
   v
5. QUALITY SCORING
   |  Composite score (see Section 3.2)
   |  Reject items scoring below 0.30 threshold
   v
6. DEDUPLICATION (three layers)
   |  Layer 1: URL exact match (seen this URL before?)
   |  Layer 2: SHA-256 content hash (same content, different URL?)
   |  Layer 3: Semantic similarity > 0.92 (paraphrase of existing item?)
   v
7. EMBED + INDEX
   |  Ollama nomic-embed-text (768 dimensions)
   |  Store in LanceDB with metadata (source, date, score, tags)
   v
   DONE -- item available for retrieval
```

### 3.2 Quality Scoring Formula

Each ingested item receives a composite quality score:

```
quality = 0.30 * authority
        + 0.25 * engagement
        + 0.20 * recency
        + 0.15 * density
        + 0.10 * uniqueness
```

**Component definitions**:

- **authority** (0.30): Source reputation. GitHub official releases = 1.0,
  arXiv papers = 0.9, official blog posts = 0.85, HN front page = 0.7,
  Reddit top posts = 0.5, random blog = 0.3.

- **engagement** (0.25): Normalized interaction metrics. GitHub stars/forks,
  HN points, Reddit upvotes. Normalized to [0, 1] using log scale with
  source-specific thresholds (e.g., HN: log(points)/log(500)).

- **recency** (0.20): Time decay factor. 1.0 for today, 0.5 at the
  category-specific half-life (see Section 4).

- **density** (0.15): Information density. Ratio of substantive content to
  total content. Code blocks, technical terms, and structured data increase
  density. Boilerplate, self-promotion, and filler decrease it. Measured by
  LLM classification or heuristic (code block ratio + technical term frequency).

- **uniqueness** (0.10): 1.0 minus the maximum cosine similarity to any
  existing item in the index. Completely novel content scores 1.0. Near-duplicates
  of existing content score close to 0.0.

### 3.3 Three-Layer Deduplication

Deduplication is critical because the same announcement often appears on GitHub,
HN, Reddit, and multiple blogs within hours.

**Layer 1 -- URL exact match**: Maintain a set of all previously seen URLs. If the
URL has been fetched before, skip it. Fast (O(1) lookup) but misses mirrors and
reposts.

**Layer 2 -- SHA-256 content hash**: Hash the extracted text content. If the hash
matches an existing item, skip it. Catches exact reposts across different URLs.

**Layer 3 -- Semantic similarity**: Embed the new item and compare against the
most recent N items (e.g., N=1000) in the index. If cosine similarity > 0.92 to
any existing item, flag as a near-duplicate. The operator can configure whether
near-duplicates are auto-rejected or merged (keeping the higher-scoring version).

---

## 4. Freshness Decay Formula

Information value decays at different rates depending on category. A critical
security advisory is stale within days; a foundational research paper remains
relevant for months.

### 4.1 Formula

```
score = 0.7 * cosine_similarity + 0.3 * decay_factor

decay_factor = 0.5 ^ (age_days / half_life)
```

Where:
- `cosine_similarity`: Relevance of the item to the query (0 to 1)
- `age_days`: Days since the item was published
- `half_life`: Category-specific decay constant (days)

### 4.2 Half-Lives by Category

| Category             | Half-Life | Rationale                                     |
|----------------------|-----------|-----------------------------------------------|
| Security advisories  | 7 days    | Patch urgency; stale after fix is available    |
| Software releases    | 30 days   | Relevant until the next release                |
| Research papers      | 90 days   | Ideas remain relevant; implementation lags     |
| Documentation        | 180 days  | Stable APIs rarely change; docs age slowly     |

### 4.3 Example

A security advisory published 14 days ago, with cosine_similarity = 0.85:

```
decay_factor = 0.5 ^ (14 / 7) = 0.5 ^ 2 = 0.25
score = 0.7 * 0.85 + 0.3 * 0.25 = 0.595 + 0.075 = 0.67
```

The same advisory when 2 days old:

```
decay_factor = 0.5 ^ (2 / 7) = 0.5 ^ 0.286 = 0.82
score = 0.7 * 0.85 + 0.3 * 0.82 = 0.595 + 0.246 = 0.84
```

The freshness bonus adds 0.17 points for a 2-day-old advisory vs. a 14-day-old one.

---

## 5. Storage Budget

### 5.1 Monthly Estimates

| Component        | Raw Size  | Processed | Embeddings | Total/Month |
|------------------|-----------|-----------|------------|-------------|
| GitHub releases  | 4 MB      | 2 MB      | 4 MB       | 6 MB        |
| arXiv abstracts  | 2 MB      | 1 MB      | 2 MB       | 3 MB        |
| HN/Reddit        | 4 MB      | 1.5 MB    | 3 MB       | 4.5 MB      |
| Security (NVD)   | 1 MB      | 0.5 MB    | 1 MB       | 1.5 MB      |
| Blogs/RSS        | 3 MB      | 1 MB      | 2 MB       | 3 MB        |
| PyPI/npm         | 2 MB      | 1 MB      | 2 MB       | 3 MB        |
| **Monthly total**| **16 MB** | **7 MB**  | **14 MB**  | **21 MB**   |

### 5.2 Annual Projection

- **Before pruning**: ~250 MB/year (12 months of accumulated data)
- **After pruning**: ~150 MB/year (remove items with quality score < 0.20 after
  their half-life has elapsed 4 times)
- **5-year projection**: ~750 MB before pruning, ~400 MB after. Easily fits on
  any modern disk.

### 5.3 Embedding Storage Detail

Embeddings at 768 dimensions (nomic-embed-text) with float32 = 3,072 bytes per
vector. For ~4,500 items/month: 4,500 * 3,072 = ~14 MB/month of raw vector data.
LanceDB adds ~10% overhead for indexing metadata.

### 5.4 Bandwidth

- **HTTP requests**: 3,000-5,000 per month across all sources
- **Total bandwidth**: ~50-100 MB/month download (mostly JSON/text)
- **Network impact**: Negligible on any connection

---

## 6. Legal and Ethical Compliance

### 6.1 Allowed Sources

| Source          | API Available | ToS Allows Scraping | Rate Limit              |
|-----------------|---------------|---------------------|-------------------------|
| GitHub API      | Yes           | Yes (with token)    | 5,000 req/hour          |
| Hacker News     | Yes (Algolia) | Yes                 | No documented limit     |
| arXiv           | Yes           | Yes                 | 3s delay required       |
| PyPI            | Yes           | Yes                 | No documented limit     |
| NVD             | Yes           | Yes (public data)   | 50 req/30s (with key)   |
| crates.io       | Yes           | Yes                 | 1 req/second            |

### 6.2 Restricted Sources

| Source          | Issue                                         | Recommendation    |
|-----------------|-----------------------------------------------|-------------------|
| Medium          | Paywalled, ToS explicitly prohibits scraping   | Do not scrape     |
| Stack Overflow  | CC BY-SA license requires attribution          | Attribute or skip |
| Twitter/X       | API is paid ($100/month basic), ToS restrictive| Skip              |

### 6.3 Conditional Sources

| Source          | Condition                                           |
|-----------------|-----------------------------------------------------|
| Reddit          | Free tier requires OAuth app registration            |
|                 | 100 requests/minute rate limit                       |
|                 | Must not bypass rate limits or impersonate users     |
|                 | Must identify as bot in User-Agent                   |

### 6.4 Best Practices

1. **Always respect robots.txt**: Check before scraping any new domain
2. **Rate limiting**: 2-3 second delay between requests to any single domain
3. **User-Agent identification**: Set User-Agent to `JCoderBot/1.0 (+https://github.com/your-repo)`
4. **Caching**: Never re-fetch content that has not changed (use ETags, If-Modified-Since)
5. **Storage**: Store only processed summaries and metadata, not full copyrighted articles
6. **Attribution**: Maintain source URLs and authors for all indexed content

---

## 7. Architecture Summary

```
+------------------+     +------------------+     +------------------+
|  TIER 1 SOURCES  |     |  TIER 2 SOURCES  |     |  TIER 3 SOURCES  |
|  GitHub, arXiv   |     |  HN, Blogs, RSS  |     |  Reddit, npm     |
|  NVD, PyPI       |     |  Changelog, TLDR |     |  crates.io       |
+--------+---------+     +--------+---------+     +--------+---------+
         |                        |                        |
         |  hourly/daily          |  weekly                |  bi-weekly
         v                        v                        v
+--------+------------------------+------------------------+---------+
|                        APScheduler                                 |
|                  (in-process job scheduler)                        |
+-+------------------------------------------------------------------+
  |
  v
+-+------------------------------------------------------------------+
|                     FETCH LAYER                                    |
|  httpx (APIs)  |  Crawl4AI (web)  |  feedparser (RSS)              |
+-+------------------------------------------------------------------+
  |
  v
+-+------------------------------------------------------------------+
|                     PROCESS LAYER                                  |
|  trafilatura (HTML->text)  |  code block extraction                |
|  phi4-mini summarization   |  quality scoring                      |
+-+------------------------------------------------------------------+
  |
  v
+-+------------------------------------------------------------------+
|                     DEDUP LAYER                                    |
|  URL set  |  SHA-256 hash  |  semantic similarity (>0.92)          |
+-+------------------------------------------------------------------+
  |
  v
+-+------------------------------------------------------------------+
|                     INDEX LAYER                                    |
|  nomic-embed-text (768d)  |  LanceDB (hybrid BM25 + vector)       |
+-+------------------------------------------------------------------+
  |
  v
+-+------------------------------------------------------------------+
|                     QUERY LAYER                                    |
|  Freshness-weighted hybrid search  |  LLM-generated answers       |
+--------------------------------------------------------------------+
```

---

## 8. Sources

- Crawl4AI: docs.crawl4ai.com
- trafilatura: trafilatura.readthedocs.io
- Hacker News API: hn.algolia.com/api
- arXiv API: info.arxiv.org/help/api
- nvdlib: nvdlib.com
- APScheduler: apscheduler.readthedocs.io
- Freshness decay research: arxiv.org/abs/2509.19376
- GitHub REST API: docs.github.com/en/rest
- PyPI JSON API: warehouse.pypa.io/api-reference
- Reddit API: reddit.com/dev/api
