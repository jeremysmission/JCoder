# tools/

Standalone utility scripts and one-off helpers. These are **not** part of
the core JCoder system and are not imported by any module in `core/`,
`scripts/`, or `tests/`.

They exist for manual, ad-hoc use during development and experimentation:

| Script | Purpose |
|--------|---------|
| `build_faiss_indexes.py` | Build FAISS vector indexes from embedded data |
| `coding_challenge.py` | Generate coding challenge prompts for eval |
| `fts5_demo.py` | Interactive demo of FTS5 full-text search |
| `gpt5_digest.py` | Summarise content using GPT models |
| `hard_challenges.py` | Generate harder coding challenges for eval |
| `ingest_all_pending.py` | Batch-ingest any pending raw data |
| `ingest_datasets.py` | One-shot dataset ingestion helper |
| `learning_test.py` | Manual test harness for learning pipeline |
| `monkey_brain_test.py` | Stress-test / fuzz harness |
| `scrape_agentic_sources.py` | Scrape agentic-AI resource lists |
| `site_wiki_builder.py` | Build a local wiki from scraped sites |
| `systematic_digest.py` | Produce systematic knowledge digests |
| `trick_science_quiz.py` | Generate tricky science quiz questions |
| `wiki_builder.py` | Wiki builder (v1) |
| `wiki_builder_v2.py` | Wiki builder (v2, improved) |

If any of these scripts become regularly used, they should be promoted into
`scripts/` (for automation) or `core/` (for library code) with proper tests.
