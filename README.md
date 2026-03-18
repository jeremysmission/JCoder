# JCoder

Fully local, offline CLI AI coding assistant with autonomous agent, self-learning pipeline, and model cascade routing. Ingests codebases, builds hybrid vector+keyword indexes (FAISS + SQLite FTS5), answers questions using a local LLM served via vLLM, and continuously improves through automated learning cycles.

**Updated:** 2026-03-17
**Location:** `C:\Users\jerem\JCoder\` (migrated from D: to C: NVMe SSD, March 2026)

## Requirements

- Python 3.10+
- CUDA-capable GPU (dual RTX 3090 recommended for full model stack)
- vLLM for model serving
- Data directory: set `JCODER_DATA` env var (defaults to `data/` in repo root)

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

## Configuration

Config files live in `config/`:

| File | Purpose |
|------|---------|
| `models.yaml` | Model names, quantization, tensor parallelism |
| `ports.yaml` | Port assignments for vLLM servers |
| `policies.yaml` | Hard caps, timeouts, GPU margins, eval settings |
| `default.yaml` | Storage paths, retrieval params, chunking strategy |
| `agent.yaml` | Agent backend, self-learning, reasoning, cascade, distillation |
| `memory.yaml` | Persistent memory and experience replay settings |
| `profiles.yaml` | User profiles and preferences |
| `download_queue.json` | Download manager backlog |

Override config location: `--config-dir <path>` or `JCODER_CONFIG_DIR` env var.

### Agent Configuration (`config/agent.yaml`)

The agent config controls 15+ modules organized by sprint:

| Section | Key Settings |
|---------|-------------|
| `backend` | `openai`, `anthropic`, or `ollama` |
| `self_learning` | `pipeline_enabled`, `continual_learner_enabled`, regression margin |
| `reasoning` | `star_enabled`, `reflection_enabled`, `best_of_n_enabled` |
| `corrective` | `corrective_retrieval_enabled`, `smart_orchestrator_enabled`, knowledge graph |
| `evolution` | `prompt_evolver_enabled`, `adversarial_self_play_enabled`, rapid digest, stigmergy |
| `cascade` | `cascade_enabled` (complexity-based model routing) |
| `distillation` | `enabled`, model, budget_usd, top-N weak topics |

## Usage

### Core Commands

```bash
# Check environment readiness
python main.py doctor

# Ingest a repository
python main.py ingest /path/to/repo --index-name myproject

# Ask a question
python main.py ask "How does authentication work?" --index-name myproject

# Run evaluation benchmark
python main.py eval --benchmark evaluation/golden_questions_v1.json

# CPU-only testing (no vLLM needed)
python main.py --mock ask "What does the chunker do?"

# Launch the GUI command center
python main.py gui
# or
jcoder-gui

# Alternative entry point
python -m jcoder
```

The GUI uses the HybridRAG3 dark palette and builds its forms from the live
Click command tree, so it stays aligned with the CLI as commands change.
Standard commands stream output into the window. The `interactive` REPL is
launched in a separate console window because it needs live terminal input.

### Self-Learning Pipeline

```bash
# Run full 6-phase learning cycle
python scripts/learning_cycle.py --eval-set evaluation/agent_eval_set_200.json

# Run specific phases only
python scripts/learning_cycle.py --eval-set evaluation/agent_eval_set_200.json --phases 1,2,3

# View report from a completed cycle
python scripts/learning_cycle.py --report-only --cycle-dir logs/learning_cycles/cycle_001

# Distill weak topics using a strong model
python scripts/distill_weak_topics.py --top 20 --budget 2.0
```

**Learning Cycle Phases:**
1. **Baseline eval** — Record scores per category from 200-question golden set
2. **Generate study queries** — Identify weak categories, produce targeted queries
3. **Study engine** — Run queries through retrieval, generate telemetry + experience
4. **Distillation** — Send weak topics to strong model (e.g. GPT-5) for expert explanations
5. **Re-eval** — Record new scores after learning
6. **Compare & report** — Delta analysis across all categories

### Weekly Knowledge Scraper

```bash
python scripts/weekly_scraper.py                      # All scrapers (RSS, changelog, SE)
python scripts/weekly_scraper.py --scraper rss         # RSS feeds only
python scripts/weekly_scraper.py --scraper changelog   # Python/PyPI changelogs
python scripts/weekly_scraper.py --scraper se          # Stack Exchange answers
python scripts/weekly_scraper.py --dry-run              # Preview without ingesting
```

Pulls from: Python blog, Real Python, PEP index, HN top stories, PyPI changelogs, Stack Exchange. Outputs markdown chunks into `fresh_knowledge.fts5.db`.

### Data Downloads

```bash
# Phase 6 datasets (46+ datasets across 5 categories)
python scripts/download_phase6_datasets.py

# Validate Stack Exchange downloads
python scripts/validate_se_downloads.py

# Sanitize Stack Exchange data (parallel)
python scripts/parallel_sanitize_se.py

# Build FTS5 indexes from downloads
python scripts/build_fts5_indexes.py
python scripts/build_se_indexes.py
```

**Phase 6 Dataset Categories:**
- **A: Code Reasoning** — SWE-Bench, CodeComplex, OpenHands trajectories
- **B: API Mastery** — API documentation, usage patterns
- **C: Security** — Vulnerability datasets, secure coding
- **D: Instruction** — Coding instruction and tutorial corpora
- **E: DevOps** — CI/CD, deployment, infrastructure

## Architecture

### Core RAG Pipeline

```
CLI (Click) -> Orchestrator -> RetrievalEngine -> EmbeddingEngine (vLLM)
                    |                |
                    |          IndexEngine (FAISS + FTS5 + RRF)
                    |                |
                    |           Reranker (vLLM cross-encoder)
                    |
               Runtime (vLLM chat completions)
```

### Autonomous Agent System (`agent/`)

```
AgentBridge -> Agent (core.py) -> ToolRegistry (40+ tools)
     |              |
     |         GoalManager -> Session persistence
     |              |
     |         LLM Backend (OpenAI / Anthropic / Ollama)
     |
     +-- SmartOrchestrator (query routing)
     +-- ModelCascade (complexity-based model selection)
     +-- CorrectiveRetriever (dynamic failure correction)
     +-- MetaCognitiveController (strategy selection)
     +-- ExperienceReplay (learning from past queries)
     +-- KnowledgeGraph (code structure understanding)
     +-- PromptEvolver (prompt optimization)
     +-- AdversarialSelfPlay (self-hardening)
     +-- RapidDigester (fast knowledge absorption)
     +-- StigmergicBooster (indirect coordination signals)
```

All self-learning modules are optional — the agent degrades gracefully if any module is unavailable.

### Advanced Reasoning Modules (`core/`)

| Module | Purpose |
|--------|---------|
| `cascade.py` | Routes queries through model hierarchy based on complexity |
| `smart_orchestrator.py` | Intelligent query routing with confidence scoring |
| `star_reasoner.py` | Self-Taught Algorithmic Reasoning (STaR) |
| `best_of_n.py` | Generates N candidates, selects best by scoring |
| `reflection.py` | Self-reflection and error correction |
| `corrective_retrieval.py` | Detects and fixes retrieval failures dynamically |
| `adversarial_self_play.py` | Hardens system via adversarial examples |
| `knowledge_graph.py` | Builds code knowledge graphs for structural understanding |
| `prompt_evolver.py` | Evolves prompts based on performance feedback |

### Data Pipeline (`ingestion/`)

| Module | Purpose |
|--------|---------|
| `chunker.py` | Smart code/text chunking with overlap |
| `repo_loader.py` | Repository loading and file discovery |
| `ast_fts5_builder.py` | AST-based FTS5 index construction |
| `sanitizer.py` | Stack Exchange data sanitization |
| `pii_scanner.py` | PII detection and removal |
| `dedup.py` | Content deduplication |
| `corpus_pipeline.py` | Batch processing for large corpora |

### Research Pipeline (`core/`)

| Module | Purpose |
|--------|---------|
| `adaptive_research.py` | PRISMA-compliant evidence gathering |
| `research_sprint.py` | Time-boxed research sprints |
| `claim_verifier.py` | Claim verification with evidence |
| `devils_advocate.py` | Counterargument generation |
| `evidence_weighter.py` | Source credibility scoring |
| `synthesis_matrix.py` | Cross-source synthesis (8 themes max) |
| `source_scorer.py` | Source reliability scoring |

### Evaluation Infrastructure (`evaluation/`)

- **200-question golden set** (`agent_eval_set_200.json`)
- **Scoring subscores:** `has_code`, `has_correct_api`, `has_imports`, `is_runnable`, `cites_source`
- **Category-based analysis** for targeted improvement
- **Eval comparison** (`scripts/eval_compare.py`) for before/after analysis

## Testing

```bash
# Full test suite
pytest tests/ -v

# Eval smoke tests (no Ollama needed)
pytest tests/test_eval_smoke.py -v

# Weekly scraper tests
pytest tests/test_weekly_scraper_extended.py -v
```

## Inclusive Downloader

All new JCoder acquisition work should go through the inclusive downloader in
`core/download_manager.py`. Do not add fresh per-script `httpx.stream(...)`
download loops.

To run the current backlog through the shared downloader path:

```bash
python scripts/run_download_queue.py --list
python scripts/run_download_queue.py
```

Operator details are in `docs/INCLUSIVE_DOWNLOADER.md`.
