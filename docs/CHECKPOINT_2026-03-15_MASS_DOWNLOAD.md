# Checkpoint: Mass Download Session -- 2026-03-15
**Context:** Firewall temporarily disabled for maximum download speed
**TARGET: 500 GB of indexed coding knowledge**
**LANGUAGE PRIORITY: Python first, Rust second (user preference)**
**Starting indexes:** 175 FTS5, 42.1 GB
**Current (estimated):** 200+ FTS5, ~60+ GB (many jobs still building indexes)
**Disk consumed this session:** ~56 GB and growing (parquet downloads + FTS5 builds)
**Last disk check:** 2,900 GB free
**All download jobs are idempotent** -- safe to re-run after reboot, they skip existing indexes
**Free space:** ~2,950 GB on D:

## Downloads Launched This Session

### CoRNStack (Nomic AI, Apache 2.0) -- CODE RETRIEVAL TRIPLETS
- cornstack-python-v1 (423K+ triplets)
- cornstack-java-v1
- cornstack-javascript-v1
- cornstack-go-v1
- cornstack-php-v1
- cornstack-ruby-v1

### AI Tuning + Reasoning
- allenai/tulu-3-sft-mixture (multi-task instruction tuning)
- Open-Orca/SlimOrca (500K cleaned instructions)
- argilla/distilabel-reasoning-prompts
- AI-MO/NuminaMath-CoT (math chain-of-thought)
- open-r1/OpenR1-Math-220k (math reasoning traces)
- nvidia/HelpSteer2 (response quality)
- Anthropic/hh-rlhf (human preference)
- open-r1/codeforces-cots solutions_py (Python competitive programming + reasoning)
- HuggingFaceTB/smoltalk apigen-80k (function calling)

### Python Code Mirrors
- bigcode/python-stack-v1-functions-filtered
- Nan-Do/code-search-net-python
- Vezora/Tested-143k-Python-Alpaca
- iamtarun/python_code_instructions_18k_alpaca

### Code Quality + Instructions
- WizardLM/evol_instruct_V2_196k
- google/code_x_glue_ct_code_to_text (Python)
- bigcode/self-oss-instruct-sc2-exec-filter-50k
- sahil2801/CodeAlpaca-20k
- open-r1/codeforces (problems)

### Previously Completed
- hermes_function_calling (64 MB) -- multi-config fix applied
- All phase6 ungated (ran full script)
- All tier1 (ran full script)
- All code_corpora (ran full script)

## MEGA Downloads (NEW FINDS -- biggest value)
- **meryyllebr543/stack-edu-huggingface Python**: 25.3M files, 27 GB raw -> DOWNLOADING
- **meryyllebr543/stack-edu-huggingface JavaScript**: 13.3M files, 13 GB raw -> DOWNLOADING
- **meryyllebr543/stack-edu-huggingface TypeScript**: 4.3M files, 3.7 GB raw -> DOWNLOADING
- **meryyllebr543/stack-edu-huggingface Rust**: 1.1M files, 1.8 GB raw -> DOWNLOADING
- **Leon-Leee/unofficial-pyedu**: 7.7M Python files, 5.7 GB -> DOWNLOADING
- **codeparrot/github-code-clean Python**: 645K files, 1.7 GB -> DOWNLOADING
- **MatrixStudio/Codeforces-Python-Submissions**: 690K solutions -> DOWNLOADING
- **microsoft/EpiCoder-func-380k**: 380K verified Python functions -> DOWNLOADING
- **petrpan26/typescript-code**: 380K TS files -> DOWNLOADING
- **Tesslate/Rust_Dataset**: 47K Rust instruction pairs -> DOWNLOADING

## Batch 3: Code Quality + DevOps + SQL (15 datasets)
- JetBrains commit-chronicle (10.7M commits, 61 parquet files)
- github-top-code (1.3M files, 80+ languages)
- python-code-500k, LeetCode, CodeXGLUE (refinement, defect, translation)
- Code review, Security DPO, CodeRM UnitTest
- Bash commands, NL2Bash, text-to-SQL (x2), Rosetta Code

## Batch 4: Agent Trajectories + Reasoning + DevOps (13 datasets)
- open-r1/codeforces-submissions (12.7M real submissions, 36 files -- capped at 20)
- Replete-AI/code_bagel (2.2M, 800M tokens, 100+ languages)
- KodCode/KodCode-V1 (487K verifiable solutions)
- open-thoughts/OpenThoughts2-1M (1M reasoning traces)
- Z1-Code-Reasoning-107K (107K coding reasoning)
- substratusai/the-stack-yaml-k8s (276K K8s configs)
- interstellarninja/hermes_reasoning_tool_use (51K tool-call conversations)
- JetBrains/git_good_bench-train (17.5K git workflows)
- b-mc2/cli-commands-explained (16K CLI commands)
- ComponentSoft/k8s-kubectl-35k (35K kubectl commands)
- Mubeen161/DEVOPS (42K DevOps instructions)
- nebius/SWE-agent-trajectories (80K agent traces)
- Kwai-Klear/SWE-smith-mini trajectories (66K, capped at 20 files)

## Completed This Session
- code_x_glue_python: 252K entries, 455 MB
- self_oss_instruct_50k: 51K entries, 115 MB
- code_alpaca_20k: 20K entries, 14 MB
- codeforces_openr1: 9.5K entries, 18 MB
- hermes_function_calling: 64 MB
- fresh_knowledge (weekly scraper): 97 chunks

## Gated (Need HF_TOKEN)
- bigcode/the-stack-v2-dedup Python (191 GB)
- nomic-ai/cornstack (original, gated -- but per-language versions are OPEN)
- Cyborg-AI/commit-chronicle (10.7M commits)
- bigcode/starcoderdata
- PatronusAI/TRAIL
- nvidia/Nemotron-Agentic-v1
- nampdn-ai/tiny-codes
- Salesforce/xlam-function-calling-60k

## Sprint R12: Retrieval Quality Revolution (IN PROGRESS)
- See: docs/SPRINT_R12_RETRIEVAL_QUALITY.md
- R12.1: ingestion/ast_fts5_builder.py CREATED (AST-aware FTS5 index builder)
- R12.1: tests/test_ast_fts5_builder.py CREATED (11 tests, running)
- Research briefs saved:
  - docs/RESEARCH_BRIEF_2026-03-15_RAG_ANTIPATTERNS.md
  - docs/RESEARCH_BRIEF_2026-03-15_CODE_SEARCH_TECH.md
  - docs/RESEARCH_BRIEF_2026-03-15_FULL_REVIEW.md

## Key Strategic Insight (from contrarian research)
**Quality > Quantity for RAG.** More data with naive chunking HURTS performance.
- AST chunking: 0.82 faithfulness vs 0.47 for naive (75% improvement)
- Industry uses trigram indexes (Zoekt) not FTS5/BM25 for code
- Cursor uses tree-sitter + Turbopuffer, not FTS5
- vLLM is 16x faster than Ollama on GPU hardware
- Next priority: infrastructure quality, not more volume

## Code Changes This Session
- ModelCascade + SmartOrchestrator WIRED in bridge.py
- agent/tools.py refactored (657->428L class body)
- ingestion/sanitizer.py refactored (620->535L)
- SQL injection fixed (weekly_knowledge_update.py)
- bridge.py silent error fixed
- orchestrator.py tests (13), runtime.py tests (18)
- psutil added as optional dep
- 1730 tests pass, 0 failures

## Research Findings Saved
- docs/RESEARCH_BRIEF_2026-03-15_FULL_REVIEW.md (competitive analysis + techniques)
- docs/CHECKPOINT_2026-03-15_DPI_QA_SPRINT_QUEUE.md (prioritized fix queue)
- memory/jcoder_research_2026-03-15.md (persistent memory)
