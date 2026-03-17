# Download Queue: Next Wave (After Current Batch Completes)
**Date:** 2026-03-15
**Priority:** Python + Rust, then general coding, then reasoning/math

## WAVE 2 (Launch after current 9 jobs finish)

### Code Pretraining (MASSIVE)
- `nvidia/Nemotron-Pretraining-Code-v2` -- 427B code tokens, CC-BY-4.0
- `bigcode/commitpack` -- 4 TB, 350+ languages (filter to Python+Rust)
- `codeparrot/github-code` -- 1 TB, 30 languages (Python-all config)

### Code Reasoning
- `nvidia/OpenCodeReasoning` -- 735K reasoning traces (config=split_0)
- `open-r1/codeforces-submissions` -- 12.7M real submissions (already in batch)

### Instruction Tuning
- `microsoft/orca-agentinstruct-1M-v1` -- 1M with tool calling
- `nvidia/Nemotron-Post-Training-Dataset-v1` -- code/math/reasoning SFT
- `nvidia/Nemotron-Post-Training-Dataset-v2` -- v2 with multilingual
- `BAAI/Infinity-Instruct` -- 7M instructions
- `cognitivecomputations/dolphin` -- 4.5M completions
- `Open-Orca/OpenOrca` -- 4.2M completions

### Documentation
- `recursal/MDN` -- Mozilla Developer Network (web dev docs)
- `HuggingFaceTB/cosmopedia` -- 28B tokens synthetic textbooks

### Math/Reasoning
- `nvidia/OpenMathInstruct-1` -- 1.8M problem-solution pairs
- `EleutherAI/proof-pile-2` -- 55B tokens math/science
- `open-web-math/open-web-math` -- 14.7B tokens math

### Agent/Tool Calling
- `Team-ACE/ToolACE` -- 26K APIs
- `microsoft/Taskbench` -- task decomposition + tool invocation
- `gorilla-llm/APIBench` -- API call dataset

### Security
- `AlicanKiraz0/All-CVE-Records-Training-Dataset` -- all CVEs
- `darkknight25/Exploit_Database_Dataset` -- Exploit-DB entries

### Notebooks
- `codeparrot/github-jupyter-code-to-text` -- Jupyter Python+docstring
- `codeparrot/apps` -- 10K coding problems

## VERIFIED OPEN (Ready to download immediately)
- `nvidia/OpenMathInstruct-1` -- 10 files, 1.8M math problem-solution pairs
- `Open-Orca/OpenOrca` -- 2 files, 4.2M completions
- `codeparrot/github-jupyter-code-to-text` -- 1 file, Jupyter Python+docstring
- `cognitivecomputations/dolphin` (flan1m-alpaca-uncensored) -- 4 files
- `HuggingFaceTB/cosmopedia` (web_samples_v1) -- 139 files! Synthetic textbooks

## NOT DOWNLOADABLE via Parquet API (need datasets library streaming)
- nvidia/OpenCodeReasoning -- HF API returns keys but no parquet URLs
- microsoft/orca-agentinstruct-1M-v1 -- same issue
- nvidia/Nemotron-Post-Training-Dataset-v1 -- same issue
- recursal/MDN -- same issue

## Curated Lists for Ongoing Discovery
- github.com/mlabonne/llm-datasets (40+ datasets)
- github.com/codefuse-ai/Awesome-Code-LLM (TMLR-published)
- github.com/argilla-io/awesome-llm-datasets

## Overlap Notes (Don't Re-Download)
- Stack-Edu Python covers codeparrot/github-code Python
- opencodeinstruct overlaps many smaller instruction sets
- CoRNStack is unique (retrieval triplets, not raw code)
- commit-chronicle overlaps commitpack for commits
