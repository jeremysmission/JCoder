# JCoder Handover -- Sanitized KB Staging (2026-03-01)

## Goal
Prepare raw coding dumps for safe indexing with permanent sanitization:

`raw -> sanitize -> clean archive -> chunk/embed/index`

No raw indexing. No raw deletion without explicit user approval.

## Current Repo State
- Working repo: `D:\JCoder`
- Uncommitted sanitizer/staging changes exist in:
  - `ingestion/sanitizer.py`
  - `cli/commands.py`
  - `core/config.py`
  - `config/default.yaml`
  - `scripts/prep_stage_for_index.py`
  - `requirements.txt`
- New format-readiness assets added:
  - `scripts/build_format_smoke_pack.py`
  - `tests/test_sanitizer_formats.py`

## Data Staging Status
- Source roots (approved):
  - `D:\Projects\KnowledgeBase\stackexchange_20251231`
  - `D:\Projects\reddit`
  - `D:\Projects\KnowledgeBase\sources\ragas`
- Staging target root:
  - `D:\JCoder\data\source_dumps`

Latest verified counts:
- StackExchange: source `220 files / 89.419 GB`, staged `220 files / 89.419 GB` (complete)
- Reddit: source `2 files / 42.967 GB`, staged `1 file / 28.483 GB` (incomplete)
- Ragas: source `694 files / 0.046 GB`, staged `0 files` (not copied yet)

## Safety Constraints (Confirmed)
- Do not touch `D:\HybridRAG3`.
- Only process approved in-scope sources unless user approves extras.
- Never index raw directly.
- Stop before any raw deletion and wait for explicit approval.

## Validation Artifacts
- Clean archive root: `D:\JCoder_Data\clean_source`
- Sanitizer logs: `D:\JCoder_Data\clean_source\_logs\sanitize_*.json`
- Prep reports: `D:\JCoder_Data\prep_stage\prep_*\prep_report.{json,csv}`
- Validation starter pack provided by user:
  - `D:\JCoder\docs\RAG_VALIDATION_STARTER\`

## Immediate Next Commands
1. Finish incomplete raw staging:
   - `robocopy D:\Projects\reddit D:\JCoder\data\source_dumps\reddit\raw_zst /E /Z /R:2 /W:2 /NP /NFL /NDL /XO`
   - `robocopy D:\Projects\KnowledgeBase\sources\ragas D:\JCoder\data\source_dumps\git_github\repo_snapshots\ragas /E /Z /R:2 /W:2 /NP /NFL /NDL /XO`
2. Re-check source vs staged counts.
3. Run prep script (sanitize + chunk-readiness report, no heavy indexing):
   - `D:\JCoder\.venv\Scripts\python D:\JCoder\scripts\prep_stage_for_index.py`
4. Spot-check 20 random clean files for PII markers.
5. Update handover with exact pass/fail and pending items.

## Known Operational Risks
- Long copy operations can timeout in this environment.
- Reddit `.zst` files may contain invalid/corrupt placeholders in some locations.
- AV can quarantine markdown files containing malware-like code signatures (false-positive risk for corpus text).
