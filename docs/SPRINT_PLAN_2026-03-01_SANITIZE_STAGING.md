# Sprint Plan -- Sanitized KB Preparedness

## Sprint Objective
By end of sprint, have raw dumps fully staged and sanitized outputs validated so beast machine can index without surprises.

## Scope
- In-scope sources:
  - StackExchange dumps
  - Reddit dumps
  - GitHub/ragas snapshots
- Out of scope:
  - Any `D:\HybridRAG3` modifications
  - Raw deletion without explicit approval

## Definition of Done
1. Approved raw sources are staged in `D:\JCoder\data\source_dumps`.
2. Sanitizer pipeline runs from `ingest` automatically and produces clean archive.
3. Prep report exists with chunk-readiness and archive integrity.
4. Parser format smoke tests pass.
5. 20-file PII spot check report produced.
6. Handover updated with exact results and remaining risks.

## Work Breakdown
1. Staging Completion
- Resume-safe copy for missing Reddit + ragas files.
- Verify file/size parity source vs staged.

2. Preparedness Validation
- Run `scripts/build_format_smoke_pack.py`.
- Run `tests/test_sanitizer_formats.py`.
- Confirm `.7z/.zip/.zst/invalid-header` handling behavior is logged.

3. Sanitization + Prep
- Run `scripts/prep_stage_for_index.py`.
- Capture `prep_report.json` and `prep_report.csv`.
- Record bad archive header counts and skipped reasons.

4. Quality/PII Gate
- Sample 20 random clean files.
- Scan for email/URL/@mention residue and record flags.
- If flags exist, log and queue sanitizer tuning before full indexing.

5. Final Readiness Pack
- Update handover doc.
- Provide one-shot command list for beast machine indexing day.

## Execution Commands
- `D:\JCoder\.venv\Scripts\python D:\JCoder\scripts\build_format_smoke_pack.py`
- `D:\JCoder\.venv\Scripts\python -m pytest -q D:\JCoder\tests\test_sanitizer_formats.py`
- `D:\JCoder\.venv\Scripts\python D:\JCoder\scripts\prep_stage_for_index.py`

## Risk Controls
- Keep operations restart-safe (robocopy `/Z` + re-check counts).
- Never overwrite raw source-of-truth.
- Keep sanitizer logs and manifest snapshots for auditability.
