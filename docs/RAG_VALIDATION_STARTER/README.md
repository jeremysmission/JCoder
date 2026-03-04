# RAG Validation Starter Pack (JCoder)

Purpose:
- Validate parser readiness and retrieval quality before tuning.
- Separate retrieval correctness from model reasoning quality.

Files in this pack:
- parser_coverage_matrix.csv
- golden_query_bank_template.json
- validation_runbook.md

Recommended flow:
1. Build parser coverage matrix for your real corpus.
2. Run parser smoke against every enabled extension.
3. Index a small deterministic golden corpus.
4. Run offline baseline first.
5. Run online model A/B with same question IDs.
6. Score fact/citation/behavior/reasoning separately.

Output gates (suggested):
- Retrieval lane pass rate >= 0.90
- Unanswerable handling >= 0.90
- Injection resistance >= 0.95
- Reasoning lane >= offline baseline + 10 percentage points
