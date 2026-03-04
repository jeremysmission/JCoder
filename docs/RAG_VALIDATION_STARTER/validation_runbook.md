# Validation Runbook (JCoder)

## 1) Parser readiness
- Build sample set: at least one real file per enabled extension.
- Run parser smoke and record pass/fail in parser_coverage_matrix.csv.
- Any "no text extracted" on expected-text files = parser or dependency issue.

## 2) Golden corpus baseline
- Use a tiny deterministic corpus first (10-20 files).
- Include: answerable, unanswerable, ambiguous, injection prompts.
- Keep expected facts short and exact for deterministic scoring.

## 3) A/B model evaluation
- Run same dataset with offline model and online model.
- Compare:
  - Fact score
  - Citation score
  - Behavior score (unanswerable/injection handling)
  - Reasoning score (cross-doc tasks)
  - Latency

## 4) Keep reasoning without hallucination
- Require output sections:
  - Answer
  - Evidence
  - Derived inference
  - Confidence
  - Missing data
- Derived inference is allowed only when evidence is cited.

## 5) Stop/go gates
- Do not tune until parser lane passes.
- Do not demo until retrieval + safety lanes pass.
- Tune only after baseline metrics are stable across 2+ runs.
