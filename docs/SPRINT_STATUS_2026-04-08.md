# JCoder Sprint Status -- 2026-04-08

Last updated: 2026-04-08 (demo-readiness planning refresh)

## Current Sprint Position

The active forward sprint queue is now demo-first.

Primary plan:
- `docs/SPRINT_PLAN_R28_R32_2026-04-08.md`

This queue supersedes the older near-term assumption that Beast saturation,
ingest throughput, and self-learning should land before the operator-facing
demo proof. Those lanes still matter, but they no longer lead the sequence.

## Why The Queue Changed

Repo review across the README, runbook, handover docs, validation starter, and
the current `scripts/demo.py` showed a mismatch:

- JCoder's mission is a grounded local coding assistant.
- The current documented "demo" path is still mostly a walkthrough.
- Recovery docs require trusted corpus -> reindex -> query validation ->
  regression/button-smash -> only then demo or retune.
- Historical sprint docs mixed real demo-prep work with broader autonomy and
  scale ambitions.

The result is that demo readiness needs a stricter, more operator-facing spine.

## Active Sprint Queue

| Sprint | Status | Purpose |
|--------|--------|---------|
| `R28` | QUEUED | Canonical demo contract, deterministic demo corpus, operator checklist, cleanup of the current demo script |
| `R29` | QUEUED | Retrieval trust and safety gate before any demo-ready claim |
| `R30` | QUEUED | End-to-end coding-assistant proof, not just a search walkthrough |
| `R31` | QUEUED | GUI and operator hardening aligned to the CLI proof path |
| `R32` | QUEUED | Live-stack credibility and honest separation of stretch features |
| `R33` | STAGED | Post-demo learning-loop proof after the core demo is stable |

## Current Execution Order

`R28 -> R29 -> R30 -> R31 -> R32`

`R33` is explicitly post-demo unless the earlier queue is complete and green.

## What Counts As Demo Readiness

JCoder is demo-ready only when all of the following are true:

1. The operator can launch the documented CLI and GUI paths without guessing.
2. The corpus and index state used for the demo is explicit and trusted.
3. The system answers at least one grounded repo question with evidence.
4. The system completes at least one bounded coding-assistant task with verification.
5. The system handles at least one unsafe or unanswerable case correctly.
6. A fresh independent QA pass confirms the above.

## Historical Docs Still Kept

These remain useful for historical context and older execution intent:

- `docs/SPRINT_COMPLETION_PLAN_2026-03-13.md`
- `docs/SPRINT_PLAN_R23_R27_2026-03-24.md`
- `docs/SPRINT_STATUS_2026-03-19.md`

They should not be treated as the current forward sprint queue unless they are
explicitly re-synced.
