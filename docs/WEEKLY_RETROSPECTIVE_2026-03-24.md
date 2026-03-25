# Weekly Retrospective -- 2026-03-24

## Time Frame

Past week covered by this note: 2026-03-17 through 2026-03-24.

## Greatest Achievements

- Locked in the extension allowlist cleanup as a single-source-of-truth refactor, including the `.cs` taxonomy correction and the `JCODER_DATA` path cleanup.
- Unblocked and verified the self-learning pipeline by fixing the FTS5 `search_content` vs `content` mismatch.
- Split the oversized agent modules into smaller units without breaking the runtime import path.
- Brought the retrieval baseline back to a measurable state and tied it to the repo self-index.
- Proved the ingest lane can run end-to-end on real corpora, including resumable long runs on `coderforge`.

## Greatest Lessons

- Baseline scripts must load the same corpus the questions were written against. A zero-score eval can be a wiring bug, not a model failure.
- FTS5 schema drift is a high-risk failure mode. One column mismatch can silently turn a learning pipeline into a no-op.
- Long-running ingest work needs resumable checkpoints or the session window becomes the bottleneck.
- GPU availability is not enough. The load has to be split intentionally so one lane does not starve the other.
- Tempdir and filesystem permissions can invalidate pytest even when the code path is correct, so environment health needs its own check.

## What Changed the Most

- The repo is now more crash-resilient because the handoff docs and sprint queue were refreshed during active work.
- The retrieval path is now measurable again, with a real baseline against [evaluation/golden_questions_v1.json](/C:/Users/jerem/JCoder/evaluation/golden_questions_v1.json).
- Heavy ingest runs now leave resumable progress artifacts instead of failing silently.

## Operating Summary

The main shift this week was from cleanup and repair into measurable throughput work. The codebase now has clearer boundaries, the retrieval harness can actually score the intended corpus, and the ingest path can survive long-running stress without losing state.

Signed: Codex
Repo: JCoder
Date: 2026-03-24
Time: 08:30 MDT

