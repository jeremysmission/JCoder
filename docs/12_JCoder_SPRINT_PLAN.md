# JCoder Sprint Plan (2026-03-07)

## Context
- Tracker: `docs/JCoder_Coding_Data_Dumps_Tracker_2026-03-01.xlsx` reports ~3 TB of candidate dumps, 139 GiB already deduped/ingested, and the recommended strategy to prioritize high-signal QA/code corpora on this workstation before delegating ultra-large archives to the Beast machine.
- Current download note: DS-1000 series has been downloaded to `D:/JCoder_Data/downloads/ds1000` but awaits sanitizer validation; CodeQA and OctoPack also staged in downloads. Proxy-free Beast downloads are still an option for the larger payloads.
- Self-learning foundation is in place (`_active_learn`, `_experience`, `_procedural_memory`, `_meta_cog`) and needs fresh data ingestion + regression/QA cycles to validate the latest materials.

## Outstanding Objectives
1. **Data acquisition**
   - Continue downloading the ranked dataset list from the tracker workbook, starting with the high-signal QA+code corpora (Shell/Rust/TypeScript augmenters) already flagged for this laptop. Defer multi-hundred GB dumps until the Beast machine can run them in parallel with HybridRAG.
   - Record per-download metadata (source URL, format, size, license, sanitizer status) and log completions in both this doc and `primary_to_secondary.md` for the multi-agent handoff.
2. **Ingestion pipeline**
   - Sanitize and chunk the new corpora (DS-1000, CodeQA, OctoPack) via `tools/ingest_datasets.py`. Confirm chunk counts for each language chunk to ensure Shell/Rust/TypeScript gaps are closing.
   - Re-run the FTS5 indexing & embedding tasks so the new chunks are available for retrieval evaluation; capture ingestion metrics (number of chunks, coverage) in the handoff log.
3. **Self-learning readiness**
   - Ensure the weekly "agentic self-learning research" feed is on autopilot: the scraper should collect the top 10 sources identified earlier and append metadata + summaries to `docs/11_RESEARCH_AGENTIC_SELF_LEARNING.md`, `docs/11a_CITATIONS_RANKED.md`, and the new `docs/11b_DEEP_DIVE_SUBAGENT_FINDINGS.md`.
   - Validate PRAXIS/RISE/GEPA/SEC/SSR patterns within the experience stores and meta-cognitive controllers; document any tuning knobs or regressions.
4. **Regression & clone testing**
   - Update the `jcoder` clone with the latest ingest/agent changes and run targeted regression suites (`tests/test_bridge.py`, `tests/test_agent_cmd.py`, `tests/test_ingest*.py`). Snapshot results before the big tuning pass finishes.
   - Run `pytest tests/ --maxfail=1` on any variant that touches the new ingestion tooling so we can quickly catch regressions introduced by Claude’s updates.
5. **Website scraping & monitoring**
   - Schedule a weekly Harvester/Spider job (document command) that scrapes the top-agentic research sites (GEPA, PRAXIS, Self-Play, etc.) and writes canonical notes into the repo docs. Log job status for the next session.
   - Ensure the crawler respects the `docs` folder structure and updates the `docs/Research` suite so future QA passes can cross-reference the latest findings.

## Immediate Claude Asks
- Finish the offline tuning run winners (tk4_ms10_np384, tk4_ms15_np384) and capture their metrics in `primary_to_secondary.md`.
- Push the staged commit (`da1ec2c`) once the tuning results + ingestion status are recorded.
- Continue pulling the remaining ranked downloads per the tracker and, for each, run the ingest pipeline + cross-check coverage for the Shell/Rust/TypeScript weak spots.
- Kick off the clone regression tests after the latest ingest patches land so we have a clean baseline for the next QA cycle.
- Keep updating `primary_to_secondary.md` with download metadata, ingest stats, sprint status, and regression logs so Codex/Claude can pick up where you left off without repeating work.
