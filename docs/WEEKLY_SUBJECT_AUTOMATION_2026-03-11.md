# Weekly Subject Automation -- 2026-03-11

## Purpose

Keep the reviewed weekly subject summary flowing into `agent_memory` on a fixed
weekly schedule instead of relying on ad hoc manual ingestion.

## Files

- `data/weekly_subject_watchlist.json`
  Canonical subject-to-site watchlist for the 7 weekly review buckets.
- `docs/WEEKLY_SUBJECT_SUMMARY_YYYY-MM-DD.md`
  Human-reviewed summary document. This remains the ingestion source of truth.
- `scripts/weekly_knowledge_update.py`
  Ingests the latest reviewed summary into `agent_memory`.
- `scripts/run_weekly_subject_update.cmd`
  Task Scheduler entrypoint that sets the repo working directory and runs the
  ingest script.
- `scripts/run_weekly_subject_update.ps1`
  PowerShell runner for interactive use.
- `scripts/install_weekly_subject_update_task.ps1`
  Registers the weekly Windows task.

## Commands

Review what would be ingested:

```powershell
cd D:\JCoder
.venv\Scripts\python.exe scripts\weekly_knowledge_update.py --latest --dry-run
```

Ingest the newest reviewed summary:

```powershell
cd D:\JCoder
.venv\Scripts\python.exe scripts\weekly_knowledge_update.py --latest
```

Register the weekly task:

```powershell
cd D:\JCoder
powershell -ExecutionPolicy Bypass -File scripts\install_weekly_subject_update_task.ps1
```

## Schedule

- Default task name: `JCoder Weekly Subject Update`
- Default cadence: every Sunday at `02:00`
- Default behavior: ingest the newest `WEEKLY_SUBJECT_SUMMARY_*.md` file if it
  has not already been ingested

## Outputs

- Memory entries in `data/indexes/agent_memory.*`
- Human-readable memory backups in `data/agent_knowledge/`
- Run receipts in `logs/weekly_subject_updates/`

## Notes

- The scheduled job ingests the latest reviewed summary. It does not replace
  the human review step that produces the summary document.
- Duplicate runs are blocked by summary-file SHA256 unless `--force` is used.
