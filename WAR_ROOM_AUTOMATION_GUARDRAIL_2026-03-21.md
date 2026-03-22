# Automation Guardrail Notice - 2026-03-21

## Incident summary

System-level automation created recurring `CodexWarRoom*` scheduled tasks that launched `pwsh.exe` against wrapper scripts under `C:\Users\jerem\.ai_handoff\teams\...`.

Observed user impact:
- sporadic bursts of roughly 8-12 PowerShell windows or tabs
- PowerShell 7 remained affected after reinstall because the scheduled tasks persisted outside the app install
- regular shell usage became unusable
- emergency recovery required administrative `cmd.exe`
- fallback shell paths were effectively unavailable to the user during recovery

This behavior is not acceptable. The user did not want autonomous regeneration or autonomous team creation without a prompt.
The user must also retain an independent way to cancel or disable the automation without relying on the same shell stack being modified.

## Likely failure mode

The current automation path appears to have conflated at least two actions:
- regenerate a stale team member
- generate or relaunch an entire team with recurring automation

That distinction is safety-critical. A stale-member repair path must not silently expand into multi-window or recurring team launch behavior.

## Required guardrail going forward

Do not install scheduled tasks, startup hooks, wrapper launchers, profile edits, or recurring automation without an explicit user confirmation prompt.
Do not remove, disable, or effectively strand the user's fallback administration path.

The prompt must state:
- target repo
- exact action
- whether it creates one member or a full team
- whether it creates scheduled tasks or startup hooks
- launch cadence or trigger condition
- exact files and system locations changed
- exact backup cancel path that does not depend on the modified shell
- exact rollback command or rollback file path

## Required approval schema

Before generation or installation, the user must be shown a prompt equivalent to:

`Repo: <repo>`
`Action: <single member regeneration | full team generation | scheduler install>`
`Creates scheduled automation: <yes/no>`
`Creates shell startup changes: <yes/no>`
`Creates or opens terminal windows automatically: <yes/no>`
`Backup cancel path outside modified shell: <exact command or UI path>`
`Rollback: <exact command>`
`Proceed? <yes/no>`

No autonomous expansion from member generation to team generation is allowed without a second explicit prompt.

## Remediation completed on 2026-03-21

- cleared user PowerShell startup profile files
- removed `CodexWarRoom*` scheduled tasks from Windows Task Scheduler

## Source notes

The scheduled-task installation pattern is documented in the HybridRAG war-room automation tooling, especially the installers that call `schtasks /Create` and the team installer that provisions monitor, rotation, and headcount jobs.
