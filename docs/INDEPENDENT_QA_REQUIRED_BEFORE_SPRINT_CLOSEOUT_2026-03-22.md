# Independent QA Required Before Sprint Closeout - 2026-03-22

## Read this first

Tonight's work improved containment, recovery posture, documentation, and handoff quality.

It does **not** count as final sprint closeout by itself.

Morning independent QA is still required.

## Why

The recent failures touched:
- shell startup behavior
- scheduled automation
- terminal behavior
- source-data integrity
- retrieval trust
- demo readiness

Those are high-blast-radius surfaces. They require a fresh, independent verification pass before anyone claims sprint closeout or demo readiness.

## Mandatory morning independent QA

A different reviewer or fresh operator perspective should verify:

### Shell and CLI

- `cmd.exe` opens and remains stable
- Node.js command prompt opens and remains stable
- Windows PowerShell opens and remains stable
- PowerShell 7 opens and remains stable
- no extra windows or tabs spawn
- `claude` resolves
- `codex` resolves
- help commands run

### Recovery path

- `disable_ai_shell_automation.cmd` is present
- the recovery instructions are readable
- fallback access paths are still available

### HybridRAG

- trusted source repair status is explicit
- canonical corpus rebuild status is explicit
- reindex status is explicit
- query validation status is explicit
- real-GUI button-smash status is explicit
- regression status is explicit
- retuning has not started prematurely

### JCoder and other repos

- handoff notes are present
- unattended objectives are constrained
- no scheduler/startup automation is being reintroduced
- no cross-repo fanout is happening without approval

## Idiot-proof rule

No one gets to say:
- "it seems fine"
- "it probably works"
- "the demo is probably okay"
- "retuning can continue"

until the morning independent QA checklist is actually completed.

## Required closeout statement

Before sprint closeout, someone must explicitly state:

`Independent QA completed. Shell stability verified. Recovery path verified. Source-repair/reindex order preserved. No unauthorized automation reintroduced.`

If that statement cannot be truthfully made, sprint closeout is not complete.

Signed: General Codex
Date: 2026-03-22
