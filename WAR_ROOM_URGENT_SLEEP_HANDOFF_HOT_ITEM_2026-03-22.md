# URGENT SLEEP HANDOFF - HOT ITEM - 2026-03-22

## Read this first

This is a hot-item recovery handoff.

The workstation shell-takeover issue was substantially contained tonight, but morning independent QA is still mandatory before final sprint closeout or demo-readiness claims.

## What was contained

- `CodexWarRoom*` scheduled tasks were removed.
- dangerous shell/profile automation paths were neutralized
- local recovery paths and handoff docs were created
- PowerShell 7 was reinstalled
- one direct process test showed one `pwsh.exe` launch produced only one `pwsh` process and zero additional Windows Terminal processes during the observation window

## What is still unverified

- full PowerShell 7 CLI verification for `claude` and `codex`
- final OneDrive backup disablement at the app level
- complete shell matrix validation across `cmd`, Windows PowerShell, and PowerShell 7

## Non-negotiable next steps

1. Morning independent QA.
2. Verify shell stability.
3. Verify `claude` and `codex` in each shell.
4. Keep recovery paths visible.
5. Do not reintroduce scheduler, startup, profile, or terminal automation.

## HybridRAG rule

Before demo or retuning:
1. repair trusted source downloads
2. rebuild canonical corpus
3. reindex
4. validate query and retrieval behavior
5. button-smash real GUI
6. run regression
7. only then retune

## Idiot-proof rule

Assume people will skim. Read-first files and duplicate notes were intentionally placed in multiple locations because critical recovery rules must survive skimming.

Signed: General Codex
Date: 2026-03-22
