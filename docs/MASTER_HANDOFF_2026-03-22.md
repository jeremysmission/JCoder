# Master Handoff - 2026-03-22

## Current system truth

This handoff captures the minimum important truth needed for the next crew to resume without missing critical context.

## What was fixed

- Removed the `CodexWarRoom*` scheduled tasks that were launching `pwsh.exe` and contributing to sporadic PowerShell window or tab spawning.
- Neutralized the active OneDrive-based PowerShell profile hooks.
- Redirected the `Documents` shell-folder path away from OneDrive at the registry level.
- Created clean local PowerShell profile files under `C:\Users\jerem\Documents\...`.
- Installed PowerShell 7 again.
- Wrote recovery, guardrail, and unattended-team guidance into multiple repos.

## What is strongly believed to be true

- The original shell-takeover behavior was caused primarily by a bad automation design using local scheduled tasks, shell/profile integration, and terminal automation.
- OneDrive-backed `Documents` was a real source of PowerShell profile instability and confusion.
- The old multi-window/tab behavior was not a normal PowerShell 7 default.

## What is still not fully verified

- End-to-end PowerShell 7 CLI validation for `claude` and `codex`.
- Clean PowerShell 7 startup and exit behavior under automated probes.
- Final OneDrive backup disablement at the app level, not just registry redirection.
- Full shell matrix verification across:
  - `cmd.exe`
  - Windows PowerShell
  - PowerShell 7

## Safe assumptions for next crew

- Do not treat PowerShell 7 as fully cleared until CLI validation is complete.
- Do not assume `claude` and `codex` are working in every shell yet.
- Do not re-enable any scheduler, profile, terminal, or startup automation.
- Keep plain `cmd.exe` and the Node.js command prompt as fallback access paths.

## Immediate priorities

### Current sprint queue

The next five execution slices are now prioritized as:
1. R23: Dual-GPU Beast validation
2. R24: Ingest saturation pass
3. R25: Retrieval baseline and regression check
4. R26: Self-learning cycle
5. R27: Demo hardening

Reference plan: [docs/SPRINT_PLAN_R23_R27_2026-03-24.md](/C:/Users/jerem/JCoder/docs/SPRINT_PLAN_R23_R27_2026-03-24.md)

Weekly retrospective: [docs/WEEKLY_RETROSPECTIVE_2026-03-24.md](/C:/Users/jerem/JCoder/docs/WEEKLY_RETROSPECTIVE_2026-03-24.md)

### 1. Shell and CLI recovery verification

Verify in this order:
1. `cmd.exe`
2. Node.js command prompt
3. Windows PowerShell
4. PowerShell 7

For each shell, verify:
- shell opens normally
- no extra windows/tabs spawn
- `claude` resolves
- `codex` resolves
- `--help` runs

### 2. HybridRAG pre-demo recovery

Do this in order:
1. repair source downloads from trusted sources
2. rebuild the clean canonical corpus
3. reindex
4. validate query and retrieval behavior
5. run button-smash testing on real GUI
6. run regression tests
7. only then retune retrieval, ranking, and query logic

### 3. JCoder recovery

Prioritize:
- source-trust classification
- quarantine of weak dumps
- promotion discipline for self-learning outputs
- no self-learning promotion without baseline wins

## Critical rules

- Do not optimize on top of an untrusted substrate.
- Do not automate on top of an untrusted control plane.
- No scheduler installs by default.
- No shell/profile/startup edits by default.
- No terminal spawning from recurring automation.
- No cross-repo fanout without explicit approval.
- No retuning before source repair and reindex.

## Recovery and safety artifacts

- Emergency disable script:
  [disable_ai_shell_automation.cmd](/C:/Users/jerem/bin/disable_ai_shell_automation.cmd)
- PowerShell recovery note:
  [POWER_SHELL_RECOVERY_2026-03-22.md](/C:/Users/jerem/POWER_SHELL_RECOVERY_2026-03-22.md)
- Unattended team template:
  [UNATTENDED_REPO_TEAM_OBJECTIVE_TEMPLATE_2026-03-22.md](/C:/Users/jerem/UNATTENDED_REPO_TEAM_OBJECTIVE_TEMPLATE_2026-03-22.md)

## Highest-risk things to avoid

- pretending retuning can fix corpus corruption
- trusting VM success as demo readiness
- allowing local shell automation to act as orchestrator
- letting teams regenerate themselves or fan out silently
- merging self-learning outputs without controlled eval wins

## Resume principle

Containment happened first. Durability comes next.

The next crew should prefer truth, verification, and recovery order over speed.

Signed: General Codex
Date: 2026-03-22
