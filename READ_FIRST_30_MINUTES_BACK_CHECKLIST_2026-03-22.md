# First 30 Minutes Back Checklist - 2026-03-22

## Goal

Use this checklist to resume safely and avoid missing critical verification steps.

## First 10 minutes

1. Open plain `cmd.exe`.
2. Confirm fallback tools still exist:
   - Node.js command prompt
   - `disable_ai_shell_automation.cmd`
3. Confirm no obvious shell swarm behavior appears.
4. Read:
   - [MASTER_HANDOFF_2026-03-22.md](/C:/Users/jerem/MASTER_HANDOFF_2026-03-22.md)
   - [POWER_SHELL_RECOVERY_2026-03-22.md](/C:/Users/jerem/POWER_SHELL_RECOVERY_2026-03-22.md)

## Next 10 minutes

Verify shells in this order:
1. `cmd.exe`
2. Node.js command prompt
3. Windows PowerShell
4. PowerShell 7

For each shell, check:
- opens normally
- no extra windows or tabs
- command prompt is responsive

Then check CLI resolution:
```text
where claude
where codex
```

For PowerShell shells:
```powershell
Get-Command claude
Get-Command codex
```

## Next 10 minutes

If shells are stable, verify:
```text
claude --help
codex --help
```

Then review repo priorities:
- HybridRAG source repair and reindex order
- JCoder source-trust and self-learning promotion discipline
- repo handoff notes in key repos

## Stop conditions

Stop and do not proceed with retuning or new automation if:
- any shell spawns extra windows or tabs
- `pwsh` hangs again
- `claude` or `codex` resolution is inconsistent
- OneDrive appears to still be reclaiming `Documents`
- source corpus integrity is still uncertain

## Rule

Do not try to “work around” unresolved shell instability by adding more automation.
