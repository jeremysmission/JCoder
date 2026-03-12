$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$PythonExe = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$ScriptPath = Join-Path $PSScriptRoot "weekly_knowledge_update.py"

Set-Location $RepoRoot
& $PythonExe $ScriptPath --latest
exit $LASTEXITCODE
