param(
    [string]$TaskName = "JCoder Weekly Subject Update",
    [string]$DayOfWeek = "SUN",
    [string]$StartTime = "02:00"
)

$ErrorActionPreference = "Stop"

$Runner = Join-Path $PSScriptRoot "run_weekly_subject_update.cmd"
$QuotedTaskName = "`"$TaskName`""
$QuotedRunner = "`"$Runner`""

$Process = Start-Process -FilePath "schtasks.exe" -ArgumentList @(
    "/Create",
    "/F",
    "/SC", "WEEKLY",
    "/D", $DayOfWeek,
    "/ST", $StartTime,
    "/TN", $QuotedTaskName,
    "/TR", $QuotedRunner
) -NoNewWindow -PassThru -Wait

if ($Process.ExitCode -ne 0) {
    throw "Failed to register scheduled task: $TaskName"
}

Write-Host "[OK] Registered task '$TaskName' for $DayOfWeek at $StartTime"
