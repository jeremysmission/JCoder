# JCoder Data Acquisition Plan (3 TB)

Date: 2026-03-01

## Security Policy (Official-Only)

Use only official or primary-source domains by default:
- `communitydatadump.com` / Stack Exchange-owned channels for SE dumps
- `datasets.softwareheritage.org` / `archive.softwareheritage.org`
- `zenodo.org` (for SOTorrent releases)
- `github.com/github/CodeSearchNet` and official linked storage
- `gharchive.org` / `data.gharchive.org`

Avoid mirrors/torrents unless you explicitly approve for that dataset.

## Current Environment Limitation

On this laptop environment, direct HTTP downloads are currently blocked by proxy authentication (`Invoke-WebRequest` returns authentication failed).
So commands are prepared and validated, but large download execution should be run on your beast machine/network context.

## Priority Order

1. Stack Exchange coding sites (P0)
2. SOTorrent (P1)
3. CodeSearchNet (P2)
4. GH Archive recent year (P1 metadata)
5. Software Heritage selective shards (P0/P1 depending subset)
6. Reddit programming (P2, strict code-only keep)

## Commands (PowerShell)

### 1) Stack Exchange (official catalog first)
Open official catalog and select coding sites:
```powershell
Start-Process "https://communitydatadump.com/index.html"
```

Download selected files to staging with your browser/download manager.
Then verify magic bytes (`7z` should start with `377abcaf271c`):
```powershell
Get-ChildItem D:\DataLake\raw\stackexchange\2025Q4\*.7z | % {
  $h=(Get-Content -Encoding Byte -TotalCount 6 $_.FullName | % { "{0:x2}" -f $_ }) -join ""
  [PSCustomObject]@{Name=$_.Name;Size=$_.Length;Magic=$h}
}
```

### 2) SOTorrent (Zenodo)
```powershell
New-Item -ItemType Directory -Force D:\DataLake\raw\sotorrent\latest | Out-Null
cd D:\DataLake\raw\sotorrent\latest
# Example record page (choose latest file URL from Zenodo UI)
Start-Process "https://zenodo.org/records/4415593"
```

### 3) CodeSearchNet
```powershell
New-Item -ItemType Directory -Force D:\DataLake\raw\codesearchnet | Out-Null
cd D:\DataLake\raw\codesearchnet
$langs=@("python","javascript","java","ruby","go","php")
foreach($l in $langs){
  Invoke-WebRequest -Uri "https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/$l.zip" -OutFile "$l.zip"
}
```

### 4) GH Archive (recent year)
```powershell
New-Item -ItemType Directory -Force D:\DataLake\raw\gharchive\2025 | Out-Null
cd D:\DataLake\raw\gharchive\2025
# Example single hour file
Invoke-WebRequest -Uri "https://data.gharchive.org/2025-01-01-0.json.gz" -OutFile "2025-01-01-0.json.gz"
```

### 5) Software Heritage
```powershell
Start-Process "https://datasets.softwareheritage.org/"
```
Pick selective shards first; full corpus is multi-TB.

## Ingest Rule

Always ingest via sanitizer pipeline:
```powershell
cd D:\JCoder
D:\JCoder\.venv\Scripts\python main.py --mock ingest <RAW_OR_ARCHIVE_PATH> --index-name <INDEX_NAME>
```

Never index raw directly without sanitization.

## Deletion Rule

Do not delete raw dumps without explicit user approval after verification.

## Lightweight E2E Test (Completed)

- Ingested tiny sample archive in mock mode
- Queried index and retrieved source-backed answer successfully
- Note: CLI `ask` command has a known post-response SQLite thread-close issue; workaround script path works for smoke tests.
