# Inclusive Downloader

## What Changed

JCoder had several separate downloader implementations spread across dataset
scripts. Each one handled retries, partial files, and HuggingFace manifests a
little differently. The project now has one shared path:

- `core/download_manager.py`

That manager owns:

- retry and backoff behavior
- resumable `.part` downloads
- atomic promotion from staging into the final cache
- ledger records for success and failure
- shared CA bundle and timeout handling through `core/http_factory.py`
- shared HuggingFace parquet manifest resolution

## Why This Exists

The old per-script downloaders were cheap and duplicated. That caused drift:

- one script would retry and another would not
- one script would keep partial data and another would discard it
- one script would follow the first HuggingFace config and another would fail
- operator fixes had to be repeated in every downloader script

The inclusive downloader fixes that by making acquisition behavior consistent
across archive downloads, RFC text pulls, CodeSearchNet zips, and parquet
dataset fetches.

## Rule For New Work

If you add or update any network acquisition code in JCoder:

1. Use `DownloadManager` from `core/download_manager.py`.
2. Use `fetch_huggingface_parquet_urls()` for HuggingFace parquet manifests.
3. Do not add raw `httpx.stream(...)` loops to new dataset scripts.
4. Do not add one-off retry logic in scripts unless it wraps the inclusive
   downloader for a source-specific reason.

## Backlog Queue

To pressure-test the inclusive downloader against the remaining backlog, use:

```bash
python scripts/run_download_queue.py --list
python scripts/run_download_queue.py
```

The queue definition lives in:

- `config/download_queue.json`

Update that queue when the backlog changes, but keep new entries pointed at
scripts that already use the inclusive downloader.
