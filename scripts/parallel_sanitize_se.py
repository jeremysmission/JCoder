"""
Parallel SE archive sanitization.

Splits archives across N workers for faster processing.
Each worker extracts and sanitizes independently, results merge at the end.

Usage:
    python scripts/parallel_sanitize_se.py [--workers N] [--se-root PATH]
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ingestion.sanitizer import (
    MAGIC_7Z,
    SanitizationConfig,
    SanitizationPipeline,
    SanitizationStats,
)

DEFAULT_SE_ROOT = Path(os.environ.get("JCODER_SE_ROOT", r"D:\Projects\KnowledgeBase\stackexchange_20251231"))
CLEAN_ROOT = Path(os.environ.get("JCODER_DATA", r"D:\JCoder_Data")) / "clean_source"


def _find_valid_archives(se_root: Path):
    """Return list of .7z files with valid headers."""
    archives = []
    for f in sorted(se_root.glob("*.7z")):
        try:
            with open(f, "rb") as fh:
                head = fh.read(6)
            if head == MAGIC_7Z:
                archives.append(f)
        except Exception:
            pass
    return archives


def _process_batch(batch_args):
    """Process a batch of archives in one worker. Returns (stats_dict, run_dir)."""
    archive_paths, batch_id, run_root = batch_args
    cfg = SanitizationConfig()
    pipeline = SanitizationPipeline(cfg)

    # Each worker gets its own run subdirectory
    run_dir = Path(run_root) / f"worker_{batch_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    stats = SanitizationStats()
    for i, arc_path in enumerate(archive_paths, 1):
        arc = Path(arc_path)
        try:
            pipeline._process_file(arc, run_dir, stats)
            print(
                f"  [W{batch_id}] {i}/{len(archive_paths)} {arc.stem} "
                f"(entries={stats.entries_written}, blocks={stats.code_blocks_kept})",
                flush=True,
            )
        except Exception as e:
            stats.errors.append(f"{arc}: {e}")
            print(f"  [W{batch_id}] {i}/{len(archive_paths)} {arc.stem} [ERROR] {e}", flush=True)

    return asdict(stats), str(run_dir)


def _merge_stats(all_stats):
    """Merge multiple SanitizationStats dicts into one."""
    merged = SanitizationStats()
    for s in all_stats:
        merged.files_seen += s["files_seen"]
        merged.entries_written += s["entries_written"]
        merged.code_blocks_kept += s["code_blocks_kept"]
        merged.pii_replacements += s["pii_replacements"]
        merged.non_english_removed += s["non_english_removed"]
        merged.compressed_skipped += s["compressed_skipped"]
        merged.errors.extend(s["errors"])
        merged.skipped_files.extend(s["skipped_files"])
    return merged


def _merge_run_dirs(worker_dirs, final_dir):
    """Merge worker output directories into one."""
    final_dir.mkdir(parents=True, exist_ok=True)
    moved = 0
    for wdir in worker_dirs:
        wpath = Path(wdir)
        if not wpath.exists():
            continue
        for sub in wpath.iterdir():
            if sub.is_dir():
                dest = final_dir / sub.name
                if dest.exists():
                    # Merge files into existing dir
                    for f in sub.iterdir():
                        target = dest / f.name
                        if not target.exists():
                            shutil.move(str(f), str(target))
                            moved += 1
                else:
                    shutil.move(str(sub), str(dest))
                    moved += len(list(dest.iterdir())) if dest.is_dir() else 1
        shutil.rmtree(wpath, ignore_errors=True)
    return moved


def main():
    parser = argparse.ArgumentParser(description="Parallel SE archive sanitization")
    parser.add_argument("--workers", type=int, default=5, help="Number of parallel workers")
    parser.add_argument("--se-root", type=Path, default=DEFAULT_SE_ROOT)
    args = parser.parse_args()

    archives = _find_valid_archives(args.se_root)
    print(f"[OK] Found {len(archives)} valid .7z archives")

    if not archives:
        print("[WARN] No valid archives to process")
        return 1

    # Sort by size descending for better load balancing (round-robin assignment)
    archives.sort(key=lambda f: f.stat().st_size, reverse=True)

    # Round-robin distribute to workers for balanced load
    batches = [[] for _ in range(args.workers)]
    for i, arc in enumerate(archives):
        batches[i % args.workers].append(str(arc))

    for i, batch in enumerate(batches):
        total_mb = sum(Path(f).stat().st_size for f in batch) / 1e6
        print(f"  Worker {i}: {len(batch)} archives, {total_mb:.0f} MB")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = CLEAN_ROOT / "_ingest_runs" / f"parallel_{ts}"
    run_root.mkdir(parents=True, exist_ok=True)

    batch_args = [(batch, i, str(run_root)) for i, batch in enumerate(batches)]

    print(f"\n[OK] Starting {args.workers} workers...")
    t0 = time.time()

    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_process_batch, ba): ba[1] for ba in batch_args}
        for future in as_completed(futures):
            batch_id = futures[future]
            try:
                stats_dict, run_dir = future.result()
                results.append((stats_dict, run_dir))
                print(
                    f"\n[OK] Worker {batch_id} done: "
                    f"{stats_dict['entries_written']} entries, "
                    f"{stats_dict['code_blocks_kept']} code blocks"
                )
            except Exception as e:
                print(f"\n[FAIL] Worker {batch_id}: {e}")

    elapsed = time.time() - t0
    print(f"\n[OK] All workers done in {elapsed:.0f}s ({elapsed/60:.1f}m)")

    # Merge results
    all_stats = [r[0] for r in results]
    worker_dirs = [r[1] for r in results]
    merged = _merge_stats(all_stats)

    final_dir = CLEAN_ROOT / "_ingest_runs" / f"{ts}_parallel_se"
    moved = _merge_run_dirs(worker_dirs, final_dir)
    shutil.rmtree(run_root, ignore_errors=True)

    # Write log
    log = {
        "generated_at": datetime.now().isoformat(),
        "raw_root": str(args.se_root),
        "run_dir": str(final_dir),
        "workers": args.workers,
        "elapsed_seconds": round(elapsed, 1),
        "archives_processed": len(archives),
        "stats": asdict(merged),
    }
    log_path = CLEAN_ROOT / "_logs" / f"sanitize_{ts}_parallel.json"
    log_path.write_text(json.dumps(log, indent=2), encoding="utf-8")

    print(f"\nResults:")
    print(f"  Archives:       {len(archives)}")
    print(f"  Entries:        {merged.entries_written}")
    print(f"  Code blocks:    {merged.code_blocks_kept}")
    print(f"  PII replaced:   {merged.pii_replacements}")
    print(f"  Non-English:    {merged.non_english_removed}")
    print(f"  Skipped:        {merged.compressed_skipped}")
    print(f"  Errors:         {len(merged.errors)}")
    print(f"  Files merged:   {moved}")
    print(f"  Output:         {final_dir}")
    print(f"  Log:            {log_path}")
    if merged.errors:
        print(f"\nErrors:")
        for e in merged.errors[:20]:
            print(f"  - {e}")
        if len(merged.errors) > 20:
            print(f"  ... and {len(merged.errors) - 20} more")
    return 0


if __name__ == "__main__":
    sys.exit(main())
