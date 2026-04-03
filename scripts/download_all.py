"""
Master download script for JCoder knowledge corpus.

Runs all data download and preparation scripts in sequence:
1. CodeSearchNet (function + docstring pairs, 6 languages)
2. Python standard library documentation
3. RFC documents (networking/web standards)

Checks disk space before starting, reports summary at end.
Safe to re-run (idempotent -- each sub-script has resume support).

Usage:
    cd C:\\Users\\jerem\\JCoder
    .venv\\Scripts\\python scripts\\download_all.py
"""
from __future__ import annotations

import io
import os
import shutil
import sys
import time
from pathlib import Path

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Wrap stdout for Windows Unicode safety
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_ROOT = Path(os.environ.get("JCODER_DATA", "data"))
MIN_FREE_GB = 30  # Minimum free disk space required


def _check_disk_space() -> bool:
    """Check that the target drive has enough free space.

    Returns True if enough space, False otherwise.
    """
    try:
        data_path = str(DATA_ROOT.resolve())
        usage = shutil.disk_usage(data_path)
        free_gb = usage.free / (1024 ** 3)
        total_gb = usage.total / (1024 ** 3)
        used_pct = (usage.used / usage.total) * 100

        print(f"  Data path: {data_path}")
        print(f"  Total: {total_gb:.1f} GB")
        print(f"  Free:  {free_gb:.1f} GB")
        print(f"  Used:  {used_pct:.1f}%")

        if free_gb < MIN_FREE_GB:
            print(f"[FAIL] Not enough disk space. Need {MIN_FREE_GB} GB free, "
                  f"have {free_gb:.1f} GB")
            return False

        print(f"[OK] Disk space sufficient ({free_gb:.1f} GB free, "
              f"need {MIN_FREE_GB} GB)")
        return True

    except Exception as exc:
        print(f"[WARN] Could not check disk space: {exc}")
        print("     Proceeding anyway...")
        return True


def _count_output_files() -> dict:
    """Count files in the clean_source output directory."""
    output_dir = DATA_ROOT / "clean_source"
    counts = {}

    if not output_dir.exists():
        return counts

    for subdir in sorted(output_dir.iterdir()):
        if subdir.is_dir():
            count = sum(1 for f in subdir.rglob("*.md") if f.is_file())
            if count > 0:
                counts[subdir.name] = count

    return counts


def _dir_size_mb(path: Path) -> float:
    """Calculate total size of directory in MB."""
    if not path.exists():
        return 0.0
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return total / (1024 * 1024)


def main() -> int:
    """Run all download scripts in sequence."""
    print("=" * 70)
    print("JCoder Knowledge Corpus -- Master Download Script")
    print(f"Data root: {DATA_ROOT}")
    print("=" * 70)

    t0 = time.time()

    # Pre-flight: disk space check
    print("\n--- PRE-FLIGHT: Disk Space ---")
    if not _check_disk_space():
        return 1

    # Show existing state
    print("\n--- EXISTING DATA ---")
    existing = _count_output_files()
    if existing:
        for name, count in existing.items():
            print(f"  {name}: {count:,} files")
    else:
        print("  (no existing data)")

    results = {}
    failures = []

    # -----------------------------------------------------------------------
    # 1. CodeSearchNet
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 1/3: CodeSearchNet Dataset")
    print("=" * 70)

    try:
        from scripts.download_codesearchnet import download_codesearchnet
        csn_result = download_codesearchnet()
        results["codesearchnet"] = csn_result
        if csn_result.get("total_files", 0) > 0:
            print(f"[OK] CodeSearchNet complete: {csn_result['total_files']:,} files")
        else:
            print("[WARN] CodeSearchNet produced no files")
            failures.append("codesearchnet")
    except Exception as exc:
        print(f"[FAIL] CodeSearchNet: {exc}")
        failures.append("codesearchnet")
        results["codesearchnet"] = {"error": str(exc)}

    # -----------------------------------------------------------------------
    # 2. Python Docs
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 2/3: Python Standard Library Documentation")
    print("=" * 70)

    try:
        from scripts.download_python_docs import download_python_docs
        pydocs_result = download_python_docs()
        results["python_docs"] = pydocs_result
        if pydocs_result.get("files_converted", 0) > 0:
            print(f"[OK] Python docs complete: "
                  f"{pydocs_result['files_converted']} files")
        else:
            print("[WARN] Python docs produced no files")
            failures.append("python_docs")
    except Exception as exc:
        print(f"[FAIL] Python docs: {exc}")
        failures.append("python_docs")
        results["python_docs"] = {"error": str(exc)}

    # -----------------------------------------------------------------------
    # 3. RFC Documents
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 3/3: RFC Documents")
    print("=" * 70)

    try:
        from scripts.download_rfc import download_rfc
        rfc_result = download_rfc()
        results["rfc"] = rfc_result
        if rfc_result.get("files_written", 0) > 0:
            print(f"[OK] RFC docs complete: {rfc_result['files_written']} files")
        else:
            print("[WARN] RFC docs produced no files")
            failures.append("rfc")
    except Exception as exc:
        print(f"[FAIL] RFC docs: {exc}")
        failures.append("rfc")
        results["rfc"] = {"error": str(exc)}

    # -----------------------------------------------------------------------
    # Final Summary
    # -----------------------------------------------------------------------
    elapsed = time.time() - t0

    print("\n" + "=" * 70)
    print("DOWNLOAD SUMMARY")
    print("=" * 70)

    # Count final output
    final_counts = _count_output_files()
    total_files = sum(final_counts.values())
    total_size = _dir_size_mb(DATA_ROOT / "clean_source")
    download_size = _dir_size_mb(DATA_ROOT / "downloads")

    print(f"\n  Output directory: {DATA_ROOT / 'clean_source'}")
    print(f"  Downloads cache:  {DATA_ROOT / 'downloads'}")
    print("")

    for name, count in sorted(final_counts.items()):
        subdir_size = _dir_size_mb(DATA_ROOT / "clean_source" / name)
        print(f"  {name:25s} {count:>8,} files  ({subdir_size:>8.1f} MB)")

    print(f"  {'':25s} {'------':>8s}         {'------':>8s}")
    print(f"  {'TOTAL':25s} {total_files:>8,} files  ({total_size:>8.1f} MB)")
    print(f"\n  Downloads cache size: {download_size:.1f} MB")
    print(f"  Time elapsed: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    # Disk space after
    try:
        drive = str(DATA_ROOT)[:3]
        free_gb = shutil.disk_usage(drive).free / (1024 ** 3)
        print(f"  Disk space remaining: {free_gb:.1f} GB")
    except Exception:
        pass

    # Status
    if failures:
        print(f"\n[WARN] {len(failures)} source(s) had issues: {', '.join(failures)}")
        print("     Re-run this script to retry failed downloads.")
    else:
        print("\n[OK] All downloads completed successfully")

    print("\nNext steps:")
    print("  1. Run: jcoder ingest-corpus codesearchnet "
          f"{DATA_ROOT / 'clean_source' / 'codesearchnet'}")
    print("  2. Run: jcoder ingest-corpus docs "
          f"{DATA_ROOT / 'clean_source' / 'python_docs'}")
    print("  3. Run: jcoder ingest-corpus docs "
          f"{DATA_ROOT / 'clean_source' / 'rfc'}")
    print("=" * 70)

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
