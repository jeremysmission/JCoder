"""
Post-download validation for reacquired SE archives.

Checks all .7z files in the SE root, compares before/after integrity,
and produces a summary report.

Usage:
    python scripts/validate_se_downloads.py [--se-root PATH]
"""

import argparse
import json
import os
import sys
from pathlib import Path

MAGIC_7Z = bytes.fromhex("377abcaf271c")
DEFAULT_SE_ROOT = Path(os.environ.get("JCODER_SE_ROOT", r"D:\Projects\KnowledgeBase\stackexchange_20251231"))
INTEGRITY_LOG = Path(os.environ.get("JCODER_DATA", "data")) / "clean_source" / "_logs" / "stackexchange_archive_integrity_20260301.json"


def check_7z(path: Path) -> dict:
    """Return header info for a .7z file."""
    try:
        size = path.stat().st_size
        with open(path, "rb") as f:
            head = f.read(6)
        valid = head == MAGIC_7Z
        all_zero = head == b"\x00" * 6
        return {
            "path": str(path),
            "name": path.name,
            "size": size,
            "valid": valid,
            "all_zero": all_zero,
            "magic_hex": head.hex(),
        }
    except Exception as e:
        return {
            "path": str(path),
            "name": path.name,
            "size": 0,
            "valid": False,
            "all_zero": False,
            "magic_hex": f"error:{e}",
        }


def load_old_integrity() -> dict:
    """Load the previous integrity log for comparison."""
    if not INTEGRITY_LOG.exists():
        return {}
    try:
        data = json.loads(INTEGRITY_LOG.read_text(encoding="utf-8"))
        return {r["path"]: r for r in data.get("rows", [])}
    except Exception:
        return {}


def main():
    parser = argparse.ArgumentParser(description="Validate SE archive downloads")
    parser.add_argument(
        "--se-root",
        type=Path,
        default=DEFAULT_SE_ROOT,
        help=f"SE archive root (default: {DEFAULT_SE_ROOT})",
    )
    args = parser.parse_args()

    se_root = args.se_root
    if not se_root.exists():
        print(f"[FAIL] SE root not found: {se_root}")
        return 1

    old_integrity = load_old_integrity()

    archives = sorted(se_root.glob("*.7z"))
    print(f"[OK] Found {len(archives)} .7z files in {se_root}")
    print()

    valid = []
    still_bad = []
    fixed = []
    new_bad = []

    for arc in archives:
        info = check_7z(arc)
        old = old_integrity.get(str(arc), {})
        was_bad = old.get("status") == "all_zero_header"

        if info["valid"]:
            valid.append(info)
            if was_bad:
                fixed.append(info)
        else:
            still_bad.append(info)
            if not was_bad:
                new_bad.append(info)

    print(f"  Valid:     {len(valid)}/{len(archives)}")
    print(f"  Invalid:   {len(still_bad)}/{len(archives)}")
    print(f"  Fixed:     {len(fixed)} (were corrupt, now valid)")
    if new_bad:
        print(f"  New bad:   {len(new_bad)} (were valid, now corrupt!)")
    print()

    if fixed:
        total_fixed_bytes = sum(f["size"] for f in fixed)
        print(f"[OK] Fixed archives ({len(fixed)}, {total_fixed_bytes / 1e9:.2f} GB):")
        for f in sorted(fixed, key=lambda x: x["name"]):
            print(f"  + {f['name']} ({f['size'] / 1e6:.0f} MB)")
        print()

    if still_bad:
        total_bad_bytes = sum(old_integrity.get(b["path"], {}).get("size", b["size"]) for b in still_bad)
        print(f"[WARN] Still corrupt ({len(still_bad)}, ~{total_bad_bytes / 1e9:.2f} GB original sizes):")
        for b in sorted(still_bad, key=lambda x: x["name"])[:20]:
            print(f"  - {b['name']} (magic={b['magic_hex']})")
        if len(still_bad) > 20:
            print(f"  ... and {len(still_bad) - 20} more")
        print()

    # Write updated integrity log
    from datetime import datetime
    new_rows = []
    for arc in archives:
        info = check_7z(arc)
        new_rows.append({
            "path": str(arc),
            "size": info["size"],
            "magic_hex": info["magic_hex"],
            "status": "valid" if info["valid"] else ("all_zero_header" if info["all_zero"] else "bad_header"),
        })

    new_log = {
        "generated_at": datetime.now().isoformat(),
        "se_root": str(se_root),
        "total": len(archives),
        "valid": len(valid),
        "invalid": len(still_bad),
        "fixed_this_run": len(fixed),
        "rows": new_rows,
    }

    out_path = INTEGRITY_LOG.parent / f"stackexchange_archive_integrity_{datetime.now().strftime('%Y%m%d')}.json"
    out_path.write_text(json.dumps(new_log, indent=2), encoding="utf-8")
    print(f"[OK] Updated integrity log: {out_path}")

    print(f"\nSummary: {len(valid)}/{len(archives)} valid ({len(fixed)} fixed this session)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
