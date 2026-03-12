"""
Re-acquire corrupt StackExchange .7z archives from archive.org.

Reads the integrity log to identify archives with all-zero headers,
downloads replacements, and validates the 7z magic bytes after download.

Usage:
    python scripts/reacquire_se_archives.py [--coding-only] [--exclude-coding] [--site SITE] [--dry-run]

Flags:
    --coding-only   Only download coding/tech-relevant sites (default: all)
    --exclude-coding
                    Only download archives outside the coding/tech priority set
    --site SITE     Limit to one or more specific archive site stems
    --dry-run       Print what would be downloaded without downloading
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

from core.download_manager import DownloadManager

MAGIC_7Z = bytes.fromhex("377abcaf271c")
BASE_URL = "https://archive.org/download/stackexchange"
_JCODER_DATA_DIR = Path(
    os.environ.get("JCODER_DATA")
    or os.environ.get("JCODER_DATA_DIR", "D:/JCoder_Data")
)
INTEGRITY_LOG = _JCODER_DATA_DIR / "clean_source" / "_logs" / "stackexchange_archive_integrity_20260301.json"
_DOWNLOADERS: dict[Path, DownloadManager] = {}

# Sites with significant coding/tech content worth prioritizing.
CODING_SITES = {
    "askubuntu.com",
    "codegolf.stackexchange.com",
    "codereview.stackexchange.com",
    "crypto.stackexchange.com",
    "cs.stackexchange.com",
    "cstheory.stackexchange.com",
    "datascience.stackexchange.com",
    "dba.meta.stackexchange.com",
    "drupal.stackexchange.com",
    "electronics.stackexchange.com",
    "engineering.stackexchange.com",
    "gis.stackexchange.com",
    "math.stackexchange.com",
    "mathematica.stackexchange.com",
    "meta.askubuntu.com",
    "meta.stackoverflow.com",
    "networkengineering.stackexchange.com",
    "quantumcomputing.stackexchange.com",
    "retrocomputing.stackexchange.com",
    "reverseengineering.stackexchange.com",
    "robotics.stackexchange.com",
    "salesforce.stackexchange.com",
    "scicomp.stackexchange.com",
    "security.stackexchange.com",
    "serverfault.com",
    "sitecore.stackexchange.com",
    "sqa.stackexchange.com",
    "stats.stackexchange.com",
    "superuser.com",
    "tex.stackexchange.com",
    "unix.stackexchange.com",
    "ux.stackexchange.com",
    "webmasters.stackexchange.com",
    "ai.stackexchange.com",
    "bioinformatics.stackexchange.com",
    "blender.stackexchange.com",
    "civicrm.stackexchange.com",
    "craftcms.stackexchange.com",
    "elementaryos.stackexchange.com",
    "iot.stackexchange.com",
    "joomla.stackexchange.com",
    "monero.stackexchange.com",
    "tor.stackexchange.com",
    "es.stackoverflow.com",
    "ja.stackoverflow.com",
    "pt.stackoverflow.com",
    "ru.meta.stackoverflow.com",
}


def load_bad_archives():
    """Load list of archives with all-zero headers from integrity log."""
    with open(INTEGRITY_LOG, "r") as f:
        data = json.load(f)
    return [r for r in data["rows"] if r.get("status") == "all_zero_header"]


def archive_site(row):
    return Path(row["path"]).stem


def select_archives(rows, *, coding_only=False, exclude_coding=False, sites=None):
    selected = list(rows)
    if coding_only:
        selected = [r for r in selected if archive_site(r) in CODING_SITES]
    elif exclude_coding:
        selected = [r for r in selected if archive_site(r) not in CODING_SITES]

    wanted = {site.strip() for site in (sites or []) if site and site.strip()}
    if wanted:
        selected = [r for r in selected if archive_site(r) in wanted]

    return selected


def validate_7z(path):
    """Check if file starts with valid 7z magic bytes."""
    try:
        with open(path, "rb") as f:
            head = f.read(6)
        return head == MAGIC_7Z
    except Exception:
        return False


def resolve_download_target(dest: str | Path) -> tuple[Path, Path]:
    dest_path = Path(dest)
    return dest_path.parent, Path(dest_path.name)


def _get_downloader(cache_root: Path, timeout: float = 600.0) -> DownloadManager:
    cache_root = cache_root.resolve()
    downloader = _DOWNLOADERS.get(cache_root)
    if downloader is None:
        suffix = str(os.getpid())
        downloader = DownloadManager(
            cache_root,
            read_timeout_s=timeout,
            ledger_name=f"_download_ledger_{suffix}.sqlite3",
            staging_dir_name=f"_staging_{suffix}",
        )
        _DOWNLOADERS[cache_root] = downloader
    return downloader


def _close_downloader() -> None:
    global _DOWNLOADERS
    for downloader in _DOWNLOADERS.values():
        downloader.close()
    _DOWNLOADERS = {}


def download_archive(url, dest, timeout=600):
    """Stream-download a file with progress reporting."""
    dest_path = Path(dest)
    cache_root, relative_path = resolve_download_target(dest_path)

    result = _get_downloader(cache_root, timeout).download_file(
        url,
        relative_path,
        overwrite=True,
        chunk_size=1024 * 1024,
        progress_label=dest_path.name,
        progress_every_bytes=1024 * 1024,
    )
    if result.ok:
        return True

    print(f"\n  [FAIL] {result.error}")
    return False


def main():
    parser = argparse.ArgumentParser(description="Re-acquire corrupt SE archives")
    parser.add_argument("--coding-only", action="store_true", help="Only coding/tech sites")
    parser.add_argument(
        "--exclude-coding",
        action="store_true",
        help="Only sites outside the coding/tech priority set",
    )
    parser.add_argument(
        "--site",
        action="append",
        default=[],
        help="Limit to one or more archive site stems (repeatable)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print plan without downloading")
    args = parser.parse_args()

    if args.coding_only and args.exclude_coding:
        parser.error("--coding-only and --exclude-coding cannot be combined")

    bad = load_bad_archives()
    print(f"[OK] {len(bad)} corrupt archives found in integrity log")

    bad = select_archives(
        bad,
        coding_only=args.coding_only,
        exclude_coding=args.exclude_coding,
        sites=args.site,
    )

    if args.coding_only:
        print(f"[OK] {len(bad)} are coding/tech-relevant")
    elif args.exclude_coding:
        print(f"[OK] {len(bad)} are outside the coding/tech priority set")
    if args.site:
        print(f"[OK] Filtered to {len(bad)} requested site(s)")

    total_bytes = sum(r["size"] for r in bad)
    print(f"[OK] Total to download: {total_bytes / 1e9:.2f} GB")
    print()

    if args.dry_run:
        for r in sorted(bad, key=lambda x: x["size"], reverse=True):
            fname = os.path.basename(r["path"])
            site = fname.replace(".7z", "")
            url = f"{BASE_URL}/{fname}"
            print(f"  {site}: {r['size'] / 1e6:.0f} MB -> {url}")
        print(f"\n[DRY-RUN] Would download {len(bad)} files ({total_bytes / 1e9:.2f} GB)")
        return

    ok = 0
    fail = 0
    for i, r in enumerate(sorted(bad, key=lambda x: x["size"]), 1):
        fname = os.path.basename(r["path"])
        dest = r["path"]
        url = f"{BASE_URL}/{fname}"
        print(f"[{i}/{len(bad)}] {fname} ({r['size'] / 1e6:.0f} MB)")

        if download_archive(url, dest):
            if validate_7z(dest):
                print(f"  [OK] Valid 7z header")
                ok += 1
            else:
                print(f"  [WARN] Downloaded but header still invalid")
                fail += 1
        else:
            fail += 1

        # Rate limit: 1 second between downloads
        if i < len(bad):
            time.sleep(1)

    print(f"\nDone: {ok} OK, {fail} failed out of {len(bad)}")


if __name__ == "__main__":
    try:
        main()
    finally:
        _close_downloader()
