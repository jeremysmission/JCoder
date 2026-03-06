"""
Re-acquire corrupt StackExchange .7z archives from archive.org.

Reads the integrity log to identify archives with all-zero headers,
downloads replacements, and validates the 7z magic bytes after download.

Usage:
    python scripts/reacquire_se_archives.py [--coding-only] [--dry-run]

Flags:
    --coding-only   Only download coding/tech-relevant sites (default: all)
    --dry-run       Print what would be downloaded without downloading
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import httpx

MAGIC_7Z = bytes.fromhex("377abcaf271c")
BASE_URL = "https://archive.org/download/stackexchange"
_JCODER_DATA_DIR = Path(os.environ.get("JCODER_DATA_DIR", "D:/JCoder_Data"))
INTEGRITY_LOG = _JCODER_DATA_DIR / "clean_source" / "_logs" / "stackexchange_archive_integrity_20260301.json"

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


def validate_7z(path):
    """Check if file starts with valid 7z magic bytes."""
    try:
        with open(path, "rb") as f:
            head = f.read(6)
        return head == MAGIC_7Z
    except Exception:
        return False


def download_archive(url, dest, timeout=600):
    """Stream-download a file with progress reporting."""
    tmp = dest + ".partial"
    try:
        with httpx.stream("GET", url, timeout=timeout, follow_redirects=True) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))
            downloaded = 0
            with open(tmp, "wb") as f:
                for chunk in resp.iter_bytes(chunk_size=1024 * 1024):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        pct = downloaded / total * 100
                        print(f"\r  {downloaded / 1e6:.0f}/{total / 1e6:.0f} MB ({pct:.0f}%)", end="", flush=True)
            print()
        # Rename on success
        if os.path.exists(dest):
            os.remove(dest)
        os.rename(tmp, dest)
        return True
    except Exception as e:
        print(f"\n  [FAIL] {e}")
        if os.path.exists(tmp):
            os.remove(tmp)
        return False


def main():
    parser = argparse.ArgumentParser(description="Re-acquire corrupt SE archives")
    parser.add_argument("--coding-only", action="store_true", help="Only coding/tech sites")
    parser.add_argument("--dry-run", action="store_true", help="Print plan without downloading")
    args = parser.parse_args()

    bad = load_bad_archives()
    print(f"[OK] {len(bad)} corrupt archives found in integrity log")

    if args.coding_only:
        bad = [r for r in bad if Path(r["path"]).stem in CODING_SITES]
        print(f"[OK] {len(bad)} are coding/tech-relevant")

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
    main()
