"""
Download government XLS/PPT/XLSX files from open sources.
-----------------------------------------------------------
Fetches spreadsheets and presentations from:
  - Data.gov CKAN API (.xls, .xlsx datasets)
  - NIST publications catalog (.xlsx)
  - World Bank data catalog (.xlsx)
  - ProjectManagementDocs.com (free PM templates)

Usage:
    python scripts/download_gov_office_files.py [--output data/raw_downloads/rare_formats/gov_office]
"""

import json
import os
import sys
import time
import urllib.request
from pathlib import Path


# Direct download URLs for known .xls/.xlsx/.ppt files
DIRECT_DOWNLOADS = [
    # NIST
    {
        "url": "https://csrc.nist.gov/csrc/media/Projects/risk-management/800-53-controls/documents/sp800-53-controls.xlsx",
        "filename": "nist_sp800_53_controls.xlsx",
        "source": "NIST",
        "notes": "SP 800-53 security controls catalog",
    },
    # World Bank
    {
        "url": "https://databank.worldbank.org/data/download/world_bank_data_catalog.xlsx",
        "filename": "world_bank_data_catalog.xlsx",
        "source": "World Bank",
        "notes": "Full World Bank data catalog",
    },
    # MITRE ATT&CK
    {
        "url": "https://attack.mitre.org/docs/enterprise-attack-v16.1/enterprise-attack-v16.1-techniques.xlsx",
        "filename": "mitre_attack_techniques.xlsx",
        "source": "MITRE",
        "notes": "ATT&CK enterprise techniques matrix",
    },
    {
        "url": "https://attack.mitre.org/docs/enterprise-attack-v16.1/enterprise-attack-v16.1-mitigations.xlsx",
        "filename": "mitre_attack_mitigations.xlsx",
        "source": "MITRE",
        "notes": "ATT&CK enterprise mitigations",
    },
    {
        "url": "https://attack.mitre.org/docs/enterprise-attack-v16.1/enterprise-attack-v16.1-software.xlsx",
        "filename": "mitre_attack_software.xlsx",
        "source": "MITRE",
        "notes": "ATT&CK enterprise software catalog",
    },
]


def download_file(url: str, dest: Path, source: str = "") -> bool:
    """Download a single file. Returns True on success."""
    if dest.exists() and dest.stat().st_size > 500:
        print(f"  [SKIP] {dest.name} (already exists)")
        return True
    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "JCoder-Research/1.0 (AI research)",
        })
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()
            if len(data) < 100:
                print(f"  [WARN] {dest.name}: too small ({len(data)} bytes)")
                return False
            dest.write_bytes(data)
            print(f"  [OK] {dest.name} ({len(data):,} bytes) from {source}")
            return True
    except Exception as exc:
        print(f"  [FAIL] {dest.name}: {exc}")
        return False


def fetch_datagov_xls(output_dir: Path, max_items: int = 30) -> int:
    """Query Data.gov CKAN API for .xls format datasets."""
    api_url = "https://catalog.data.gov/api/3/action/package_search"
    params = f"?q=format:xls&rows={max_items}"
    success = 0

    try:
        req = urllib.request.Request(api_url + params, headers={
            "User-Agent": "JCoder-Research/1.0",
        })
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        results = data.get("result", {}).get("results", [])
        print(f"  Data.gov: found {len(results)} datasets with XLS resources")

        for dataset in results:
            for resource in dataset.get("resources", []):
                fmt = (resource.get("format") or "").lower()
                url = resource.get("url", "")
                if fmt in ("xls", "xlsx") and url:
                    name = resource.get("name", "") or dataset.get("name", "")
                    safe_name = "".join(c if c.isalnum() or c in "-_." else "_" for c in name)[:80]
                    ext = ".xlsx" if "xlsx" in fmt else ".xls"
                    dest = output_dir / f"datagov_{safe_name}{ext}"
                    if download_file(url, dest, "Data.gov"):
                        success += 1
                    time.sleep(0.5)
                    if success >= max_items:
                        return success
    except Exception as exc:
        print(f"  [ERROR] Data.gov API: {exc}")

    return success


def main():
    output = Path("data/raw_downloads/rare_formats/gov_office")

    for i, arg in enumerate(sys.argv[1:]):
        if arg == "--output" and i + 1 < len(sys.argv) - 1:
            output = Path(sys.argv[i + 2])

    output.mkdir(parents=True, exist_ok=True)
    total = 0

    # Phase 1: Direct downloads (known URLs)
    print(f"\n=== Phase 1: Direct downloads ({len(DIRECT_DOWNLOADS)} files) ===")
    for item in DIRECT_DOWNLOADS:
        dest = output / item["filename"]
        if download_file(item["url"], dest, item["source"]):
            total += 1
        time.sleep(0.5)

    # Phase 2: Data.gov CKAN API
    print("\n=== Phase 2: Data.gov XLS datasets ===")
    total += fetch_datagov_xls(output, max_items=20)

    print(f"\nDone: {total} files downloaded to {output}")


if __name__ == "__main__":
    os.chdir(os.environ.get("JCODER_ROOT", str(Path(__file__).resolve().parent.parent)))
    main()
