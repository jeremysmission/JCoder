"""
Download URL Recovery Script
-------------------------------
Scans Side Hustle RAG Business metadata sidecars and JCoder download
manifests to build a recovery manifest of original download URLs.

Also queries ALTERNATIVE_DATA_SOURCES.md bulk sources for formats
not found in existing metadata (.xls, .ppt, .epub, .odt, .ods, .odp).

Usage:
    python scripts/recover_download_urls.py [--output recovery_manifest.json]
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

# Target rare formats we need to recover sources for
TARGET_FORMATS = {".xls", ".ppt", ".epub", ".odt", ".ods", ".odp",
                  ".rst", ".svg", ".drawio", ".dia", ".tsv"}

# Known bulk sources for each format (from ALTERNATIVE_DATA_SOURCES.md)
BULK_SOURCES = {
    ".xls": [
        {
            "name": "Data.gov CKAN API",
            "url": "https://catalog.data.gov/api/3/action/package_search?q=format:xls&rows=100",
            "method": "api",
            "notes": "311K datasets, many with .xls exports",
        },
        {
            "name": "NIST SP 800-53 Control Catalog",
            "url": "https://csrc.nist.gov/publications",
            "method": "direct",
            "notes": "Master XLSX spreadsheet of all FIPS/SP publications",
        },
        {
            "name": "Internet Archive Military Manuals",
            "cmd": 'ia download MManuals --glob="*.xls"',
            "method": "ia_cli",
            "notes": "Requires: pip install internetarchive && ia configure",
        },
        {
            "name": "World Bank Data Catalog",
            "url": "https://databank.worldbank.org/data/download/world_bank_data_catalog.xlsx",
            "method": "direct",
            "notes": "Full catalog as XLSX",
        },
        {
            "name": "MITRE ATT&CK Excel Export",
            "cmd": "pip install mitreattack-python && python -c \"from mitreattack.stix20 import MitreAttackData; MitreAttackData.attackToExcel()\"",
            "method": "python",
            "notes": "Generates .xlsx from STIX data",
        },
    ],
    ".ppt": [
        {
            "name": "Internet Archive Military Manuals",
            "cmd": 'ia download MManuals --glob="*.ppt"',
            "method": "ia_cli",
        },
        {
            "name": "SANS Reading Room Papers",
            "url": "https://www.sans.org/white-papers/",
            "method": "scrape",
            "notes": "3000+ cybersecurity papers, some in PPT format",
        },
        {
            "name": "ProjectManagementDocs.com Templates",
            "url": "https://www.projectmanagementdocs.com/",
            "method": "scrape",
            "notes": "Free PM templates in DOCX/XLSX/PPT",
        },
    ],
    ".epub": [
        {
            "name": "Project Gutenberg",
            "url": "https://www.gutenberg.org/robot/harvest?filetypes[]=epub",
            "method": "wget",
            "notes": "70,000 public domain books in EPUB",
        },
        {
            "name": "OpenStax Free Textbooks",
            "url": "https://openstax.org/subjects",
            "method": "direct",
            "notes": "90 free college textbooks in EPUB",
        },
        {
            "name": "Internet Archive EPUB Collections",
            "cmd": 'ia search "mediatype:texts AND format:epub" --itemlist | head -100 | while read id; do ia download "$id" --glob="*.epub"; done',
            "method": "ia_cli",
        },
    ],
    ".odt": [
        {
            "name": "Anna's Archive (Public Domain)",
            "url": "https://annas-archive.gd/",
            "method": "torrent",
            "notes": "RARE format. Convert EPUB to ODT via LibreOffice CLI if needed",
        },
    ],
    ".ods": [
        {
            "name": "Anna's Archive (Public Domain)",
            "url": "https://annas-archive.gd/",
            "method": "torrent",
            "notes": "RARE format. Some government datasets available in ODS",
        },
    ],
    ".odp": [
        {
            "name": "Anna's Archive (Public Domain)",
            "url": "https://annas-archive.gd/",
            "method": "torrent",
            "notes": "RARE format. Convert PPT to ODP via LibreOffice CLI if needed",
        },
    ],
    ".rst": [
        {
            "name": "Python Documentation Source",
            "cmd": "git clone https://github.com/python/cpython.git --depth=1 && find cpython/Doc -name '*.rst'",
            "method": "git",
            "notes": "Thousands of .rst files from CPython docs",
        },
        {
            "name": "Linux Kernel Documentation",
            "cmd": "git clone https://github.com/torvalds/linux.git --depth=1 && find linux/Documentation -name '*.rst'",
            "method": "git",
        },
        {
            "name": "LLVM Documentation",
            "cmd": "git clone https://github.com/llvm/llvm-project.git --depth=1 && find llvm-project/llvm/docs -name '*.rst'",
            "method": "git",
        },
    ],
    ".svg": [
        {
            "name": "OWASP Cheat Sheet Series",
            "cmd": "git clone https://github.com/OWASP/CheatSheetSeries.git",
            "method": "git",
            "notes": "SVG diagrams embedded in security cheat sheets",
        },
        {
            "name": "USGS FGDC Geological Symbols",
            "url": "http://pubs.usgs.gov/tm/2006/11A02",
            "method": "direct",
        },
        {
            "name": "QGIS Geology Symbology SVGs",
            "url": "https://sourceforge.net/projects/qgisgeologysymbology/files/svg/",
            "method": "direct",
        },
    ],
    ".drawio": [
        {
            "name": "draw.io Templates",
            "cmd": "git clone https://github.com/jgraph/drawio-diagrams.git",
            "method": "git",
        },
        {
            "name": "draw.io Shape Libraries",
            "cmd": "git clone https://github.com/jgraph/drawio-libs.git",
            "method": "git",
        },
    ],
    ".dia": [
        {
            "name": "Dia Additional Shapes",
            "cmd": "git clone https://github.com/sdteffen/dia-additional-shapes.git",
            "method": "git",
        },
        {
            "name": "DIA Shapes Collection",
            "cmd": "git clone https://github.com/retgoat/DIA-shapes.git",
            "method": "git",
        },
    ],
    ".tsv": [
        {
            "name": "HathiTrust Hathifiles",
            "url": "https://www.hathitrust.org/hathifiles",
            "method": "wget",
            "notes": "Tab-delimited inventory of 1.4M+ US Federal Documents",
        },
        {
            "name": "Data.gov TSV Datasets",
            "url": "https://catalog.data.gov/api/3/action/package_search?q=format:tsv&rows=100",
            "method": "api",
        },
    ],
}


def scan_side_hustle_metadata(root: Path) -> List[Dict[str, Any]]:
    """Scan Side Hustle source_feed for metadata with download URLs."""
    results = []
    source_feed = root / "research" / "source_feed"
    if not source_feed.exists():
        return results
    for meta_file in sorted(source_feed.rglob("*.metadata.json")):
        try:
            data = json.loads(meta_file.read_text(encoding="utf-8"))
            results.append({
                "source": "side_hustle",
                "id": data.get("id", ""),
                "url": data.get("url", ""),
                "final_url": data.get("final_url", ""),
                "status": data.get("status", ""),
                "content_type": data.get("content_type", ""),
                "saved_relpath": data.get("saved_relpath", ""),
                "sha256": data.get("sha256", ""),
            })
        except Exception:
            continue
    return results


def scan_jcoder_download_queue(root: Path) -> List[Dict[str, Any]]:
    """Scan JCoder download queue config for source entries."""
    results = []
    queue_file = root / "config" / "download_queue.json"
    if not queue_file.exists():
        return results
    try:
        data = json.loads(queue_file.read_text(encoding="utf-8"))
        items = data if isinstance(data, list) else data.get("queue", [])
        for item in items:
            results.append({
                "source": "jcoder_queue",
                "id": item.get("id", ""),
                "description": item.get("description", ""),
                "command": item.get("command", []),
                "tags": item.get("tags", []),
            })
    except Exception:
        pass
    return results


def build_recovery_manifest(
    side_hustle_root: Path,
    jcoder_root: Path,
    output_path: Path,
) -> Dict[str, Any]:
    """Build a complete recovery manifest combining all sources."""
    manifest = {
        "generated_by": "recover_download_urls.py",
        "target_formats": sorted(TARGET_FORMATS),
        "existing_metadata": [],
        "bulk_sources_by_format": {},
        "jcoder_queue_entries": [],
    }

    # Scan existing metadata
    manifest["existing_metadata"] = scan_side_hustle_metadata(side_hustle_root)

    # Scan JCoder queue
    manifest["jcoder_queue_entries"] = scan_jcoder_download_queue(jcoder_root)

    # Add bulk sources for each target format
    for fmt in sorted(TARGET_FORMATS):
        manifest["bulk_sources_by_format"][fmt] = BULK_SOURCES.get(fmt, [])

    # Write manifest
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    return manifest


def main():
    side_hustle = Path(os.environ.get(
        "SIDE_HUSTLE_ROOT",
        "C:/Users/jerem/Side Hustle RAG Business",
    ))
    jcoder = Path(os.environ.get("JCODER_ROOT", "C:/Users/jerem/JCoder"))

    output = Path(sys.argv[1]) if len(sys.argv) > 1 else jcoder / "data" / "recovery_manifest.json"

    manifest = build_recovery_manifest(side_hustle, jcoder, output)

    print(f"Recovery manifest written to: {output}")
    print(f"  Existing metadata entries: {len(manifest['existing_metadata'])}")
    print(f"  JCoder queue entries: {len(manifest['jcoder_queue_entries'])}")
    print(f"  Target formats: {len(manifest['target_formats'])}")
    total_sources = sum(len(v) for v in manifest["bulk_sources_by_format"].values())
    print(f"  Bulk source entries: {total_sources}")


if __name__ == "__main__":
    main()
