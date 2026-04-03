"""
Download programming-relevant RFC documents and convert to markdown.

Downloads individual RFC text files from rfc-editor.org and converts
them to markdown compatible with corpus_pipeline.ingest_markdown_docs.

Output: $JCODER_DATA\\clean_source\\rfc\\

Usage:
    cd C:\\Users\\jerem\\JCoder
    python scripts/download_rfc.py
"""
from __future__ import annotations

import io
import os
import re
import sys
import time
from pathlib import Path

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from core.download_manager import DownloadManager

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Primary: IETF datatracker (more permissive). Fallback: rfc-editor.org
RFC_URLS = [
    "https://www.ietf.org/rfc/rfc{number}.txt",
    "https://raw.githubusercontent.com/nicktimko/rfc-archive/master/rfc{number}.txt",
]

DATA_ROOT = Path(os.environ.get("JCODER_DATA", "data"))
DOWNLOAD_DIR = DATA_ROOT / "downloads" / "rfc"
OUTPUT_DIR = DATA_ROOT / "clean_source" / "rfc"

MAX_RETRIES = 3

RELEVANT_RFCS = {
    1035: "DNS - Domain Names - Implementation and Specification",
    2616: "HTTP/1.1 (original)",
    2818: "HTTP Over TLS (HTTPS)",
    3986: "URI - Uniform Resource Identifier",
    5246: "TLS 1.2",
    6455: "The WebSocket Protocol",
    6570: "URI Template",
    6749: "OAuth 2.0 Authorization Framework",
    7230: "HTTP/1.1 - Message Syntax and Routing",
    7231: "HTTP/1.1 - Semantics and Content",
    7232: "HTTP/1.1 - Conditional Requests",
    7233: "HTTP/1.1 - Range Requests",
    7234: "HTTP/1.1 - Caching",
    7235: "HTTP/1.1 - Authentication",
    7519: "JSON Web Token (JWT)",
    7540: "HTTP/2",
    7807: "Problem Details for HTTP APIs",
    8259: "The JSON Data Interchange Format",
    8446: "TLS 1.3",
    9110: "HTTP Semantics",
}


def _download_rfc(number: int, downloader: DownloadManager) -> tuple[str, bool]:
    """Download a single RFC text file. Returns text content or empty string."""
    txt_path = DOWNLOAD_DIR / f"rfc{number}.txt"
    if txt_path.exists() and txt_path.stat().st_size > 100:
        return txt_path.read_text(encoding="utf-8", errors="replace"), True

    for url_template in RFC_URLS:
        url = url_template.format(number=number)
        result = downloader.download_file(
            url,
            txt_path.name,
            min_existing_bytes=100,
            chunk_size=64 * 1024,
        )
        if not result.ok:
            continue
        text = result.path.read_text(encoding="utf-8", errors="replace")
        if len(text) > 500:
            return text, False
    print(f"[WARN] RFC {number}: download failed from all sources")
    return "", False


def _rfc_to_markdown(number: int, title: str, text: str) -> str:
    """Convert raw RFC text to markdown."""
    lines = text.splitlines()
    # Strip page break headers/footers (lines with form feed or [Page X])
    cleaned = []
    for line in lines:
        if "\f" in line:
            continue
        if re.match(r"^\s*\[Page \d+\]\s*$", line):
            continue
        if re.match(r"^RFC \d+\s+.+\s+\w+ \d{4}\s*$", line):
            continue
        cleaned.append(line)

    body = "\n".join(cleaned).strip()

    # Convert section headers to markdown
    body = re.sub(r"^(\d+\.[\d.]*)\s+(.+)$", r"## \1 \2", body, flags=re.MULTILINE)

    header = f"# RFC {number}: {title}\n\n- source: rfc-editor.org\n- rfc_number: {number}\n\n"
    return header + body


def main():
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nRFC Download: {len(RELEVANT_RFCS)} programming-relevant documents")
    print("=" * 50)

    stats = {"downloaded": 0, "cached": 0, "failed": 0, "converted": 0}
    t0 = time.monotonic()

    with DownloadManager(
        DOWNLOAD_DIR,
        user_agent="JCoder/1.0 (educational)",
        max_retries=MAX_RETRIES,
        read_timeout_s=120.0,
    ) as downloader:
        for number, title in sorted(RELEVANT_RFCS.items()):
            md_path = OUTPUT_DIR / f"rfc{number}.md"
            if md_path.exists() and md_path.stat().st_size > 100:
                stats["cached"] += 1
                print(f"[OK] RFC {number} already converted, skipping")
                continue

            text, from_cache = _download_rfc(number, downloader)
            if not text:
                stats["failed"] += 1
                continue

            if from_cache:
                stats["cached"] += 1
            else:
                stats["downloaded"] += 1

            md = _rfc_to_markdown(number, title, text)
            md_path.write_text(md, encoding="utf-8")
            stats["converted"] += 1
            print(f"[OK] RFC {number}: {title} ({len(md):,} chars)")
            time.sleep(0.5)  # Be polite to the server

    elapsed = time.monotonic() - t0
    print("=" * 50)
    print(f"[OK] Done in {elapsed:.1f}s")
    print(f"  Downloaded: {stats['downloaded']}")
    print(f"  From cache: {stats['cached']}")
    print(f"  Converted:  {stats['converted']}")
    print(f"  Failed:     {stats['failed']}")


if __name__ == "__main__":
    main()
