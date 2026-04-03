"""Build a local multi-format smoke pack for sanitizer/parser readiness checks.

Creates a small corpus with representative formats:
- StackExchange Posts.xml (plain + .7z + .zip)
- Reddit JSONL + valid/invalid .zst
- GitHub-style source files with comments/docstrings
- Markdown/HTML code-block text

Usage:
  `.venv\\Scripts\\python scripts\\build_format_smoke_pack.py`
  `.venv\\Scripts\\python scripts\\build_format_smoke_pack.py --out data\\format_smoke_pack`
"""

from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path

import py7zr
import pyzstd


POSTS_XML = """<?xml version="1.0" encoding="utf-8"?>
<posts>
  <row Id="100" PostTypeId="1" Title="How do I reverse a Python list?" Tags="&lt;python&gt;&lt;list&gt;" AcceptedAnswerId="101" />
  <row Id="101" PostTypeId="2" ParentId="100" Score="15" CreationDate="2024-01-01T00:00:00.000" Body="&lt;p&gt;Use slicing.&lt;/p&gt;&lt;pre&gt;&lt;code&gt;items[::-1]
&lt;/code&gt;&lt;/pre&gt;" />
  <row Id="102" PostTypeId="2" ParentId="100" Score="9" CreationDate="2024-01-02T00:00:00.000" Body="&lt;p&gt;Use reversed().&lt;/p&gt;&lt;pre&gt;&lt;code&gt;list(reversed(items))
&lt;/code&gt;&lt;/pre&gt;" />
</posts>
"""


def _write(path: Path, data: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(data, encoding="utf-8")


def build(out: Path) -> None:
    if out.exists():
        shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True, exist_ok=True)

    # Plain StackExchange XML
    se_dir = out / "stackexchange"
    _write(se_dir / "Posts.xml", POSTS_XML)

    # Valid 7z containing Posts.xml
    with py7zr.SevenZipFile(se_dir / "stackoverflow.com.7z", mode="w") as zf:
        zf.write(str(se_dir / "Posts.xml"), arcname="Posts.xml")

    # Invalid 7z signature
    (se_dir / "broken.7z").write_bytes(b"\x00" * 64)

    # Zip with Posts.xml
    with zipfile.ZipFile(se_dir / "stackexchange.zip", mode="w") as zf:
        zf.writestr("Posts.xml", POSTS_XML)

    # Reddit JSONL + zst
    reddit_dir = out / "reddit"
    reddit_lines = [
        {
            "title": "Python dict merge",
            "selftext": "Use backticks: ```python\\n{**a, **b}\\n``` and avoid @user mentions",
            "subreddit": "python",
        },
        {
            "title": "No code post",
            "selftext": "This should be ignored by sanitizer",
            "subreddit": "programming",
        },
    ]
    jsonl = "\n".join(json.dumps(x, ensure_ascii=True) for x in reddit_lines) + "\n"
    _write(reddit_dir / "RS_test.jsonl", jsonl)
    with pyzstd.open(reddit_dir / "RS_test.zst", "wt", encoding="utf-8") as f:
        f.write(jsonl)
    (reddit_dir / "broken.zst").write_bytes(b"\x00" * 64)

    # GitHub-style source
    gh_dir = out / "github"
    _write(
        gh_dir / "example.py",
        '''"""Utilities for parsing payloads.

Maintainer: Jane Doe
Contact: jane@example.com
"""

# Parse headers from payload and normalize values.
def parse_headers(payload):
    return payload.get("headers", {})
''',
    )

    # Generic docs with code blocks
    docs_dir = out / "docs"
    _write(
        docs_dir / "howto.md",
        """# Deploy notes

Author by John Smith
Use:

```bash
python main.py --mock ingest .
```

Inline: `pip install package`.
""",
    )

    print(f"[smoke-pack] built: {out}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default=os.path.join("data", "format_smoke_pack"),
        help="Output folder for synthetic smoke pack",
    )
    args = parser.parse_args()
    build(Path(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
