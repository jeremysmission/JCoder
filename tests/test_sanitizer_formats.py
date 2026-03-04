"""Sanitizer format readiness smoke tests.

These tests verify we can sanitize representative source/container formats
before large-scale indexing.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

py7zr = pytest.importorskip("py7zr")
pyzstd = pytest.importorskip("pyzstd")

from ingestion.sanitizer import SanitizationConfig, SanitizationPipeline


def _write(path: Path, data: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(data, encoding="utf-8")


def _build_pack(root: Path) -> None:
    posts_xml = """<?xml version="1.0" encoding="utf-8"?>
<posts>
  <row Id="1" PostTypeId="1" Title="Python merge dicts" Tags="&lt;python&gt;&lt;dict&gt;" AcceptedAnswerId="2" />
  <row Id="2" PostTypeId="2" ParentId="1" Score="10" CreationDate="2024-01-01T00:00:00.000" Body="&lt;p&gt;Use unpacking.&lt;/p&gt;&lt;pre&gt;&lt;code&gt;{**a, **b}
&lt;/code&gt;&lt;/pre&gt;" />
</posts>
"""
    se = root / "stackexchange"
    _write(se / "Posts.xml", posts_xml)
    with py7zr.SevenZipFile(se / "site.7z", mode="w") as zf:
        zf.write(str(se / "Posts.xml"), arcname="Posts.xml")
    with zipfile.ZipFile(se / "site.zip", mode="w") as zf:
        zf.writestr("Posts.xml", posts_xml)
    (se / "broken.7z").write_bytes(b"\x00" * 32)

    reddit = root / "reddit"
    rows = [
        {"title": "Code sample", "selftext": "```python\nprint('x')\n```", "subreddit": "python"},
        {"title": "No code", "selftext": "hello world", "subreddit": "programming"},
    ]
    jsonl = "\n".join(json.dumps(x) for x in rows) + "\n"
    _write(reddit / "RS_test.jsonl", jsonl)
    with pyzstd.open(reddit / "RS_test.zst", "wt", encoding="utf-8") as f:
        f.write(jsonl)
    (reddit / "broken.zst").write_bytes(b"\x00" * 32)

    gh = root / "github"
    _write(
        gh / "sample.py",
        '''"""Contact me at person@example.com."""

# profile https://example.com/user/name
def f(x):
    # @john compute square
    return x * x
''',
    )

    _write(
        root / "docs" / "guide.md",
        "Use this:\n```bash\npython main.py --mock ingest .\n```\n",
    )


def test_sanitizer_handles_core_formats(tmp_path: Path):
    raw = tmp_path / "raw"
    clean = tmp_path / "clean"
    _build_pack(raw)

    pipeline = SanitizationPipeline(
        SanitizationConfig(enabled=True, clean_archive_dir=str(clean), langdetect_threshold=0.8)
    )
    run_dir, stats, log_path = pipeline.run(str(raw), index_name="format_smoke")

    assert Path(run_dir).exists()
    assert Path(log_path).exists()
    assert stats.entries_written > 0
    assert stats.code_blocks_kept > 0

    logs = Path(log_path).read_text(encoding="utf-8")
    assert "invalid_7z_magic" in logs
    assert "invalid_zst_magic" in logs

    # Ensure language/source buckets are produced.
    produced = list(clean.rglob("*.md"))
    assert produced, "No sanitized markdown outputs were produced"

