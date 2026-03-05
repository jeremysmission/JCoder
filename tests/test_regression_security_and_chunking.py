"""Regression tests for security guards and Unicode-safe chunking."""

from __future__ import annotations

import zipfile
from types import SimpleNamespace

import pytest

from core.telemetry import TelemetryStore
from ingestion.chunker import Chunker
from ingestion.sanitizer import SanitizationConfig, SanitizationPipeline, SanitizationStats


def test_telemetry_rejects_invalid_operator(tmp_path):
    store = TelemetryStore(str(tmp_path / "telemetry.db"))
    with pytest.raises(ValueError):
        store._query_by_confidence("!=", 0.5, 5)


def test_zip_safe_extract_blocks_path_traversal(tmp_path):
    pipeline = SanitizationPipeline(SanitizationConfig(clean_archive_dir=str(tmp_path / "clean")))
    stats = SanitizationStats()
    archive = tmp_path / "input.zip"
    extract_root = tmp_path / "extract"
    extract_root.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../../escape.txt", "evil")
        zf.writestr("safe/posts.xml", "<root/>")

    with zipfile.ZipFile(archive, "r") as zf:
        pipeline._extract_zip_members_safe(
            zf=zf,
            names=zf.namelist(),
            dest_dir=extract_root,
            stats=stats,
            archive_label=str(archive),
        )

    assert not (tmp_path / "escape.txt").exists()
    assert (extract_root / "safe" / "posts.xml").exists()
    assert any("zip_path_traversal_blocked" in e for e in stats.errors)


def test_chunk_by_ast_uses_byte_offsets_safely_with_unicode():
    # Multi-byte unicode in preamble previously caused byte-index slicing bugs.
    content = "éé preamble\n\ndef foo():\n    return 1\n"
    start_char = content.index("def foo")
    start_byte = len(content[:start_char].encode("utf-8"))
    end_byte = len(content.encode("utf-8"))

    child = SimpleNamespace(type="function_definition", start_byte=start_byte, end_byte=end_byte)
    root = SimpleNamespace(children=[child])
    tree = SimpleNamespace(root_node=root)
    parser = SimpleNamespace(parse=lambda _b: tree)

    chunker = Chunker(max_chars=4000)
    chunker._parser_cache["python"] = (object(), parser)

    chunks = chunker._chunk_by_ast(content, "sample.py", "python")
    texts = [c["content"] for c in chunks]

    assert any("éé preamble" in t for t in texts)
    assert any("def foo():" in t for t in texts)
