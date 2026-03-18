"""Extended tests for the data ingestion pipeline.

Covers: RepoLoader file discovery/validation, MinHash dedup edge cases,
and sanitizer integration scenarios.
"""

import os
import struct
import tempfile
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ingestion.dedup import MinHashDedup, DedupStats
from ingestion.repo_loader import (
    FileValidator,
    RepoLoader,
    SKIP_DIRS,
    _BINARY_SIGS,
    DEFAULT_MAX_FILE_KB,
)
from ingestion.sanitizer import (
    SanitizationConfig,
    SanitizationPipeline,
    SanitizationStats,
    _extract_code_blocks,
    _strip_pii,
    _strip_markup,
)


# ------------------------------------------------------------------ #
#  Helpers
# ------------------------------------------------------------------ #

def _make_tree(base: Path, spec: dict):
    """Create a directory tree from a nested dict.

    Keys ending with '/' are dirs; values are bytes or str for files.
    """
    for name, content in spec.items():
        path = base / name
        if isinstance(content, dict):
            path.mkdir(parents=True, exist_ok=True)
            _make_tree(path, content)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(content, bytes):
                path.write_bytes(content)
            else:
                path.write_text(content, encoding="utf-8")


def _mock_chunker():
    """Return a Chunker mock that echoes file paths as chunks."""
    chunker = MagicMock()
    chunker.chunk_file.side_effect = lambda p: [{"file": p, "text": "chunk"}]
    return chunker


# ------------------------------------------------------------------ #
#  RepoLoader -- file discovery by extension
# ------------------------------------------------------------------ #

class TestFileDiscovery:
    def test_discovers_supported_extensions(self, tmp_path):
        _make_tree(tmp_path, {
            "main.py": "print('hi')",
            "app.js": "console.log('hi')",
            "readme.md": "# Hello",
            "notes.txt": "plain text",
            "logo.svg": "<svg/>",
        })
        chunker = _mock_chunker()
        loader = RepoLoader(chunker)
        chunks = loader.load(str(tmp_path))
        chunked_files = {c["file"] for c in chunks}
        # .py, .js, .md, .txt are in LANGUAGE_MAP
        assert any("main.py" in f for f in chunked_files)
        assert any("app.js" in f for f in chunked_files)
        # .svg is not supported
        assert not any("logo.svg" in f for f in chunked_files)

    def test_skip_dirs_pruned(self, tmp_path):
        _make_tree(tmp_path, {
            "src": {"core.py": "x = 1"},
            "node_modules": {"dep.py": "y = 2"},
            "__pycache__": {"cached.py": "z = 3"},
            ".git": {"config.py": "w = 4"},
        })
        chunker = _mock_chunker()
        loader = RepoLoader(chunker)
        chunks = loader.load(str(tmp_path))
        chunked_files = {c["file"] for c in chunks}
        assert any("core.py" in f for f in chunked_files)
        for skip in ("node_modules", "__pycache__", ".git"):
            assert not any(skip in f for f in chunked_files)

    def test_nonexistent_root_raises(self):
        chunker = _mock_chunker()
        loader = RepoLoader(chunker)
        with pytest.raises(FileNotFoundError):
            loader.load("/nonexistent/path/abc123")


# ------------------------------------------------------------------ #
#  RepoLoader -- binary file detection
# ------------------------------------------------------------------ #

class TestBinaryDetection:
    @pytest.mark.parametrize("sig", _BINARY_SIGS)
    def test_binary_signatures_rejected(self, tmp_path, sig):
        fpath = tmp_path / "binary_file.py"
        fpath.write_bytes(sig + b"\x00" * 100)
        v = FileValidator()
        assert v.is_valid(str(fpath)) is False
        assert v.skip_counts.get("binary", 0) >= 1

    def test_text_file_accepted(self, tmp_path):
        fpath = tmp_path / "good.py"
        fpath.write_text("def foo(): pass\n", encoding="utf-8")
        v = FileValidator()
        assert v.is_valid(str(fpath)) is True


# ------------------------------------------------------------------ #
#  RepoLoader -- symlink handling
# ------------------------------------------------------------------ #

class TestSymlinkHandling:
    def test_symlink_to_valid_file_processed(self, tmp_path):
        real = tmp_path / "real.py"
        real.write_text("x = 1\n", encoding="utf-8")
        link = tmp_path / "link.py"
        try:
            link.symlink_to(real)
        except OSError:
            pytest.skip("Symlinks not supported on this OS/config")
        v = FileValidator()
        assert v.is_valid(str(link)) is True

    def test_broken_symlink_rejected(self, tmp_path):
        link = tmp_path / "broken.py"
        try:
            link.symlink_to(tmp_path / "does_not_exist.py")
        except OSError:
            pytest.skip("Symlinks not supported on this OS/config")
        v = FileValidator()
        assert v.is_valid(str(link)) is False


# ------------------------------------------------------------------ #
#  RepoLoader -- max file size enforcement
# ------------------------------------------------------------------ #

class TestMaxFileSize:
    def test_oversized_file_rejected(self, tmp_path):
        fpath = tmp_path / "huge.py"
        # Write slightly over default 1 MB
        fpath.write_bytes(b"x" * (DEFAULT_MAX_FILE_KB * 1024 + 1))
        v = FileValidator()
        assert v.is_valid(str(fpath)) is False
        assert v.skip_counts["too_large"] == 1

    def test_custom_max_size(self, tmp_path):
        fpath = tmp_path / "medium.py"
        fpath.write_bytes(b"x" * 600)
        # Allow only 0.5 KB
        v = FileValidator(max_file_kb=1)
        assert v.is_valid(str(fpath)) is True
        # Now 2 KB content with 1 KB limit
        fpath.write_bytes(b"x" * 2048)
        assert v.is_valid(str(fpath)) is False

    def test_empty_file_rejected(self, tmp_path):
        fpath = tmp_path / "empty.py"
        fpath.write_bytes(b"")
        v = FileValidator()
        assert v.is_valid(str(fpath)) is False
        assert v.skip_counts["empty"] == 1


# ------------------------------------------------------------------ #
#  MinHash Dedup -- identical content
# ------------------------------------------------------------------ #

class TestDedupIdentical:
    def test_exact_duplicate_detected(self):
        d = MinHashDedup()
        text = "def hello_world():\n    print('hello world')\n"
        assert d.is_duplicate(text) is False
        assert d.is_duplicate(text) is True
        s = d.stats()
        assert s.exact_dupes == 1
        assert s.unique == 1

    def test_exact_duplicate_with_explicit_ids(self):
        d = MinHashDedup()
        text = "import os\nimport sys\n"
        assert d.is_duplicate(text, doc_id="doc1") is False
        assert d.is_duplicate(text, doc_id="doc2") is True


# ------------------------------------------------------------------ #
#  MinHash Dedup -- similar content above threshold
# ------------------------------------------------------------------ #

class TestDedupSimilar:
    def test_near_duplicate_flagged(self):
        d = MinHashDedup(threshold=0.5)
        base = "the quick brown fox jumps over the lazy dog " * 5
        tweaked = base.replace("fox", "cat")
        assert d.is_duplicate(base) is False
        assert d.is_duplicate(tweaked) is True
        assert d.stats().near_dupes >= 1

    def test_appended_whitespace_still_duplicate(self):
        d = MinHashDedup(threshold=0.5)
        text = "a long enough sentence for shingling to work properly here"
        assert d.is_duplicate(text) is False
        assert d.is_duplicate(text + "   ") is True


# ------------------------------------------------------------------ #
#  MinHash Dedup -- distinct content passes
# ------------------------------------------------------------------ #

class TestDedupDistinct:
    def test_completely_different_content_passes(self):
        d = MinHashDedup()
        assert d.is_duplicate("alpha bravo charlie delta echo foxtrot") is False
        assert d.is_duplicate("one two three four five six seven eight") is False
        assert d.stats().unique == 2
        assert d.stats().exact_dupes == 0
        assert d.stats().near_dupes == 0


# ------------------------------------------------------------------ #
#  MinHash Dedup -- empty document handling
# ------------------------------------------------------------------ #

class TestDedupEmpty:
    def test_empty_string_not_duplicate_of_itself(self):
        d = MinHashDedup()
        # First empty -> unique, second empty -> exact dupe (SHA match)
        assert d.is_duplicate("") is False
        assert d.is_duplicate("") is True

    def test_empty_shingles_produce_max_hash_signature(self):
        d = MinHashDedup()
        sig = d._minhash(set())
        assert (sig == np.full(d.num_perm, (1 << 32) - 1, dtype=np.uint32)).all()


# ------------------------------------------------------------------ #
#  MinHash Dedup -- shingle size configuration
# ------------------------------------------------------------------ #

class TestDedupShingleConfig:
    def test_default_shingle_size_5(self):
        shingles = MinHashDedup._shingle("abcdefgh")
        # "abcde", "bcdef", "cdefg", "defgh" = 4 shingles
        assert len(shingles) == 4

    def test_custom_shingle_size(self):
        shingles = MinHashDedup._shingle("abcdefgh", k=3)
        # "abc", "bcd", "cde", "def", "efg", "fgh" = 6
        assert len(shingles) == 6

    def test_short_text_below_shingle_size(self):
        shingles = MinHashDedup._shingle("ab", k=5)
        # Text shorter than k -> single shingle of the whole text
        assert shingles == {"ab"}

    def test_empty_text_no_shingles(self):
        shingles = MinHashDedup._shingle("")
        assert shingles == set()

    def test_shingle_lowercases(self):
        s1 = MinHashDedup._shingle("ABCDE")
        s2 = MinHashDedup._shingle("abcde")
        assert s1 == s2


# ------------------------------------------------------------------ #
#  Sanitizer Integration -- full pipeline mock
# ------------------------------------------------------------------ #

class TestSanitizerIntegration:
    def test_full_pipeline_load_sanitize(self, tmp_path):
        """Simulate: load files -> sanitize -> verify output structure."""
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        code_file = raw_dir / "example.py"
        code_file.write_text(textwrap.dedent("""\
            # This module handles data parsing
            def parse(data):
                \"\"\"Parse incoming data stream.\"\"\"
                return data.split(',')
        """), encoding="utf-8")

        cfg = SanitizationConfig(
            enabled=True,
            clean_archive_dir=str(tmp_path / "clean"),
        )
        pipeline = SanitizationPipeline(cfg)
        run_dir, stats, log_path = pipeline.run(str(raw_dir), "test_index")

        assert stats.files_seen >= 1
        assert Path(run_dir).exists()
        assert Path(log_path).exists()

    def test_sanitizer_preserves_code_structure(self, tmp_path):
        """Code blocks in markdown survive sanitization."""
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        md_file = raw_dir / "guide.md"
        md_file.write_text(textwrap.dedent("""\
            # Setup Guide

            Install with:

            ```python
            import jcoder
            jcoder.init()
            ```

            Then run the tests.
        """), encoding="utf-8")

        cfg = SanitizationConfig(
            enabled=True,
            clean_archive_dir=str(tmp_path / "clean"),
        )
        pipeline = SanitizationPipeline(cfg)
        run_dir, stats, _ = pipeline.run(str(raw_dir), "test_idx")

        # Should have extracted the code block
        assert stats.code_blocks_kept >= 1
        assert stats.entries_written >= 1

    def test_pii_stripped_from_explanation(self):
        stats = SanitizationStats()
        text = "Contact admin@example.com or visit https://secret.io for details"
        cleaned = _strip_pii(text, stats)
        assert "admin@example.com" not in cleaned
        assert "https://secret.io" not in cleaned
        assert stats.pii_replacements >= 2

    def test_large_file_streaming_no_crash(self, tmp_path):
        """A large-ish code file should not crash the sanitizer."""
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        big_file = raw_dir / "big.py"
        # 50K of comments -- enough to exercise the pipeline
        lines = ["# Comment line number {}\n".format(i) for i in range(2000)]
        lines.append("def main(): pass\n")
        big_file.write_text("".join(lines), encoding="utf-8")

        cfg = SanitizationConfig(
            enabled=True,
            clean_archive_dir=str(tmp_path / "clean"),
        )
        pipeline = SanitizationPipeline(cfg)
        run_dir, stats, _ = pipeline.run(str(raw_dir), "big_test")
        assert stats.files_seen == 1

    def test_extract_code_blocks_backtick_and_inline(self):
        text = textwrap.dedent("""\
            Here is code:
            ```python
            x = 1
            ```
            And inline: `some_func()` is useful.
        """)
        blocks = _extract_code_blocks(text)
        langs = [b[0] for b in blocks]
        codes = [b[1] for b in blocks]
        assert "python" in langs
        assert "x = 1" in codes[0]

    def test_strip_markup_removes_html_and_markdown(self):
        text = "<b>bold</b> and [link](http://x.com) and **strong**"
        cleaned = _strip_markup(text)
        assert "<b>" not in cleaned
        assert "http://x.com" not in cleaned
        assert "link" in cleaned

    def test_nonexistent_raw_root_raises(self, tmp_path):
        cfg = SanitizationConfig(
            enabled=True,
            clean_archive_dir=str(tmp_path / "clean"),
        )
        pipeline = SanitizationPipeline(cfg)
        with pytest.raises(FileNotFoundError):
            pipeline.run("/does/not/exist/xyz", "test")
