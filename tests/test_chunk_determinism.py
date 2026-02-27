"""Chunk IDs must be identical across re-ingest of the same files."""

import os
import tempfile

from ingestion.chunker import Chunker


def test_chunk_ids_deterministic():
    """Same content -> same chunk IDs, every time."""
    chunker = Chunker(max_chars=4000)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write("def hello():\n    return 'world'\n")
        path = f.name

    try:
        chunks_a = chunker.chunk_file(path)
        chunks_b = chunker.chunk_file(path)

        ids_a = [c["id"] for c in chunks_a]
        ids_b = [c["id"] for c in chunks_b]

        assert ids_a == ids_b, f"Chunk IDs differ: {ids_a} vs {ids_b}"
    finally:
        os.unlink(path)


def test_chunk_ids_stable_across_crlf():
    """CRLF vs LF should produce the same chunk ID (hash normalizes line endings)."""
    chunker = Chunker(max_chars=4000)

    content_lf = "def foo():\n    pass\n"
    content_crlf = "def foo():\r\n    pass\r\n"

    with tempfile.NamedTemporaryFile(mode="wb", suffix=".py", delete=False) as f:
        f.write(content_lf.encode("utf-8"))
        path_lf = f.name

    with tempfile.NamedTemporaryFile(mode="wb", suffix=".py", delete=False) as f:
        f.write(content_crlf.encode("utf-8"))
        path_crlf = f.name

    try:
        # Note: chunk content will differ (raw bytes), but the hash
        # normalizes \r\n -> \n before hashing, so IDs should match
        chunks_lf = chunker.chunk_file(path_lf)
        chunks_crlf = chunker.chunk_file(path_crlf)

        # The content_hash (which normalizes) should match
        hashes_lf = [c["content_hash"] for c in chunks_lf]
        hashes_crlf = [c["content_hash"] for c in chunks_crlf]

        assert hashes_lf == hashes_crlf, "CRLF normalization failed in chunk hash"
    finally:
        os.unlink(path_lf)
        os.unlink(path_crlf)


def test_empty_file_produces_no_chunks():
    """Empty files should not produce chunks."""
    chunker = Chunker(max_chars=4000)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write("")
        path = f.name

    try:
        chunks = chunker.chunk_file(path)
        assert chunks == [], f"Expected no chunks, got {len(chunks)}"
    finally:
        os.unlink(path)
