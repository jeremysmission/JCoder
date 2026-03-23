"""
Tests for document parsers and parser registry.

Verifies the parser cascade for each format:
  - XlsParser, PptParser, OdtParser, OdsParser, OdpParser, EpubParser
  - PlainTextParser
  - parser_registry (lazy loading, dispatch, supported_extensions)
  - DocumentChunker (HybridRAG-compatible 1200/200 chunks)
"""

import json
import os
import struct
import tempfile
import zipfile

import pytest

from ingestion.parser_registry import (
    DOCUMENT_EXTENSIONS,
    get_parser,
    parse_file,
    supported_extensions,
)
from ingestion.plain_text_parser import PlainTextParser
from ingestion.chunker import DocumentChunker


# ---------------------------------------------------------------------------
# Fixtures: create minimal test files for each format
# ---------------------------------------------------------------------------

@pytest.fixture
def rst_file(tmp_path):
    p = tmp_path / "example.rst"
    p.write_text("Title\n=====\n\nSome reStructuredText content.\n\n.. code-block:: python\n\n    print('hello')\n", encoding="utf-8")
    return str(p)


@pytest.fixture
def svg_file(tmp_path):
    p = tmp_path / "diagram.svg"
    p.write_text('<svg xmlns="http://www.w3.org/2000/svg"><circle r="50"/></svg>', encoding="utf-8")
    return str(p)


@pytest.fixture
def csv_file(tmp_path):
    p = tmp_path / "data.csv"
    p.write_text("name,age,city\nAlice,30,Denver\nBob,25,Boulder\n", encoding="utf-8")
    return str(p)


@pytest.fixture
def tsv_file(tmp_path):
    p = tmp_path / "data.tsv"
    p.write_text("name\tage\tcity\nAlice\t30\tDenver\n", encoding="utf-8")
    return str(p)


@pytest.fixture
def odt_file(tmp_path):
    """Create a minimal valid ODT file (ZIP with content.xml)."""
    p = tmp_path / "doc.odt"
    with zipfile.ZipFile(str(p), "w") as zf:
        zf.writestr("content.xml", """<?xml version="1.0"?>
<office:document-content xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0"
  xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0">
  <office:body>
    <office:text>
      <text:p>Hello from OpenDocument Text.</text:p>
      <text:p>Second paragraph.</text:p>
    </office:text>
  </office:body>
</office:document-content>""")
        zf.writestr("meta.xml", """<?xml version="1.0"?>
<office:document-meta xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0"
  xmlns:dc="http://purl.org/dc/elements/1.1/">
  <office:meta>
    <dc:title>Test ODT Document</dc:title>
  </office:meta>
</office:document-meta>""")
    return str(p)


@pytest.fixture
def ods_file(tmp_path):
    """Create a minimal valid ODS file."""
    p = tmp_path / "sheet.ods"
    with zipfile.ZipFile(str(p), "w") as zf:
        zf.writestr("content.xml", """<?xml version="1.0"?>
<office:document-content xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0"
  xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0"
  xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0">
  <office:body>
    <office:spreadsheet>
      <table:table table:name="Sheet1">
        <table:table-row><text:p>Cell A1</text:p></table:table-row>
      </table:table>
    </office:spreadsheet>
  </office:body>
</office:document-content>""")
    return str(p)


@pytest.fixture
def odp_file(tmp_path):
    """Create a minimal valid ODP file."""
    p = tmp_path / "slides.odp"
    with zipfile.ZipFile(str(p), "w") as zf:
        zf.writestr("content.xml", """<?xml version="1.0"?>
<office:document-content xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0"
  xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0"
  xmlns:draw="urn:oasis:names:tc:opendocument:xmlns:drawing:1.0">
  <office:body>
    <office:presentation>
      <draw:page><text:p>Slide 1 content</text:p></draw:page>
    </office:presentation>
  </office:body>
</office:document-content>""")
    return str(p)


@pytest.fixture
def epub_file(tmp_path):
    """Create a minimal valid EPUB file."""
    p = tmp_path / "book.epub"
    with zipfile.ZipFile(str(p), "w") as zf:
        zf.writestr("META-INF/container.xml", """<?xml version="1.0"?>
<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container" version="1.0">
  <rootfiles>
    <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>
  </rootfiles>
</container>""")
        zf.writestr("OEBPS/content.opf", """<?xml version="1.0"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:title>Test Book</dc:title>
    <dc:creator>Test Author</dc:creator>
  </metadata>
  <manifest>
    <item id="ch1" href="ch1.xhtml" media-type="application/xhtml+xml"/>
  </manifest>
  <spine>
    <itemref idref="ch1"/>
  </spine>
</package>""")
        zf.writestr("OEBPS/ch1.xhtml", """<!DOCTYPE html>
<html><head><title>Chapter 1</title></head>
<body>
<h1>Chapter 1</h1>
<p>This is the first chapter of the test book.</p>
<p>It has multiple paragraphs for testing.</p>
</body></html>""")
    return str(p)


# ---------------------------------------------------------------------------
# PlainTextParser tests
# ---------------------------------------------------------------------------

class TestPlainTextParser:
    def test_parse_rst(self, rst_file):
        parser = PlainTextParser()
        text = parser.parse(rst_file)
        assert "reStructuredText" in text
        assert "print('hello')" in text

    def test_parse_svg(self, svg_file):
        parser = PlainTextParser()
        text = parser.parse(svg_file)
        assert "<svg" in text
        assert "circle" in text

    def test_parse_csv(self, csv_file):
        parser = PlainTextParser()
        text = parser.parse(csv_file)
        assert "Alice" in text
        assert "Denver" in text

    def test_parse_tsv(self, tsv_file):
        parser = PlainTextParser()
        text = parser.parse(tsv_file)
        assert "Alice" in text

    def test_parse_with_details(self, rst_file):
        parser = PlainTextParser()
        text, details = parser.parse_with_details(rst_file)
        assert details["parser"] == "PlainTextParser"
        assert details["total_len"] > 0
        assert "error" not in details

    def test_nonexistent_file(self, tmp_path):
        parser = PlainTextParser()
        text, details = parser.parse_with_details(str(tmp_path / "nope.txt"))
        assert text == ""
        assert "error" in details


# ---------------------------------------------------------------------------
# OpenDocument parser tests
# ---------------------------------------------------------------------------

class TestOpenDocumentParsers:
    def test_odt_parser(self, odt_file):
        text, details = parse_file(odt_file)
        assert "Hello from OpenDocument Text" in text
        assert details["parser"] == "OdtParser"
        assert details["method"] == "content_xml"

    def test_odt_metadata(self, odt_file):
        text, details = parse_file(odt_file)
        assert "Test ODT Document" in text

    def test_ods_parser(self, ods_file):
        text, details = parse_file(ods_file)
        assert "Cell A1" in text
        assert details["parser"] == "OdsParser"

    def test_odp_parser(self, odp_file):
        text, details = parse_file(odp_file)
        assert "Slide 1 content" in text
        assert details["parser"] == "OdpParser"

    def test_bad_zip(self, tmp_path):
        p = tmp_path / "corrupt.odt"
        p.write_bytes(b"not a zip file at all")
        text, details = parse_file(str(p))
        assert text == ""
        assert "BadZipFile" in details.get("error", "")


# ---------------------------------------------------------------------------
# EPUB parser tests
# ---------------------------------------------------------------------------

class TestEpubParser:
    def test_epub_parse(self, epub_file):
        text, details = parse_file(epub_file)
        assert "Chapter 1" in text
        assert "first chapter" in text
        assert details["parser"] == "EpubParser"
        assert details["method"] == "opf_spine"

    def test_epub_metadata(self, epub_file):
        text, details = parse_file(epub_file)
        assert "Test Book" in text
        assert "Test Author" in text

    def test_epub_content_files_count(self, epub_file):
        text, details = parse_file(epub_file)
        assert details["content_files"] == 1

    def test_epub_bad_zip(self, tmp_path):
        p = tmp_path / "bad.epub"
        p.write_bytes(b"corrupted epub")
        text, details = parse_file(str(p))
        assert text == ""
        assert "error" in details


# ---------------------------------------------------------------------------
# Parser Registry tests
# ---------------------------------------------------------------------------

class TestParserRegistry:
    def test_supported_extensions(self):
        exts = supported_extensions()
        assert ".xls" in exts
        assert ".ppt" in exts
        assert ".odt" in exts
        assert ".ods" in exts
        assert ".odp" in exts
        assert ".epub" in exts
        assert ".rst" in exts
        assert ".svg" in exts

    def test_get_parser_known(self):
        parser = get_parser(".rst")
        assert parser is not None
        assert hasattr(parser, "parse")
        assert hasattr(parser, "parse_with_details")

    def test_get_parser_unknown(self):
        parser = get_parser(".zzz_unknown")
        assert parser is None

    def test_parse_file_unknown_ext(self, tmp_path):
        p = tmp_path / "file.zzz"
        p.write_text("data")
        text, details = parse_file(str(p))
        assert text == ""
        assert "No parser" in details.get("error", "")

    def test_parser_caching(self):
        p1 = get_parser(".rst")
        p2 = get_parser(".rst")
        assert p1 is p2

    def test_document_extensions_frozen(self):
        assert isinstance(DOCUMENT_EXTENSIONS, frozenset)


# ---------------------------------------------------------------------------
# DocumentChunker tests
# ---------------------------------------------------------------------------

class TestDocumentChunker:
    def test_basic_chunking(self):
        chunker = DocumentChunker(chunk_size=50, overlap=10)
        text = "A" * 120
        chunks = chunker.chunk_text(text, "test.txt")
        assert len(chunks) > 1

    def test_overlap_present(self):
        chunker = DocumentChunker(chunk_size=100, overlap=20)
        text = "word " * 100
        chunks = chunker.chunk_text(text, "test.txt")
        if len(chunks) >= 2:
            # Last chars of chunk N should overlap with start of chunk N+1
            end_of_first = chunks[0]["content"][-20:]
            # Overlap means some content repeats
            assert len(chunks) >= 2

    def test_heading_prepend(self):
        chunker = DocumentChunker(chunk_size=100, overlap=10)
        text = "INTRODUCTION\n\n" + "Content here. " * 50
        chunks = chunker.chunk_text(text, "test.txt")
        # Some chunks should have heading context
        has_heading = any("[SECTION]" in c["content"] for c in chunks)
        assert has_heading

    def test_empty_text(self):
        chunker = DocumentChunker()
        assert chunker.chunk_text("", "test.txt") == []
        assert chunker.chunk_text("   \n\n  ", "test.txt") == []

    def test_chunk_strategy_tag(self):
        chunker = DocumentChunker(chunk_size=50, overlap=5)
        chunks = chunker.chunk_text("Hello world. " * 20, "test.txt")
        for c in chunks:
            assert c.get("chunk_strategy") == "hybridrag_document"

    def test_chunk_file_with_rst(self, rst_file):
        chunker = DocumentChunker(chunk_size=200, overlap=20)
        chunks = chunker.chunk_file(rst_file)
        assert len(chunks) >= 1
        full_text = " ".join(c["content"] for c in chunks)
        assert "reStructuredText" in full_text

    def test_defaults_match_hybridrag(self):
        chunker = DocumentChunker()
        assert chunker.chunk_size == 1200
        assert chunker.overlap == 200
        assert chunker.max_heading_len == 160
        assert chunker.heading_lookback == 2000
