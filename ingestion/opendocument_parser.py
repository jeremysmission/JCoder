"""
OpenDocument Parsers (.odt, .ods, .odp)
-----------------------------------------
Extracts text from OpenDocument format files (LibreOffice/OpenOffice).

These are ZIP archives containing XML. Main content is in content.xml,
metadata in meta.xml. Uses only stdlib (zipfile + xml.etree).
"""

import logging
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Any, Dict, Tuple

log = logging.getLogger(__name__)

# OpenDocument namespace prefixes
_OD_NAMESPACES = {
    "text": "urn:oasis:names:tc:opendocument:xmlns:text:1.0",
    "table": "urn:oasis:names:tc:opendocument:xmlns:table:1.0",
    "office": "urn:oasis:names:tc:opendocument:xmlns:office:1.0",
    "draw": "urn:oasis:names:tc:opendocument:xmlns:drawing:1.0",
    "presentation": "urn:oasis:names:tc:opendocument:xmlns:presentation:1.0",
    "dc": "http://purl.org/dc/elements/1.1/",
    "meta": "urn:oasis:names:tc:opendocument:xmlns:meta:1.0",
}

# Block-level elements that get newline separation
_BLOCK_TAGS = {
    f"{{{_OD_NAMESPACES['text']}}}p",
    f"{{{_OD_NAMESPACES['text']}}}h",
    f"{{{_OD_NAMESPACES['table']}}}table-row",
    f"{{{_OD_NAMESPACES['text']}}}list-item",
    f"{{{_OD_NAMESPACES['draw']}}}frame",
}


def _walk_text(element: ET.Element) -> str:
    """Recursively extract text from an OD XML element tree."""
    parts = []
    tag = element.tag

    if element.text:
        parts.append(element.text)

    for child in element:
        parts.append(_walk_text(child))
        if child.tail:
            parts.append(child.tail)

    text = "".join(parts)
    if tag in _BLOCK_TAGS:
        text = text.strip() + "\n"
    return text


def _extract_metadata(zf: zipfile.ZipFile) -> str:
    """Extract metadata fields from meta.xml."""
    try:
        meta_xml = zf.read("meta.xml").decode("utf-8", errors="ignore")
        root = ET.fromstring(meta_xml)
    except (KeyError, ET.ParseError):
        return ""

    fields = []
    ns_dc = _OD_NAMESPACES["dc"]
    ns_meta = _OD_NAMESPACES["meta"]

    for tag, label in [
        (f"{{{ns_dc}}}title", "Title"),
        (f"{{{ns_dc}}}subject", "Subject"),
        (f"{{{ns_meta}}}keyword", "Keyword"),
        (f"{{{ns_dc}}}description", "Description"),
        (f"{{{ns_dc}}}creator", "Creator"),
    ]:
        for elem in root.iter(tag):
            if elem.text and elem.text.strip():
                fields.append(f"{label}: {elem.text.strip()}")
    return "\n".join(fields)


def _parse_opendocument(file_path: str, parser_name: str) -> Tuple[str, Dict[str, Any]]:
    """Shared parsing logic for all OpenDocument formats."""
    path = Path(file_path)
    details: Dict[str, Any] = {"file": str(path), "parser": parser_name}

    try:
        with zipfile.ZipFile(str(path), "r") as zf:
            parts = []

            # Metadata
            meta_text = _extract_metadata(zf)
            if meta_text:
                parts.append(meta_text)

            # Main content
            try:
                content_xml = zf.read("content.xml").decode("utf-8", errors="ignore")
                root = ET.fromstring(content_xml)
                body_text = _walk_text(root).strip()
                if body_text:
                    parts.append(body_text)
                details["method"] = "content_xml"
            except (KeyError, ET.ParseError) as exc:
                details["method"] = "no_content_xml"
                details["content_error"] = f"{type(exc).__name__}: {exc}"

            text = "\n\n".join(parts)
            details["total_len"] = len(text)
            return text, details

    except zipfile.BadZipFile as exc:
        details["error"] = f"BadZipFile: {exc}"
        return "", details
    except Exception as exc:
        details["error"] = f"RUNTIME_ERROR: {type(exc).__name__}: {exc}"
        return "", details


class OdtParser:
    """Parse OpenDocument Text (.odt) files."""

    def parse(self, file_path: str) -> str:
        text, _ = self.parse_with_details(file_path)
        return text

    def parse_with_details(self, file_path: str) -> Tuple[str, Dict[str, Any]]:
        return _parse_opendocument(file_path, "OdtParser")


class OdsParser:
    """Parse OpenDocument Spreadsheet (.ods) files."""

    def parse(self, file_path: str) -> str:
        text, _ = self.parse_with_details(file_path)
        return text

    def parse_with_details(self, file_path: str) -> Tuple[str, Dict[str, Any]]:
        return _parse_opendocument(file_path, "OdsParser")


class OdpParser:
    """Parse OpenDocument Presentation (.odp) files."""

    def parse(self, file_path: str) -> str:
        text, _ = self.parse_with_details(file_path)
        return text

    def parse_with_details(self, file_path: str) -> Tuple[str, Dict[str, Any]]:
        return _parse_opendocument(file_path, "OdpParser")
