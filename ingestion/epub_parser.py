"""
EPUB Parser (.epub)
---------------------
Extracts text from EPUB eBook files.

EPUBs are ZIP archives containing XHTML content files. The reading
order is defined by the OPF package file's spine element.

Uses only stdlib: zipfile, xml.etree, html.parser.
"""

import logging
import xml.etree.ElementTree as ET
import zipfile
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Tuple

log = logging.getLogger(__name__)


class _HTMLTextExtractor(HTMLParser):
    """Strip HTML tags and extract plain text."""

    _SKIP_TAGS = {"script", "style", "head"}
    _BREAK_TAGS = {"p", "div", "br", "h1", "h2", "h3", "h4", "h5", "h6",
                   "li", "tr", "dt", "dd", "blockquote", "section", "article"}

    def __init__(self):
        super().__init__()
        self._parts: List[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in self._SKIP_TAGS:
            self._skip_depth += 1
        elif tag in self._BREAK_TAGS and self._skip_depth == 0:
            self._parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self._SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)

    def handle_data(self, data):
        if self._skip_depth == 0:
            self._parts.append(data)

    def get_text(self) -> str:
        return "".join(self._parts)


def _strip_html(html_content: str) -> str:
    """Convert HTML to plain text."""
    extractor = _HTMLTextExtractor()
    try:
        extractor.feed(html_content)
    except Exception:
        pass
    return extractor.get_text()


class EpubParser:
    """Parse EPUB eBook files into plain text."""

    def parse(self, file_path: str) -> str:
        text, _ = self.parse_with_details(file_path)
        return text

    def parse_with_details(self, file_path: str) -> Tuple[str, Dict[str, Any]]:
        path = Path(file_path)
        details: Dict[str, Any] = {"file": str(path), "parser": "EpubParser"}

        try:
            with zipfile.ZipFile(str(path), "r") as zf:
                parts = []

                # Find OPF file
                opf_path = self._find_opf(zf)
                if not opf_path:
                    # Fallback: glob for HTML files
                    text = self._fallback_glob(zf, details)
                    details["total_len"] = len(text)
                    return text, details

                # Parse spine order
                content_files = self._parse_spine(zf, opf_path, details)

                # Extract metadata from OPF
                meta = self._extract_opf_metadata(zf, opf_path)
                if meta:
                    parts.append(meta)

                # Read content in spine order
                for cf in content_files:
                    try:
                        html_bytes = zf.read(cf)
                        html_text = html_bytes.decode("utf-8", errors="ignore")
                        plain = _strip_html(html_text).strip()
                        if plain:
                            parts.append(plain)
                    except (KeyError, Exception):
                        continue

                details["content_files"] = len(content_files)
                details["method"] = "opf_spine"
                text = "\n\n".join(parts)
                details["total_len"] = len(text)
                return text, details

        except zipfile.BadZipFile as exc:
            details["error"] = f"BadZipFile: {exc}"
            return "", details
        except Exception as exc:
            details["error"] = f"RUNTIME_ERROR: {type(exc).__name__}: {exc}"
            return "", details

    @staticmethod
    def _find_opf(zf: zipfile.ZipFile) -> str:
        """Locate the OPF package file via container.xml or by searching."""
        try:
            container = zf.read("META-INF/container.xml").decode("utf-8", errors="ignore")
            root = ET.fromstring(container)
            ns = {"c": "urn:oasis:names:tc:opendocument:xmlns:container"}
            for rf in root.iter():
                if rf.tag.endswith("rootfile"):
                    fp = rf.get("full-path", "")
                    if fp:
                        return fp
        except (KeyError, ET.ParseError):
            pass
        # Fallback: search for .opf files
        for name in zf.namelist():
            if name.endswith(".opf"):
                return name
        return ""

    @staticmethod
    def _parse_spine(zf: zipfile.ZipFile, opf_path: str, details: Dict) -> List[str]:
        """Parse OPF spine to get content files in reading order."""
        opf_dir = str(PurePosixPath(opf_path).parent)
        opf_xml = zf.read(opf_path).decode("utf-8", errors="ignore")
        root = ET.fromstring(opf_xml)

        # Build manifest: id -> href
        manifest = {}
        for item in root.iter():
            if item.tag.endswith("}item") or item.tag == "item":
                item_id = item.get("id", "")
                href = item.get("href", "")
                media = item.get("media-type", "")
                if item_id and href:
                    manifest[item_id] = (href, media)

        # Walk spine
        content_files = []
        for itemref in root.iter():
            if itemref.tag.endswith("}itemref") or itemref.tag == "itemref":
                idref = itemref.get("idref", "")
                if idref in manifest:
                    href, media = manifest[idref]
                    if "html" in media or href.endswith((".html", ".xhtml", ".htm")):
                        full = f"{opf_dir}/{href}" if opf_dir and opf_dir != "." else href
                        content_files.append(full)
        return content_files

    @staticmethod
    def _extract_opf_metadata(zf: zipfile.ZipFile, opf_path: str) -> str:
        """Extract title, author, language from OPF."""
        try:
            opf_xml = zf.read(opf_path).decode("utf-8", errors="ignore")
            root = ET.fromstring(opf_xml)
            fields = []
            for elem in root.iter():
                tag = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
                if tag == "title" and elem.text:
                    fields.append(f"Title: {elem.text.strip()}")
                elif tag == "creator" and elem.text:
                    fields.append(f"Author: {elem.text.strip()}")
                elif tag == "language" and elem.text:
                    fields.append(f"Language: {elem.text.strip()}")
            return "\n".join(fields)
        except Exception:
            return ""

    @staticmethod
    def _fallback_glob(zf: zipfile.ZipFile, details: Dict) -> str:
        """Fallback: read all HTML files in the archive."""
        details["method"] = "fallback_glob"
        parts = []
        for name in sorted(zf.namelist()):
            if name.endswith((".html", ".xhtml", ".htm")):
                try:
                    html = zf.read(name).decode("utf-8", errors="ignore")
                    plain = _strip_html(html).strip()
                    if plain:
                        parts.append(plain)
                except Exception:
                    continue
        return "\n\n".join(parts)
