"""
Legacy PowerPoint (.ppt) Parser
---------------------------------
Extracts text from PowerPoint 97-2003 binary (.ppt) files.

Strategy cascade:
  1. olefile (OLE2 container -- parse PPT record stream)
  2. Raw binary scan (last resort -- text runs from bytes)

PPT binary format stores text in two record types:
  - 0x0FA0 = TextCharsAtom (UTF-16LE encoded)
  - 0x0FA8 = TextBytesAtom (ASCII/Latin-1 encoded)
"""

import logging
import struct
from pathlib import Path
from typing import Any, Dict, List, Tuple

log = logging.getLogger(__name__)

_TEXT_CHARS_ATOM = 0x0FA0
_TEXT_BYTES_ATOM = 0x0FA8


class PptParser:
    """Parse legacy .ppt files into plain text."""

    def parse(self, file_path: str) -> str:
        text, _ = self.parse_with_details(file_path)
        return text

    def parse_with_details(self, file_path: str) -> Tuple[str, Dict[str, Any]]:
        path = Path(file_path)
        details: Dict[str, Any] = {"file": str(path), "parser": "PptParser"}

        # Strategy 1: olefile with PPT record parsing
        try:
            import olefile
            details["olefile_available"] = True
            text = self._parse_olefile(path, details)
            if text.strip():
                details["method"] = "olefile"
                details["total_len"] = len(text)
                return text, details
        except ImportError:
            details["olefile_available"] = False
        except Exception as exc:
            details["olefile_error"] = f"{type(exc).__name__}: {exc}"

        # Strategy 2: raw binary text extraction
        try:
            text = self._parse_raw_binary(path)
            details["method"] = "raw_binary"
            details["total_len"] = len(text)
            return text, details
        except Exception as exc:
            details["error"] = f"RUNTIME_ERROR: {type(exc).__name__}: {exc}"
            return "", details

    @staticmethod
    def _extract_ppt_text_records(data: bytes) -> List[str]:
        """Walk the PPT binary record stream and extract text atoms."""
        texts = []
        pos = 0
        while pos + 8 <= len(data):
            try:
                ver_inst, rec_type, rec_len = struct.unpack_from("<HHI", data, pos)
            except struct.error:
                break
            pos += 8
            if pos + rec_len > len(data):
                break
            if rec_type == _TEXT_CHARS_ATOM:
                try:
                    text = data[pos:pos + rec_len].decode("utf-16-le", errors="ignore")
                    text = text.strip()
                    if text:
                        texts.append(text)
                except Exception:
                    pass
            elif rec_type == _TEXT_BYTES_ATOM:
                try:
                    text = data[pos:pos + rec_len].decode("latin-1", errors="ignore")
                    text = text.strip()
                    if text:
                        texts.append(text)
                except Exception:
                    pass
            pos += rec_len
        return texts

    @classmethod
    def _parse_olefile(cls, path: Path, details: Dict) -> str:
        import olefile
        ole = olefile.OleFileIO(str(path))
        parts = []

        # Extract metadata
        meta = ole.get_metadata()
        for field in ("title", "subject", "author", "keywords", "comments"):
            val = getattr(meta, field, None)
            if val:
                decoded = val.decode("utf-8", errors="ignore") if isinstance(val, bytes) else str(val)
                if decoded.strip():
                    parts.append(f"{field.title()}: {decoded.strip()}")

        # Extract text from PowerPoint Document stream
        ppt_stream = "PowerPoint Document"
        if ole.exists(ppt_stream):
            data = ole.openstream(ppt_stream).read()
            texts = cls._extract_ppt_text_records(data)
            details["text_records_found"] = len(texts)
            parts.extend(texts)

        ole.close()
        return "\n\n".join(parts)

    @staticmethod
    def _parse_raw_binary(path: Path) -> str:
        data = path.read_bytes()
        runs = []
        current = []
        for byte in data:
            ch = chr(byte) if 32 <= byte < 127 or byte in (10, 13, 9) else None
            if ch:
                current.append(ch)
            else:
                if len(current) >= 8:
                    runs.append("".join(current).strip())
                current = []
        if len(current) >= 8:
            runs.append("".join(current).strip())
        return "\n".join(r for r in runs if r)
