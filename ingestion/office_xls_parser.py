"""
Legacy Excel (.xls) Parser
---------------------------
Extracts text from Excel 97-2003 binary (.xls) files.

Strategy cascade:
  1. xlrd (best quality -- structured cell extraction)
  2. olefile (OLE2 metadata + binary stream text)
  3. Raw binary scan (last resort -- text runs from bytes)
"""

import logging
from pathlib import Path
from typing import Any, Dict, Tuple

log = logging.getLogger(__name__)


class XlsParser:
    """Parse legacy .xls files into plain text."""

    def parse(self, file_path: str) -> str:
        text, _ = self.parse_with_details(file_path)
        return text

    def parse_with_details(self, file_path: str) -> Tuple[str, Dict[str, Any]]:
        path = Path(file_path)
        details: Dict[str, Any] = {"file": str(path), "parser": "XlsParser"}

        # Strategy 1: xlrd
        try:
            import xlrd
            details["xlrd_available"] = True
            text = self._parse_xlrd(path, details)
            if text.strip():
                details["method"] = "xlrd"
                details["total_len"] = len(text)
                return text, details
        except ImportError:
            details["xlrd_available"] = False
        except Exception as exc:
            details["xlrd_error"] = f"{type(exc).__name__}: {exc}"

        # Strategy 2: olefile
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

        # Strategy 3: raw binary text extraction
        try:
            text = self._parse_raw_binary(path)
            details["method"] = "raw_binary"
            details["total_len"] = len(text)
            return text, details
        except Exception as exc:
            details["error"] = f"RUNTIME_ERROR: {type(exc).__name__}: {exc}"
            return "", details

    @staticmethod
    def _parse_xlrd(path: Path, details: Dict) -> str:
        import xlrd
        wb = xlrd.open_workbook(str(path), on_demand=True)
        details["sheet_count"] = wb.nsheets
        parts = []
        for idx in range(wb.nsheets):
            sheet = wb.sheet_by_index(idx)
            parts.append(f"[Sheet: {sheet.name}]")
            for row_idx in range(sheet.nrows):
                cells = [str(sheet.cell_value(row_idx, col)) for col in range(sheet.ncols)]
                row_text = "\t".join(cells).strip()
                if row_text:
                    parts.append(row_text)
        return "\n".join(parts)

    @staticmethod
    def _parse_olefile(path: Path, details: Dict) -> str:
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
        # Extract text from streams
        for stream_name in ole.listdir():
            try:
                data = ole.openstream(stream_name).read()
                text = data.decode("utf-8", errors="ignore")
                clean = "".join(ch for ch in text if ch.isprintable() or ch in "\n\t")
                if len(clean.strip()) > 20:
                    parts.append(clean.strip())
            except Exception:
                continue
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
