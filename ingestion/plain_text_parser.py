"""
Plain Text Parser
-------------------
Handles text-based formats that need no special parsing:
  .rst, .csv, .tsv, .svg, .drawio, .dia, .log, .ini, .cfg, .conf

Simply reads the file as UTF-8 text with error-ignoring encoding.
"""

from pathlib import Path
from typing import Any, Dict, Tuple


class PlainTextParser:
    """Read any text-based file as-is."""

    def parse(self, file_path: str) -> str:
        text, _ = self.parse_with_details(file_path)
        return text

    def parse_with_details(self, file_path: str) -> Tuple[str, Dict[str, Any]]:
        path = Path(file_path)
        details: Dict[str, Any] = {"file": str(path), "parser": "PlainTextParser"}
        try:
            data = path.read_text(encoding="utf-8", errors="ignore")
            details["total_len"] = len(data)
            return data, details
        except Exception as exc:
            details["error"] = f"RUNTIME_ERROR: {type(exc).__name__}: {exc}"
            return "", details
