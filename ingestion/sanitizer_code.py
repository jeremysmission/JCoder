"""Code comment extraction for the sanitization pipeline.

Extracted from sanitizer.py to keep module under 500 LOC limit.
Contains: Python docstring/comment extraction and generic comment parsing.
"""
from __future__ import annotations

import ast
import re
import tokenize
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, List, Tuple

if TYPE_CHECKING:
    from ingestion.sanitizer import SanitizationStats


class CodeCommentMixin:
    """Mixin providing code comment extraction for SanitizationPipeline."""

    def _process_code_comments(self, fp: Path, run_dir: Path, stats: SanitizationStats) -> None:
        from ingestion.sanitizer import (
            CODE_EXT_TO_LANG, _strip_markup, _strip_pii, _is_english_or_unknown,
            detect_langs,
        )

        ext = fp.suffix.lower()
        lang = CODE_EXT_TO_LANG.get(ext, "unknown")
        try:
            content = fp.read_text(encoding="utf-8", errors="replace")
        except Exception:
            stats.skipped_files.append(f"{fp} [read_error]")
            return

        snippets: List[str] = []
        if ext == ".py":
            snippets.extend(self._extract_python_docstrings(content))
            snippets.extend(self._extract_python_comments(content))
        else:
            snippets.extend(self._extract_generic_comments(content))

        clean_snippets = []
        for s in snippets:
            t = _strip_markup(s)
            t = _strip_pii(t, stats)
            if not t:
                continue
            if not _is_english_or_unknown(
                t, self.cfg.langdetect_threshold, stats, detect_langs,
            ):
                continue
            clean_snippets.append(t)
        if not clean_snippets:
            stats.skipped_files.append(f"{fp} [no_clean_comments]")
            return

        entry = self._build_md_entry(
            title=fp.name,
            source_path=str(fp),
            source_kind="github",
            language=lang,
            explanation="\n".join(f"- {x}" for x in clean_snippets[:200]),
            code_blocks=[],
            tags=[],
        )
        self._write_entry(entry, lang, "github", fp.stem, run_dir, stats)

    @staticmethod
    def _extract_python_docstrings(text: str) -> List[str]:
        out: List[str] = []
        try:
            tree = ast.parse(text)
        except Exception:
            return out
        module_doc = ast.get_docstring(tree)
        if module_doc:
            out.append(module_doc)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                d = ast.get_docstring(node)
                if d:
                    out.append(d)
        return out

    @staticmethod
    def _extract_python_comments(text: str) -> List[str]:
        out: List[str] = []
        try:
            for tok in tokenize.tokenize(BytesIO(text.encode("utf-8")).readline):
                if tok.type == tokenize.COMMENT:
                    out.append(tok.string.lstrip("# ").strip())
        except Exception:
            return out
        return out

    @staticmethod
    def _extract_generic_comments(text: str) -> List[str]:
        out = []
        out.extend([m.group(1).strip() for m in re.finditer(r"//\s*(.+)", text)])
        out.extend([m.group(1).strip() for m in re.finditer(r"#\s*(.+)", text)])
        out.extend([m.group(1).strip() for m in re.finditer(r"/\*\s*(.*?)\s*\*/", text, re.DOTALL)])
        return [x for x in out if x]
