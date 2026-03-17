"""
AST-Aware FTS5 Index Builder
-----------------------------
Wraps the tree-sitter Chunker for use with FTS5 index building scripts.
Instead of naive text[:MAX_CHARS] slicing, this splits code at function/class
boundaries, preserving semantic units.

Research basis: cAST (EMNLP 2025) -- +4.3 Recall@5 vs naive chunking.

Usage:
    from ingestion.ast_fts5_builder import ASTIndexBuilder

    builder = ASTIndexBuilder(db_path, max_chunk_chars=2000)
    builder.add_code("def hello(): pass", "source.py", language="python")
    builder.add_text("Some documentation", "doc_id_123")
    entries, chunks, size_mb = builder.finish()
"""

from __future__ import annotations

import hashlib
import re
import sqlite3
from pathlib import Path
from typing import List, Optional, Tuple

from ingestion.chunker import Chunker, LANGUAGE_MAP


_NORMALIZE_RE = re.compile(r"[_\-./\\:]")
_PYTHON_BOUNDARY_RE = re.compile(
    r"(?m)^(?=async\s+def\s+\w+\s*\(|def\s+\w+\s*\(|class\s+\w+\s*[:(])"
)


def _normalize_for_fts5(text: str) -> str:
    """Normalize text for FTS5 indexing (lowercase, simplify separators)."""
    out = _NORMALIZE_RE.sub(" ", text)
    return out.lower()


def _detect_language(source_id: str) -> Optional[str]:
    """Detect language from source ID or file extension."""
    for ext, lang in LANGUAGE_MAP.items():
        if source_id.endswith(ext):
            return lang
    return None


def _heuristic_code_chunks(
    code: str,
    source_id: str,
    language: Optional[str],
) -> List[dict]:
    """Fallback structural chunking when tree-sitter grammars are unavailable."""
    stripped = code.strip()
    if not stripped:
        return []

    if language != "python":
        return [{"content": stripped}]

    matches = list(_PYTHON_BOUNDARY_RE.finditer(code))
    if not matches:
        return [{"content": stripped}]

    chunks: List[dict] = []
    first_start = matches[0].start()
    preamble = code[:first_start].strip()
    if preamble:
        chunks.append({"content": preamble})

    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(code)
        content = code[start:end].strip()
        if content:
            chunks.append({"content": content})

    return chunks


class ASTIndexBuilder:
    """FTS5 index builder that uses tree-sitter AST chunking for code."""

    def __init__(
        self,
        db_path: Path,
        max_chunk_chars: int = 2000,
        batch_size: int = 5000,
    ):
        self.db_path = Path(db_path)
        self._chunker = Chunker(max_chars=max_chunk_chars)
        self._batch_size = batch_size
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS chunks "
            "USING fts5(search_content, source_path, chunk_id)"
        )
        self._batch: List[Tuple[str, str, str]] = []
        self.total_entries = 0
        self.total_chunks = 0

    def add_code(
        self,
        code: str,
        source_id: str,
        language: Optional[str] = None,
    ) -> int:
        """Add code content using AST-aware chunking.

        Returns number of chunks produced.
        """
        if not code or not code.strip():
            return 0

        self.total_entries += 1
        lang = language or _detect_language(source_id)

        if lang:
            # Use AST chunking -- split at function/class boundaries
            chunks = self._chunker._chunk_by_ast(code, source_id, lang)
            if not chunks:
                # AST parse failed, fall back to lightweight structural chunking
                # before collapsing to naive char windows.
                chunks = _heuristic_code_chunks(code, source_id, lang)
            if not chunks:
                chunks = self._chunker._chunk_by_chars(code, source_id)
        else:
            # Unknown language, use char chunking
            chunks = self._chunker._chunk_by_chars(code, source_id)
            if not chunks:
                chunks = _heuristic_code_chunks(code, source_id, lang)

        normalized_chunks: List[dict] = []
        for chunk in chunks:
            content = chunk.get("content", "")
            if (
                content
                and len(content) > self._chunker.max_chars
                and chunk.get("node_type") not in {"preamble", "trailing"}
            ):
                normalized_chunks.extend(
                    self._chunker._chunk_by_chars(content, source_id)
                )
            else:
                normalized_chunks.append(chunk)

        added = 0
        for chunk in normalized_chunks:
            content = chunk.get("content", "")
            if content and content.strip():
                cid = chunk.get("content_hash", "") or hashlib.sha256(
                    f"{source_id}:{added}".encode()
                ).hexdigest()
                self._batch.append((
                    _normalize_for_fts5(content),
                    source_id,
                    cid,
                ))
                added += 1
                if len(self._batch) >= self._batch_size:
                    self._flush()

        return added

    def add_text(self, text: str, source_id: str) -> int:
        """Add non-code text (docs, Q&A, etc.) using char chunking.

        Returns number of chunks produced.
        """
        if not text or len(text.strip()) < 30:
            return 0

        self.total_entries += 1
        chunks = self._chunker._chunk_by_chars(text, source_id)
        added = 0
        for chunk in chunks:
            content = chunk.get("content", "")
            if content and len(content.strip()) > 30:
                cid = chunk.get("content_hash", "") or hashlib.sha256(
                    f"{source_id}:{added}".encode()
                ).hexdigest()
                self._batch.append((
                    _normalize_for_fts5(content),
                    source_id,
                    cid,
                ))
                added += 1
                if len(self._batch) >= self._batch_size:
                    self._flush()

        return added

    def _flush(self) -> None:
        if self._batch:
            self._conn.executemany(
                "INSERT INTO chunks(search_content, source_path, chunk_id) "
                "VALUES (?, ?, ?)",
                self._batch,
            )
            self._conn.commit()
            self.total_chunks += len(self._batch)
            self._batch = []

    def finish(self) -> Tuple[int, int, float]:
        """Flush remaining batch and close. Returns (entries, chunks, size_mb)."""
        self._flush()
        self._conn.close()
        size_mb = self.db_path.stat().st_size / 1e6
        return self.total_entries, self.total_chunks, size_mb

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.finish()
