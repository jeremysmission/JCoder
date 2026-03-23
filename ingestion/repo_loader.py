"""
Repo Loader
-----------
Scans a repository and extracts supported code files.

Non-programmer explanation:
This module walks through your project folder,
finds code files by extension, skips junk directories
(node_modules, .git, __pycache__), and passes each file
to the chunker for splitting.
"""

import os
from typing import Dict, List, Set

from .chunker import Chunker, DocumentChunker, LANGUAGE_MAP

try:
    from .parser_registry import DOCUMENT_EXTENSIONS, parse_file
    _HAS_PARSER_REGISTRY = True
except ImportError:
    _HAS_PARSER_REGISTRY = False
    DOCUMENT_EXTENSIONS = frozenset()

# ZIP-based document formats that start with PK signature but are NOT binary
_ZIP_DOCUMENT_EXTS = {".epub", ".odt", ".ods", ".odp", ".docx", ".pptx", ".xlsx"}

# OLE2-based document formats that start with 0xD0CF signature
_OLE2_DOCUMENT_EXTS = {".xls", ".ppt", ".doc"}

# Directories that should never be scanned
SKIP_DIRS: Set[str] = {
    ".git", ".hg", ".svn",
    "node_modules", "vendor", "venv", ".venv",
    "__pycache__", ".mypy_cache", ".pytest_cache",
    "dist", "build", ".tox", ".eggs",
    ".idea", ".vscode",
    "data", "logs",
    "evaluation",
}

# Binary signatures (first bytes)
_BINARY_SIGS = [b"\x89PNG", b"GIF8", b"\xff\xd8\xff", b"PK\x03\x04",
                b"\x7fELF", b"MZ", b"\x00asm"]

DEFAULT_MAX_FILE_KB = 1024  # 1 MB


class FileValidator:
    """Validates files before chunking: skips binary and oversized files."""

    def __init__(self, max_file_kb: int = DEFAULT_MAX_FILE_KB):
        self.max_file_bytes = max_file_kb * 1024
        self.skip_counts: Dict[str, int] = {}

    def is_valid(self, path: str) -> bool:
        """Return True if the file should be chunked."""
        # Size check
        try:
            size = os.path.getsize(path)
        except OSError:
            self._count("os_error")
            return False

        if size > self.max_file_bytes:
            self._count("too_large")
            return False

        if size == 0:
            self._count("empty")
            return False

        # Binary check -- skip for known document formats that have binary headers
        ext = os.path.splitext(path)[1].lower()
        if ext in _ZIP_DOCUMENT_EXTS or ext in _OLE2_DOCUMENT_EXTS:
            return True  # Document parsers handle these formats

        try:
            with open(path, "rb") as f:
                header = f.read(8)
            if any(header.startswith(sig) for sig in _BINARY_SIGS):
                self._count("binary")
                return False
        except OSError:
            self._count("os_error")
            return False

        return True

    def _count(self, reason: str):
        self.skip_counts[reason] = self.skip_counts.get(reason, 0) + 1

    def print_summary(self):
        if not self.skip_counts:
            return
        total = sum(self.skip_counts.values())
        print(f"[WARN] Skipped {total} files:")
        for reason, count in sorted(self.skip_counts.items()):
            print(f"  {reason}: {count}")


class RepoLoader:
    """
    Recursively scans a directory tree and chunks all supported source files.
    """

    def __init__(self, chunker: Chunker, max_file_kb: int = DEFAULT_MAX_FILE_KB,
                 include_documents: bool = True):
        self.chunker = chunker
        self.supported_extensions = set(LANGUAGE_MAP.keys())
        if include_documents and _HAS_PARSER_REGISTRY:
            self.supported_extensions |= set(DOCUMENT_EXTENSIONS)
        self.validator = FileValidator(max_file_kb)
        self._doc_chunker = DocumentChunker() if include_documents else None

    @staticmethod
    def _should_skip_dir(dirname: str) -> bool:
        """Skip generated/runtime folders that should not be self-ingested."""
        return dirname in SKIP_DIRS or dirname.startswith(".tmp_pytest")

    def load(self, root_path: str) -> List[dict]:
        """
        Walk the directory tree, chunk every supported file.
        Returns flat list of chunk dicts ready for embedding.
        """
        if not os.path.isdir(root_path):
            raise FileNotFoundError(f"Repository path does not exist: {root_path}")

        all_chunks = []
        files_processed = 0

        for dirpath, dirnames, filenames in os.walk(root_path):
            # Prune skipped directories in-place (prevents os.walk from descending)
            dirnames[:] = [d for d in dirnames if not self._should_skip_dir(d)]

            for filename in filenames:
                ext = os.path.splitext(filename)[1].lower()
                if ext not in self.supported_extensions:
                    continue

                full_path = os.path.join(dirpath, filename)

                if not self.validator.is_valid(full_path):
                    continue

                try:
                    # Use document parser for non-code formats
                    if (ext in DOCUMENT_EXTENSIONS and _HAS_PARSER_REGISTRY
                            and self._doc_chunker is not None):
                        chunks = self._doc_chunker.chunk_file(full_path)
                    else:
                        chunks = self.chunker.chunk_file(full_path)
                    all_chunks.extend(chunks)
                    files_processed += 1
                except Exception as e:
                    print(f"[WARN] Failed to chunk {full_path}: {e}")

        print(f"[OK] Processed {files_processed} files, {len(all_chunks)} chunks")
        self.validator.print_summary()
        return all_chunks
