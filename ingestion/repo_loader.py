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
from typing import List, Set

from .chunker import Chunker, LANGUAGE_MAP

# Directories that should never be scanned
SKIP_DIRS: Set[str] = {
    ".git", ".hg", ".svn",
    "node_modules", "vendor", "venv", ".venv",
    "__pycache__", ".mypy_cache", ".pytest_cache",
    "dist", "build", ".tox", ".eggs",
    ".idea", ".vscode",
}


class RepoLoader:
    """
    Recursively scans a directory tree and chunks all supported source files.
    """

    def __init__(self, chunker: Chunker):
        self.chunker = chunker
        self.supported_extensions = set(LANGUAGE_MAP.keys())

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
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]

            for filename in filenames:
                ext = os.path.splitext(filename)[1].lower()
                if ext not in self.supported_extensions:
                    continue

                full_path = os.path.join(dirpath, filename)

                try:
                    chunks = self.chunker.chunk_file(full_path)
                    all_chunks.extend(chunks)
                    files_processed += 1
                except Exception as e:
                    # Log but don't crash on individual file failures
                    print(f"[WARN] Failed to chunk {full_path}: {e}")

        print(f"[OK] Processed {files_processed} files, {len(all_chunks)} chunks")
        return all_chunks
