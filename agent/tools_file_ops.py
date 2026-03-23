"""File operation tools mixin for ToolRegistry."""

from __future__ import annotations

import fnmatch
import os
import re
from pathlib import Path
from typing import Any, List, Optional


# Avoid circular import -- ToolResult and MAX_OUTPUT_BYTES live in tools.py
# The mixin accesses them via self (ToolResult) or module-level import at
# class-body time.  We do a lazy import inside each method instead.


class FileOpsMixin:
    """Mixin providing file I/O and search tool implementations."""

    # These attributes are provided by ToolRegistry (the concrete class).
    working_dir: str
    _allowed_dirs: List[str]

    def _resolve_path(self, path: str) -> str:  # type: ignore[override]
        ...  # provided by ToolRegistry

    # -- File tools --------------------------------------------------------

    def _read_file(self, path: str, max_lines: int = 0) -> Any:
        from agent.tools import ToolResult, MAX_OUTPUT_BYTES

        resolved = self._resolve_path(path)
        try:
            with open(resolved, "r", encoding="utf-8", errors="replace") as f:
                if max_lines > 0:
                    lines = []
                    for i, line in enumerate(f):
                        if i >= max_lines:
                            break
                        lines.append(line)
                    content = "".join(lines)
                else:
                    content = f.read()
        except FileNotFoundError:
            return ToolResult(False, "", f"File not found: {resolved}")
        except OSError as exc:
            return ToolResult(False, "", f"Could not read file: {exc}")

        # Truncate huge files
        if len(content) > MAX_OUTPUT_BYTES:
            content = content[:MAX_OUTPUT_BYTES] + "\n... [truncated]"

        return ToolResult(True, content)

    def _write_file(self, path: str, content: str) -> Any:
        from agent.tools import ToolResult

        resolved = self._resolve_path(path)
        parent = os.path.dirname(resolved)
        os.makedirs(parent, exist_ok=True)

        with open(resolved, "w", encoding="utf-8", newline="\n") as f:
            f.write(content)

        return ToolResult(True, f"Wrote {len(content)} bytes to {resolved}")

    def _edit_file(
        self, path: str, old_text: str, new_text: str
    ) -> Any:
        from agent.tools import ToolResult

        resolved = self._resolve_path(path)
        try:
            with open(resolved, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
        except FileNotFoundError:
            return ToolResult(False, "", f"File not found: {resolved}")
        except OSError as exc:
            return ToolResult(False, "", f"Could not read file: {exc}")

        count = content.count(old_text)
        if count == 0:
            return ToolResult(False, "", "old_text not found in file")
        if count > 1:
            return ToolResult(
                False, "",
                f"old_text found {count} times -- must be unique. "
                "Provide more surrounding context.",
            )

        new_content = content.replace(old_text, new_text, 1)
        try:
            with open(resolved, "w", encoding="utf-8", newline="\n") as f:
                f.write(new_content)
        except OSError as exc:
            return ToolResult(False, "", f"Could not write file: {exc}")

        return ToolResult(True, f"Edited {resolved} (1 replacement)")

    # -- Search tools ------------------------------------------------------

    def _search_files(
        self, pattern: str, directory: str = ""
    ) -> Any:
        from agent.tools import ToolResult

        root = self._resolve_path(directory) if directory else self.working_dir
        if not os.path.isdir(root):
            return ToolResult(False, "", f"Directory not found: {root}")

        matches = []
        root_path = Path(root)
        for p in root_path.rglob(pattern.lstrip("**/") if "**" in pattern else pattern):
            if p.is_file():
                matches.append(str(p))
            if len(matches) >= 200:
                break

        # Try full glob if rglob didn't work well
        if not matches:
            for p in root_path.glob(pattern):
                if p.is_file():
                    matches.append(str(p))
                if len(matches) >= 200:
                    break

        if not matches:
            return ToolResult(True, "No files matched the pattern.")

        return ToolResult(True, "\n".join(matches))

    def _search_content(
        self,
        pattern: str,
        directory: str = "",
        file_pattern: str = "",
        max_results: int = 50,
    ) -> Any:
        from agent.tools import ToolResult

        root = self._resolve_path(directory) if directory else self.working_dir
        if not os.path.isdir(root):
            return ToolResult(False, "", f"Directory not found: {root}")

        try:
            regex = re.compile(pattern, re.IGNORECASE)
        except re.error as e:
            return ToolResult(False, "", f"Invalid regex: {e}")

        matches = []
        root_path = Path(root)
        glob_pat = file_pattern or "*"

        for fpath in root_path.rglob(glob_pat):
            if not fpath.is_file():
                continue
            # Skip binary files and large files
            if fpath.stat().st_size > 1_000_000:
                continue
            try:
                text = fpath.read_text(encoding="utf-8", errors="replace")
            except (OSError, PermissionError):
                continue

            for i, line in enumerate(text.splitlines(), 1):
                if regex.search(line):
                    matches.append(f"{fpath}:{i}: {line.strip()}")
                    if len(matches) >= max_results:
                        break
            if len(matches) >= max_results:
                break

        if not matches:
            return ToolResult(True, "No matches found.")

        return ToolResult(True, "\n".join(matches))

    # -- Directory listing tool --------------------------------------------

    def _list_directory(
        self,
        path: str = "",
        recursive: bool = False,
        max_depth: int = 2,
    ) -> Any:
        from agent.tools import ToolResult, MAX_OUTPUT_BYTES

        root = self._resolve_path(path) if path else self.working_dir
        if not os.path.isdir(root):
            return ToolResult(False, "", f"Directory not found: {root}")

        entries: list[str] = []
        cap = 500

        def _format_entry(entry_path: str, depth: int) -> str:
            """Format a single directory entry with size and type."""
            name = os.path.basename(entry_path)
            prefix = "  " * depth
            if os.path.isdir(entry_path):
                return f"{prefix}[DIR]  {name}/"
            try:
                size = os.path.getsize(entry_path)
            except OSError:
                size = 0
            if size < 1024:
                size_str = f"{size} B"
            elif size < 1_048_576:
                size_str = f"{size / 1024:.1f} KB"
            else:
                size_str = f"{size / 1_048_576:.1f} MB"
            ext = os.path.splitext(name)[1] or "(no ext)"
            return f"{prefix}[FILE] {name}  ({size_str}, {ext})"

        def _walk(dir_path: str, depth: int) -> None:
            if len(entries) >= cap:
                return
            try:
                items = sorted(os.listdir(dir_path))
            except PermissionError:
                entries.append(f"{'  ' * depth}[PERM DENIED] {dir_path}")
                return

            for item in items:
                if len(entries) >= cap:
                    return
                full = os.path.join(dir_path, item)
                entries.append(_format_entry(full, depth))
                if recursive and os.path.isdir(full) and depth < max_depth:
                    _walk(full, depth + 1)

        _walk(root, 0)

        if len(entries) >= cap:
            entries.append(f"\n... [capped at {cap} entries]")

        header = f"Directory: {root} ({len(entries)} entries)\n"
        output = header + "\n".join(entries)
        if len(output) > MAX_OUTPUT_BYTES:
            output = output[:MAX_OUTPUT_BYTES] + "\n... [truncated]"
        return ToolResult(True, output)
