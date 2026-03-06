"""
Tool Registry
--------------
Defines the tools available to the JCoder agent and handles execution.
Each tool has a JSON schema (for LLM function calling) and an execute method.

Tools:
- read_file: Read file contents
- write_file: Create or overwrite a file
- edit_file: Surgical string replacement in a file
- run_command: Execute shell command (with safety rails)
- search_files: Find files by glob pattern
- search_content: Search file contents (grep)
- rag_query: Query JCoder's retrieval engine
- memory_search: Search the agent's personal knowledge base
- memory_store: Store learned knowledge for future reference
- web_search: Search the web (when online)
- web_fetch: Fetch and read a web page (when online)
- list_directory: List files and subdirectories with sizes/types
- git_status: Show branch, changed files, recent commits
- git_diff: Show staged or unstaged diff output
- git_commit: Stage files and commit with validation
- run_tests: Run pytest with timeout and summary extraction
"""

from __future__ import annotations

import fnmatch
import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


# ---------------------------------------------------------------------------
# Safety
# ---------------------------------------------------------------------------

# Commands that are always blocked (case-insensitive first token)
_BLOCKED_COMMANDS = frozenset({
    "format", "diskpart", "shutdown", "reboot", "poweroff",
    "mkfs", "dd", "fdisk", "parted",
})

# Patterns that are blocked anywhere in the command string
_BLOCKED_PATTERNS = [
    re.compile(r"rm\s+(-[a-zA-Z]*f[a-zA-Z]*\s+)?/\s*$"),  # rm -rf /
    re.compile(r"del\s+/[sS]\s+/[qQ]\s+[A-Z]:\\", re.IGNORECASE),  # del /S /Q C:\
    re.compile(r":\(\)\{.*:\|:&.*\};:"),  # fork bomb
    re.compile(r">\s*/dev/sd[a-z]"),  # write to raw disk
]

MAX_OUTPUT_BYTES = 100_000  # 100KB max output per command


def _is_command_safe(command: str) -> tuple[bool, str]:
    """Check if a shell command is safe to execute. Returns (safe, reason)."""
    stripped = command.strip()
    if not stripped:
        return False, "Empty command"

    first_token = stripped.split()[0].lower()
    if first_token in _BLOCKED_COMMANDS:
        return False, f"Blocked command: {first_token}"

    for pattern in _BLOCKED_PATTERNS:
        if pattern.search(stripped):
            return False, f"Blocked pattern: {pattern.pattern}"

    return True, ""


# ---------------------------------------------------------------------------
# Tool result
# ---------------------------------------------------------------------------

@dataclass
class ToolResult:
    """Result of executing a tool."""
    success: bool
    output: str
    error: str = ""
    elapsed_s: float = 0.0


# ---------------------------------------------------------------------------
# Tool definitions (JSON schemas for LLM function calling)
# ---------------------------------------------------------------------------

TOOL_SCHEMAS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file. Returns the file text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute or relative path to the file",
                    },
                    "max_lines": {
                        "type": "integer",
                        "description": "Max lines to read (default: all)",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Create or overwrite a file with the given content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to write",
                    },
                    "content": {
                        "type": "string",
                        "description": "Full content to write to the file",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": (
                "Replace a specific string in a file with new text. "
                "The old_text must appear exactly once in the file."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to edit",
                    },
                    "old_text": {
                        "type": "string",
                        "description": "The exact text to find and replace",
                    },
                    "new_text": {
                        "type": "string",
                        "description": "The replacement text",
                    },
                },
                "required": ["path", "old_text", "new_text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": (
                "Execute a shell command and return its output. "
                "Use for running tests, installing packages, git commands, etc."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to execute",
                    },
                    "timeout_s": {
                        "type": "integer",
                        "description": "Timeout in seconds (default: 120)",
                    },
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": (
                "Find files matching a glob pattern. "
                "Returns a list of matching file paths."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern (e.g. '**/*.py', 'src/**/*.ts')",
                    },
                    "directory": {
                        "type": "string",
                        "description": "Root directory to search in (default: working dir)",
                    },
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_content",
            "description": (
                "Search file contents for a regex pattern. "
                "Returns matching lines with file paths and line numbers."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Regex pattern to search for",
                    },
                    "directory": {
                        "type": "string",
                        "description": "Root directory to search in (default: working dir)",
                    },
                    "file_pattern": {
                        "type": "string",
                        "description": "Only search files matching this glob (e.g. '*.py')",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Max matches to return (default: 50)",
                    },
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rag_query",
            "description": (
                "Query JCoder's knowledge base (Stack Overflow, docs, code). "
                "Returns relevant code snippets and explanations from the indexed corpus."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language question about code",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "task_complete",
            "description": (
                "Signal that the current task is complete. "
                "Call this when you have finished the assigned work."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "Brief summary of what was accomplished",
                    },
                },
                "required": ["summary"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_search",
            "description": (
                "Search the agent's personal knowledge base for past solutions "
                "and learned information."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language search query",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to return (default: 5)",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_store",
            "description": (
                "Store a piece of learned knowledge in the agent's memory "
                "for future reference."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The knowledge or information to store",
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional tags for categorization",
                    },
                },
                "required": ["content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web for programming documentation, examples, "
                "and solutions. Only available when online."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query (e.g. 'python asyncio tutorial')",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum results to return (default: 5)",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_fetch",
            "description": (
                "Fetch and read a web page. Returns the text content. "
                "Only available when online."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "URL of the page to fetch",
                    },
                    "max_chars": {
                        "type": "integer",
                        "description": "Max characters to return (default: 50000)",
                    },
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": (
                "List files and subdirectories in a directory. "
                "Shows file sizes and types. Capped at 500 entries."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory path (default: working dir)",
                    },
                    "recursive": {
                        "type": "boolean",
                        "description": "List recursively (default: false)",
                    },
                    "max_depth": {
                        "type": "integer",
                        "description": "Max recursion depth (default: 2)",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "git_status",
            "description": (
                "Show git repository status: current branch, changed files, "
                "and last 5 commits."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "git_diff",
            "description": (
                "Show git diff output. Can show staged or unstaged changes, "
                "optionally scoped to a specific path."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "staged": {
                        "type": "boolean",
                        "description": "Show staged changes only (default: false)",
                    },
                    "path": {
                        "type": "string",
                        "description": "Limit diff to this file or directory",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "git_commit",
            "description": (
                "Stage specific files and create a git commit. "
                "Refuses if files list or message is empty."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "files": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of file paths to stage and commit",
                    },
                    "message": {
                        "type": "string",
                        "description": "Commit message",
                    },
                },
                "required": ["files", "message"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_tests",
            "description": (
                "Run pytest on a file or directory. Returns pass/fail counts "
                "and test output."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "target": {
                        "type": "string",
                        "description": "Test file or directory (default: all tests)",
                    },
                    "verbose": {
                        "type": "boolean",
                        "description": "Verbose output (default: true)",
                    },
                    "timeout_s": {
                        "type": "integer",
                        "description": "Timeout in seconds (default: 300)",
                    },
                },
                "required": [],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Tool executor
# ---------------------------------------------------------------------------

class ToolRegistry:
    """Executes tools by name with safety guards."""

    def __init__(
        self,
        working_dir: str = ".",
        rag_callback: Optional[Callable[[str], str]] = None,
        allowed_dirs: Optional[List[str]] = None,
        memory: Optional[Any] = None,
        web_searcher: Optional[Any] = None,
    ):
        self.working_dir = os.path.abspath(working_dir)
        self._rag_callback = rag_callback
        self._memory = memory
        self._web = web_searcher
        self._allowed_dirs = [os.path.abspath(d) for d in (allowed_dirs or [])]
        if self.working_dir not in self._allowed_dirs:
            self._allowed_dirs.append(self.working_dir)

        self._dispatch: Dict[str, Callable] = {
            "read_file": self._read_file,
            "write_file": self._write_file,
            "edit_file": self._edit_file,
            "run_command": self._run_command,
            "search_files": self._search_files,
            "search_content": self._search_content,
            "rag_query": self._rag_query,
            "memory_search": self._memory_search,
            "memory_store": self._memory_store,
            "web_search": self._web_search,
            "web_fetch": self._web_fetch,
            "task_complete": self._task_complete,
            "list_directory": self._list_directory,
            "git_status": self._git_status,
            "git_diff": self._git_diff,
            "git_commit": self._git_commit,
            "run_tests": self._run_tests,
        }

    @property
    def schemas(self) -> List[Dict[str, Any]]:
        """Return tool JSON schemas for LLM function calling."""
        return TOOL_SCHEMAS

    def execute(self, name: str, arguments: Dict[str, Any]) -> ToolResult:
        """Execute a tool by name. Returns ToolResult."""
        fn = self._dispatch.get(name)
        if not fn:
            return ToolResult(
                success=False, output="", error=f"Unknown tool: {name}"
            )
        t0 = time.monotonic()
        try:
            result = fn(**arguments)
            result.elapsed_s = time.monotonic() - t0
            return result
        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=f"{type(e).__name__}: {e}",
                elapsed_s=time.monotonic() - t0,
            )

    def _resolve_path(self, path: str) -> str:
        """Resolve a path relative to working_dir, enforce allowed dirs."""
        if os.path.isabs(path):
            resolved = os.path.abspath(path)
        else:
            resolved = os.path.abspath(os.path.join(self.working_dir, path))

        if self._allowed_dirs:
            if not any(
                resolved.startswith(d) for d in self._allowed_dirs
            ):
                raise PermissionError(
                    f"Path {resolved} is outside allowed directories"
                )
        return resolved

    # -- File tools --------------------------------------------------------

    def _read_file(self, path: str, max_lines: int = 0) -> ToolResult:
        resolved = self._resolve_path(path)
        if not os.path.isfile(resolved):
            return ToolResult(False, "", f"File not found: {resolved}")

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

        # Truncate huge files
        if len(content) > MAX_OUTPUT_BYTES:
            content = content[:MAX_OUTPUT_BYTES] + "\n... [truncated]"

        return ToolResult(True, content)

    def _write_file(self, path: str, content: str) -> ToolResult:
        resolved = self._resolve_path(path)
        parent = os.path.dirname(resolved)
        os.makedirs(parent, exist_ok=True)

        with open(resolved, "w", encoding="utf-8", newline="\n") as f:
            f.write(content)

        return ToolResult(True, f"Wrote {len(content)} bytes to {resolved}")

    def _edit_file(
        self, path: str, old_text: str, new_text: str
    ) -> ToolResult:
        resolved = self._resolve_path(path)
        if not os.path.isfile(resolved):
            return ToolResult(False, "", f"File not found: {resolved}")

        with open(resolved, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

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
        with open(resolved, "w", encoding="utf-8", newline="\n") as f:
            f.write(new_content)

        return ToolResult(True, f"Edited {resolved} (1 replacement)")

    # -- Shell tool --------------------------------------------------------

    def _run_command(
        self, command: str, timeout_s: int = 120
    ) -> ToolResult:
        safe, reason = _is_command_safe(command)
        if not safe:
            return ToolResult(False, "", f"Safety block: {reason}")

        try:
            proc = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=min(timeout_s, 600),
                cwd=self.working_dir,
                env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            )
            output = proc.stdout
            if proc.stderr:
                output += f"\n[stderr]\n{proc.stderr}"
            if len(output) > MAX_OUTPUT_BYTES:
                output = output[:MAX_OUTPUT_BYTES] + "\n... [truncated]"

            return ToolResult(
                success=proc.returncode == 0,
                output=output,
                error="" if proc.returncode == 0 else f"Exit code: {proc.returncode}",
            )
        except subprocess.TimeoutExpired:
            return ToolResult(False, "", f"Command timed out after {timeout_s}s")

    # -- Search tools ------------------------------------------------------

    def _search_files(
        self, pattern: str, directory: str = ""
    ) -> ToolResult:
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
    ) -> ToolResult:
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

    # -- RAG tool ----------------------------------------------------------

    def _rag_query(self, query: str) -> ToolResult:
        if not self._rag_callback:
            return ToolResult(
                False, "",
                "RAG not configured. Set rag_callback in ToolRegistry.",
            )
        try:
            answer = self._rag_callback(query)
            return ToolResult(True, answer)
        except Exception as e:
            return ToolResult(False, "", f"RAG query failed: {e}")

    # -- Memory tools ------------------------------------------------------

    def _memory_search(self, query: str, top_k: int = 5) -> ToolResult:
        if not self._memory:
            return ToolResult(
                False, "",
                "Agent memory not configured. Memory module is unavailable.",
            )
        try:
            results = self._memory.search(query, top_k=top_k)
            if not results:
                return ToolResult(True, "No relevant memories found.")
            lines = []
            for i, r in enumerate(results, 1):
                lines.append(
                    f"[{i}] (score={r.get('score', 0):.3f}, "
                    f"confidence={r.get('confidence', 0):.2f}) "
                    f"{r.get('content', '')[:500]}"
                )
                if r.get("source_task"):
                    lines.append(f"    Source: {r['source_task'][:200]}")
            return ToolResult(True, "\n".join(lines))
        except Exception as e:
            return ToolResult(False, "", f"Memory search failed: {e}")

    def _memory_store(self, content: str, tags: Optional[List[str]] = None) -> ToolResult:
        if not self._memory:
            return ToolResult(
                False, "",
                "Agent memory not configured. Memory module is unavailable.",
            )
        try:
            entry = self._memory.ingest(
                content=content,
                source_task="agent_tool_store",
                tags=tags or [],
                confidence=0.8,
                tokens_used=0,
            )
            return ToolResult(
                True,
                f"Stored in memory (id={getattr(entry, 'id', 'unknown')})",
            )
        except Exception as e:
            return ToolResult(False, "", f"Memory store failed: {e}")

    # -- Web tools ---------------------------------------------------------

    def _web_search(self, query: str, max_results: int = 5) -> ToolResult:
        if not self._web:
            return ToolResult(
                False, "",
                "Web search not available. WebSearcher is not configured.",
            )
        try:
            results = self._web.search_duckduckgo(query, max_results=max_results)
            if not results:
                return ToolResult(True, "No results found.")
            lines = []
            for i, r in enumerate(results, 1):
                lines.append(
                    f"[{i}] {r['title']}\n"
                    f"    URL: {r['url']}\n"
                    f"    {r['snippet']}"
                )
            return ToolResult(True, "\n\n".join(lines))
        except PermissionError as e:
            return ToolResult(False, "", str(e))
        except Exception as e:
            return ToolResult(False, "", f"Web search failed: {e}")

    def _web_fetch(self, url: str, max_chars: int = 50_000) -> ToolResult:
        if not self._web:
            return ToolResult(
                False, "",
                "Web fetch not available. WebSearcher is not configured.",
            )
        try:
            text = self._web.fetch_page(url, max_chars=max_chars)
            if not text:
                return ToolResult(True, "[Page returned no text content]")
            return ToolResult(True, text)
        except PermissionError as e:
            return ToolResult(False, "", str(e))
        except Exception as e:
            return ToolResult(False, "", f"Web fetch failed: {e}")

    # -- Directory listing tool --------------------------------------------

    def _list_directory(
        self,
        path: str = "",
        recursive: bool = False,
        max_depth: int = 2,
    ) -> ToolResult:
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

    # -- Git tools ---------------------------------------------------------

    def _git_status(self) -> ToolResult:
        parts: list[str] = []

        # Current branch
        try:
            branch = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True, text=True, timeout=15,
                cwd=self.working_dir,
            )
            parts.append(f"Branch: {branch.stdout.strip()}")
        except Exception as e:
            parts.append(f"Branch: (error: {e})")

        # Porcelain status
        try:
            status = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True, text=True, timeout=30,
                cwd=self.working_dir,
            )
            status_text = status.stdout.strip()
            if status_text:
                parts.append(f"\nChanged files:\n{status_text}")
            else:
                parts.append("\nWorking tree clean.")
        except Exception as e:
            parts.append(f"\nStatus error: {e}")

        # Recent commits
        try:
            log = subprocess.run(
                ["git", "log", "--oneline", "-5"],
                capture_output=True, text=True, timeout=15,
                cwd=self.working_dir,
            )
            log_text = log.stdout.strip()
            if log_text:
                parts.append(f"\nRecent commits:\n{log_text}")
        except Exception as e:
            parts.append(f"\nLog error: {e}")

        output = "\n".join(parts)
        if len(output) > MAX_OUTPUT_BYTES:
            output = output[:MAX_OUTPUT_BYTES] + "\n... [truncated]"
        return ToolResult(True, output)

    def _git_diff(
        self, staged: bool = False, path: str = ""
    ) -> ToolResult:
        cmd = ["git", "diff"]
        if staged:
            cmd.append("--staged")
        if path:
            cmd.append("--")
            cmd.append(path)

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True, text=True, timeout=30,
                cwd=self.working_dir,
            )
            output = proc.stdout
            if proc.stderr:
                output += f"\n[stderr]\n{proc.stderr}"
            if not output.strip():
                label = "staged" if staged else "unstaged"
                return ToolResult(True, f"No {label} changes.")
            if len(output) > MAX_OUTPUT_BYTES:
                output = output[:MAX_OUTPUT_BYTES] + "\n... [truncated]"
            return ToolResult(True, output)
        except subprocess.TimeoutExpired:
            return ToolResult(False, "", "git diff timed out after 30s")
        except Exception as e:
            return ToolResult(False, "", f"git diff failed: {e}")

    def _git_commit(
        self, files: List[str], message: str
    ) -> ToolResult:
        # Safety: refuse empty inputs
        if not files:
            return ToolResult(False, "", "Refusing to commit: files list is empty.")
        if not message or not message.strip():
            return ToolResult(False, "", "Refusing to commit: message is empty.")

        # Validate all file paths are within allowed dirs
        resolved_files: list[str] = []
        for f in files:
            try:
                resolved_files.append(self._resolve_path(f))
            except PermissionError as e:
                return ToolResult(False, "", str(e))

        # Stage files
        try:
            add_proc = subprocess.run(
                ["git", "add", "--"] + resolved_files,
                capture_output=True, text=True, timeout=30,
                cwd=self.working_dir,
            )
            if add_proc.returncode != 0:
                return ToolResult(
                    False, "",
                    f"git add failed (exit {add_proc.returncode}): "
                    f"{add_proc.stderr.strip()}",
                )
        except subprocess.TimeoutExpired:
            return ToolResult(False, "", "git add timed out after 30s")

        # Commit
        try:
            commit_proc = subprocess.run(
                ["git", "commit", "-m", message.strip()],
                capture_output=True, text=True, timeout=30,
                cwd=self.working_dir,
            )
            output = commit_proc.stdout
            if commit_proc.stderr:
                output += f"\n[stderr]\n{commit_proc.stderr}"
            if len(output) > MAX_OUTPUT_BYTES:
                output = output[:MAX_OUTPUT_BYTES] + "\n... [truncated]"
            return ToolResult(
                success=commit_proc.returncode == 0,
                output=output,
                error="" if commit_proc.returncode == 0
                else f"git commit failed (exit {commit_proc.returncode})",
            )
        except subprocess.TimeoutExpired:
            return ToolResult(False, "", "git commit timed out after 30s")

    # -- Test runner tool --------------------------------------------------

    def _run_tests(
        self,
        target: str = "",
        verbose: bool = True,
        timeout_s: int = 300,
    ) -> ToolResult:
        cmd = ["python", "-m", "pytest"]
        if target:
            cmd.append(target)
        if verbose:
            cmd.append("-v")
        cmd.extend(["--tb=short"])

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True, text=True,
                timeout=min(timeout_s, 600),
                cwd=self.working_dir,
                env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            )
            output = proc.stdout
            if proc.stderr:
                output += f"\n[stderr]\n{proc.stderr}"

            # Extract summary line (e.g. "5 passed, 1 failed in 2.3s")
            summary_line = ""
            for line in reversed(output.splitlines()):
                stripped = line.strip()
                if "passed" in stripped or "failed" in stripped or "error" in stripped:
                    summary_line = stripped
                    break

            if summary_line:
                output += f"\n\n=== SUMMARY: {summary_line} ==="

            if len(output) > MAX_OUTPUT_BYTES:
                output = output[:MAX_OUTPUT_BYTES] + "\n... [truncated]"

            return ToolResult(
                success=proc.returncode == 0,
                output=output,
                error="" if proc.returncode == 0
                else f"Tests failed (exit {proc.returncode})",
            )
        except subprocess.TimeoutExpired:
            return ToolResult(
                False, "", f"Tests timed out after {timeout_s}s"
            )

    # -- Control tool ------------------------------------------------------

    def _task_complete(self, summary: str) -> ToolResult:
        return ToolResult(True, f"TASK_COMPLETE: {summary}")
