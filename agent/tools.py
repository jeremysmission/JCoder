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

import os
import re
import shlex
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from agent.tool_defs import (
    ToolResultCache,
    _cache_key,
    _validate_tool_args,
    TOOL_PARAM_SCHEMAS as _TOOL_PARAM_SCHEMAS,
)
from agent.tools_file_ops import FileOpsMixin
from agent.tools_shell_ops import ShellOpsMixin
from agent.tools_knowledge_ops import KnowledgeOpsMixin


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
_SHELL_OPERATOR_TOKENS = frozenset({"&&", "||", "|", ";", "&", ">", ">>", "<", "<<"})

MAX_OUTPUT_BYTES = 100_000  # 100KB max output per command


def _split_command_args(command: str) -> list[str]:
    """Split a command string into argv without invoking a shell."""
    parts = shlex.split(command, posix=os.name != "nt")
    if os.name == "nt":
        cleaned: list[str] = []
        for part in parts:
            if len(part) >= 2 and part[0] == part[-1] and part[0] in {"'", '"'}:
                part = part[1:-1]
            cleaned.append(part)
        return cleaned
    return parts


def _is_within_directory(path: str, allowed_dir: str) -> bool:
    """Return True when path is inside allowed_dir after canonicalization."""
    try:
        common = os.path.commonpath(
            [os.path.normcase(path), os.path.normcase(allowed_dir)]
        )
    except ValueError:
        return False
    return common == os.path.normcase(allowed_dir)


def _is_command_safe(command: str) -> tuple[bool, str]:
    """Check if a shell command is safe to execute. Returns (safe, reason)."""
    stripped = command.strip()
    if not stripped:
        return False, "Empty command"

    try:
        argv = _split_command_args(stripped)
    except ValueError as exc:
        return False, f"Could not parse command: {exc}"

    if not argv:
        return False, "Empty command"

    first_token = os.path.splitext(os.path.basename(argv[0]))[0].lower()
    if first_token in _BLOCKED_COMMANDS:
        return False, f"Blocked command: {first_token}"

    for pattern in _BLOCKED_PATTERNS:
        if pattern.search(stripped):
            return False, f"Blocked pattern: {pattern.pattern}"

    bad_token = next((token for token in argv if token in _SHELL_OPERATOR_TOKENS), "")
    if bad_token:
        return False, f"Shell operator not allowed: {bad_token}"

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
# Canonical schemas live in agent/tool_schemas.py (single source of truth).
from agent.tool_schemas import TOOL_SCHEMAS  # noqa: F401, E402


# ---------------------------------------------------------------------------
# Tool executor
# ---------------------------------------------------------------------------

class ToolRegistry(FileOpsMixin, ShellOpsMixin, KnowledgeOpsMixin):
    """Executes tools by name with safety guards."""

    def __init__(
        self,
        working_dir: str = ".",
        rag_callback: Optional[Callable[[str], str]] = None,
        allowed_dirs: Optional[List[str]] = None,
        memory: Optional[Any] = None,
        web_searcher: Optional[Any] = None,
    ):
        self.working_dir = os.path.realpath(os.path.abspath(working_dir))
        self._rag_callback = rag_callback
        self._memory = memory
        self._web = web_searcher
        self._allowed_dirs = [
            os.path.realpath(os.path.abspath(d)) for d in (allowed_dirs or [])
        ]
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
            resolved = os.path.realpath(os.path.abspath(path))
        else:
            resolved = os.path.realpath(
                os.path.abspath(os.path.join(self.working_dir, path))
            )

        if self._allowed_dirs:
            if not any(
                _is_within_directory(resolved, allowed_dir)
                for allowed_dir in self._allowed_dirs
            ):
                raise PermissionError(
                    f"Path {resolved} is outside allowed directories"
                )
        return resolved
