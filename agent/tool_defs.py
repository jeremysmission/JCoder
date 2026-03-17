"""
Tool Definitions: Types, Safety, Caching, Validation
-----------------------------------------------------
Shared types and utility functions used by the tool system.
"""

from __future__ import annotations

import hashlib
import json as _json
import re
import shlex
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from agent.tool_schemas import TOOL_PARAM_SCHEMAS


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_OUTPUT_BYTES = 100_000


# ---------------------------------------------------------------------------
# Safety
# ---------------------------------------------------------------------------

_BLOCKED_COMMANDS = frozenset({
    "format", "diskpart", "shutdown", "reboot", "poweroff",
    "mkfs", "dd", "fdisk", "parted",
    "cipher", "bcdedit", "bootrec",
    "shred", "cfdisk", "sfdisk", "badblocks",
})

_BLOCKED_PATTERNS = [
    re.compile(r"rm\s+(-[a-zA-Z]*f[a-zA-Z]*\s+)?/\s*$"),
    re.compile(r"del\s+/[sS]\s+/[qQ]\s+[A-Z]:\\", re.IGNORECASE),
    re.compile(r":\(\)\{.*:\|:&.*\};:"),
    re.compile(r">\s*/dev/sd[a-z]"),
]


def _split_command_args(command: str) -> list[str]:
    """Split a command string into argv, handling both Unix and Windows."""
    try:
        return shlex.split(command, posix=True)
    except ValueError:
        return shlex.split(command, posix=False)


def _is_within_directory(path: str, allowed_dir: str) -> bool:
    """Check if a resolved path is within an allowed directory."""
    import os
    norm_path = os.path.normcase(os.path.normpath(path))
    norm_dir = os.path.normcase(os.path.normpath(allowed_dir))
    return norm_path == norm_dir or norm_path.startswith(norm_dir + os.sep)


_SHELL_OPERATORS = re.compile(r"[;&|]|&&|\|\|")


def _is_command_safe(command: str) -> tuple[bool, str]:
    """Check if a command is safe to execute."""
    stripped = command.strip()
    if not stripped:
        return False, "Empty command"

    # Block shell operators (&&, ||, ;, |) -- subprocess runs without shell
    if _SHELL_OPERATORS.search(stripped):
        return False, "Shell operator detected -- use separate commands"

    try:
        tokens = _split_command_args(stripped)
    except ValueError:
        tokens = stripped.split()

    if not tokens:
        return False, "Could not parse command"

    first_token = tokens[0].lower()
    base = first_token.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    if base in _BLOCKED_COMMANDS:
        return False, f"Blocked command: '{base}'"

    for pat in _BLOCKED_PATTERNS:
        if pat.search(stripped):
            return False, f"Blocked pattern: {pat.pattern}"

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
# Tool result cache
# ---------------------------------------------------------------------------

CACHEABLE_TOOLS = frozenset({
    "read_file", "search_files", "search_content",
    "rag_query", "memory_search", "list_directory",
    "git_status", "git_diff",
})

CACHE_TTL: Dict[str, float] = {
    "read_file": 30.0,
    "search_files": 30.0,
    "search_content": 30.0,
    "rag_query": 120.0,
    "memory_search": 120.0,
    "list_directory": 15.0,
    "git_status": 10.0,
    "git_diff": 10.0,
}

_CACHE_MAX_ENTRIES = 256


def _cache_key(name: str, arguments: Dict[str, Any]) -> str:
    """Deterministic cache key from tool name + sorted arguments."""
    raw = _json.dumps({"t": name, "a": arguments}, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()


class ToolResultCache:
    """TTL-bounded LRU cache for deterministic tool results."""

    def __init__(self, max_entries: int = _CACHE_MAX_ENTRIES):
        self._store: Dict[str, Tuple[float, ToolResult]] = {}
        self._max = max_entries
        self.hits = 0
        self.misses = 0

    def get(self, key: str, ttl: float) -> Optional[ToolResult]:
        entry = self._store.get(key)
        if entry is None:
            self.misses += 1
            return None
        ts, result = entry
        if time.monotonic() - ts > ttl:
            del self._store[key]
            self.misses += 1
            return None
        self.hits += 1
        return result

    def put(self, key: str, result: ToolResult) -> None:
        if len(self._store) >= self._max:
            oldest_key = min(self._store, key=lambda k: self._store[k][0])
            del self._store[oldest_key]
        self._store[key] = (time.monotonic(), result)

    def invalidate(self) -> None:
        self._store.clear()

    def stats(self) -> Dict[str, Any]:
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 3) if total else 0.0,
            "entries": len(self._store),
        }


# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------

def _validate_tool_args(
    name: str, arguments: Dict[str, Any]
) -> Tuple[bool, str, Dict[str, Any]]:
    """Validate and coerce tool arguments against the JSON schema.

    Returns (valid, error_msg, cleaned_arguments).
    """
    schema = TOOL_PARAM_SCHEMAS.get(name)
    if schema is None:
        return True, "", arguments

    properties = schema.get("properties", {})
    required = set(schema.get("required", []))

    missing = required - set(arguments.keys())
    if missing:
        return False, f"Missing required parameters: {', '.join(sorted(missing))}", arguments

    cleaned = {}
    for key, value in arguments.items():
        if key not in properties:
            continue
        prop_schema = properties[key]
        expected_type = prop_schema.get("type", "string")

        if expected_type == "integer" and isinstance(value, str):
            try:
                value = int(value)
            except (ValueError, TypeError):
                return False, f"Parameter '{key}' must be integer, got '{value}'", arguments
        elif expected_type == "integer" and isinstance(value, float):
            value = int(value)
        elif expected_type == "string" and not isinstance(value, str):
            value = str(value)
        elif expected_type == "boolean" and isinstance(value, str):
            value = value.lower() in ("true", "1", "yes")

        cleaned[key] = value

    return True, "", cleaned
