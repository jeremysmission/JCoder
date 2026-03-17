"""
Tool JSON Schemas for LLM Function Calling
-------------------------------------------
Defines the function-calling schemas that describe each tool's
name, description, and parameters for the LLM.
"""

from __future__ import annotations

from typing import Any, Dict, List


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
                "Execute a command and return its output. "
                "Runs a single program with arguments (no shell operators "
                "like |, &&, ;, or redirects). "
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


def build_param_schema_map() -> Dict[str, Dict[str, Any]]:
    """Extract parameter schemas from TOOL_SCHEMAS for quick lookup."""
    mapping: Dict[str, Dict[str, Any]] = {}
    for entry in TOOL_SCHEMAS:
        fn = entry.get("function", {})
        name = fn.get("name", "")
        params = fn.get("parameters", {})
        if name and params:
            mapping[name] = params
    return mapping


TOOL_PARAM_SCHEMAS: Dict[str, Dict[str, Any]] = build_param_schema_map()
