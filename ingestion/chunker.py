"""
Chunker
-------
Splits source code into structured chunks using tree-sitter AST parsing.

Non-programmer explanation:
Instead of chopping code at arbitrary character boundaries, this module
understands code structure. It uses tree-sitter to parse the file into
an Abstract Syntax Tree (AST) -- basically a map of every function,
class, and block. It then splits at those logical boundaries so each
chunk is a complete, meaningful piece of code.

Falls back to character splitting for languages without a grammar.
"""

import hashlib
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

CHUNKER_VERSION = "1.0"

# Map file extensions to tree-sitter language names and their pip packages.
# Install with: pip install tree-sitter-python tree-sitter-javascript etc.
LANGUAGE_MAP = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".java": "java",
    ".cpp": "cpp",
    ".c": "c",
    ".cs": "c_sharp",
    ".go": "go",
    ".rs": "rust",
    ".rb": "ruby",
    ".php": "php",
    ".kt": "kotlin",
    # Config files -- no AST grammar, char-fallback only
    ".yaml": None,
    ".yml": None,
}

# AST node types that represent logical chunk boundaries per language.
# When we see one of these, we start a new chunk.
BOUNDARY_NODES = {
    "python": {"function_definition", "class_definition", "decorated_definition"},
    "javascript": {"function_declaration", "class_declaration", "export_statement",
                    "lexical_declaration", "expression_statement"},
    "typescript": {"function_declaration", "class_declaration", "export_statement",
                    "lexical_declaration", "interface_declaration", "type_alias_declaration"},
    "tsx": {"function_declaration", "class_declaration", "export_statement"},
    "java": {"method_declaration", "class_declaration", "interface_declaration"},
    "cpp": {"function_definition", "class_specifier", "namespace_definition"},
    "c": {"function_definition", "struct_specifier"},
    "c_sharp": {"method_declaration", "class_declaration", "interface_declaration"},
    "go": {"function_declaration", "method_declaration", "type_declaration"},
    "rust": {"function_item", "impl_item", "struct_item", "enum_item", "trait_item"},
    "ruby": {"method", "class", "module"},
    "php": {"function_definition", "class_declaration", "method_declaration"},
    "kotlin": {"function_declaration", "class_declaration", "object_declaration"},
}


def _try_load_language(lang_name: str):
    """
    Attempt to import the tree-sitter grammar for a language.
    Returns (Language, Parser) or (None, None) if not installed.
    """
    try:
        from tree_sitter import Language, Parser

        # Each language is a separate pip package: tree_sitter_python, etc.
        module = __import__(f"tree_sitter_{lang_name}")
        language = Language(module.language())
        parser = Parser(language)
        return language, parser
    except (ImportError, AttributeError, Exception):
        return None, None


class Chunker:
    """
    Splits source files into indexed chunks.
    Uses tree-sitter AST when grammar is available, character fallback otherwise.
    """

    def __init__(self, max_chars: int = 4000):
        self.max_chars = max_chars
        self._parser_cache: Dict[str, Optional[Tuple]] = {}

    def _get_parser(self, lang_name: str):
        """Load and cache tree-sitter parser for a language."""
        if lang_name not in self._parser_cache:
            lang, parser = _try_load_language(lang_name)
            self._parser_cache[lang_name] = (lang, parser) if parser else None
        return self._parser_cache[lang_name]

    def _hash(self, text: str) -> str:
        # Normalize line endings so chunk IDs are stable across OS
        return hashlib.sha256(text.replace("\r\n", "\n").encode("utf-8")).hexdigest()

    def _make_chunk(self, text: str, file_path: str, **extra) -> Dict:
        """Build a chunk metadata dict."""
        return {
            "id": self._hash(text),
            "content": text,
            "source_path": file_path,
            "source_type": os.path.splitext(file_path)[1],
            "ingestion_date": datetime.now(timezone.utc).isoformat(),
            "content_hash": self._hash(text),
            "chunker_version": CHUNKER_VERSION,
            **extra,
        }

    def _chunk_by_ast(self, content: str, file_path: str, lang_name: str) -> List[Dict]:
        """
        Split file at AST boundaries (functions, classes, etc.).
        If a single node exceeds max_chars, fall back to character split for that node.
        """
        cached = self._get_parser(lang_name)
        if cached is None:
            return []

        _language, parser = cached
        tree = parser.parse(content.encode("utf-8"))
        root = tree.root_node
        boundaries = BOUNDARY_NODES.get(lang_name, set())

        chunks = []
        last_end = 0

        for child in root.children:
            if child.type in boundaries:
                # Capture any code between the last boundary and this one (imports, etc.)
                preamble = content[last_end:child.start_byte].strip()
                if preamble:
                    chunks.append(self._make_chunk(
                        preamble, file_path, node_type="preamble",
                    ))

                node_text = content[child.start_byte:child.end_byte]
                if len(node_text) <= self.max_chars:
                    chunks.append(self._make_chunk(
                        node_text, file_path, node_type=child.type,
                    ))
                else:
                    # Oversized node -- character-split it
                    chunks.extend(self._chunk_by_chars(node_text, file_path))

                last_end = child.end_byte

        # Trailing code after last boundary
        trailing = content[last_end:].strip()
        if trailing:
            chunks.append(self._make_chunk(trailing, file_path, node_type="trailing"))

        return chunks

    def _chunk_by_chars(self, content: str, file_path: str) -> List[Dict]:
        """
        Fallback: split at character boundaries.
        Tries to break at newlines when possible.
        """
        chunks = []
        start = 0

        while start < len(content):
            end = min(start + self.max_chars, len(content))

            # Try to break at a newline instead of mid-line
            if end < len(content):
                newline_pos = content.rfind("\n", start, end)
                if newline_pos > start:
                    end = newline_pos + 1

            chunk_text = content[start:end]
            if chunk_text.strip():
                chunks.append(self._make_chunk(chunk_text, file_path))

            start = end

        return chunks

    def chunk_file(self, file_path: str) -> List[Dict]:
        """
        Chunk a single file. Uses AST parsing when possible, character split otherwise.
        """
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        if not content.strip():
            return []

        ext = os.path.splitext(file_path)[1].lower()
        lang_name = LANGUAGE_MAP.get(ext)

        if lang_name:
            ast_chunks = self._chunk_by_ast(content, file_path, lang_name)
            if ast_chunks:
                return ast_chunks

        # No grammar available or AST parse returned nothing -- character fallback
        return self._chunk_by_chars(content, file_path)
