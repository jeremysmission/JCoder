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
import importlib
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
    ".cc": "cpp",
    ".cxx": "cpp",
    ".cs": "c_sharp",
    ".go": "go",
    ".rs": "rust",
    ".rb": "ruby",
    ".php": "php",
    ".kt": "kotlin",
    # Config files -- no AST grammar, char-fallback only
    ".yaml": None,
    ".yml": None,
    # Documentation / harvested markdown -- char-fallback only
    ".md": None,
    ".txt": None,
    ".json": None,
    # Document formats -- parsed by ingestion.parser_registry, char-fallback here
    ".rst": None,
    ".csv": None,
    ".tsv": None,
    ".svg": None,
    ".drawio": None,
    ".dia": None,
    ".log": None,
    ".html": None,
    ".htm": None,
    ".xml": None,
    ".ini": None,
    ".cfg": None,
    ".conf": None,
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
        module = importlib.import_module(f"tree_sitter_{lang_name}")
        language = Language(module.language())
        parser = Parser(language)
        return language, parser
    except (ImportError, AttributeError, OSError):
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
        h = self._hash(text)
        return {
            "id": h,
            "content": text,
            "source_path": file_path,
            "source_type": os.path.splitext(file_path)[1],
            "ingestion_date": datetime.now(timezone.utc).isoformat(),
            "content_hash": h,
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
        content_bytes = content.encode("utf-8")
        tree = parser.parse(content_bytes)
        root = tree.root_node
        boundaries = BOUNDARY_NODES.get(lang_name, set())

        chunks = []
        last_end = 0

        for child in root.children:
            if child.type in boundaries:
                # Capture any code between the last boundary and this one (imports, etc.)
                preamble = content_bytes[last_end:child.start_byte].decode("utf-8", errors="replace").strip()
                if preamble:
                    chunks.append(self._make_chunk(
                        preamble, file_path, node_type="preamble",
                    ))

                node_text = content_bytes[child.start_byte:child.end_byte].decode("utf-8", errors="replace")
                if len(node_text) <= self.max_chars:
                    chunks.append(self._make_chunk(
                        node_text, file_path, node_type=child.type,
                    ))
                else:
                    # Oversized node -- character-split it
                    chunks.extend(self._chunk_by_chars(node_text, file_path))

                last_end = child.end_byte

        # Trailing code after last boundary
        trailing = content_bytes[last_end:].decode("utf-8", errors="replace").strip()
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
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
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


class DocumentChunker:
    """HybridRAG-compatible document chunker.

    Splits parsed document text using the same algorithm as HybridRAG3:
      - 1200-char chunks with 200-char overlap
      - Smart boundary detection: paragraph > sentence > newline > hard cut
      - Section heading prepend for context preservation

    Use this when producing indexes that must be queryable by HybridRAG3.
    """

    def __init__(self, chunk_size: int = 1200, overlap: int = 200,
                 max_heading_len: int = 160, heading_lookback: int = 2000):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.max_heading_len = max_heading_len
        self.heading_lookback = heading_lookback

    @staticmethod
    def _is_heading(line: str, max_len: int) -> bool:
        """Detect if a line is a section heading."""
        stripped = line.strip()
        if not stripped or len(stripped) > max_len:
            return False
        import re
        # ALL CAPS lines (>3 chars)
        if len(stripped) > 3 and stripped == stripped.upper() and stripped[0].isalpha():
            return True
        # Numbered sections: 1.2.3 Title
        if re.match(r"^\d+(\.\d+)*\s+", stripped):
            return True
        # Lines ending with colon
        if stripped.endswith(":"):
            return True
        return False

    def _find_nearest_heading(self, text: str, pos: int) -> str:
        """Search backwards from pos to find the nearest heading."""
        search_start = max(0, pos - self.heading_lookback)
        region = text[search_start:pos]
        lines = region.split("\n")
        for line in reversed(lines):
            if self._is_heading(line, self.max_heading_len):
                return line.strip()
        return ""

    def _find_break_point(self, text: str, start: int, end: int) -> int:
        """Find best break point using priority: paragraph > sentence > newline > hard."""
        if end >= len(text):
            return end
        # Paragraph break (double newline)
        para_pos = text.rfind("\n\n", start, end)
        if para_pos > start:
            return para_pos + 2
        # Sentence end
        sent_pos = text.rfind(". ", start, end)
        if sent_pos > start:
            return sent_pos + 2
        # Single newline
        nl_pos = text.rfind("\n", start, end)
        if nl_pos > start:
            return nl_pos + 1
        # Hard cut
        return end

    def chunk_text(self, text: str, file_path: str) -> List[Dict]:
        """Split document text into overlapping chunks with heading context."""
        if not text or not text.strip():
            return []

        chunks = []
        start = 0
        chunker = Chunker(max_chars=self.chunk_size)

        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            end = self._find_break_point(text, start, end)

            chunk_text = text[start:end].strip()
            if chunk_text:
                # Prepend section heading for context
                heading = self._find_nearest_heading(text, start)
                if heading:
                    chunk_text = f"[SECTION] {heading}\n{chunk_text}"

                chunk = chunker._make_chunk(chunk_text, file_path)
                chunk["chunk_strategy"] = "hybridrag_document"
                chunks.append(chunk)

            # Advance with overlap
            new_start = max(end - self.overlap, start + 1)
            start = new_start

        return chunks

    def chunk_file(self, file_path: str) -> List[Dict]:
        """Parse and chunk a document file using the parser registry."""
        from ingestion.parser_registry import get_parser, DOCUMENT_EXTENSIONS
        ext = os.path.splitext(file_path)[1].lower()

        if ext in DOCUMENT_EXTENSIONS:
            parser = get_parser(ext)
            if parser:
                text, details = parser.parse_with_details(file_path)
                if text.strip():
                    return self.chunk_text(text, file_path)

        # Fallback: read as text
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            return self.chunk_text(content, file_path)
        except Exception:
            return []
