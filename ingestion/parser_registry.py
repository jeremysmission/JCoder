"""
Parser Registry
-----------------
Single source of truth for file extension -> parser mapping.

All modules that need to know which file types are supported MUST
import from here. No hardcoded extension lists elsewhere.

Parser classes are loaded lazily to avoid import-time dependencies
on optional packages (xlrd, olefile, etc.).
"""

from typing import Any, Dict, Optional, Tuple

# Extension -> (module_path, class_name) for lazy loading.
# Parsers are only imported when first needed.
_PARSER_MAP = {
    # Legacy Office binary (97-2003)
    ".xls": ("ingestion.office_xls_parser", "XlsParser"),
    ".ppt": ("ingestion.office_ppt_parser", "PptParser"),
    # OpenDocument
    ".odt": ("ingestion.opendocument_parser", "OdtParser"),
    ".ods": ("ingestion.opendocument_parser", "OdsParser"),
    ".odp": ("ingestion.opendocument_parser", "OdpParser"),
    # eBook
    ".epub": ("ingestion.epub_parser", "EpubParser"),
    # Plain text formats (no special parsing needed)
    ".rst": ("ingestion.plain_text_parser", "PlainTextParser"),
    ".csv": ("ingestion.plain_text_parser", "PlainTextParser"),
    ".tsv": ("ingestion.plain_text_parser", "PlainTextParser"),
    ".svg": ("ingestion.plain_text_parser", "PlainTextParser"),
    ".drawio": ("ingestion.plain_text_parser", "PlainTextParser"),
    ".dia": ("ingestion.plain_text_parser", "PlainTextParser"),
    ".log": ("ingestion.plain_text_parser", "PlainTextParser"),
    ".ini": ("ingestion.plain_text_parser", "PlainTextParser"),
    ".cfg": ("ingestion.plain_text_parser", "PlainTextParser"),
    ".conf": ("ingestion.plain_text_parser", "PlainTextParser"),
    ".html": ("ingestion.plain_text_parser", "PlainTextParser"),
    ".htm": ("ingestion.plain_text_parser", "PlainTextParser"),
    ".xml": ("ingestion.plain_text_parser", "PlainTextParser"),
}

# All supported document extensions (union of parser registry + LANGUAGE_MAP)
DOCUMENT_EXTENSIONS = frozenset(_PARSER_MAP.keys())

# Cache of instantiated parsers
_parser_cache: Dict[str, Any] = {}


def get_parser(ext: str) -> Optional[Any]:
    """Get a parser instance for a file extension.

    Returns None if no parser is registered for the extension.
    Parser classes are instantiated once and cached.
    """
    ext = ext.lower()
    if ext not in _PARSER_MAP:
        return None

    if ext not in _parser_cache:
        module_path, class_name = _PARSER_MAP[ext]
        try:
            import importlib
            mod = importlib.import_module(module_path)
            cls = getattr(mod, class_name)
            _parser_cache[ext] = cls()
        except (ImportError, AttributeError) as exc:
            import logging
            logging.getLogger(__name__).warning(
                "Failed to load parser %s.%s for %s: %s",
                module_path, class_name, ext, exc,
            )
            return None
    return _parser_cache[ext]


def parse_file(file_path: str) -> Tuple[str, Dict[str, Any]]:
    """Parse a file using the appropriate registered parser.

    Returns (text, details_dict). If no parser is registered,
    returns empty string with an error in details.
    """
    import os
    ext = os.path.splitext(file_path)[1].lower()
    parser = get_parser(ext)
    if parser is None:
        return "", {"file": file_path, "error": f"No parser for {ext}"}
    return parser.parse_with_details(file_path)


def supported_extensions() -> frozenset:
    """Return all extensions that have a registered parser."""
    return DOCUMENT_EXTENSIONS
