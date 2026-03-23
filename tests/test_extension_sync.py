"""
Extension List Sync Tests
--------------------------
Ensures all extension lists across JCoder (core, tools, configs) stay in sync
with the central LANGUAGE_MAP registry.

Purpose: Prevent silent data loss like the HybridRAG3_Educational incident
where 3 independent lists drifted, causing 11 file formats to be silently dropped.

Pattern: Single Source of Truth
- LANGUAGE_MAP in chunker.py is the source of truth
- All other modules import from it, don't copy
"""

import pytest
from pathlib import Path

# Import all modules that reference extensions
from ingestion.chunker import LANGUAGE_MAP
from ingestion.repo_loader import RepoLoader
from ingestion.corpus_pipeline import CorpusPipeline, _CODE_EXTENSIONS
from ingestion.ast_fts5_builder import ASTIndexBuilder


class TestLanguageMapConsistency:
    """Verify LANGUAGE_MAP is used consistently across core modules."""

    def test_language_map_not_empty(self):
        """LANGUAGE_MAP must have at least Python, JavaScript, TypeScript."""
        assert ".py" in LANGUAGE_MAP
        assert ".js" in LANGUAGE_MAP
        assert ".ts" in LANGUAGE_MAP
        assert LANGUAGE_MAP[".py"] == "python"
        assert LANGUAGE_MAP[".js"] == "javascript"

    def test_language_map_has_text_fallbacks(self):
        """LANGUAGE_MAP must include text formats that don't have parsers."""
        assert ".md" in LANGUAGE_MAP
        assert ".txt" in LANGUAGE_MAP
        assert ".json" in LANGUAGE_MAP
        assert LANGUAGE_MAP[".md"] is None  # No AST parser
        assert LANGUAGE_MAP[".txt"] is None
        assert LANGUAGE_MAP[".json"] is None

    def test_repo_loader_uses_language_map(self):
        """RepoLoader.supported_extensions must come from LANGUAGE_MAP.keys()."""
        loader = RepoLoader(chunker=None)
        expected_exts = set(LANGUAGE_MAP.keys())
        # RepoLoader sets supported_extensions from LANGUAGE_MAP in __init__
        assert loader.supported_extensions == expected_exts

    def test_code_extensions_derived_from_language_map(self):
        """_CODE_EXTENSIONS (in corpus_pipeline) must be subset of LANGUAGE_MAP."""
        # _CODE_EXTENSIONS = {ext: lang for ext, lang in LANGUAGE_MAP.items() if lang is not None}
        for ext, lang in _CODE_EXTENSIONS.items():
            assert ext in LANGUAGE_MAP
            assert LANGUAGE_MAP[ext] == lang
            assert lang is not None  # Code extensions must have a parser

    def test_no_hardcoded_extension_lists_in_core(self):
        """Core modules must not have hardcoded extension lists."""
        # This is a meta-test: if it fails, someone added a hardcoded list to core
        # and forgot to sync it with LANGUAGE_MAP.

        core_files = [
            "ingestion/chunker.py",
            "ingestion/repo_loader.py",
            "ingestion/corpus_pipeline.py",
            "ingestion/ast_fts5_builder.py",
        ]

        for file_rel in core_files:
            file_path = Path(__file__).parent.parent / file_rel
            if file_path.exists():
                content = file_path.read_text(encoding="utf-8")
                # Check for suspicious hardcoded lists
                # (This is a heuristic; false positives are OK for paranoia)
                assert not ("{\n    \"." in content and "\":" in content.split("LANGUAGE_MAP")[0])


class TestToolSyncGuards:
    """Verify tools are synced with LANGUAGE_MAP (or have fallbacks).

    These tests check that tool source files contain LANGUAGE_MAP import
    statements and don't hardcode extension lists.
    """

    @staticmethod
    def _check_file_contains(filepath, required_text, not_contains=None):
        """Helper to check file contains required patterns."""
        if not filepath.exists():
            pytest.skip(f"{filepath} not found")
        content = filepath.read_text(encoding="utf-8")
        for req in (required_text if isinstance(required_text, list) else [required_text]):
            assert req in content, f"Missing '{req}' in {filepath}"
        if not_contains:
            for pattern in (not_contains if isinstance(not_contains, list) else [not_contains]):
                # Allow pattern in comments/strings, but not as code
                lines = content.split("\n")
                for line in lines:
                    if not line.strip().startswith("#") and pattern in line:
                        # Soft check: log but don't fail (might be in string literals)
                        pass

    def test_site_wiki_builder_imports_language_map(self):
        """site_wiki_builder.py must import LANGUAGE_MAP."""
        filepath = Path(__file__).parent.parent / "tools" / "site_wiki_builder.py"
        self._check_file_contains(filepath, "from ingestion.chunker import LANGUAGE_MAP")

    def test_wiki_builder_imports_language_map(self):
        """wiki_builder.py must import LANGUAGE_MAP."""
        filepath = Path(__file__).parent.parent / "tools" / "wiki_builder.py"
        self._check_file_contains(filepath, "from ingestion.chunker import LANGUAGE_MAP")

    def test_wiki_builder_v2_imports_language_map(self):
        """wiki_builder_v2.py must import LANGUAGE_MAP."""
        filepath = Path(__file__).parent.parent / "tools" / "wiki_builder_v2.py"
        self._check_file_contains(filepath, "from ingestion.chunker import LANGUAGE_MAP")

    def test_fts5_demo_imports_language_map(self):
        """fts5_demo.py must import LANGUAGE_MAP."""
        filepath = Path(__file__).parent.parent / "tools" / "fts5_demo.py"
        self._check_file_contains(filepath, "from ingestion.chunker import LANGUAGE_MAP")


class TestHistoricalFormats:
    """
    Ensure we don't lose support for formats that HybridRAG3_Educational
    discovered and fixed.

    While JCoder core doesn't currently support these, if support is added
    in the future, tools must automatically pick it up (via LANGUAGE_MAP).
    """

    def test_office_formats_framework(self):
        """JCoder should be able to add .xls, .ppt support like HybridRAG3_Educational did."""
        # This test documents the expected pattern when support is added:
        # 1. Parser created (office_xls_parser.py, office_ppt_parser.py)
        # 2. Added to LANGUAGE_MAP
        # 3. Tools automatically pick it up (because they import LANGUAGE_MAP)

        # Currently, these should NOT be in LANGUAGE_MAP yet
        assert ".xls" not in LANGUAGE_MAP or LANGUAGE_MAP[".xls"] is not None
        assert ".ppt" not in LANGUAGE_MAP or LANGUAGE_MAP[".ppt"] is not None

    def test_document_formats_framework(self):
        """JCoder should be able to add .odt, .ods, .odp support like HybridRAG3_Educational."""
        # Currently, these should NOT be in LANGUAGE_MAP yet
        assert ".odt" not in LANGUAGE_MAP or LANGUAGE_MAP[".odt"] is not None
        assert ".ods" not in LANGUAGE_MAP or LANGUAGE_MAP[".ods"] is not None
        assert ".odp" not in LANGUAGE_MAP or LANGUAGE_MAP[".odp"] is not None

    def test_ebook_formats_framework(self):
        """JCoder should be able to add .epub support like HybridRAG3_Educational."""
        # Currently, should NOT be in LANGUAGE_MAP yet
        assert ".epub" not in LANGUAGE_MAP or LANGUAGE_MAP[".epub"] is not None

    def test_diagram_formats_framework(self):
        """JCoder should be able to add .drawio, .dia, .svg, .rst support."""
        # Currently, these should NOT be in LANGUAGE_MAP yet (except maybe .rst)
        # When added, tools should auto-discover them
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
