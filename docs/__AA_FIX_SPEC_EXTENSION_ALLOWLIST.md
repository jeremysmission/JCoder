# CANONICAL FIX SPEC — Extension Allowlist Sync
# Authority: Direct operator order | Severity: CRITICAL
# Applies to: ALL repos that use file discovery, download, or indexing
# Date: 2026-03-22

---

## THE RULE

**There must be ONE source of truth for supported extensions: the parser registry.**

Every other list (config, downloader, fallback, indexer) must derive from the
registry at runtime. No hardcoded copies. No fallback lists that drift.

If the registry supports it, the downloader picks it up and the indexer processes it.
If the registry doesn't support it, nobody touches it. Period.

---

## THE PROBLEM (recap)

Three independent extension lists existed:

1. `registry.py` — parser registry (was correct but incomplete)
2. `config.py` — indexer config (was wrong, drifted)
3. `bulk_transfer_v2.py` — downloader fallback (was wrong, copied from config)

A sync test checked lists #2 and #3 against each other. Both were wrong the
same way, so the test passed green. Nobody checked against #1.

Result: 11 file formats silently skipped during download. No error. No log. Gone.

---

## THE FIX (mandatory for all repos)

### Architecture: Single Source of Truth

```
registry.py (ONLY source of truth)
    |
    +--> config.py imports from registry at runtime
    |
    +--> bulk_transfer_v2.py imports from registry at runtime
    |
    +--> tests verify all consumers match registry
```

### Step 1: Create missing parsers (new files, each < 500 LOC)

Each parser is its own module. No god classes. Each under 500 lines.

| Extension | Parser Class | File | Strategy |
|-----------|-------------|------|----------|
| `.xls` | `XlsParser` | `parsers/xls_parser.py` | OLE2 binary, use `olefile` or `xlrd` |
| `.ppt` | `PptParser` | `parsers/ppt_parser.py` | OLE2 binary, use `olefile` |
| `.odt` | `OdtParser` | `parsers/opendocument_parser.py` | ZIP + XML, stdlib `zipfile` + `xml.etree` |
| `.ods` | `OdsParser` | `parsers/opendocument_parser.py` | Same module, different class |
| `.odp` | `OdpParser` | `parsers/opendocument_parser.py` | Same module, different class |
| `.epub` | `EpubParser` | `parsers/epub_parser.py` | ZIP + XHTML, stdlib only |
| `.rst` | (PlainTextParser) | (existing) | Just register the extension |
| `.tsv` | (PlainTextParser) | (existing) | Just register the extension |
| `.drawio` | (PlainTextParser) | (existing) | XML-based, text-readable |
| `.svg` | (PlainTextParser) | (existing) | XML-based, text-readable |
| `.dia` | (PlainTextParser) | (existing) | XML-based, text-readable |

**Coding rules:**
- Each parser class < 500 LOC (comments don't count)
- No patches — full class rewrites if modifying existing parsers
- Portable — stdlib where possible, minimal dependencies
- Modular — one parser per concern, shared base class

### Step 2: Register in registry.py

Add to `ParserRegistry.__init__()`:

```python
# Legacy Office formats (OLE2 binary)
self.register(".xls", "Excel 97-2003", XlsParser)
self.register(".ppt", "PowerPoint 97-2003", PptParser)

# OpenDocument formats (ZIP + XML)
self.register(".odt", "OpenDocument Text", OdtParser)
self.register(".ods", "OpenDocument Spreadsheet", OdsParser)
self.register(".odp", "OpenDocument Presentation", OdpParser)

# eBook
self.register(".epub", "EPUB eBook", EpubParser)

# Plain text variants (already have PlainTextParser)
self.register(".rst", "reStructuredText", PlainTextParser)
self.register(".tsv", "Tab-Separated Values", PlainTextParser)
self.register(".drawio", "draw.io Diagram (XML)", PlainTextParser)
self.register(".svg", "SVG Vector Graphic", PlainTextParser)
self.register(".dia", "Dia Diagram (XML)", PlainTextParser)
```

### Step 3: Eliminate hardcoded extension lists

**config.py** — Replace the hardcoded `supported_extensions` list with:

```python
@dataclass
class IndexingConfig:
    @property
    def supported_extensions(self) -> list[str]:
        from src.parsers.registry import REGISTRY
        return REGISTRY.supported_extensions()
```

If a static default is needed for serialization, generate it from the registry
at class definition time, not by hand.

**bulk_transfer_v2.py** — The `_RAG_EXTENSIONS_FALLBACK` should be removed
entirely. Replace with:

```python
def _resolve_rag_extensions() -> set[str]:
    from src.parsers.registry import REGISTRY
    return set(REGISTRY.supported_extensions())

_RAG_EXTENSIONS: set[str] = _resolve_rag_extensions()
```

No fallback list. If the registry can't import, that's a real error — don't
silently degrade to a wrong list.

### Step 4: Remove the integrity gate

The `_REQUIRED_EXTENSIONS_GATE` and `_block_bugged_downloader()` in
`bulk_transfer_v2.py` were a temporary safety net. Once the fix is applied
and tests pass, remove them. The sync test (Step 5) replaces the gate
permanently.

### Step 5: Add sync tests (mandatory)

```python
# tests/test_extension_allowlist_sync.py

def test_config_matches_registry():
    """Config supported_extensions must equal registry."""
    from src.parsers.registry import REGISTRY
    from src.core.config import IndexingConfig
    cfg = IndexingConfig()
    assert set(cfg.supported_extensions) == set(REGISTRY.supported_extensions())

def test_downloader_matches_registry():
    """Downloader extension set must equal registry."""
    from src.parsers.registry import REGISTRY
    from src.tools.bulk_transfer_v2 import _RAG_EXTENSIONS
    assert _RAG_EXTENSIONS == set(REGISTRY.supported_extensions())

def test_critical_formats_registered():
    """The 11 previously-missing formats must have parsers."""
    from src.parsers.registry import REGISTRY
    critical = [".xls", ".ppt", ".odt", ".ods", ".odp", ".epub",
                ".rst", ".tsv", ".drawio", ".svg", ".dia"]
    for ext in critical:
        assert REGISTRY.get(ext) is not None, f"{ext} missing from registry"

def test_no_hardcoded_extension_lists():
    """Verify no file contains a hardcoded extension list that could drift."""
    # Grep for patterns like EXTENSIONS = [...] or _FALLBACK = {...}
    # that aren't importing from registry
    pass  # Implement as static analysis
```

### Step 6: File validation (recommended)

Add to `file_validator.py` or equivalent:

```python
OLE2_MAGIC = b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1'  # .xls, .ppt, .doc
ZIP_MAGIC = b'PK\x03\x04'                            # .odt, .ods, .odp, .epub

MIN_FILE_SIZES = {
    ".xls": 2048, ".ppt": 5000, ".doc": 2048,
    ".odt": 1500, ".ods": 1500, ".odp": 2000,
    ".epub": 1000,
}
```

---

## PORTING TO OTHER REPOS

Each repo should:

1. Check if it has file discovery, download, or indexing
2. If yes: adopt the SINGLE SOURCE OF TRUTH pattern above
3. Copy the parser modules (`xls_parser.py`, `ppt_parser.py`, etc.)
4. Update its registry equivalent to import from parsers
5. Remove any hardcoded extension lists
6. Add the sync tests
7. Run the downloader on existing source paths (incremental — dedup handles it)
8. Re-index
9. Validate queries hit content from new formats

**Do NOT:**
- Copy extension lists by hand into config files
- Create "fallback" lists that can drift
- Skip the sync tests
- Patch existing lists — rewrite the consumer to import from registry

---

## VERIFICATION CHECKLIST

After applying to any repo:

- [ ] All 11 extensions registered in parser registry
- [ ] Parser classes exist and are < 500 LOC each
- [ ] Config imports from registry (no hardcoded list)
- [ ] Downloader imports from registry (no hardcoded list)
- [ ] Integrity gate removed (replaced by sync tests)
- [ ] Sync tests pass
- [ ] Critical format test passes
- [ ] Downloader re-run picks up previously skipped files
- [ ] Re-index completed
- [ ] Query validation confirms new format content is retrievable
- [ ] War room COMPLETE note posted with file counts

---

## REFERENCE

- Original bug: HybridRAG3_Educational, 3 independent lists drifted
- Fix commit: 8f6046f (partial — parsers not created)
- Gate commit: 422f9fe (temporary safety net)
- Briefing: PRIORITY1_COORDINATOR_BRIEFING_PACKET.md (in all repo war rooms)
- This spec: C:\Users\jerem\AgentTeam\FIX_SPEC_EXTENSION_ALLOWLIST.md

Signed: Claude | Authority: Direct operator order | 2026-03-22
